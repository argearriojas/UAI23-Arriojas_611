from glob import glob
import pickle
import sys
from contextlib import closing
import itertools

import math
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import pandas as pd

from gym import Env, spaces
from gym.utils import seeding

from gym import utils
from gym.wrappers import TimeLimit

import matplotlib.pyplot as plt
import seaborn as sns
from six import StringIO


def windy_cliff_experiment(beta = 20, max_steps=1000):

    desc = np.array([ 'HGR', 'HFF', 'HFF', 'HFF', 'HSR', ], dtype='c')
    wind_field = generate_wind_field(desc, direction=np.pi*0.75, strength=0.7)

    env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-5)
    env = TimeLimit(env, max_episode_steps=max_steps)

    dynamics, rewards = get_dynamics_and_rewards(env)
    prior_policy = np.matrix(np.ones((env.nS, env.nA))) / env.nA

    _, _, _, optimistic_policy, _, optimistic_estimated_distribution = solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1000)
    optimistic_true_distribution = compute_policy_induced_distribution(dynamics, optimistic_policy, max_steps)

    ground_truth_policy = solve_maxent_value_policy_iteration(beta, env, prior_policy, max_steps)
    _, _, _, optimal_policy, _, optimal_estimated_distribution, biasing_info = solve_biased_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1000, ground_truth_policy=ground_truth_policy, evaluate=True, env=env)
    optimal_true_distribution = compute_policy_induced_distribution(dynamics, optimal_policy, max_steps)

    fig = plot_dist(desc,
        wind_field, optimistic_policy, None,
        None, optimal_policy, None,
        titles=[
            'Windy cliff', f'Optimistic policy', None,
            None, f'Optimal policy', None,
        ], ncols=3, show_plot=False)

    axes = fig.get_axes()
    for axis, estimated_distribution, true_distribution in zip(
            [axes[2], axes[5]],
            [optimistic_estimated_distribution, optimal_estimated_distribution],
            [optimistic_true_distribution, optimal_true_distribution]
        ):
        x = np.array(list(range(env.nS)), dtype=float)
        axis.bar(x - 0.20, estimated_distribution, width=0.3, alpha=1, label='Estimated')
        axis.bar(x + 0.20, true_distribution, width=0.3, alpha=1, label='True')
        # axis.set_title('State distribution\ncomparison')
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        # axis.legend()
    # axes[2].legend(loc=2, ncol=2, bbox_to_anchor=(0., 1.15), prop={'size': 14})
    axes[2].legend(loc=2, ncol=1, bbox_to_anchor=(0., 1.30), prop={'size': 16})
    fig.delaxes(axes[3])
    plt.show()

    for label, data in biasing_info.items():
        print(label)
        print(data)
        if type(data) == list and len(data) > 0:
            plt.plot(data, label=label)
            plt.title(label)
            plt.show()


def random_maze_experiment(size=10, traps=10, wind_strength=2, beta=30, max_steps=1000, bias_max_it=200, random_seed=None, show_plots=True):

    np.random.seed(random_seed)
    env = generate_random_environment(size=size, n_traps=traps, rv_strength_params=(wind_strength, 2), max_steps=max_steps)
    np.random.seed()

    dynamics, rewards = get_dynamics_and_rewards(env)
    prior_policy = np.matrix(np.ones((env.nS, env.nA))) / env.nA

    ground_truth_policy = solve_maxent_value_policy_iteration(beta, env, prior_policy, max_it=1000)

    _, _, _, optimal_policy, _, _, info = solve_biased_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1000, bias_max_it=bias_max_it, ground_truth_policy=ground_truth_policy, quiet=not show_plots)
    optimal_true_distribution = compute_policy_induced_distribution(dynamics, optimal_policy, max_steps)

    error_policy = kl_divergence(optimal_policy, ground_truth_policy, axis=1).max()

    if show_plots:
        plot_dist(env.desc,
            env.unwrapped.wind_field, optimal_policy, optimal_true_distribution,
            titles=[
                'Random windy maze',
                f'Policy computed w/ biased unconstrained inference\nKL divergence wrt. ground truth = {error_policy:.4e}\n'+fr'$\beta$ = {beta}',
                'Induced distribution'
            ]
        )
    else:
        return info


def random_maze_n_iters(size=10, traps=10, wind_strength=2, beta=30, bias_max_it=200):
    env = generate_random_environment(size=size, n_traps=traps, rv_strength_params=(wind_strength, 2))
    dynamics, rewards = get_dynamics_and_rewards(env)
    return solve_biased_unconstrained(beta, dynamics, rewards, eig_max_it=1000, bias_max_it=bias_max_it, quiet=True)[-1]['iterations_completed']


def plot_dkl_evolution_for_random_maze_experiment(n=10):

    fig, axes = plt.subplots(1, 2, sharey=True)

    for _ in range(n):
        info = random_maze_experiment(show_plots=False)
        y, z = info['error_policy'], info['error_dynamics']
        _ = axes[0].plot(y)
        _ = axes[1].plot(z)
    
    axes[0].set_title('Policy KL divergence')
    axes[1].set_title('Dynamics KL divergence')
    
    axes[0].set_yscale('log')
    axes[0].set_ylabel('KL Divergence')

    for axis in axes:
        axis.set_xlabel('iteration #')
        axis.grid(which='both', axis='y', alpha=0.2)

    fig.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.show()


def failure_mode_experiment(beta=15, wind_strength=0.5, max_steps=1000):
    desc = np.array([
        "HRRG",
        "FRHR",
        "FHHR",
        "FRRS",
    ], dtype='c')
    wind_field = generate_wind_field(desc, direction=1.5 * np.pi, strength=wind_strength)

    prior_env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-2, action_success_prob=[0.99, 0.99, 0.99, 0.99])
    prior_env = TimeLimit(prior_env, max_episode_steps=max_steps)

    prior_dyn, rewards = get_dynamics_and_rewards(prior_env)
    prior_pol = np.matrix(np.ones((prior_env.nS, prior_env.nA))) / prior_env.nA

    # introduce a failure in which action 'up' is no longer effective
    target_env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-2, action_success_prob=[0.99, 0.99, 0.99, 0.0001])
    target_env = TimeLimit(target_env, max_episode_steps=max_steps)

    target_dyn, rewards = get_dynamics_and_rewards(target_env)

    # first we solve the initial problem where all actions work
    _, _, _, policy, _, _, _ = solve_biased_unconstrained(beta, prior_dyn, rewards, prior_pol, target_dynamics=prior_dyn)
    healthy_agent_distribution = compute_policy_induced_distribution(prior_dyn, policy, max_steps, prior_env.isd)
    failing_agent_distribution = compute_policy_induced_distribution(target_dyn, policy, max_steps, target_env.isd)
    healthy_agent_evaluation = evaluate_policy(prior_env, policy, max_steps)
    failing_agent_evaluation = evaluate_policy(target_env, policy, max_steps)

    # now we consider the prior dynamics to solve the case where action 'up' has been suppressed
    _, _, _, policy, _, _, _ = solve_biased_unconstrained(beta, prior_dyn, rewards, prior_pol, target_dynamics=target_dyn)
    failing_agent_alt_distribution = compute_policy_induced_distribution(target_dyn, policy, max_steps, target_env.isd)
    failing_agent_alt_evaluation = evaluate_policy(target_env, policy, max_steps)

    plot_dist(desc,
        wind_field, healthy_agent_distribution, failing_agent_distribution, failing_agent_alt_distribution,
        titles=[
            f"Maze Layout\nHorizon = {max_steps} steps",
            f"Solution with\nall actions available\nMean rewards = {healthy_agent_evaluation:.2f}",
            f"Failure introduced breaks solution\nMean rewards = {failing_agent_evaluation:.2f}",
            f"Solution with action\n'up' suppressed\nMean rewards = {failing_agent_alt_evaluation:.2f}"
        ],
        ncols=2
    )


def solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000):
    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = np.matrix(np.diag(np.exp(beta * np.array(rewards).flatten())))
    # The twisted matrix (biased problem)
    M = P.dot(T)

    # left eigenvector
    u = np.matrix(np.ones((1, nS * nA)))
    # right eigenvector
    v = np.matrix(np.ones((nS * nA, 1)))

    for i in range(eig_max_it):

        uk = u.dot(M)
        lu = np.linalg.norm(uk)
        uk = uk / lu

        vk = M.dot(v)
        lv = np.linalg.norm(vk)
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))/ beta).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))/ beta).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk

        if u_err <= 1e-8 and v_err <= 1e-8:
            break

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    optimal_dynamics = dynamics.multiply(chi)
    optimal_dynamics = optimal_dynamics.multiply(1 / optimal_dynamics.sum(axis=0)).tocsr()

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return lu, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_biased_unconstrained(beta, prior_dynamics, rewards, prior_policy=None, target_dynamics=None, eig_max_it=10000, bias_max_it=200, ground_truth_policy=None, evaluate=False, env=None, quiet=False):

    nS, nSnA = prior_dynamics.shape
    nA = nSnA // nS

    if prior_policy is None:
        prior_policy = np.matrix(np.ones((nS, nA))) / nA

    if target_dynamics is None:
        target_dynamics = prior_dynamics

    ### initialization ###
    biased_dynamics = prior_dynamics.copy()
    biased_rewards = rewards

    error_policy_list = []
    error_dynamics_list = []
    evaluation_list = []
    for i in range(bias_max_it):

        l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solve_unconstrained(beta, biased_dynamics, biased_rewards, prior_policy, eig_max_it=eig_max_it)
        if evaluate:
            assert env is not None, "Must provide environment for policy evaluation"
            evaluation = evaluate_policy(env, optimal_policy, env._max_episode_steps)
            evaluation_list.append(evaluation)
        if ground_truth_policy is not None:
            error_policy = kl_divergence(optimal_policy, ground_truth_policy, axis=1).sum()
            error_policy_list.append(error_policy)

        kl_err = kl_divergence(optimal_dynamics, target_dynamics, axis=0)
        error_dynamics_list.append(kl_err.sum())
        if kl_err.max() < 1e-6:
            if not quiet:
                print(f'Solved in {i} iterations')
            break

        chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
        biased_dynamics = target_dynamics.multiply(1. / chi)
        biased_dynamics = biased_dynamics.multiply(1 / biased_dynamics.sum(axis=0))
        biased_rewards = rewards + kl_divergence(target_dynamics, biased_dynamics) / beta
        biased_rewards -= biased_rewards.max()
    else:
        if not quiet:
            print(f'Did not finish after {i} iterations')

    info = dict(
        error_policy=error_policy_list,
        error_dynamics=error_dynamics_list,
        evaluation=evaluation_list,
        iterations_completed=i,
    )
    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution, info


def solve_maxent_value_policy_iteration(beta, env, prior_policy, max_it):
    dynamics, rewards = get_dynamics_and_rewards(env)
    rewards = rewards.reshape((env.nS, env.nA))

    prior_policy = np.matrix(prior_policy)
    policy = np.matrix(prior_policy)

    log_policy_over_prior = np.log(policy / prior_policy)

    Q = np.matrix(np.zeros((env.nS, env.nA)))
    V = np.matrix(np.zeros((env.nS, 1)))
    for k in range(1, max_it+1):
        Q = rewards + dynamics.multiply(V).sum(axis=0).reshape((env.nS, env.nA))
        V = np.multiply(policy, (Q - log_policy_over_prior / beta)).sum(axis=1)

        if k % 30 == 0:
            prev_policy = policy.copy()
            policy = np.multiply(np.exp(beta * (Q - V)), prior_policy)
            policy = (policy / policy.sum(axis=1))

            err = kl_divergence(policy, prev_policy, axis=1).max()
            if err < 1e-8:
                break
            
            log_policy_over_prior = np.array(np.zeros_like(policy)).flatten()
            mask = np.array(policy).flatten() > 0
            log_policy_over_prior[mask] = np.array(np.log(np.array(policy).flatten()[mask] / np.array(prior_policy).flatten()[mask]))
            log_policy_over_prior = np.matrix(log_policy_over_prior.reshape((env.nS, env.nA)))

    return np.array(policy)


def kl_divergence(dist_a, dist_b, axis=0):
    numer = csr_matrix(dist_a)
    denom = coo_matrix(dist_b)
    kldiv = denom.copy()
    numer = np.array(numer[denom.row, denom.col]).flatten()
    kldiv.data = np.log(numer / denom.data) * numer
    kldiv = kldiv.sum(axis=axis)

    return kldiv


def compute_policy_induced_distribution(dynamics, policy, steps, isd=None):
    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    mdp_generator = get_mdp_transition_matrix(dynamics, policy)

    if isd is not None:
        x = np.multiply(np.matrix(isd).T, policy).flatten().T
    else:
        x = np.matrix(np.ones((nS * nA, 1))) / nS / nA

    for _ in range(steps):
        x = mdp_generator.dot(x)

    return np.array(x).reshape((nS, nA)).sum(axis=1)


def _evaluate_policy(dynamics, rewards, isd, policy, steps, full_value_function=False):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    Q = np.matrix(np.zeros((1, nS * nA)))
    V = np.matrix(np.zeros((nS, 1)))
    for _ in range(steps):
        Q = rewards + dynamics.multiply(V).sum(axis=0)
        V = np.matrix((policy * (np.array(Q).reshape((nS, nA)))).sum(axis=1)).T

    if full_value_function:
        return V
    else:
        return (np.array(V).flatten() * isd).sum()


def evaluate_policy(env, policy, steps, full_value_function=False):

    dynamics, rewards = get_dynamics_and_rewards(env)
    return _evaluate_policy(dynamics, rewards, env.isd, policy, steps, full_value_function)


def get_mdp_transition_matrix(transition_dynamics, policy):

    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA + np.array(list(range(nA)))).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)), (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows ,cols)), shape=(nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix


def get_dynamics_and_rewards(env):

    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    assert (dynamics.sum(axis=0).round(15) == 1.).all()

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)), shape=shape).sum(axis=0)

    return dynamics, rewards


def generate_random_environment(size=10, n_traps=10, rv_strength_params=(2, 2), max_steps=1000):

    admissible = False
    while not admissible:
        #create base layout
        desc = np.array(['F' * size]*size, dtype='c')

        desc[-2, 1] = b'S'
        desc[1, -2] = b'G'

        #place traps randomly
        chosen_idx = np.random.choice((desc == b'F').sum(), n_traps, replace=False)
        desc[tuple(row[chosen_idx] for row in np.where(desc == b'F'))] = b'H'

        #check that start and goal are not surrounded by traps
        admissible = True
        for s in [b'S', b'G']:
            x = np.argwhere(desc == s)
            c = 0
            for di, dj in itertools.product((-1, 0, 1), (-1, 0, 1)):
                i, j = (x + [di, dj])[0]
                c += int(desc[i, j] == b'H')
            if c > 0:
                #found at least one trap in the neighborhood of either S or G
                #try again
                admissible = False

    #generate wind
    wind_field = generate_wind_field(desc, rv_strength_params=rv_strength_params)

    env_src = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-2)
    env = TimeLimit(env_src, max_episode_steps=max_steps)
    
    return env


def plot_dist(desc, *paths_list, ncols=4, filename=None, titles=None, main_title=None, figsize=None, show_values=False, show_plot=True, symbols_in_color = True, symbol_size=180):
    desc = np.asarray(desc, dtype='c')

    if len(paths_list) == 0:
        paths_list = [desc]
        axes = [plt.gca()]
    elif len(paths_list) == 1:
        fig = plt.figure(figsize=figsize)
        axes = [plt.gca()]
    elif len(paths_list) > 1:
        n_axes = len(paths_list)

        ncols = min(ncols, n_axes)
        nrows = (n_axes-1)//ncols+1

        figsize = (5*ncols, 5*nrows) if figsize is None else figsize
        fig, axes = plt.subplots(nrows, ncols, sharey=False, figsize=figsize)
        axes = axes.ravel()
    else:
        raise ValueError("Missing required parameter: path")

    if titles is not None:
        assert type(titles) == list
        assert len(titles) == len(paths_list)
    else:
        titles = [None] * len(paths_list)

    for axi, paths, title in zip(axes, paths_list, titles):
        if paths is None:
            # fig.delaxes(axi)
            continue
        draw_paths(desc, axi, paths, title, show_values, symbols_in_color, symbol_size)

    if main_title is not None:
        plt.suptitle(main_title)
    if filename is not None:
        plt.savefig(filename, dpi=300)
        return plt.gcf()
    elif show_plot:
        plt.show()
    else:
        return plt.gcf()


def draw_paths(desc, axi, paths, title=None, show_values=False, symbols_in_color = True, symbol_size=120):
    if paths is None:
        return
    nrow, ncol = desc.shape
    nsta = nrow * ncol
    out = np.ones(desc.shape + (3,), dtype=float)

    show_whole_maze = (desc.shape == paths.shape) and (desc == paths).all()
    if paths.shape in [desc.shape, (nsta,)] and not show_whole_maze:
        paths = paths - paths.min()
        if paths.max() > 0:
            paths = paths / paths.max()
        paths = paths.reshape(desc.shape)

        # Path: blue
        out[:, :, 0] = out[:, :, 1] = 1 - paths

    out = add_layout(desc, out)

    axi.imshow(out)

    # show symbols for some special states
    axi.scatter(*np.argwhere(desc.T == b'S').T, color='#00CD00' if symbols_in_color else 'k', s=symbol_size, marker='o')
    axi.scatter(*np.argwhere(desc.T == b'G').T, color='#E6CD00' if symbols_in_color else 'k', s=symbol_size, marker='*')
    axi.scatter(*np.argwhere(desc.T == b'H').T, color='#E60000' if symbols_in_color else 'k', s=symbol_size, marker='X')
    axi.scatter(*np.argwhere(desc.T == b'C').T, color='#FF8000' if symbols_in_color else 'k', s=symbol_size, marker='D')
    axi.scatter(*np.argwhere(desc.T == b'N').T, color='#808080' if symbols_in_color else 'k', s=symbol_size, marker=6)

    if len(paths.shape) == 2 and paths.shape[0] == nsta:
        # looks like a policy, lets try to illustrate it with arrows
        # axi.scatter(*np.argwhere(desc.T == b'F').T, color='#FFFFFF', s=10)

        nact = paths.shape[1]

        if nact in [2, 3]:
            direction = ['left', 'right', 'stay']
        elif nact in [4, 5]:
            direction = ['left', 'down', 'right', 'up', 'stay']
        elif nact in [8, 9]:
            direction = ['left', 'down', 'right', 'up', 'stay', 'leftdown', 'downright', 'rightup', 'upleft']
        else:
            raise NotImplementedError

        for state, row in enumerate(paths):
            for action, prob in enumerate(row):
                action_str = direction[action]
                if action_str == 'stay':
                    continue
                if action_str == 'left':
                    d_x, d_y = -prob, 0
                if action_str == 'down':
                    d_x, d_y = 0, prob
                if action_str == 'right':
                    d_x, d_y = prob, 0
                if action_str == 'up':
                    d_x, d_y = 0, -prob
                if action_str == 'leftdown':
                    d_x, d_y = -prob / np.sqrt(2), prob / np.sqrt(2)
                if action_str == 'downright':
                    d_x, d_y = prob / np.sqrt(2), prob / np.sqrt(2)
                if action_str == 'rightup':
                    d_x, d_y = prob / np.sqrt(2), -prob / np.sqrt(2)
                if action_str == 'upleft':
                    d_x, d_y = -prob / np.sqrt(2), -prob / np.sqrt(2)
                if desc[state // ncol, state % ncol] not in [b'W', b'G', b'H']:
                    axi.arrow(state % ncol, state // ncol, d_x*0.4, d_y*0.4,
                             width=0.001, head_width=0.2*prob, head_length=0.2*prob,
                             fc='k', ec='k')

    elif paths.shape == desc.shape and show_values:
        for i, row in enumerate(paths):
            for j, value in enumerate(row):
                # if desc[state // ncol, state % ncol] not in [b'W', b'G']:
                if value != 0:
                    axi.text(j-0.4, i-0.15, f"{value:.2f}", c='k', fontsize=10.)

    elif paths.shape == (2, nrow, ncol):
        # this is the signature for a force field. Let's plot this with arrows
        dx = np.cos(paths[0]) * paths[1] * 0.4
        dy = np.sin(paths[0]) * paths[1] * 0.4

        for row in range(nrow):
            for col in range(ncol):
                size = paths[1, row, col]
                axi.arrow(col, row, dx[row, col], dy[row, col], width=0.001, head_width=0.15*size, head_length=0.15*size, fc='k', ec='k')

    if title is not None:
        axi.set_title(title)

    axi.set_xlim(-0.5, ncol - 0.5)
    axi.set_ylim(nrow - 0.5, -0.5)
    axi.get_xaxis().set_visible(False)
    axi.get_yaxis().set_visible(False)


def add_layout(desc, out):

    walls = (desc == b'W')

    # Walls: black
    out[walls] = [0, 0, 0]

    return out



def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})


class WindyFrozenLake(DiscreteEnv):

    def __init__(
            self, desc=None, map_name="4x4", n_action=4,
            cyclic_mode=True,
            goal_attractor=0.,
            max_reward=0., min_reward=-1.5, step_penalization=1.,
            restart_tiles=b'H',
            reward_dict=None,
            diagonal_cost=np.sqrt(2),
            wind_field=None,
            action_success_prob=[1.] * 9,
            ):

        self.reward_dict = {
            b'H': min_reward,
            b'G': max_reward,
            b'C': - step_penalization / 2.,
            b'N': - step_penalization * 1.5,
        }

        if reward_dict is not None:
            self.reward_dict.update(reward_dict)

        goal_attractor = float(goal_attractor)

        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (min_reward, max_reward)

        if wind_field is None:
            wind_field = np.zeros((2, nrow, ncol))
        assert wind_field.shape == (2, nrow, ncol)
        self.wind_field = wind_field

        if n_action in [4, 5, 8, 9]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_stay = 4
            a_leftdown = 5
            a_downright = 6
            a_rightup = 7
            a_upleft = 8
        else:
            raise NotImplementedError(f'n_action:{n_action}')

        all_actions = set(list(range(n_action)))
        self.n_state = n_state = nrow * ncol
        self.n_action = n_action

        isd = np.array(desc == b'S').astype('float64').ravel()
        if isd.sum() == 0:
            isd = np.array(desc == b'F').astype('float64').ravel()
        isd /= isd.sum()
        self.isd = isd

        transition_dynamics = {s : {a : [] for a in all_actions}
                               for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, action):
            if action == a_left:
                col = max(col - 1, 0)

            elif action == a_down:
                row = min(row + 1, nrow - 1)

            elif action == a_right:
                col = min(col + 1, ncol - 1)

            elif action == a_up:
                row = max(row - 1, 0)

            elif action == a_leftdown:
                col = max(col - 1, 0)
                row = min(row + 1, nrow - 1)

            elif action == a_downright:
                row = min(row + 1, nrow - 1)
                col = min(col + 1, ncol - 1)

            elif action == a_rightup:
                col = min(col + 1, ncol - 1)
                row = max(row - 1, 0)

            elif action == a_upleft:
                row = max(row - 1, 0)
                col = max(col - 1, 0)

            elif action == a_stay:
                pass

            else:
                raise ValueError("Invalid action provided")

            return (row, col)

        def compute_transition_dynamics(wind_action_probs, action_intended):

            restart = letter in restart_tiles and cyclic_mode

            is_in_goal = letter == b'G'

            is_diagonal_step = action_intended in [5, 6, 7, 8]
            diagonal_adjust = diagonal_cost if is_diagonal_step else 1.
            rew = self.reward_dict.get(letter, - step_penalization) * diagonal_adjust

            if restart:
                done = False
                for ini_state, start_prob in enumerate(isd):
                    if start_prob > 0.0:
                        sat_li.append((start_prob, ini_state, rew, done))

                return

            elif is_in_goal:
                p = goal_attractor
                if p > 0:
                    sat_li.append((p, state, rew, False))

                done = not cyclic_mode
                for ini_state, start_prob in enumerate(isd):
                    p = start_prob * (1 - goal_attractor)
                    if p > 0.0:
                        sat_li.append((p, ini_state, rew, done))

                return

            for wind_action, prob in enumerate(wind_action_probs):
                if prob == 0.:
                    continue

                # intermediate step due to wind
                # wind acts first
                introw, intcol = inc(row, col, wind_action)
                if desc[introw, intcol] == b'W':
                    introw, intcol = row, col
                intstate = to_s(introw, intcol)

                # execute intended action with probability step_success_prob
                newrow, newcol = inc(introw, intcol, action_intended)
                if desc[newrow, newcol] == b'W':
                    newrow, newcol = introw, intcol
                    
                newstate = to_s(newrow, newcol)

                done = False
                sat_li.append((prob * action_success_prob[action_intended], newstate, rew, done))
                if action_success_prob[action_intended] < 1:
                    sat_li.append((prob * (1. - action_success_prob[action_intended]), intstate, rew, done))

            return


        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)
                letter = desc[row, col]

                wind_angle, displacement_prob = wind_field[:, row, col]
                assert displacement_prob <= 1.
                h_component = np.sqrt(displacement_prob) * np.cos(wind_angle)
                v_component = np.sqrt(displacement_prob) * np.sin(wind_angle)

                if letter == b'F':
                    wind_action_probs = [
                        max(0., -h_component) ** 2,
                        max(0., v_component) ** 2,
                        max(0., h_component) ** 2,
                        max(0., -v_component) ** 2,
                    ]
                    wind_action_probs = [round(x, 12) for x in wind_action_probs]
                else:
                    wind_action_probs = [0., 0., 0., 0.]

                wind_action_probs.append(1. - sum(wind_action_probs))

                for action_intended in all_actions:
                    sat_li = transition_dynamics[state][action_intended]
                    compute_transition_dynamics(wind_action_probs, action_intended)

        super().__init__(n_state, n_action, transition_dynamics, isd)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        else:
            return None


def generate_wind_field(desc, direction=None, strength=None, direction_seed=None, rv_strength_params=(2, 2),
        direction_smoothness=5, strength_smoothness=1):

    assert len(rv_strength_params) == 2
    assert type(direction_smoothness) == int
    assert type(strength_smoothness) == int

    nrow, ncol = desc.shape

    data = np.zeros((2, nrow, ncol), dtype=float)

    if direction is not None:
        data[0, :, :] = direction
    else:
        data[0, :, :] = np.random.uniform(2 * np.pi) if direction_seed is None else direction_seed
        n = direction_smoothness
        for i in range(1, n + 1):
            for row in range(nrow):
                for col in range(ncol):
                    data[0, row, col] = (data[0, row, col] + data[0, row-1, col] + data[0, (row+1) % nrow, col] + data[0, row, col-1] + data[0, row, (col+1) % ncol]) / 5 + np.random.normal(0, 2 * np.pi * 0.3 / i)

    if strength is not None:
        assert strength <= 1.
        data[1, :, :] = strength
    else:
        data[1, :, :] = np.random.beta(rv_strength_params[0], rv_strength_params[1], size=(nrow, ncol))
        n = strength_smoothness
        for i in range(1, n + 1):
            for row in range(nrow):
                for col in range(ncol):
                    data[1, row, col] = (data[1, row, col] + data[1, row-1, col] + data[1, (row+1) % nrow, col] + data[1, row, col-1] + data[1, row, (col+1) % ncol]) / 5
                
    data[:, desc == b'W'] = 0
    data[:, desc == b'S'] = 0
    data[:, desc == b'G'] = 0
    data[:, desc == b'H'] = 0
    data[:, desc == b'R'] = 0

    return data


def generate_random_maze_n_iters_data(n_replicas=10, name="001"):

    params_tuples = (10, 20, 30), (0.10, 0.15, 0.20, 0.25), (1, 2, 4), (2, 5, 20, 50)
    total_combinations = math.prod(len(t) for t in params_tuples)

    params_list = []
    data = []
    for i, params in enumerate(itertools.product(*params_tuples)):
        print(f"{i+1: 4d}/{total_combinations}", params, end="\t", flush=True)
        size, trap_density, wind_strength, beta = params
        traps = int(np.round(size * trap_density))
        d = [0] * n_replicas
        for j in range(n_replicas):
            d[j] = random_maze_n_iters(size, traps, wind_strength, beta)
            print("#", end="", flush=True)
        print()

        params_list.append(params)
        data.append(d)

    with open(f"random_maze_n_iters_data_{name}.pkl", "wb") as file:
        pickle.dump((params_list, data), file)

def show_random_maze_n_iters_violinplots():
    params_list = []
    data = []
    for filename in glob('random_maze_n_iters_data*.pkl'):
        with open(filename, "rb") as file:
            p, d = pickle.load(file)
        params_list.extend(p)
        data.extend(d)

    keys = ("size", "trap density", "wind level", "beta")
    rows = []
    for p, d in zip(params_list, data):
        for v in d:
            r = dict(zip(keys, p))
            r['N biasing iterations'] = v
            rows.append(r)

    df = pd.DataFrame(rows)
    # print(pd.DataFrame({
    #     "n replicas": df.groupby(['size','trap density'])['N biasing iterations'].count(),
    #     "mean iterations": df.groupby(['size','trap density'])['N biasing iterations'].mean(),
    #     "max iterations": df.groupby(['size','trap density'])['N biasing iterations'].apply(lambda x: x.sort_values().tolist()[-3:]),
    # }).style.set_precision(2).to_latex())
    # len(df)
    
    g = sns.FacetGrid(df.loc[df['N biasing iterations'] < 50], row="size", col="wind level")
    g = g.map(sns.violinplot, "beta", "N biasing iterations", "trap density", inner=None, linewidth=0.1, split=False, bw=.5, scale="area")
    g.fig.set_size_inches(11.12, 6.82)
    g.fig.get_axes()[0].legend(title= 'trap density', loc='upper left')
    plt.ylim(1.5, 10)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    windy_cliff_experiment(beta=50)
    random_maze_experiment()
    plot_dkl_evolution_for_random_maze_experiment()
    failure_mode_experiment()

    # takes too long. Comment this by default
    # generate_random_maze_n_iters_data()
    # show_random_maze_n_iters_violinplots()

