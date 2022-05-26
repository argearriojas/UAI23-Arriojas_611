import hashlib
import json
from collections import Counter
import os
from joblib import Parallel, delayed, cpu_count
import pickle
import time
import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv, PendulumEnv, AcrobotEnv
from gym.wrappers import TimeLimit, TransformObservation, TransformReward
from gym import spaces, ActionWrapper, Env
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

SHORT_EXPERIMENTS = [
    # env_name,  nbins,  beta
    ('Pendulum',    12,     1),
    ('MountainCar',  8,     2),
    ('CartPole',     4,    10),
    ('Acrobot',      5,    15),
]

FULL_EXPERIMENTS = [
    # env_name,  nbins,  beta
    ('Pendulum',    16,     1),
    ('MountainCar', 12,     2),
    ('CartPole',     8,    10),
    ('Acrobot',     12,    25),
]
EXPERIMENTS = SHORT_EXPERIMENTS

class ExtendedCartPoleEnv(CartPoleEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if self.steps_beyond_done == 0:
            # need to make this an instantaneus reward drop when done
            reward = 0.

        return next_state, reward, done, info


class ExtendedMountainCarEnv(MountainCarEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if self.state[0] >= self.goal_position:
            # need to make this an instantaneus reward when done
            reward = 0.
        return next_state, reward, done, info


class ExtendedPendulum(PendulumEnv):
    def __init__(self, g=10):
        super().__init__(g)
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # have to deal with non-standard observation manipulation
    def _get_obs(self):
        th, thdot = self.state

        # make the angle periodic
        th = (th + np.pi) % (2 * np.pi) - np.pi

        return np.array([th, thdot], dtype=np.float32)


class ExtendedAcrobot(AcrobotEnv):
    def __init__(self):
        super().__init__()
        high = np.array([np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _get_ob(self):
        th1, th2, th1dot, th2dot = self.state
        th1 = (th1 + np.pi) % (2 * np.pi) - np.pi
        th2 = (th2 + np.pi) % (2 * np.pi) - np.pi

        return np.array([th1, th2, th1dot, th2dot], dtype=np.float32)


class DiscretizeObservation(TransformObservation):
    def __init__(self, env, nbins, lo_clip, hi_clip, indep=None):
        assert type(nbins) in [list, tuple, int, np.ndarray]

        nvars, = env.observation_space.shape
        if type(nbins) == int:
            nbins = (nbins, ) * nvars
        else:
            assert len(nbins) == nvars
            assert all([type(b) == int for b in nbins])
            nbins = tuple(nbins)

        if indep is None:
            # assume all variables are independent
            self.indep = np.array([1] * nvars, dtype=bool)
        else:
            assert len(indep) == nvars
            self.indep = np.array(indep, dtype=bool)

        self.nbins = nbins
        self.bounds = bounds = [np.linspace(l, h, n+1).tolist() for l, h, n in zip(lo_clip, hi_clip, nbins)]
        n = [1] + [np.prod(nbins[:i]) for i in range(1, nvars)]
        bin_edges = [b[1:-1] for b in bounds]
        def f(state):
            x = [np.digitize(s, e) for s, e in zip(state, bin_edges)]
            idx = sum([n[i] * x[i] for i in range(nvars)])
            return idx

        super().__init__(env, f)


class DiscretizeAction(ActionWrapper):
    def __init__(self, env: Env, nbins: int) -> None:
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Box)
        assert len(env.action_space.shape) == 1

        self.ndim_actions, = env.action_space.shape
        self.powers = [nbins ** (i-1) for i in range(self.ndim_actions, 0, -1)]

        low = env.action_space.low
        high = env.action_space.high
        self.action_mapping = np.linspace(low, high, nbins)
        self.action_space = spaces.Discrete(nbins ** self.ndim_actions)
    
    def action(self, action):
        
        a = action
        unwrapped_action = np.zeros((self.ndim_actions,), dtype=float)

        for i, p in enumerate(self.powers):
            idx, a = a // p, a % p
            unwrapped_action[i] = self.action_mapping[idx, i]

        return unwrapped_action


def get_model_estimation(env_name, nbins, n_tries=256, force=False):
    os.makedirs(env_name, exist_ok=True)

    env = get_environment(env_name, nbins)
    data = dict(
        bounds=env.bounds,
        nbins=env.nbins,
    )
    if hasattr(env, 'sticky_actions_n'):
        data['sticky_actions_n'] = env.unwrapped.sticky_actions_n
    hash = hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

    output_file = os.path.join(env_name, f"model_estimation_{hash}.pkl")
    if os.path.exists(output_file) and not force:
        print(f'loading previously computed model for {env_name} with bins {nbins}')
        with open(output_file, 'rb') as file:
            dynamics, rewards = pickle.load(file)
        return dynamics, rewards

    n_jobs = cpu_count()

    ss = SeedSequence()
    child_seeds = ss.spawn(n_jobs)
    rngs = [default_rng(s) for s in child_seeds]

    def work(nbins, rng):
        
        env = get_environment(env_name, nbins, max_episode_steps=200)
        env.reset()

        bounds = env.bounds
        nbins = env.nbins
        n_vars = len(bounds)

        cnts = Counter()
        rwds = Counter()

        def sweep_and_do(depth):
            if depth == 0:
                low, high = np.array([b[:2] for b in bounds]).T
                for action in range(env.action_space.n):
                    for _ in range(n_tries//n_jobs):
                        env.unwrapped.state = rng.uniform(low=low, high=high)
                        s = env.f(env.unwrapped.state)
                        ns, r, d, _ = env.step(action)

                        i = s * env.action_space.n + action
                        j = ns
                        cnts[(j, i)] += 1
                        rwds[(0, i)] += r

                        if d:
                            env.reset()

            else:
                for _ in range(nbins[depth-1]):
                    sweep_and_do(depth-1)
                    bounds[depth-1] = np.roll(bounds[depth-1], -1)
                bounds[depth-1] = np.roll(bounds[depth-1], -1)

        sweep_and_do(n_vars)

        state_space_size = np.prod(nbins)
        N = state_space_size
        M = state_space_size * env.action_space.n

        keys, data = list(zip(*cnts.items()))
        coords = list(zip(*keys))
        cnts = coo_matrix((data, coords), shape=(N, M), dtype=float).tocsc().sorted_indices()

        keys, data = list(zip(*rwds.items()))
        coords = list(zip(*keys))
        rwds = coo_matrix((data, coords), shape=(1, M), dtype=float).todense()

        return cnts, rwds

    # result = [work(nbins, rng) for rng in rngs]
    result = Parallel(n_jobs=n_jobs)(delayed(work)(nbins, rng) for rng in rngs)
    cnts_lst, rwds_lst = list(zip(*result))

    dynamics = sum(cnts_lst).tocsc().sorted_indices()
    rewards = sum(rwds_lst)

    for i, (start, end) in enumerate(zip(dynamics.indptr, dynamics.indptr[1:])):
        if len(dynamics.data[start:end]) > 0:
            col_cnts = dynamics.data[start:end].sum()
            rewards[0, i] = rewards[0, i] / col_cnts
            dynamics.data[start:end] = dynamics.data[start:end] / col_cnts

    with open(output_file, 'wb') as file:
        pickle.dump((dynamics, rewards), file)

    return dynamics, rewards


def get_environment(env_name, nbins, max_episode_steps=0, reward_offset=0):

    if env_name == 'CartPole':
        lo_clip = np.array([-2.5, -3.5, -0.25 , -2.5])
        hi_clip = np.array([ 2.5,  3.5,  0.25 ,  2.5])
        env = ExtendedCartPoleEnv()
    elif env_name == 'MountainCar':
        lo_clip = np.array([-1.2, -0.07])
        hi_clip = np.array([ 0.6,  0.07])
        env = ExtendedMountainCarEnv()
    elif env_name == 'Pendulum':
        lo_clip = np.array([-np.pi, -8.0])
        hi_clip = np.array([ np.pi,  8.0])
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=3)
    elif env_name == 'Pendulum5':
        lo_clip = np.array([-np.pi, -8.0])
        hi_clip = np.array([ np.pi,  8.0])
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=5)
    elif env_name == 'Acrobot':
        lo_clip = np.array([-np.pi, -np.pi, -12.566371, -28.274334])
        hi_clip = np.array([ np.pi,  np.pi,  12.566371,  28.274334])
        env = ExtendedAcrobot()
    else:
        raise ValueError(f'wrong environment name {env_name}')

    env = DiscretizeObservation(env, nbins, lo_clip, hi_clip)
    if reward_offset != 0:
        env = TransformReward(env, lambda r: r + reward_offset)
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps)

    return env


def test_policy(env, policy, render=True, fps=20, quiet=False, rng=None):

    if rng is not None:
        random_choice = rng.choice
    else:
        random_choice = np.random.choice

    while True:
        state = env.reset()

        done = False
        episode_reward = 0
        while not done:
            if render:
                if not quiet:
                    print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\r', flush=True)
                _ = env.render()
                time.sleep(1/fps)

            # Sample action from action probability distribution
            action = random_choice(env.action_space.n, p=policy[state])

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        if not quiet:
            print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\n', flush=True)

        if not render:
            return episode_reward

        user_input = input("Again? [y]/n: ")
        if user_input in ['n', 'no']:
            env.close()
            break


def compute_policies_for_plots(env_name, nbins_list=[4], beta_list=[1], bias_max_it=150):
    
    policy_filename_template = os.path.join(env_name, 'optimal_policy_beta{beta:04.1f}_{nbins_str}bins.npy')
    info_filename_template = os.path.join(env_name, 'info_beta{beta:04.1f}_{nbins_str}bins.pkl')

    for nbins in nbins_list:
        env = get_environment(env_name, nbins)
        nbins = env.nbins
        nbins_str = "-".join([f'{i:02d}' for i in nbins])
        if any([not os.path.exists(policy_filename_template.format(beta=beta, nbins_str=nbins_str)) for beta in beta_list]):
            print(f"{nbins=}", flush=True)
            dynamics, rewards = get_model_estimation(env_name, nbins=nbins, n_tries=512)
        else:
            continue
        for beta in beta_list:
            filename = policy_filename_template.format(beta=beta, nbins_str=nbins_str)
            if not os.path.exists(filename):
                print(f"{nbins=}, {beta=: 5.1f}", flush=True)
                rewards -= rewards.max()
                l, u, v, optimal_policy, optimal_dynamics, estimated_distribution, info = solve_biased_unconstrained(beta, dynamics, rewards, bias_max_it=bias_max_it, tolerance=1e-6)
                if info['iterations_completed'] < bias_max_it:
                    np.save(filename, optimal_policy)
                    with open(info_filename_template.format(beta=beta, nbins_str=nbins_str), 'wb') as file:
                        pickle.dump(info, file)
                else:
                    print(f'W: no solution found for {env_name} with {nbins=} and beta = {beta:.1f}')


def compute_per_iter_evaluation(env_name, nbins, beta, max_episode_steps=500, n_episodes=128, force=False):
    env = get_environment(env_name, nbins)
    nbins = env.nbins
    nbins_str = "-".join([f'{i:02d}' for i in nbins])
    info_filename = os.path.join(env_name, f'info_beta{beta:04.1f}_{nbins_str}bins.pkl')
    results_filename = os.path.join(env_name, f'results_beta{beta:04.1f}_{nbins_str}bins.pkl')

    print(f"{nbins=}, {beta=: 5.1f}", flush=True)

    if os.path.exists(results_filename) and not force:
        print("This result is already available")
        return

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)

    ncpu = cpu_count()

    ss = SeedSequence()
    child_seeds = ss.spawn(ncpu)
    rngs = [default_rng(s) for s in child_seeds]

    def work(policy, rng):
        env = get_environment(env_name, nbins=nbins, max_episode_steps=max_episode_steps)
        return [test_policy(env, policy, render=False, quiet=True, rng=rng) for _ in range(n_episodes//ncpu)]

    results = {}
    for i, policy in enumerate(info['policy_list']):
        print(f'evaluating iteration {i+1: 3d} / {info["iterations_completed"]}')
        results[i] = sum(Parallel(n_jobs=ncpu)(delayed(work)(policy, rng) for rng in rngs), [])

    with open(results_filename, 'wb') as file:
        pickle.dump(results, file)


def get_final_result_summary(env_name, nbins, beta):
    env = get_environment(env_name, nbins)
    nbins = env.nbins
    nbins_str = "-".join([f'{i:02d}' for i in nbins])
    info_filename = os.path.join(env_name, f'info_beta{beta:04.1f}_{nbins_str}bins.pkl')
    results_filename = os.path.join(env_name, f'results_beta{beta:04.1f}_{nbins_str}bins.pkl')

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)
    with open(results_filename, 'rb') as file:
        results = pickle.load(file)

    n = info['iterations_completed']
    x = np.array(results[n-1])
    m, s = x.mean(), x.std()
    l = len(x)
    print(f"{(env_name, nbins, beta)}: Finished in {n: 3d} iterations. Rewards = {m:.2f} ({s:.2f}) in {l} episodes.")
    print()


def load_and_test_policy(env_name, nbins, beta, iteration=-1, max_episode_steps=500):
    env = get_environment(env_name, nbins, max_episode_steps=max_episode_steps)
    nbins = env.nbins
    nbins_str = "-".join([f'{i:02d}' for i in nbins])
    info_filename = os.path.join(env_name, f'info_beta{beta:04.1f}_{nbins_str}bins.pkl')

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)

    policy = info['policy_list'][iteration]
    test_policy(env, policy, render=True, fps=60)


def plot_per_iter_evaluation(env_name, nbins, beta):
    env = get_environment(env_name, nbins)
    nbins = env.nbins
    nbins_str = "-".join([f'{i:02d}' for i in nbins])
    results_filename = os.path.join(env_name, f'results_beta{beta:04.1f}_{nbins_str}bins.pkl')
    with open(results_filename, 'rb') as file:
        results = pickle.load(file)

    x = np.array(list(results.keys()))
    r = np.array(list(results.values()))
    _ = plt.scatter(x, r.mean(axis=1), label=env_name)
    plt.show()

    
def compute_solution_and_test(env_name, nbins, beta, max_episode_steps=500, render=False):
    dynamics, rewards = get_model_estimation(env_name, nbins=nbins, n_tries=512)

    env = get_environment(env_name, nbins, max_episode_steps=max_episode_steps)

    rewards -= rewards.max()
    l, u, v, optimal_policy, optimal_dynamics, estimated_distribution, _ = solve_biased_unconstrained(beta, dynamics, rewards, bias_max_it=500)
    test_policy(env, optimal_policy, render=render, fps=60)


def generate_paper_data():
    for env_name, nbins, beta in EXPERIMENTS:
        compute_policies_for_plots(env_name, nbins_list=[nbins], beta_list=[beta], bias_max_it=150)
        compute_per_iter_evaluation(env_name, nbins=nbins, beta=beta, max_episode_steps=500, n_episodes=128)
        get_final_result_summary(env_name, nbins, beta)


def figure_1(w = 5):
    results_filename_template = os.path.join('{env_name}', 'results_beta{beta:04.1f}_{nbins_str}bins.pkl')
    plt.figure(dpi=150)
    for env_name, nbins, beta in EXPERIMENTS:
        env = get_environment(env_name, nbins)
        nbins = env.nbins
        nbins_str = "-".join([f'{i:02d}' for i in nbins])
        with open(results_filename_template.format(env_name=env_name, beta=beta, nbins_str=nbins_str), 'rb') as file:
            results = pickle.load(file)
        x = np.array(list(results.keys()))
        r = np.array(list(results.values()))
        y = r.mean(axis=1)
        offset, scale = y.min(), (y - y.min()).max()
        scale = scale if scale > 0 else 1.
        
        y = np.array([y[max(i - w + 1, 0) : i + 1].mean() for i in range(len(y))])
        y -= offset
        y /= scale

        _ = plt.plot(x, y, '-', label=env_name)
    plt.ylabel('Returns (normalized)')
    plt.xlabel('Biasing iteration')
    plt.title('Biasing algorithm improves solution performance')
    plt.legend()
    plt.show()


def solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000, estimate_distribution=False):
    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    # T = csc_matrix(np.diag(np.exp(beta * np.array(rewards).flatten())))
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()

    # left eigenvector
    u = np.matrix(np.ones((nS * nA, 1)))
    u_scale = np.sum(u)

    # right eigenvector
    v = np.matrix(np.ones((nS * nA, 1))) / (nS * nA)

    for i in range(eig_max_it):

        uk = (Mt).dot(u)
        lu = np.sum(uk) / u_scale
        rescale = 1. / np.sqrt(uk.max()*uk.min())
        uk = uk / lu * rescale
        u_scale *= rescale

        if estimate_distribution:
            vk = M.dot(v)
            lv = vk.sum()
            vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))/ beta).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        if estimate_distribution:
            mask = np.logical_and(vk > 0, v > 0)
            v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))/ beta).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        if estimate_distribution:
            v = vk
            if u_err <= 1e-8 and v_err <= 1e-8:
                break
        else:
            if u_err <= 1e-8:
                break

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    if estimate_distribution:
        v = v / v.sum()
        u = u / u.dot(v)

        estimated_distribution = np.array(np.multiply(u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()
    else:
        estimated_distribution = None

    return lu, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_biased_unconstrained(beta, prior_dynamics, rewards, prior_policy=None, target_dynamics=None, eig_max_it=10000, bias_max_it=200, ground_truth_policy=None, tolerance=1e-6):

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
    policy_list = []
    for i in range(1, bias_max_it+1):

        l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solve_unconstrained(beta, biased_dynamics, biased_rewards, prior_policy, eig_max_it=eig_max_it)
        policy_list.append(optimal_policy)
        if ground_truth_policy is not None:
            error_policy = kl_divergence(optimal_policy, ground_truth_policy, axis=1).max()
            error_policy_list.append(error_policy)

        kl_err = kl_divergence(optimal_dynamics, target_dynamics).max()
        error_dynamics_list.append(kl_err)
        if kl_err < tolerance:
            print(f'Solved in {i} iterations')
            break

        chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
        biased_dynamics = target_dynamics.multiply(1. / chi)
        biased_dynamics = biased_dynamics.multiply(1 / biased_dynamics.sum(axis=0))
        biased_rewards = rewards + kl_divergence(target_dynamics, biased_dynamics) / beta
        biased_rewards -= biased_rewards.max()

    else:
        print(f'Did not finish after {i} iterations')

    info = dict(
        error_dynamics_list=error_dynamics_list,
        error_policy_list=error_policy_list,
        policy_list=policy_list,
        iterations_completed=i,
    )
    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution, info


def get_Q_table(l, u, N, nS, nA):
    return N * np.log(l) + np.log(u).reshape((nS, nA))


def kl_divergence(dist_a, dist_b, axis=0):
    numer = csr_matrix(dist_a)
    denom = coo_matrix(dist_b)
    kldiv = denom.copy()
    numer = np.array(numer[denom.row, denom.col]).flatten()
    kldiv.data = np.log(numer / denom.data) * numer
    kldiv = kldiv.sum(axis=axis)

    return kldiv


def get_mdp_transition_matrix(transition_dynamics, policy):

    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA + np.array(list(range(nA)))).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)), (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows ,cols)), shape=(nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix



if __name__ == '__main__':
    generate_paper_data()
    figure_1(w=5)

    # load_and_test_policy('Pendulum', nbins=12, beta=1., iteration=-1, max_episode_steps=500)
    # load_and_test_policy('MountainCar', nbins=8, beta=2, iteration=-1, max_episode_steps=500)
    # load_and_test_policy('CartPole', nbins=4, beta=10, iteration=-1, max_episode_steps=500)
    # load_and_test_policy('Acrobot', nbins=5, beta=15, iteration=-1, max_episode_steps=500)

    # compute_solution_and_test('Pendulum', nbins=12, beta=1, max_episode_steps=500, render=True)
    # compute_solution_and_test('MountainCar', nbins=8, beta=2, max_episode_steps=500, render=True)
    # compute_solution_and_test('CartPole', nbins=4, beta=10, max_episode_steps=500, render=True)
    # compute_solution_and_test('Acrobot', nbins=12, beta=25, max_episode_steps=500, render=True)
