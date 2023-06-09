from windy_frozen_lake import *
from utils import evaluate_policy, solve_maxent_value_policy_iteration
from tqdm import tqdm

import os
import json
import pandas as pd
import seaborn as sns

from collections import defaultdict

class MyDefaultDict(dict):
    def __init__(self, default=lambda k: None):
        self.default = default
        super().__init__()
    def __missing__(self, key):
        v = self.default(key)
        self[key] = v
        return v


def model_free_evaluate(env, policy, n_episodes=60, rng=None):
    nA = env.action_space.n
    if rng is None:
        rng = np.random.default_rng()
    total_return = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = rng.choice(nA, p=policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward
    average_return = total_return / n_episodes
    return average_return


def collect_experience(env, n_episodes, exploration_policy=None, rng=None):
    nA = env.action_space.n
    if rng is None:
        rng = np.random.default_rng()
    if exploration_policy is None:
        exploration_policy = defaultdict(lambda: np.ones(nA) / nA)

    print("Collecting experience ...")
    experience = []
    for _ in tqdm(range(n_episodes), ncols=120):
        state, _ = env.reset()
        action = rng.choice(nA, p=exploration_policy[state])

        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = rng.choice(nA, p=exploration_policy[next_state])
            experience.append((state, action, reward, next_state, next_action))
            state, action = next_state, next_action

    return experience


def model_free(env, experience, beta, alpha, prior_policy=None, b_tnsor=None, d_table=None, offset=0, n_epochs=1, save_period=10000, rescale_period=5000):
    nA = env.action_space.n
    nS = env.observation_space.n

    if prior_policy is None:
        prior_policy = defaultdict(lambda: np.ones(nA) / nA)
    if b_tnsor is None:
        b_tnsor = defaultdict(lambda: 1)
    if d_table is None:
        d_table = defaultdict(lambda: 0)

    l_alpha = alpha / nS

    l_value = 1.
    u_table = defaultdict(lambda: np.ones(nA))
    v_table = defaultdict(lambda: np.ones(nA))
    state_freq = defaultdict(lambda: 0)

    steps_trained = 0
    step_list = []
    theta_list = []

    for epoch in tqdm(range(1, n_epochs+1)):
        for state, action, reward, next_state, next_action in experience:

            bias = b_tnsor[state, action, next_state]
            reward = reward + d_table[state, action] - offset

            exp_re = np.exp(reward * beta)

            state_freq[state] += 1
            ##### the right eigenvector update #####
            v_valu = v_table[next_state][next_action]
            v_prev = v_table[state][action]
            bayes_rule = prior_policy[next_state][next_action] * state_freq[next_state] / (prior_policy[state][action] * state_freq[state])
            v_valu = v_valu + alpha * (exp_re / l_value * v_prev * bayes_rule * bias - v_valu)
            v_table[next_state][next_action] = v_valu

            ##### the left eigenvector update #####
            u_valu = u_table[state][action]
            u_next = u_table[next_state][next_action]
            u_valu = u_valu + alpha * (exp_re / l_value * u_next * bias - u_valu)
            u_table[state][action] = u_valu

            ##### the eigenvalue update #####
            l_value = l_value + l_alpha * (exp_re * u_next / u_valu * bias - l_value)
            l_value = min(l_value, 1)

            steps_trained += 1

            if steps_trained % (save_period) == 0:
                theta = -np.log(l_value) / beta - offset
                step_list.append(steps_trained)
                theta_list.append(theta)

            if steps_trained % (rescale_period) == 0:
                v_sum = sum(sum(v_table.values()))
                for k in v_table.keys():
                    v_table[k] /= v_sum
                uv_sum = sum([(u_table[k]*v_table[k]).sum() for k in u_table.keys()])
                for k in u_table.keys():
                    u_table[k] /= uv_sum
    
    v_sum = sum(sum(v_table.values()))
    for k in v_table.keys():
        v_table[k] /= v_sum
    uv_sum = sum([(u_table[k]*v_table[k]).sum() for k in u_table.keys()])
    for k in u_table.keys():
        u_table[k] /= uv_sum
    
    chi = defaultdict(lambda: 1)
    distribution = defaultdict(lambda: 0)
    policy = defaultdict(lambda: np.ones(nA) / nA)
    for state in u_table.keys():
        p = prior_policy[state] * u_table[state]
        chi[state] = p.sum()
        policy[state] = p / chi[state]
        distribution[state] = (u_table[state] * v_table[state]).sum()

    return dict(
        rho = l_value,
        theta = theta,
        chi = chi,
        policy = policy,
        distribution = distribution,
        steps_trained = steps_trained,
        step_list = step_list,
        theta_list = theta_list,
    )


def model_free_fixed_point_iteration(env, experience, beta, alpha, prior_policy=None, dynamics_adj_fn=lambda k: 1., n_epochs=10, n_bias_iter=5, p_alpha=0.1):
    observed_transitions = set()
    for state, action, _, next_state, _ in experience:
        observed_transitions.add((state, action, next_state))

    dynamics_adj = MyDefaultDict(dynamics_adj_fn)
    e_psi = defaultdict(lambda: 1)
    e_logpsi = defaultdict(lambda: 0)
    b_tnsor = None
    d_table = None
    offset = 0
    val = 1.

    for bias_iter in range(n_bias_iter+1):

        result = model_free(env, experience, beta=beta, alpha=alpha, prior_policy=prior_policy, b_tnsor=b_tnsor, d_table=d_table, offset=offset, n_epochs=n_epochs, save_period=20_000)

        psi = defaultdict(lambda: 1)
        for s, c in result['chi'].items():
            psi[s] = 1 / c

        err = float('inf')
        while err > 1e-4:
            for state, action, _, next_state, _ in experience:
                e_psi[state, action] = e_psi[state, action] + p_alpha * (psi[next_state] * dynamics_adj[state, action, next_state] - e_psi[state, action])
                e_logpsi[state, action] = e_logpsi[state, action] + p_alpha * (np.log(psi[next_state]) * dynamics_adj[state, action, next_state] - e_logpsi[state, action])
            tmp = np.array(list(e_psi.values()))
            tmp.sort()
            err = (np.abs(val - tmp) / val).max()
            val = tmp
            print(f"{err:.5f}", end=' ')
        print()

        b_tnsor = defaultdict(lambda: 1)
        for state, action, next_state in observed_transitions:
            b_tnsor[state, action, next_state] = psi[next_state] / e_psi[state, action]

        d_table = defaultdict(lambda: 0)
        for state, action in e_psi.keys():
            d_table[state, action] = (np.log(e_psi[state, action]) - e_logpsi[state, action]) / beta

        offset = max(d_table.values())

        x = result['step_list']
        y = result['theta_list']
        plt.plot(x, y)
    plt.show()

    return result


def windy_cliff_experiment_model_free():
    beta=10
    max_steps=1000
    n_episodes=500
    n_epochs=15
    n_bias_iter = 5

    rng = np.random.default_rng()

    desc = np.array([ 'HGR', 'HFF', 'HFF', 'HFF', 'HSR', ], dtype='c')
    wind_field = generate_wind_field(desc, direction=np.pi*0.75, strength=0.7)

    env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-5)
    env = TimeLimit(env, max_episode_steps=max_steps)

    # for evaluation purposes
    dynamics, rewards = get_dynamics_and_rewards(env)

    prior_policy = np.ones((env.nS, env.nA)) / env.nA

    print("Collecting experience ...")
    experience = []
    for _ in tqdm(range(n_episodes), ncols=120):
        state, _ = env.reset()
        action = rng.choice(env.nA, p=prior_policy[state])
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = rng.choice(env.nA, p=prior_policy[next_state])
            experience.append((state, action, reward, next_state, next_action))
            state, action = next_state, next_action

    psi_vctr = np.ones(env.nS)
    e_psi_table = np.ones((env.nS, env.nA))
    e_logpsi_table = np.zeros((env.nS, env.nA))

    save_period = max_steps * n_episodes // 1000
    rescale_period = max_steps * 5

    l_value = np.exp(-beta)
    b_tnsor = np.ones((env.nS, env.nA, env.nS), dtype=float)
    d_table = np.zeros((env.nS, env.nA), dtype=float)
    offset = d_table.max()

    e_psi_table = np.ones((env.nS, env.nA))
    e_logpsi_table = np.zeros((env.nS, env.nA))

    plot_list = []
    for bias_iter in range(n_bias_iter+1):
        alpha = 0.005
        l_alpha = alpha / env.nS
        p_alpha = 0.1
        err = float('inf')
        print('Estimating biases ... ', end=' ')
        while err > 1e-4:
            a = e_psi_table.copy()
            b = e_logpsi_table.copy()
            for state, action, reward, next_state, next_action in experience:
                e_psi_table[state, action] = e_psi_table[state, action] + p_alpha * (psi_vctr[next_state] - e_psi_table[state, action])
                e_logpsi_table[state, action] = e_logpsi_table[state, action] + p_alpha * (np.log(psi_vctr[next_state]) - e_logpsi_table[state, action])
            err = np.max([np.abs((e_psi_table - a)/a).max(), np.abs((e_logpsi_table - b)/(b + np.finfo(float).eps)).max()])

        b_tnsor = psi_vctr[None, None] / e_psi_table[..., None]
        d_table = (np.log(e_psi_table) - e_logpsi_table) / beta
        offset = d_table.max()
        print('done.')

        # initialize
        u_table = np.matrix(np.ones((env.nS, env.nA)))
        v_table = np.matrix(np.ones((env.nS, env.nA))) / env.nS / env.nA
        state_freq = np.zeros(env.nS) + np.finfo(float).eps

        steps_trained = 0
        theta_list = [1.]
        step_list = [0]
        err_list = [1.]
        eval_list = [(0, evaluate_policy(dynamics,rewards, env.isd, prior_policy, max_steps)/max_steps)]
        err = float('inf')
        print('Learning from experience ...')
        for epoch in tqdm(range(1, n_epochs+1), ncols=120, desc=f"Bias iter #{bias_iter: 3d}/{n_bias_iter}"):
            a = u_table.copy()
            b = v_table.copy()
            for state, action, reward, next_state, next_action in experience:

                bias = b_tnsor[state, action, next_state]
                reward = reward + d_table[state, action] - offset

                exp_re = np.exp(reward * beta)

                ## right eigenvector is not necessary
                # state_freq[state] += 1
                # ##### the right eigenvector update #####
                # v_valu = v_table[next_state, next_action]
                # v_prev = v_table[state, action]
                # bayes_rule = prior_policy[next_state, next_action] * state_freq[next_state] / (prior_policy[state, action] * state_freq[state])
                # v_valu = v_valu + alpha * (exp_re / l_value * v_prev * bayes_rule * bias - v_valu)
                # v_table[next_state, next_action] = v_valu

                ##### the left eigenvector update #####
                u_valu = u_table[state, action]
                u_next = u_table[next_state, next_action]
                u_valu = u_valu + alpha * (exp_re / l_value * u_next * bias - u_valu)
                u_table[state, action] = u_valu

                ##### the eigenvalue update #####
                l_value = l_value + l_alpha * (exp_re * u_next / u_valu * bias - l_value)
                l_value = min(l_value, 1)

                steps_trained += 1

                if steps_trained % (save_period) == 0:
                    theta = -np.log(l_value) / beta - offset
                    err = np.max([np.abs((u_table - a)/a).max(), np.abs((v_table - b)/(b + np.finfo(float).eps)).max()])

                    step_list.append(steps_trained)
                    theta_list.append(theta)
                    err_list.append(err)

                if steps_trained % (rescale_period) == 0:
                    v_table /= v_table.sum()
                    u_table /= np.multiply(u_table, v_table).sum()

            policy = np.multiply(prior_policy, u_table)
            policy /= policy.sum(axis=1)
            evaluation = evaluate_policy(dynamics, rewards, env.isd, policy.A, max_steps) / max_steps
            eval_list.append((steps_trained, evaluation))

        psi_vctr = 1 / np.multiply(prior_policy, u_table.A).sum(axis=1)

        plot_list.append((step_list, theta_list, err_list, eval_list))

    return plot_list


def multiple_windy_cliff_experiment_model_free(n_replica=5, save_images=False):
    beta=10
    max_steps=1000
    desc = np.array([ 'HGR', 'HFF', 'HFF', 'HFF', 'HSR', ], dtype='c')
    wind_field = generate_wind_field(desc, direction=np.pi*0.75, strength=0.7)
    env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-5)
    env = TimeLimit(env, max_episode_steps=max_steps)
    dynamics, rewards = get_dynamics_and_rewards(env)
    ground_truth_policy = solve_maxent_value_policy_iteration(beta, dynamics, rewards, None, max_steps)['policy']
    gt_performance = evaluate_policy(dynamics, rewards, env.isd, ground_truth_policy, max_steps) / max_steps

    if not os.path.exists('data'):
        os.makedirs('data')
    filename = 'data/multiple_windy_cliff_experiment_model_free_data.json'
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            plot_list_list = json.load(file)[:n_replica]
    else:
        plot_list_list = []

    new_replica = n_replica - len(plot_list_list)
    if new_replica > 0:
        print(f"Started with {len(plot_list_list)} replicas available. Computing {new_replica} more ...")
        for replica in range(new_replica):
            plot_list = windy_cliff_experiment_model_free()
            plot_list_list.append(plot_list)

            with open(filename, 'w') as file:
                json.dump(plot_list_list, file)
    else:
        print(f"Using {len(plot_list_list)} replicas available ...")

    data = []
    for replica, plot_list in enumerate(plot_list_list):
        for bias_iter, (step_list, theta_list, err_list, eval_list) in enumerate(plot_list):
            for steps_trained, theta in zip(step_list, theta_list):
                if steps_trained % 100000 == 0:
                    data.append({"Biasing iteration":f"Iter #{bias_iter}" + ('' if bias_iter > 0 else " (no bias --> optimistic)"), "Steps trained":steps_trained, "Theta":theta})

    df = pd.DataFrame(data)
    _ = plt.clf()
    sns.lineplot(data=df, x='Steps trained', y='Theta', hue='Biasing iteration')
    plt.ylim(0.9, 1.05)
    plt.title('Free energy at different bias iterations')
    plt.show()

    data = []
    for replica, plot_list in enumerate(plot_list_list):
        for bias_iter, (step_list, theta_list, err_list, eval_list) in enumerate(plot_list):
            if bias_iter > 4:
                continue
            for steps_trained, mean_reward_per_step in eval_list:
                if bias_iter == 0:
                    label = "Rewards for optimistic policy"
                else:
                    label = f"Rewards for policy at biasing iteration #{bias_iter}"
                data.append({"Biasing iteration": label, "Steps trained":steps_trained, "Mean reward per step":mean_reward_per_step})

    df = pd.DataFrame(data)
    _ = plt.clf()
    sns.lineplot(data=df, x='Steps trained', y='Mean reward per step', hue='Biasing iteration')
    plt.hlines(gt_performance, df['Steps trained'].min(), df['Steps trained'].max(), 'k', '--', label='Rewards for optimal policy')
    plt.title('Policy performace at intermediate bias iterations')
    plt.legend()
    if save_images:
        if not os.path.exists('images'):
            os.mkdir('images')
        plt.savefig('images/Fig5.png', dpi=600)
        # plt.savefig('images/Fig5.pdf')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-replica', type=int, default=5)
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()

    multiple_windy_cliff_experiment_model_free(args.n_replica, args.save_images)
