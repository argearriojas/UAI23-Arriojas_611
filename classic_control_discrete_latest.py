import hashlib
import json
from collections import Counter
import os
from joblib import Parallel, delayed, cpu_count
import pickle
import numpy as np
from numpy.random import SeedSequence, default_rng
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv, PendulumEnv, AcrobotEnv
from gym.wrappers import TimeLimit, TransformObservation, TransformReward
from gym import spaces, ActionWrapper, Env
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix


class ExtendedCartPoleEnv(CartPoleEnv):
    def __init__(self):
        self.metadata["render_fps"] = 60
        super().__init__()

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)

        if self.steps_beyond_terminated == 0:
            # need to make this an instantaneus reward drop when terminated
            reward = 0.

        return next_state, reward, terminated, truncated, info


class ExtendedMountainCarEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0):
        self.metadata["render_fps"] = 60
        super().__init__(goal_velocity=goal_velocity)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)

        if self.state[0] >= self.goal_position:
            # need to make this an instantaneus reward when terminated
            reward = 0.
        return next_state, reward, terminated, truncated, info


class ExtendedPendulum(PendulumEnv):
    def __init__(self, g=10):
        self.metadata["render_fps"] = 60
        super().__init__(g=g)
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
        self.metadata["render_fps"] = 60
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
        self.observation_space = spaces.Discrete(np.prod(nbins))


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


def get_model_estimation(env_name, nbins, n_tries=128, force=False):
    os.makedirs(os.path.join('data', env_name), exist_ok=True)

    env = get_environment(env_name, nbins)
    data = dict(
        bounds=env.bounds,
        nbins=env.nbins,
    )
    if hasattr(env, 'sticky_actions_n'):
        data['sticky_actions_n'] = env.unwrapped.sticky_actions_n
    hash = hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

    output_file = os.path.join('data', env_name, f"model_estimation_{hash}.pkl")
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
                        ns, r, te, tr, _ = env.step(action)
                        d = te or tr

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
    elif env_name == 'Pendulum21':
        lo_clip = np.array([-np.pi, -8.0])
        hi_clip = np.array([ np.pi,  8.0])
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=21)
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


def test_policy(env, policy, render=True, quiet=False, rng=None):
    env_render_mode = env.unwrapped.render_mode
    if render:
        env.unwrapped.render_mode = 'human'

    if rng is not None:
        random_choice = rng.choice
    else:
        random_choice = np.random.choice

    while True:
        state, _ = env.reset()

        done = False
        episode_reward = 0
        while not done:
            if render:
                if not quiet:
                    print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\r', flush=True)

            # Sample action from action probability distribution
            action = random_choice(env.action_space.n, p=policy[state])

            # Apply the sampled action in our environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        if not quiet:
            print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\n', flush=True)

        if not render:
            return episode_reward

        user_input = input("Again? [y]/n/q: ")
        if user_input in ['n', 'no', 'q', 'quit', 'exit', 'c', 'cancel']:
            env.close()
            break

    env.unwrapped.render_mode = env_render_mode
    return user_input

