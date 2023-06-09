from operator import xor
import sys
from contextlib import closing
from six import StringIO
import itertools
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import matplotlib.pyplot as plt

from gym import Env, spaces
from gym import utils as gym_utils
from gym.utils import seeding
from gym.wrappers import TimeLimit


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


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()


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
        return int(self.s), {'prob': self.isd[self.s]}

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, t, False, {"prob": p})


class WindyFrozenLake(DiscreteEnv):

    def __init__(
            self, desc=None, map_name=None, n_action=4,
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


        assert (desc is not None) ^ (map_name is not None), "Please provide either desc or map_name"
        if map_name is not None:
            if map_name in MAPS.keys():
                desc = MAPS[map_name]
            else:
                raise ValueError(f"map '{map_name}' is not available")

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

            is_in_goal = letter == b'G'

            is_diagonal_step = action_intended in [5, 6, 7, 8]
            diagonal_adjust = diagonal_cost if is_diagonal_step else 1.
            rew = self.reward_dict.get(letter, - step_penalization) * diagonal_adjust

            if letter in restart_tiles and cyclic_mode:
                done = False
                for ini_state, start_prob in enumerate(isd):
                    if start_prob > 0.0:
                        sat_li.append((start_prob, ini_state, rew, done))

                return
            elif (letter in restart_tiles or is_in_goal) and not cyclic_mode:
                done = True
                sat_li.append((1., state, rew, done))

                return

            elif is_in_goal:
                p = goal_attractor
                if p > 0:
                    sat_li.append((p, state, rew, False))

                done = False
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
        desc[row][col] = gym_utils.colorize(desc[row][col], "red", highlight=True)

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


def plot_dist(desc, *paths_list, ncols=4, filename=None, titles=None, main_title=None, figsize=None, show_values=False, show_plot=True, symbols_in_color = True, symbol_size=180, show_grid=False, do_offset=True, do_scale=True):
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
        draw_paths(desc, axi, paths, title, show_values, symbols_in_color, symbol_size, show_grid, do_offset, do_scale)

    if main_title is not None:
        plt.suptitle(main_title)
    if filename is not None:
        plt.savefig(filename, dpi=300)
        return plt.gcf()
    elif show_plot:
        plt.show()
    else:
        return plt.gcf()


def draw_paths(desc, axi, paths, title=None, show_values=False, symbols_in_color = True, symbol_size=120, show_grid = False, do_offset=True, do_scale=True):
    if paths is None:
        return
    nrow, ncol = desc.shape
    nsta = nrow * ncol
    out = np.ones(desc.shape + (3,), dtype=float)

    show_whole_maze = (desc.shape == paths.shape) and (desc == paths).all()
    if paths.shape in [desc.shape, (nsta,)] and not show_whole_maze:
        if do_offset:
            paths = paths - paths.min()
        if do_scale and paths.max() > 0:
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
    if show_grid:
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_xticklabels([])
        axi.set_yticklabels([])
        axi.set_xticks(np.arange(-.5, desc.shape[1], 1), minor=True)
        axi.set_yticks(np.arange(-.5, desc.shape[0], 1), minor=True)
        axi.grid(which='minor', color='lightgray', linestyle='--', linewidth=1)
    else:
        axi.get_xaxis().set_visible(False)
        axi.get_yaxis().set_visible(False)


def add_layout(desc, out):

    walls = (desc == b'W')

    # Walls: black
    out[walls] = [0, 0, 0]

    return out


def test_policy(env, policy, quiet=True, rng=None):

    if rng is not None:
        random_choice = rng.choice
    else:
        random_choice = np.random.choice

    state, _ = env.reset()

    done = False
    episode_reward = 0
    while not done:
        # Sample action from action probability distribution
        action = random_choice(env.action_space.n, p=policy[state])

        # Apply the sampled action in our environment
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    if not quiet:
        print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\n', flush=True)

    return episode_reward


MAPS = {
    "7x7holes": [ "FFFFFFF", "FFFSFFF", "FFFFFFF", "FFHHHFF", "FFFFFFF", "FFFGFFF", "FFFFFFF" ],
    "8x8zigzag": [ "FFFFFFFF", "FSFFFFFF", "WWWWWFFF", "FFFFFFFF", "FFFFFFFF", "FFFWWWWW", "FFFFFFGF", "FFFFFFFF" ],
    "9x9zigzag": [ "FFFFFFFFF", "FSFFFFFFF", "WWWWWWFFF", "FFFFFFFFF", "FFFFFFFFF", "FFFFFFFFF", "FFFWWWWWW", "FFFFFFFGF", "FFFFFFFFF" ],
    "9x9ridgex4": [ "FFFFFFFFF", "FFFFFFFFF", "FFFHHHFFF", "FFFFFFFFF", "FSFHHHFGF", "FFFFFFFFF", "FFFHHHFFF", "FFFFFFFFF", "FFFFFFFFF", ],
    "9x9ridgex4v2": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFHHHFFF",
        "FFFFFFFFF",
        "FSFHHHFGF",
        "FFFFFFFFF",
        "FFFWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF", ],
    "10x10empty": [ "FFFFFFFFFF", "FFFFFFFFFF", "FFSFFFFFFF", "FFFFFFFFFF", "FFFFFFFFFF", "FFFFFFFFFF", "FFFFFFFFFF", "FFFFFFFGFF", "FFFFFFFFFF", "FFFFFFFFFF", ],
    "8x8a_relearn":[ # http://alexge233.github.io/relearn/
        "SFFFFFFF", "FFFFWWFF", "FFFWFFFF", "WWWFFFFF", "FFFFFWFF", "FFFWWWFF", "FFWFFFFF", "GFWFFFFF", ],
    "8x8b_relearn":[ # http://alexge233.github.io/relearn/
        "SFFFFFFF", "FFFFWWFF", "HHHWFFFF", "WWWFFFFF", "FFFFFWFF", "FFFWWWFF", "FFWFFFFF", "GFWHHHHH", ],
    "Todorov3B":[
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFWWWFF',
        'FFFFFFFFFFFWFFF',
        'FFFFFFFWFFFWFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFWWWWWFFFFFFF',
        'FFFWFFFFFFWFFFF',
        'FFFWFFFFFFWFFFF',
        'FFFWFFFFWWWFFFF',
        'FFFFFFFFWFFFFFF',
        'FFFFFFFFFFFFFFF',
    ],
    "Tiomkin2":[
        'FFFFFFFFFFFFFFFFFFFFF',
        'FSFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'WWWWWWWWWWWWWWWFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFWWWWWWWWWWWWFFFF',
        'FFFFFWHHFFFFFFFFFFFFF',
        'FFFFFWHHFFFFFFFFFFFFF',
        'FFFFFWHHFFFFFFFFFFFFF',
        'FFFFFWWWWWFWWWWWWWWWW',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFGF',
        'FFFFFFFFFFFFFFFFFFFFF',
    ],

}

