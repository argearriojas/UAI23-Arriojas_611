import gym
from packaging import version
gym_version = version.parse(gym.__version__)

if gym_version >= version.parse('0.25.0'):
    from classic_control_discrete_latest import *
else:
    from classic_control_discrete_compat import *

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from utils import solve_biased_unconstrained

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", 'font.family': 'serif', 'text.usetex': True,
    'pgf.rcfonts': False, 'savefig.dpi': 300, 'figure.dpi': 150})


def compute_policies_for_plots(env_name, nbins_list=[4], beta_list=[1], bias_max_it=150):
    
    policy_filename_template = os.path.join('data', env_name, 'optimal_policy_beta{beta:04.1f}_{nbins_str}bins.npy')
    info_filename_template = os.path.join('data', env_name, 'info_beta{beta:04.1f}_{nbins_str}bins.pkl')

    for nbins in nbins_list:
        env = get_environment(env_name, nbins)
        nbins = env.nbins
        nbins_str = "-".join([f'{i:02d}' for i in nbins])
        if any([not os.path.exists(policy_filename_template.format(beta=beta, nbins_str=nbins_str)) for beta in beta_list]):
            print(f"{nbins=}", flush=True)
            dynamics, rewards = get_model_estimation(env_name, nbins=nbins, n_tries=256)
        else:
            continue
        for beta in beta_list:
            filename = policy_filename_template.format(beta=beta, nbins_str=nbins_str)
            if not os.path.exists(filename):
                print(f"{nbins=}, {beta=: 5.1f}", flush=True)
                rewards -= rewards.max()
                solution = solve_biased_unconstrained(beta, dynamics, rewards, bias_max_it=bias_max_it, track_policy=True, tolerance=1e-6, eig_max_it=2000)
                optimal_policy = solution['optimal_policy']
                info = solution['info']
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
    info_filename = os.path.join('data', env_name, f'info_beta{beta:04.1f}_{nbins_str}bins.pkl')
    results_filename = os.path.join('data', env_name, f'results_beta{beta:04.1f}_{nbins_str}bins.pkl')

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
    n_iter = len(info['policy_list'])
    pbar = tqdm(range(n_iter), unit='iteration')
    for i in pbar:
        policy = info['policy_list'][i]
        # print(f'evaluating iteration {i: 3d} / {info["iterations_completed"]}')
        results[i] = sum(Parallel(n_jobs=ncpu)(delayed(work)(policy, rng) for rng in rngs), [])

    with open(results_filename, 'wb') as file:
        pickle.dump(results, file)


def get_final_result_summary(env_name, nbins, beta):
    env = get_environment(env_name, nbins)
    nbins = env.nbins
    nbins_str = "-".join([f'{i:02d}' for i in nbins])
    info_filename = os.path.join('data', env_name, f'info_beta{beta:04.1f}_{nbins_str}bins.pkl')
    results_filename = os.path.join('data', env_name, f'results_beta{beta:04.1f}_{nbins_str}bins.pkl')

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
    info_filename = os.path.join('data', env_name, f'info_beta{beta:04.1f}_{nbins_str}bins.pkl')

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)

    policy = info['policy_list'][iteration]
    return test_policy(env, policy, render=True)


SHORT_EXPERIMENTS = [
    # env_name,  nbins,  beta
    ('Pendulum',    14,     1),
    ('MountainCar', 10,     1),
    ('CartPole',     6,    10),
    ('Acrobot',      6,    25),
]

FULL_EXPERIMENTS = [
    # env_name,  nbins,  beta
    ('Pendulum',    16,     1),
    ('MountainCar', 12,     1),
    ('CartPole',     8,    10),
    ('Acrobot',     10,    15),
]

EXPERIMENTS = SHORT_EXPERIMENTS


def generate_data():
    for env_name, nbins, beta in EXPERIMENTS:
        compute_policies_for_plots(env_name, nbins_list=[nbins], beta_list=[beta], bias_max_it=150)
        compute_per_iter_evaluation(env_name, nbins=nbins, beta=beta, max_episode_steps=500, n_episodes=32)
        get_final_result_summary(env_name, nbins, beta)


def figure_3(w = 5, save_images=False):
    results_filename_template = os.path.join('data', '{env_name}', 'results_beta{beta:04.1f}_{nbins_str}bins.pkl')
    # plt.figure(dpi=150)
    for env_name, nbins, beta in EXPERIMENTS:
        env = get_environment(env_name, nbins)
        nbins = env.nbins
        nbins_str = "-".join([f'{i:02d}' for i in nbins])
        with open(results_filename_template.format(env_name=env_name, beta=beta, nbins_str=nbins_str), 'rb') as file:
            results = pickle.load(file)
        x = np.array(list(results.keys()))
        r = np.array(list(results.values()))
        y = r.mean(axis=1)
        # offset, scale = y.min(), (y - y.min()).max()
        offset, scale = 0, 1
        scale = scale if scale > 0 else 1.
        
        y = np.array([y[max(i - w + 1, 0) : i + 1].mean() for i in range(len(y))])
        y -= offset
        y /= scale

        _ = plt.plot(x, y, '-', label=env_name)
        # plt.text(x[-1]-3, y[-1]+10, f"{x[-1]} iters\nR = {y[-1]:.2f}")
    plt.ylabel('Total Return')
    plt.xlabel('Biasing iteration')
    plt.title('Biasing algorithm improves solution performance')
    left, right = plt.gca().get_xlim()
    plt.hlines([-500, 0, 500], left, right, color='lightgray', linestyle='--')
    plt.ylim(-600, 600)
    plt.yticks([-500,0,500])
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    # plt.subplots_adjust(right=0.72)
    if save_images:
        fig = plt.gcf()
        # fig.set_size_inches(w=10, h=6.75)
        if not os.path.exists('images'):
            os.mkdir('images')
        fig.savefig(f'images/image_classic_control_biasing_progress.png', bbox_inches='tight')
        # fig.savefig(f'images/image_classic_control_biasing_progress.pdf', format='pdf', bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--pendulum', action='store_true')
    parser.add_argument('--mountaincar', action='store_true')
    parser.add_argument('--cartpole', action='store_true')
    parser.add_argument('--acrobot', action='store_true')
    parser.add_argument('--show-evaluation', action='store_true')
    parser.add_argument('--save-images', action='store_true')

    args = parser.parse_args()
    kvargs = vars(args)
    
    compute_full_experiments = args.full
    if compute_full_experiments:
        EXPERIMENTS = FULL_EXPERIMENTS
    else:
        EXPERIMENTS = SHORT_EXPERIMENTS

    if any([args.pendulum, args.mountaincar, args.cartpole, args.acrobot]):
        ALL_EXPERIMENTS = EXPERIMENTS
        EXPERIMENTS = []
        for env_name,  nbins,  beta in ALL_EXPERIMENTS:
            if kvargs[env_name.lower()]:
                EXPERIMENTS.append((env_name, nbins, beta))

    generate_data()
    figure_3(w=5, save_images=args.save_images)

    if args.show_evaluation:
        for env_name,  nbins,  beta in EXPERIMENTS:
            user_input = load_and_test_policy(env_name, nbins=nbins, beta=beta, iteration=-1, max_episode_steps=500)
            if user_input in ['q', 'quit', 'exit', 'c', 'cancel']:
                break

