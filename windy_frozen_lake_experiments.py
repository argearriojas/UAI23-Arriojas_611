from windy_frozen_lake import *
from utils import *
import os
from datetime import datetime
import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", 'font.family': 'serif', 'text.usetex': True,
    'pgf.rcfonts': False, 'savefig.dpi': 300, 'figure.dpi': 150})


def windy_cliff_experiment(beta=20, max_steps=1000, save_images=False):

    desc = np.array([ 'HGR', 'HFF', 'HFF', 'HFF', 'HSR', ], dtype='c')
    wind_field = generate_wind_field(desc, direction=np.pi*0.75, strength=0.7)

    env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-5)
    env = TimeLimit(env, max_episode_steps=max_steps)
    plot_dist(desc, wind_field, titles=['Windy Cliff'], show_grid=True)

    dynamics, rewards = get_dynamics_and_rewards(env)
    prior_policy = np.matrix(np.ones((env.nS, env.nA))) / env.nA

    solution = solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000)
    optimistic_policy = solution['optimal_policy']
    optimistic_estimated_distribution = solution['estimated_distribution']
    optimistic_true_distribution = compute_policy_induced_distribution(dynamics, optimistic_policy, max_steps)

    ground_truth_policy = solve_maxent_value_policy_iteration(beta, dynamics, rewards, prior_policy, max_steps)['policy']
    solution = solve_biased_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000, ground_truth_policy=ground_truth_policy, evaluate=True, evaluate_isd=env.isd, evaluate_steps=max_steps)
    optimal_policy = solution['optimal_policy']
    optimal_estimated_distribution = solution['estimated_distribution']
    biasing_info = solution['info']

    optimal_true_distribution = compute_policy_induced_distribution(dynamics, optimal_policy, max_steps)


    xticks = np.arange(env.nS)
    vlines = np.arange(env.nS+1)-0.5
    xticks_labels = desc.flatten().astype(str)
    xticks_labels[xticks_labels == 'F'] = ''
    xticks_labels[xticks_labels == 'R'] = ''
    xticks_labels[xticks_labels == 'H'] = 'X'

    fig = plot_dist(desc,
        wind_field, optimistic_policy, None,
        None, optimal_policy, None,
        titles=[
            'Windy cliff', f'Optimistic policy', None,
            None, f'Optimal policy', None,
        ], ncols=3, show_plot=False)
    axes = fig.get_axes()
    for axis in axes:
        axis.set_title(axis.get_title(), fontsize=22)
    for axis, estimated_distribution, true_distribution in zip(
            [axes[2], axes[5]],
            [optimistic_estimated_distribution, optimal_estimated_distribution],
            [optimistic_true_distribution, optimal_true_distribution]
        ):
        x = np.array(list(range(env.nS)), dtype=float)
        axis.bar(x - 0.20, estimated_distribution, width=0.3, alpha=1, label='Estimated')
        axis.bar(x + 0.20, true_distribution, width=0.3, alpha=1, label='True')
        miny, maxy = axis.get_ylim()
        axis.vlines(vlines, miny, maxy*1.2, color='lightgray', linewidth=0.5)
        axis.set_yticks([])
        axis.set_xticks(xticks, xticks_labels)
        axis.set_ylim(miny, maxy)
        axis.set_xlabel("State type", fontsize=16)
        axis.set_ylabel("State frequency", fontsize=16)
    axes[2].legend(loc=2, ncol=1, bbox_to_anchor=(-0.1, 1.10), prop={'size': 16}, framealpha=1)
    fig.delaxes(axes[3])

    if save_images:
        if not os.path.exists('images'):
            os.makedirs('images')
        fig.savefig('images/image_optimistic_vs_optimal_v3.png', bbox_inches='tight')
        # fig.savefig('images/image_optimistic_vs_optimal_v3.pdf', bbox_inches='tight')
    plt.show()

    # for label, data in biasing_info.items():
    #     print(label)
    #     print(data)
    #     if type(data) == list and len(data) > 0:
    #         plt.plot(data, label=label)
    #         plt.title(label)
    #         plt.show()


def random_maze_experiment(size=10, traps=10, wind_strength=2, beta=30, max_steps=1000, bias_max_it=200, random_seed=None, show_plots=True, save_images=False):

    np.random.seed(random_seed)
    env = generate_random_environment(size=size, n_traps=traps, rv_strength_params=(wind_strength, 2), max_steps=max_steps)
    np.random.seed()

    dynamics, rewards = get_dynamics_and_rewards(env)
    prior_policy = np.matrix(np.ones((env.nS, env.nA))) / env.nA

    ground_truth_policy = solve_maxent_value_policy_iteration(beta, dynamics, rewards, prior_policy, max_it=1000)['policy']

    solution = solve_biased_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1000, bias_max_it=bias_max_it, ground_truth_policy=ground_truth_policy, quiet=not show_plots)
    optimal_policy = solution['optimal_policy']
    optimal_true_distribution = compute_policy_induced_distribution(dynamics, optimal_policy, max_steps)

    error_policy = kl_divergence(optimal_policy, ground_truth_policy, axis=1).max()

    if show_plots:
        fig = plot_dist(env.desc,
            env.unwrapped.wind_field, optimal_policy, optimal_true_distribution,
            titles=[
                'Random windy maze',
                f'Policy computed w/ biased unconstrained inference\nKL divergence wrt. ground truth = {error_policy:.4e}\n'+fr'$\beta$ = {beta}',
                'Induced distribution'
            ],
            show_plot=False,
        )
        if save_images:
            if not os.path.exists('images'):
                os.makedirs('images')
            # at each run we get a different maze, save image with timestamp
            filename = 'image_random_maze_' + datetime.now().strftime("%Y%m%d-%H%M%S")
            fig.savefig(f'images/{filename}.png', bbox_inches='tight')
            # fig.savefig(f'images/{filename}.pdf', bbox_inches='tight')
        plt.show()
    else:
        return solution['info']


def dynamics_shift_experiment(beta=15, wind_strength=0.5, max_steps=1000, prior_action_success_prob=[0.99, 0.99, 0.99, 0.99], target_action_success_prob=[0.99, 0.99, 0.99, 0.0001], save_images=False):
    if save_images:
        os.makedirs('images', exist_ok=True)
        backend = matplotlib.get_backend()

    desc = np.array([
        "HRRG",
        "FRHR",
        "FHHR",
        "FRRS",
    ], dtype='c')
    wind_field = generate_wind_field(desc, direction=1.5 * np.pi, strength=wind_strength)

    prior_env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-2, action_success_prob=prior_action_success_prob)
    prior_env = TimeLimit(prior_env, max_episode_steps=max_steps)

    prior_dyn, rewards = get_dynamics_and_rewards(prior_env)
    prior_pol = np.matrix(np.ones((prior_env.nS, prior_env.nA))) / prior_env.nA

    # introduce a failure in which action 'up' is no longer effective
    target_env = WindyFrozenLake(desc=desc, wind_field=wind_field, min_reward=-2, action_success_prob=target_action_success_prob)
    target_env = TimeLimit(target_env, max_episode_steps=max_steps)

    target_dyn, rewards = get_dynamics_and_rewards(target_env)

    # first we solve the initial problem where all actions work
    policy_one = solve_biased_unconstrained(beta, prior_dyn, rewards, prior_pol, target_dynamics=prior_dyn)['optimal_policy']
    original_dyn_agent_distribution = compute_policy_induced_distribution(prior_dyn, policy_one, max_steps, prior_env.isd)
    shifted_dyn_agent_distribution = compute_policy_induced_distribution(target_dyn, policy_one, max_steps, target_env.isd)
    original_dyn_agent_evaluation = evaluate_policy(prior_dyn, rewards, prior_env.isd, policy_one, max_steps)
    shifted_dyn_agent_evaluation = evaluate_policy(target_dyn, rewards, target_env.isd, policy_one, max_steps)

    # now we consider the prior dynamics to solve the case where action 'up' has been suppressed
    policy_two = solve_biased_unconstrained(beta, prior_dyn, rewards, prior_pol, target_dynamics=target_dyn)['optimal_policy']
    shifted_dyn_agent_alt_distribution = compute_policy_induced_distribution(target_dyn, policy_two, max_steps, target_env.isd)
    shifted_dyn_agent_alt_evaluation = evaluate_policy(target_dyn, rewards, target_env.isd, policy_two, max_steps)

    fig = plot_dist(desc,
        wind_field, original_dyn_agent_distribution, policy_one, shifted_dyn_agent_distribution, shifted_dyn_agent_alt_distribution, policy_two,
        titles=[
            f"Maze Layout\nHorizon = {max_steps} steps",
            f"Original Solution \nMean rewards = {original_dyn_agent_evaluation:.2f}",
            f"Policy for Original Solution",
            f"Dynamics shift breaks solution\nMean rewards = {shifted_dyn_agent_evaluation:.2f}",
            f"New Solution\nMean rewards = {shifted_dyn_agent_alt_evaluation:.2f}",
            f"Policy for New Solution",
        ],
        ncols=3,
        show_plot=False,
    )

    if save_images:
        if not os.path.exists('images'):
            os.makedirs('images')
        fig.set_size_inches(w=10, h=6.75)
        fig.savefig(f'images/image_dynamics_shift.png', bbox_inches='tight')
        # fig.savefig(f'images/image_dynamics_shift.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--windy-cliff', action='store_true')
    parser.add_argument('--random-maze', action='store_true')
    parser.add_argument('--dynamics-shift', action='store_true')
    parser.add_argument('--improvement-mode', action='store_true')
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()

    none_selected = not any([args.windy_cliff, args.random_maze, args.dynamics_shift, args.improvement_mode])

    if args.windy_cliff or none_selected:
        # Figure 1
        windy_cliff_experiment(beta = 50, save_images=args.save_images)

    if args.random_maze or none_selected:
        # Figure 2
        random_maze_experiment(save_images=args.save_images)

    if args.dynamics_shift or none_selected:
        # Figure 4: failure mode
        dynamics_shift_experiment(beta=15, prior_action_success_prob=[0.99, 0.99, 0.99, 0.99], target_action_success_prob=[0.99, 0.99, 0.99, 0.01], save_images=args.save_images)
    
    if args.improvement_mode:
        # Extra: improvement mode
        dynamics_shift_experiment(beta=15, prior_action_success_prob=[0.99, 0.99, 0.99, 0.10], target_action_success_prob=[0.99, 0.99, 0.99, 0.99])

