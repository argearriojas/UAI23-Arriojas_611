import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix


def solve_unconstrained(beta, dynamics, rewards, prior_policy=None, N=0, eig_max_it=10000, tolerance=1e-8, dtype=float):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    if prior_policy is None:
        prior_policy = np.ones((nS, nA)) / nA

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr().astype(dtype)
    Mt = M.T.tocsr()

    # left eigenvector
    u = np.matrix(np.ones((nS * nA, 1)), dtype=dtype)
    u_scale = np.linalg.norm(u)

    # right eigenvector
    v = np.matrix(np.ones((nS * nA, 1)), dtype=dtype)
    v_scale = np.linalg.norm(u)

    eps = np.finfo(M.dtype).eps
    max_float = np.finfo(M.dtype).max / 200

    for i in range(eig_max_it):

        # tmp = M.dot(v)
        # rho = (u.T.dot(tmp) / u.T.dot(v))[0, 0]
        # uk = Mt.dot(u) / rho
        # vk = tmp / rho

        uk = Mt.dot(u)
        lu = np.linalg.norm(uk) / u_scale
        uk = uk / lu

        vk = M.dot(v)
        lv = np.linalg.norm(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        u_err = np.abs((np.log(uk+eps) - np.log(u+eps))/ beta).max()
        v_err = np.abs((np.log(vk+eps) - np.log(v+eps))/ beta).max()

        # update the eigenvectors
        u = uk
        v = vk

        if u_err <= tolerance and v_err <= tolerance:
            break

    else:
        print('did not converge', u_err, v_err)

    v = v / v.sum()
    u = u / u.T.dot(v)

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    optimal_dynamics = dynamics.multiply(chi)
    optimal_dynamics = optimal_dynamics.multiply(1 / optimal_dynamics.sum(axis=0)).tocsr()

    estimated_full_distribution = np.array(np.multiply(u, v).reshape((nS, nA)))
    estimated_distribution = estimated_full_distribution.sum(axis=1)

    rho = (lu + lv) / 2
    theta = - np.log(rho) / beta
    Q = - N * theta + np.log(u.reshape((nS, nA))) / beta
    V = - N * theta + np.log(chi) / beta

    return dict(
        rho = rho, theta = theta, u = u, v = v, chi = chi,
        optimal_policy = optimal_policy,
        optimal_dynamics = optimal_dynamics,
        estimated_full_distribution = estimated_full_distribution,
        estimated_distribution = estimated_distribution,
        Q = Q, V = V,
    )


def solve_biased_unconstrained(beta, prior_dynamics, rewards, prior_policy=None, target_dynamics=None, N=0,
        eig_max_it=10000, bias_max_it=200, ground_truth_policy=None, evaluate=False, evaluate_isd=None,
        evaluate_steps=None, track_policy=False, quiet=False, tolerance=1e-8, dtype=float):

    nS, nSnA = prior_dynamics.shape
    nA = nSnA // nS

    if evaluate_isd is None:
        evaluate_isd = np.ones(nS) / nS

    if prior_policy is None:
        prior_policy = np.matrix(np.ones((nS, nA))) / nA

    if target_dynamics is None:
        target_dynamics = prior_dynamics

    ### initialization ###
    biased_dynamics = prior_dynamics.copy()
    biased_rewards = rewards
    chi = np.matrix(np.ones((nS, 1)))

    error_policy_list = []
    error_dynamics_list = []
    error_chi_list = []
    evaluation_list = []
    policy_list = []
    for i in range(bias_max_it):

        solution = solve_unconstrained(beta, biased_dynamics, biased_rewards, prior_policy, N=N, eig_max_it=eig_max_it, tolerance=tolerance, dtype=dtype)
        optimal_policy = solution['optimal_policy']

        if track_policy:
            policy_list.append(optimal_policy)
        if evaluate:
            assert evaluate_steps is not None, "Must provide trajectory length for policy evaluation"
            evaluation = evaluate_policy(target_dynamics, rewards, evaluate_isd, optimal_policy, evaluate_steps)
            evaluation_list.append(evaluation)
        if ground_truth_policy is not None:
            error_policy = kl_divergence(optimal_policy, ground_truth_policy, axis=1).sum()
            error_policy_list.append(error_policy)

        kl_err = kl_divergence(solution['optimal_dynamics'], target_dynamics, axis=0)
        error_dynamics_list.append(kl_err.sum())
        chi_err = (np.abs(chi - solution['chi']) / chi).max()
        error_chi_list.append(chi_err)
        if kl_err.max() < 1e-6 or chi_err < 1e-4:
            if not quiet:
                print(f'Solved in {i} iterations. KL error: {kl_err.max()}, Chi error: {chi_err}')
            break

        chi = solution['chi']

        biased_dynamics = target_dynamics.multiply(1. / chi)
        biased_dynamics = biased_dynamics.multiply(1 / biased_dynamics.sum(axis=0))

        biased_rewards = rewards + kl_divergence(target_dynamics, biased_dynamics) / beta
        biased_rewards -= biased_rewards.max()

    else:
        if not quiet:
            print(f'Did not finish after {i} iterations')

    solution['info'] = dict(
        error_policy=error_policy_list,
        error_dynamics=error_dynamics_list,
        evaluation=evaluation_list,
        policy_list=policy_list,
        iterations_completed=i,
    )
    return solution


def solve_value_policy_iteration(beta, dynamics, rewards, prior_policy, max_it):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    rewards = rewards.reshape((nS, nA))

    if prior_policy is None:
        prior_policy = np.ones((nS, nA)) / nA

    prior_policy = np.matrix(prior_policy)
    policy = np.matrix(prior_policy)

    Q = np.matrix(np.zeros((nS, nA)))
    for k in range(1, max_it+1):
        Q = rewards + dynamics.multiply(Q.max(axis=1)).sum(axis=0).reshape((nS, nA))

        if k % 5 == 0:
            prev_policy = policy.copy()
            policy = np.multiply(prior_policy, np.exp(beta * (Q - Q.max(axis=1))))
            policy /= policy.sum(axis=1)

            err = kl_divergence(policy, prev_policy, axis=1).max()
            if err < 1e-8:
                break

    return dict(
        Q = Q,
        policy = np.array(policy),
    )


def solve_sarsa_value_policy_iteration(beta, dynamics, rewards, prior_policy, max_it):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    rewards = rewards.reshape((nS, nA))

    if prior_policy is None:
        prior_policy = np.ones((nS, nA)) / nA

    prior_policy = np.matrix(prior_policy)
    policy = np.matrix(prior_policy)

    Q = np.matrix(np.zeros((nS, nA)))
    V = np.matrix(np.zeros((nS, 1)))
    for k in range(1, max_it+1):
        Q = rewards + dynamics.multiply(V).sum(axis=0).reshape((nS, nA))
        V = np.multiply(policy, Q).sum(axis=1)

        if k % 5 == 0:
            prev_policy = policy.copy()
            policy = np.multiply(prior_policy, np.exp(beta * (Q - V)))
            policy /= policy.sum(axis=1)

            err = kl_divergence(policy, prev_policy, axis=1).max()
            if err < 1e-8:
                break

    return dict(
        Q = Q,
        policy = np.array(policy),
    )


def solve_maxent_value_policy_iteration(beta, dynamics, rewards, prior_policy, max_it):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    rewards = rewards.reshape((nS, nA))

    if prior_policy is None:
        prior_policy = np.ones((nS, nA)) / nA

    prior_policy = np.matrix(prior_policy)
    policy = np.matrix(prior_policy)

    log_policy_over_prior = np.log(policy / prior_policy)

    Q = np.matrix(np.zeros((nS, nA)))
    V = np.matrix(np.zeros((nS, 1)))
    for k in range(1, max_it+1):
        Q = rewards + dynamics.multiply(V).sum(axis=0).reshape((nS, nA))
        V = np.multiply(policy, (Q - log_policy_over_prior / beta)).sum(axis=1)

        if k % 5 == 0:
            prev_policy = policy.copy()
            policy = np.multiply(np.exp(beta * (Q - V)), prior_policy)
            policy = policy / policy.sum(axis=1)

            err = kl_divergence(policy, prev_policy, axis=1).max()
            if err < 1e-8:
                break
            
            log_policy_over_prior = np.array(np.zeros_like(policy)).flatten()
            mask = np.array(policy).flatten() > 0
            log_policy_over_prior[mask] = np.array(np.log(np.array(policy).flatten()[mask] / np.array(prior_policy).flatten()[mask]))
            log_policy_over_prior = np.matrix(log_policy_over_prior.reshape((nS, nA)))

    return dict(
        Q = Q, V = V,
        policy = np.array(policy),
    )


def dp_qlearning_backup_equation(beta, dynamics, rewards, prior_policy, max_it):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    rewards = rewards.reshape((nS, nA))
    if prior_policy is None:
        prior_policy = np.ones((nS, nA)) / nA

    Q = np.matrix(np.zeros((nS, nA)))
    for _ in range(max_it):
        Q = rewards + dynamics.multiply(Q.max(axis=1)).sum(axis=0).reshape((nS, nA))

    policy = np.multiply(prior_policy, np.exp(beta * (Q - Q.max(axis=1))))
    policy /= policy.sum(axis=1)

    return dict(
        Q = Q,
        policy = policy.A,
    )


def dp_softq_backup_equation(beta, dynamics, rewards, prior_policy, max_it):

    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    rewards = rewards.reshape((nS, nA))
    if prior_policy is None:
        prior_policy = np.ones((nS, nA)) / nA
    prior_policy = np.matrix(prior_policy)

    Q = np.matrix(np.zeros((nS, nA)))
    V = np.matrix(np.zeros((nS, 1)))
    for _ in range(max_it):
        offset = np.mean(V)
        Q = rewards + offset + np.log(dynamics.multiply(np.exp(beta * (V - offset))).sum(axis=0)).reshape((nS, nA)) / beta
        V = offset + (np.log(np.multiply(prior_policy, np.exp(beta * (Q - offset))).sum(axis=1))) / beta

    return dict(
        Q = Q, V = V,
        policy = np.multiply(prior_policy, np.exp(beta * (Q - V))).A,
    )


def get_mdp_transition_matrix(transition_dynamics, policy):

    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA + np.arange(nA)).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)), (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows ,cols)), shape=(nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix


def kl_divergence(dist_a, dist_b, axis=0):
    numer = coo_matrix(dist_a)
    numer.eliminate_zeros()
    denom = csr_matrix(dist_b)
    kldiv = numer.copy()
    denom = np.array(denom[numer.row, numer.col]).flatten()
    mask = denom > 0.
    kldiv.data[mask] = np.log(numer.data[mask] / denom[mask]) * numer.data[mask]
    kldiv.data[np.logical_not(mask)] = np.inf
    kldiv = kldiv.sum(axis=axis)

    return kldiv


def evaluate_policy(dynamics, rewards, isd, policy, steps, full_value_function=False):

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


def make_greedy(policy):
    policy = policy == policy.max(axis=1)[..., np.newaxis]
    return policy / policy.sum(axis=1)[..., np.newaxis]

