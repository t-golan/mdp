import numpy as np


def mdp_compute_PR_policy(P, R, policy):
    P_policy = np.zeros(shape=P.shape)
    PR = mdp_compute_PR(P, R)
    PR_policy = np.zeros(len(P))
    for i in range(len(P)):
        actions = np.where(policy == i)
        P_policy[actions][:] = P[actions][:][i]
        PR_policy[actions] = PR[actions][i]
    return P_policy, PR_policy


def mdp_eval_policy_iterative(P, PR, discount, policy, epsilon=0.0001):
    V = np.zeros(shape=(P[0]))
    P_policy, PR_policy = mdp_compute_PR_policy(P, R, policy)
    while True:
        V_prev = V
        V = PR_policy + discount * P_policy * V_prev
        var = np.max(np.abs(V - V_prev))
        if var < ((1 - discount)/discount) * epsilon:
            break
    return V
            




def mdp_bellman_operator(P, PR, discount, policy):
    for i in range()


def mdp_policy_iteration(P, R, discount, policy0, max_iter=1000, eval_type=0):
    PR = mdp_compute_PR(P, R)
    policy = mdp_bellman_operator(P, PR, discount, np.zeros(shape=len(P)))
    while True:
        V = mdp_eval_policy_iterative(P, PR, discount, policy)
        policy_next = mdp_bellman_operator(P, PR, discount, V)
        if policy_next==policy:
            break
        policy = policy_next

    return V, policy





def mdp_compute_PR(P, R):
    PR = np.ndarray(shape=(len(P), 2))
    for i in range(3):
        PR[:][i] = np.sum([a*b for a, b in zip(P[:][:][i], R[:][:][i])], axis=0)
    return PR


P = np.array(([[[0.5, 0], [0, 0], [0.5, 1]], [[0.7, 0], [0.1, 0.95], [0.2, 0.05]], [[0.4, 0.3], [0, 0.3], [0.6, 0.4]]]))
R = np.zeros((3, 3, 2))
R[1][0][0] = 5
R[2][0][1] = -1
print(mdp_policy_iteration(P, R))


