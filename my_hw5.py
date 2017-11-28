from gridworld import GridWorld, GridWorld_MDP
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import random

def update_matrix(mdp, m, s, a):
    transition_dict = mdp.P_snexts(s, a)
    m[s, :] = np.zeros(mdp.num_states)
    for next_state, transition_p in transition_dict.items():
        if not mdp.is_absorbing(next_state):
            m[s, next_state] = transition_p


def init_matrix(mdp, policy):
    matrix = np.zeros((mdp.num_states, mdp.num_states))
    for i in range(len(policy)):
        update_matrix(mdp, matrix, i, policy[i])
    return matrix


def policy_iteration(mdp, gamma=1, iters=5, plot=True):
    '''
    Performs policy iteration on an mdp and returns the value function and policy 
    :param mdp: mdp class (GridWorld_MDP) 
    :param gam: discount parameter should be in (0, 1] 
    :param iters: number of iterations to run policy iteration for
    :param plot: boolean for if a plot should be generated for the utilities of the start state
    :return: two numpy arrays of length |S| and one of length iters.
    Two arrays of length |S| are U and pi where U is the value function
    and pi is the policy. The third array contains U(start) for each iteration
    the algorithm.
    '''
    pi = np.zeros(mdp.num_states, dtype=np.int)
    U = np.zeros(mdp.num_states)
    Ustart = []
    identity_matrix = np.eye(11)
    matrix = init_matrix(mdp, pi)
    for itr in range(iters):
        # policy evaluation
        b = [mdp.R(s) for s in mdp.S()]
        A = identity_matrix - gamma * matrix
        A_inverse = np.linalg.pinv(A)
        U = np.dot(A_inverse, b)

        # policy improvement
        for current_state in mdp.S():
            action2U = defaultdict(float)
            for a in mdp.A(current_state):
                transition_dict = mdp.P_snexts(current_state, a)
                for next_state, transition_p in transition_dict.items():
                    if mdp.is_absorbing(next_state):
                        action2U[a] = mdp.R(next_state)
                    else:
                        updated_value = transition_p * U[next_state]
                        action2U[a] += updated_value
            max_policy = max(action2U, key=action2U.get)
            update_matrix(mdp, matrix, current_state, max_policy)
            pi[current_state] = max_policy
        start_idx = mdp.loc2state[mdp.start]
        Ustart.append(U[start_idx])

    if plot:
        fig = plt.figure()
        plt.title("Policy Iteration with $\gamma={0}$".format(gamma))
        plt.xlabel("Iteration (k)")
        plt.ylabel("Utility of Start")
        plt.ylim(-1, 1)
        plt.plot(Ustart)

        pp = PdfPages('./plots/piplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    # U and pi should be returned with the shapes and types specified
    return U, pi, np.array(Ustart)


def td_update(v, s1, r, s2, terminal, alpha, gamma):
    '''
    Performs the TD update on the value function v for one transition (s,a,r,s').
    Update to v should be in place.
    :param v: The value function, a numpy array of length |S|
    :param s1: the current state, an integer 
    :param r: reward for the transition
    :param s2: the next state, an integer
    :param terminal: bool for if the episode ended
    :param alpha: learning rate parameter
    :param gamma: discount factor
    :return: Nothing
    '''
    # you should update the value function v inplace (does not need to be returned)
    if terminal:
        v[s1] = r
    else:
        v[s1] = v[s1] + alpha * (r + gamma * v[s2] - v[s1])


def td_episode(env, pi, v, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for one episode update the value function after
    each iteration. The value function update should be done with the TD learning rule.
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S| representing the policy
    :param v: numpy array of length |S| representing the value function
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps in the episode
    :return: two floats G, v0 where G is the discounted return and v0 is the value function of the initial state (before learning)
    '''
    G = 0.
    v0 = v[env.get_state()]

    # episode ends when max_steps have been completed
    # episode ends when env is in the absorbing state
    # Learning should be done online (after every step)
    # return the discounted sum of rewards G, and the value function's estimate from the initial state v0
    # the value function estimate should be before any learn takes place in this episode

    for t in range(max_steps):
        if env.is_absorbing():
            env.reset_to_start()
            return G, v0
        else:
            current_state = env.get_state()
            is_terminal = env.is_terminal()
            action = pi[current_state]
            reward = env.Act(action)
            G += math.pow(gamma, t + 1) * reward
            next_state = env.get_state()
            td_update(v, current_state, reward, next_state, is_terminal, alpha, gamma)
    env.reset_to_start()
    return G, v0


def td_learning(env, pi, gamma, alpha, episodes=200, plot=True):
    '''
    Evaluates the policy pi in the environment by estimating the value function
    with TD updates  
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S|, representing the policy 
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to use in evaluating the policy
    :param plot: boolean for if a plot should be generated for returns and estimates
    :return: Two lists containing the returns for each episode and the value function estimates, also returns the value function
    '''
    returns, estimates = [], []
    v = np.zeros(env.num_states)
    for e in range(episodes):
        discounted_sum, temp_estimate = td_episode(env, pi, v, gamma, alpha)
        returns.append(discounted_sum)
        estimates.append(temp_estimate)

    # value function should start at 0 for all states
    # return the list of returns, and list of estimates for all episodes
    # also return the value function v

    if plot:
        fig = plt.figure()
        plt.title("TD Learning with $\gamma={0}$ and $\\alpha={1}$".format(gamma, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])

        pp = PdfPages('./plots/tdplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    return returns, estimates, v


def egreedy(q, s, eps):
    '''
    Epsilon greedy action selection for a discrete Q function.
    :param q: numpy array of size |S|X|A| representing the state action value look up table
    :param s: the current state to get an action (an integer)
    :param eps: the epsilon parameter to randomly select an action
    :return: an integer representing the action
    '''

    p = random.random()
    action_list = range(q.shape[1])
    if p <= eps:
        return random.choice(action_list)
    else:
        max_value = q[s,:][0]
        max_action = 0
        for i in range(q.shape[1]):
            if q[s,i] > max_value:
                max_value = q[s,i]
                max_action = i
        return max_action

def q_update(q, s1, a, r, s2, terminal, alpha, gamma):
    '''
    Performs the Q learning update rule for a (s,a,r,s') transition. 
    Updates to the Q values should be done inplace
    :param q: numpy array of size |S|x|A| representing the state action value table
    :param s1: current state
    :param a: action taken
    :param r: reward observed
    :param s2: next state
    :param terminal: bool for if the episode ended
    :param alpha: learning rate
    :param gamma: discount factor
    :return: None
    '''

    # update should be done inplace (not returned)
    if terminal:
        q[s1,a] = r
    else:
        max_value = max(q[s2,:])
        q[s1,a] = q[s1,a] + alpha * (r + gamma * max_value - q[s1,a])


def q_episode(env, q, eps, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for an episode update the state action value function
    online according to the Q learning update rule. Actions are taken with an epsilon greedy policy
    :param env: environment object (GridWorld)
    :param q: numpy array of size |S|x|A| for state action value function
    :param eps: epsilon greedy parameter
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps to interact with the environment
    :return: two floats: G, q0 which are the discounted return and the estimate of the return from the initial state
    '''
    G = 0.
    q0 = max(q[env.get_state(),:])

    # Return G the discounted some of rewards and q0 the estimate of G from the initial state
    for t in range(max_steps):
        if env.is_absorbing():
            env.reset_to_start()
            return G, q0
        else:
            current_state = env.get_state()
            is_terminal = env.is_terminal()
            action = egreedy(q,current_state, eps)
            reward = env.Act(action)
            G += math.pow(gamma, t + 1) * reward
            next_state = env.get_state()
            q_update(q, current_state, action, reward, next_state, is_terminal, alpha, gamma)
    env.reset_to_start()
    return G, q0


def q_learning(env, eps, gamma, alpha, episodes=200, plot=True):
    '''
    Learns a policy by estimating the state action values through interactions 
    with the environment.  
    :param env: environment object (GridWorld)
    :param eps: epsilon greedy action selection parameter
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to learn
    :param plot: boolean for if a plot should be generated returns and estimates
    :return: Two lists containing the returns for each episode and the action value function estimates of the return, also returns the Q table
    '''
    returns, estimates = [], []
    q = np.zeros((env.num_states, env.num_actions))

    for e in range(episodes):
        # print('episode {}'.format(e))
        discounted_sum, temp_estimate = q_episode(env, q, eps, gamma, alpha)
        returns.append(discounted_sum)
        estimates.append(temp_estimate)
    # return the returns and estimates for each episode and the Q table

    if plot:
        fig = plt.figure()
        plt.title("Q Learning with $\gamma={0}$, $\epsilon={1}$, and $\\alpha={2}$".format(gamma, eps, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])

        pp = PdfPages('./plots/qplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    return returns, estimates, q


if __name__ == '__main__':
    env = GridWorld()
    mdp = GridWorld_MDP()

    U, pi, Ustart = policy_iteration(mdp, gamma=1, plot=True)
    print(U)
    vret, vest, v = td_learning(env, pi, gamma=1., alpha=0.1, episodes=2000, plot=True)
    print(v)
    qret, qest, q = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    print(q)