import pandas as pd
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm


def load_data():
    transitions = genfromtxt("../data/icu-sepsis-mixed/transitionFunction.csv", delimiter=",")
    survival_reward = genfromtxt("../data/icu-sepsis-mixed/rewardFunction-survival.csv", delimiter=",")
    death_reward = genfromtxt("../data/icu-sepsis-mixed/rewardFunction-death.csv", delimiter=",")
    mixed_reward = genfromtxt("../data/icu-sepsis-mixed/rewardFunction-mixed.csv", delimiter=",")
    return transitions, survival_reward, death_reward, mixed_reward

def value_iter(P: np.ndarray, R: np.ndarray, gamma: float = 1.0, epsilon: float = 1e-4 ) -> None:
    '''
    value iteration that supports three reward variants. 
    P (transition probability): np.ndarray, (17900, 716): (i, j) corresponds to (s, a, s'). each row corresponds to a (s, a) pair, column the next state
    R_ (reward function): np.ndarray, (716, ): each column corresponds to the reward received upon transitioning into that state
    gamma: discount factor
    epsilon: threshold for convergence
    '''
    num_states = 716
    num_actions = 25
    absorbing_state = num_states - 1

    P_sa_s = P.reshape((num_states, num_actions, num_states))
    V = np.zeros(num_states)

    iteration = 0
    while True:
        V_prev = V.copy()
        for s in range(num_states):
            Q_sa = np.zeros(num_actions)
            for a in range(num_actions):
                Q_sa[a] = np.sum(P_sa_s[s, a, :] * (R + gamma * V_prev))
            V[s] = np.max(Q_sa)

        V[absorbing_state] = R[absorbing_state]

        delta = np.max(np.abs(V - V_prev))
        iteration += 1
        if delta < epsilon:
            break

    print(f"Converged after {iteration} iterations.")
    return V


if __name__ == "__main__":
    P, R_surv, R_death, R_mixed = load_data()
    V_plus = value_iter(P, R_surv)
    V_minus = value_iter(P, R_death)
    V_mixed = value_iter(P, R_mixed)
    
    # for proposition 2: Value sum in MDPs
    print("prop 2: ", np.isclose((V_plus + V_minus), V_mixed, atol=1e-8).all())

    # for proposition 4: State-value difference in MDPs with shared policy
    print("prop 4: ", np.isclose((V_plus - V_minus)[:-3], 1.00, atol=1e-10).all())  # excluding last 3 states