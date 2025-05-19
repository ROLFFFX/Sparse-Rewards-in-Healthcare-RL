from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import plot_Qs
import matplotlib as plt
import pprint

if __name__ == "__main__":
    '''
    Example for how to use LifeGate_MDP class.
    '''

    mdp_plus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="-")

    #graph = mdp_plus.visualize_mdp()
    #graph.render(directory='generated-dag-plus', view=True)
    #graph = mdp_minus.visualize_mdp()
    #graph.render(directory='generated-dag-minus', view=True)

    print("--- 2 MDP's, fixed pi (uniform random) ---")

    print()
    pprint.pp(mdp_plus.transitions)
    pprint.pp(mdp_minus.transitions)

    Q_plus = mdp_plus.compute_Qs()
    Q_minus = mdp_minus.compute_Qs()
    print()
    print("Q plus")
    pprint.pp(Q_plus)
    print()
    print("Q minus")
    pprint.pp(Q_minus)
    print()
    print("Q plus - Q minus")
    Q = {
    state: {
        action: Q_plus[state][action] - Q_minus[state].get(action, 0.0)
        for action in Q_plus[state]
    }
    for state in Q_plus
    }

    pprint.pp(Q)
    plot_Qs(Q_plus, Q_minus, "Uniform Random Toy MDP")

    print()
    print()
    print("--- 2 MDP's, policy iter ---")

    V_plus, pi_plus = mdp_plus.policy_iter()
    V_minus, pi_minus = mdp_minus.policy_iter()
    Q_plus = mdp_plus.compute_Qs(pi_plus)
    Q_minus = mdp_minus.compute_Qs(pi_minus)

    print()
    print("Q plus")
    pprint.pp(Q_plus)
    print()
    print("Q minus")
    pprint.pp(Q_minus)
    print()
    print("Q plus - Q minus")
    Q = {
    state: {
        action: Q_plus[state][action] - Q_minus[state].get(action, 0.0)
        for action in Q_plus[state]
    }
    for state in Q_plus
    }
    pprint.pp(Q)
    print()
    print("pi* plus")
    pprint.pp(pi_plus)
    print()
    print("pi* minus")
    pprint.pp(pi_minus)
    plot_Qs(Q_plus, Q_minus, "Policy Iteration Toy MDP")