from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import plot_Qs
import matplotlib as plt
import pprint

if __name__ == "__main__":
    '''
    Example for how to use LifeGate_MDP class.
    '''

    mdp_plus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="-")

    graph = mdp_plus.visualize_mdp()
    # graph.render(directory='generated-dag-plus', view=True)
    graph = mdp_minus.visualize_mdp()
    # graph.render(directory='generated-dag-minus', view=True)

    Q_plus = mdp_plus.compute_Qs()
    Q_minus = mdp_minus.compute_Qs()

    plot_Qs(Q_plus, Q_minus, "Uniform Random Large MDP")

    V_plus, pi_plus = mdp_plus.policy_iter()
    V_minus, pi_minus = mdp_minus.policy_iter()
    Q_plus = mdp_plus.compute_Qs(pi_plus)
    Q_minus = mdp_minus.compute_Qs(pi_minus)

    plot_Qs(Q_plus, Q_minus, "Policy Iteration Large MDP")