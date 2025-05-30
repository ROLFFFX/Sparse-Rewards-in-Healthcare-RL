from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import plot_Qs
import matplotlib as plt
import pprint

def break_gamma():
    mdp_plus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="-")

    for i in range(1, 4):
        g = 1.0 - i/10.0

        Q_plus = mdp_plus.compute_Qs(gamma=g)
        Q_minus = mdp_minus.compute_Qs(gamma=g)

        plot_Qs(Q_plus, Q_minus, f"uniform_random_break_gamma_{g}")

        V_plus, pi_plus = mdp_plus.policy_iter(gamma=g)
        V_minus, pi_minus = mdp_minus.policy_iter(gamma=g)
        Q_plus = mdp_plus.compute_Qs(pi_plus, gamma=g)
        Q_minus = mdp_minus.compute_Qs(pi_minus, gamma=g)

        plot_Qs(Q_plus, Q_minus, f"policy_iter_break_gamma_{g}")