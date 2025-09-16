from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import plot_Qs
from lifegate_mdp import compare_pis
import matplotlib as plt
import pprint

def invert_classes_break_gamma():
    mdp_plus = LifeGate_MDP(max_num_states=200, p_terminal=0.7, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=200, p_terminal=0.7, max_actions=2, seed=28, mode="-")
    mdp_mixed = LifeGate_MDP(max_num_states=200, p_terminal=0.7, max_actions=2, seed=28, mode="+-")

    dot = mdp_mixed.visualize_mdp(title="inverrted_classes_mdp")

    dot.render("inverted_classes_mdp", format="pdf", cleanup=True)

    for i in range(1, 4):
        g = 1.0 - i/10.0

        Q_plus = mdp_plus.compute_Qs(gamma=g)
        Q_minus = mdp_minus.compute_Qs(gamma=g)
        Q_mixed = mdp_mixed.compute_Qs(gamma=g)

        plot_Qs(Q_plus, Q_minus, f"uniform_random_break_gamma_inverted_classes{g}")

        V_plus, pi_plus = mdp_plus.policy_iter(gamma=g)
        V_minus, pi_minus = mdp_minus.policy_iter(gamma=g)
        V_plus = mdp_plus.policy_eval(pi_plus)
        V_minus = mdp_minus.policy_eval(pi_minus)
        V_mixed, pi_mixed = mdp_mixed.policy_iter(gamma=g)
        Q_plus = mdp_plus.compute_Qs(pi_plus, gamma=g)
        Q_minus = mdp_minus.compute_Qs(pi_minus, gamma=g)

        plot_Qs(Q_plus, Q_minus, f"policy_iter_break_gamma_inverted_classes{g}")

        print("number of mismatched actions for plus vs minus gamma = ", g, ": ", compare_pis(pi_plus, pi_minus, mdp_plus, mdp_minus))
        print("number of mismatched actions for plus vs mixed gamma = ", g, ": ", compare_pis(pi_plus, pi_mixed, mdp_plus, mdp_mixed))
        print("number of mismatched actions for minus vs mixed gamma = ", g, ": ", compare_pis(pi_minus, pi_mixed, mdp_minus, mdp_mixed))

invert_classes_break_gamma()