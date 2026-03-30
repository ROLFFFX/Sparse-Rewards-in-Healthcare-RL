from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import plot_Qs
from lifegate_mdp import compare_pis
import matplotlib as plt
import pprint
import copy

def flip_terminal_outcomes(mdp):
    mode = 2 if mdp.mode == "+-" else 1 # used for arithmetic to calculate new reward
    flipped = {}

    for state, actions in mdp.transitions.items():
        if state == "death":
            new_state = "recovery"
        elif state == "recovery":
            new_state = "death"
        else:
            new_state = state

        flipped[new_state] = {}

        for action, outcomes in actions.items():
            flipped[new_state][action] = {}
            for next_state, (prob, reward) in outcomes.items():
                if next_state == "death":
                    new_next_state = "recovery"
                    new_reward = reward + mode if state != "death" else 0.0
                elif next_state == "recovery":
                    new_next_state = "death"
                    new_reward = reward - mode if state != "recovery" else 0.0
                else:
                    new_next_state = next_state
                    new_reward = reward

                flipped[new_state][action][new_next_state] = (prob, new_reward)
    return flipped

def new_flipped_mdp(mdp):
    new_mdp = copy.deepcopy(mdp)
    new_mdp.transitions = flip_terminal_outcomes(mdp)
    new_mdp.tagging_dag()
    return new_mdp


def invert_classes_break_gamma():
    s = 1

    mdp_plus_pos = LifeGate_MDP(max_num_states=200, p_terminal=0.01, max_actions=2, seed=s, mode="+")
    mdp_minus_pos = LifeGate_MDP(max_num_states=200, p_terminal=0.01, max_actions=2, seed=s, mode="-")
    mdp_mixed_pos = LifeGate_MDP(max_num_states=200, p_terminal=0.01, max_actions=2, seed=s, mode="+-")

    mdp_plus_neg = new_flipped_mdp(mdp_plus_pos)
    mdp_minus_neg = new_flipped_mdp(mdp_minus_pos)
    mdp_mixed_neg = new_flipped_mdp(mdp_mixed_pos)

    for i in range(1, 4):
        g = 1.0 - i/10.0

        Q_plus_pos = mdp_plus_pos.compute_Qs(gamma=g)
        Q_minus_pos = mdp_minus_pos.compute_Qs(gamma=g)
        Q_mixed_pos = mdp_mixed_pos.compute_Qs(gamma=g)

        Q_plus_neg = mdp_plus_neg.compute_Qs(gamma=g)
        Q_minus_neg = mdp_minus_neg.compute_Qs(gamma=g)
        Q_mixed_neg = mdp_mixed_neg.compute_Qs(gamma=g)

        plot_Qs(Q_plus_pos, Q_minus_pos, f"uniform_random_positive_class{g}")
        plot_Qs(Q_plus_neg, Q_minus_neg, f"uniform_random_negative_class{g}")

        V_plus, pi_plus = mdp_plus_pos.policy_iter(gamma=g)
        V_minus, pi_minus = mdp_minus_pos.policy_iter(gamma=g)
        V_plus = mdp_plus_pos.policy_eval(pi_plus)
        V_minus = mdp_minus_pos.policy_eval(pi_minus)
        V_mixed, pi_mixed = mdp_mixed_pos.policy_iter(gamma=g)
        Q_plus = mdp_plus_pos.compute_Qs(pi_plus, gamma=g)
        Q_minus = mdp_minus_pos.compute_Qs(pi_minus, gamma=g)

        plot_Qs(Q_plus, Q_minus, f"policy_iter_positive_class{g}")

        V_plus, pi_plus = mdp_plus_neg.policy_iter(gamma=g)
        V_minus, pi_minus = mdp_minus_neg.policy_iter(gamma=g)
        V_plus = mdp_plus_neg.policy_eval(pi_plus)
        V_minus = mdp_minus_neg.policy_eval(pi_minus)
        V_mixed, pi_mixed = mdp_mixed_neg.policy_iter(gamma=g)
        Q_plus = mdp_plus_neg.compute_Qs(pi_plus, gamma=g)
        Q_minus = mdp_minus_neg.compute_Qs(pi_minus, gamma=g)

        plot_Qs(Q_plus, Q_minus, f"policy_iter_negative_class{g}")

invert_classes_break_gamma()