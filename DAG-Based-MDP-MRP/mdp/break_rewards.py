from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import run_lifegate
from lifegate_mdp import run_mixed

def break_rewards(bonus_reward=0.1):

    def add_rewards(mdp):
        terminal_states = {'death', 'recovery'}
        for state in mdp.transitions:
            if state in terminal_states:
                continue 
            for action in mdp.transitions[state]:
                for next_state in mdp.transitions[state][action]:
                    prob, reward = mdp.transitions[state][action][next_state]
                    if next_state not in terminal_states:
                        mdp.transitions[state][action][next_state] = (prob, reward + bonus_reward)

    mdp_plus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="-")
    mdp_mixed = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="+-")

    add_rewards(mdp_plus)
    add_rewards(mdp_minus)
    add_rewards(mdp_mixed)

    run_lifegate(mdp_plus, mdp_minus, "break_rewards")
    run_mixed(mdp_plus, mdp_minus, mdp_mixed, "break_rewards_mixed")