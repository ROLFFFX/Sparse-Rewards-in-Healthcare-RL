from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import run_lifegate
from lifegate_mdp import run_mixed

def run_large_mdp():
    mdp_plus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="-")
    mdp_mixed = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=28, mode="+-")

    run_lifegate(mdp_plus, mdp_minus, "large_mdp")
    run_mixed(mdp_plus, mdp_minus, mdp_mixed, "large_mixed")