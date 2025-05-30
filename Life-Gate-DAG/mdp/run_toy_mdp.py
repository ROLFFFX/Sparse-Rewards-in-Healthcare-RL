from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import run_lifegate
from lifegate_mdp import run_mixed
import pprint

def run_toy_mdp():
    mdp_plus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="-")
    mdp_mixed = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="+-") 

    run_lifegate(mdp_plus, mdp_minus, "toy_mdp")
    run_mixed(mdp_plus, mdp_minus, mdp_mixed, "toy_mixed")