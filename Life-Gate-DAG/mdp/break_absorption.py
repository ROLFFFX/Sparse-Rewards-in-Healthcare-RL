from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import run_lifegate
import matplotlib as plt
import pprint

def break_absorption():

    mdp_plus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="-")

    mdp_plus.states.append('s3')
    mdp_minus.states.append('s3')

    mdp_plus.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}
    mdp_minus.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}

    mdp_plus.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}
    mdp_plus.transitions['s1']['1']['s3'] = (0.48, 0.0)
    mdp_plus.transitions['s1']['1']['recovery'] = (0.52, 1.0)
    del mdp_plus.transitions['s1']['1']['death']
    mdp_plus.transitions['s2']['1']['recovery'] = (0.5, 1.0)
    mdp_plus.transitions['s2']['1']['s3'] = (0.5, 0.0)

    mdp_minus.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}
    mdp_minus.transitions['s1']['1']['s3'] = (0.48, 0.0)
    mdp_minus.transitions['s1']['1']['recovery'] = (0.52, 0.0)
    del mdp_minus.transitions['s1']['1']['death']
    mdp_minus.transitions['s2']['1']['recovery'] = (0.5, 0.0)
    mdp_minus.transitions['s2']['1']['s3'] = (0.5, 0.0)

    run_lifegate(mdp_plus, mdp_minus, 'break_absorption')