from lifegate_mdp import LifeGate_MDP
from lifegate_mdp import run_lifegate
from lifegate_mdp import run_mixed
import matplotlib as plt
import pprint

def break_absorption():

    mdp_plus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="+")
    mdp_minus = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="-")
    mdp_mixed = LifeGate_MDP(max_num_states=5, p_terminal=0.3, max_actions=2, seed=28, mode="+-")

    mdp_plus.states.append('s3')
    mdp_minus.states.append('s3')
    mdp_mixed.states.append('s3')
    

    mdp_plus.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}
    mdp_minus.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}
    mdp_mixed.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}

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

    mdp_mixed.transitions['s3'] = {'0': {'s3': (1.0, 0.0)}, '1': {'s3': (1.0, 0.0)}}
    mdp_mixed.transitions['s1']['1']['s3'] = (0.48, 0.0)
    mdp_mixed.transitions['s1']['1']['recovery'] = (0.52, 1.0)
    del mdp_mixed.transitions['s1']['1']['death']
    mdp_mixed.transitions['s2']['1']['recovery'] = (0.5, 1.0)
    mdp_mixed.transitions['s2']['1']['s3'] = (0.5, 0.0)

    # run_lifegate(mdp_plus, mdp_minus, 'break_absorption')
    # run_mixed(mdp_plus, mdp_minus, mdp_mixed, "break_absorption_mixed")
    
    V_plus, pi_plus = mdp_plus.policy_iter()
    V_minus, pi_minus = mdp_minus.policy_iter()
    V_mixed, pi_mixed = mdp_mixed.policy_iter()
    
    V_plus_deter, pi_plus_deter = mdp_plus.policy_iter_deter()
    V_minus_deter, pi_minus_deter = mdp_minus.policy_iter_deter()
    V_mixed_deter, pi_mixed_deter = mdp_mixed.policy_iter_deter()
    
    # Q = mdp_minus.compute_Qs()
    # pprint.pprint(Q)
    
    # print("Not deterministic policies:")
    # pprint.pprint(pi_plus)
    # pprint.pprint(pi_minus)
    # pprint.pprint(pi_mixed)
    
    # pprint.pprint(V_plus)
    # pprint.pprint(V_minus)
    
    print("-------------------------------")
    print("Deterministic policies:")
    print("Plus: ")
    pprint.pprint(pi_plus_deter)
    print("Minus: ")
    pprint.pprint(pi_minus_deter)
    print("Mixed: ")
    pprint.pprint(pi_mixed_deter)
    
    # print("-------------------------------")
    # print("Not deterministic value table:")
    # pprint.pprint(V_plus)
    # pprint.pprint(V_minus)
    # pprint.pprint(V_mixed)
    
    # print("-------------------------------")
    # print("Deterministic value table:")
    # pprint.pprint(V_plus_deter)
    # pprint.pprint(V_minus_deter)
    # pprint.pprint(V_mixed_deter)
        
break_absorption()