from lifegate_mrp import LifeGateMRP
import pprint

if __name__ == "__main__":
    '''
    Example for how to use lifegate_mrp class.
    '''
    # step1: generate dag
    mrp = LifeGateMRP(max_num_states=200, p_terminal=0.2, max_branch=4, seed=1211 )    # init mrp class
    graph = mrp.visualize_mrp()
    graph.render(directory='generated-dag', view=True)                          # display generated dag
    
    # step2: generating P (transition probabilities) and R (reward function) for each version of mrp (MRPr, MRPd)
    P_r, R_r = mrp.build_mrp_matrices(version="recovery")   
    P_d, R_d = mrp.build_mrp_matrices(version="death")      
    # pprint.pp(P_d)
    # pprint.pp(R_d)
    
    print(R_d)

    # step3: generate V tables respectively using value_iter_mrp
    # @NOTE: first two entries would be terminal states for V tables.
    V_d = mrp.value_iter_mrp(P_d, R_d)
    V_r = mrp.value_iter_mrp(P_r, R_r)
    
    print("V_r - V_d")
    pprint.pp((V_d - V_r)[2:])  # expect to see all -1
    

