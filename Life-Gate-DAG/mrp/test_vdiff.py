from lifegate_mrp import LifeGateMRP
import numpy as np
import random 
from tqdm import tqdm

def validate_vdiff(max_num_states=15, p_terminal=0.5, max_branch=4, seed=1008):
    mdp = LifeGateMRP(
        max_num_states=max_num_states,
        p_terminal=p_terminal,
        max_branch=max_branch,
        seed=seed
    )
    P_r, R_r = mdp.build_mrp_matrices(version="recovery")
    P_d, R_d = mdp.build_mrp_matrices(version="death")
    V_r = mdp.value_iter_mrp(P_r, R_r)
    V_d = mdp.value_iter_mrp(P_d, R_d)
    V_diff = V_d - V_r
    terminal_states = {"death", "recovery"}
    success = True
    for state, idx in mdp.state_idx.items():
        if state in terminal_states:
            continue
        if not np.isclose(V_diff[idx], -1.0, atol=0):
            print(f"Mismatch at {state}: V_d={V_d[idx]:.4f}, V_r={V_r[idx]:.4f}, diff={V_diff[idx]:.4f}")
            success = False
    
    # return V_diff, V_d, V_r

if __name__ == "__main__":
    for i in tqdm(range(0, 10000), desc=f"Processing..."):
        num_states = random.randint(3, 2000)     # generate random number of states
        p_terminal = random.uniform(0, 1)       # generate random p_terminal
        max_branch = random.randint(1, 20)      # generate random max_branch
        seed = random.randint(1, 1018)          # seed
        validate_vdiff(num_states, p_terminal, max_branch,seed)
    print("Testing complete.")
        

    
