import pandas as pd
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm

def get_state(row: int) -> int:
    return row // 25

def get_action(row: int) -> int:
    return row % 25

def get_s_a(row) -> tuple[int, int]:
    return (get_state(row), get_action(row))

def detect_cycles(transitions: np.ndarray) -> None:
    # 1. check for probability of self-transition.
    # 2. check for self-absorbing states.
    def self_transition(transitions: np.ndarray) -> None:
        self_transition_s_a = []
        self_transition_state = set()
        self_absorbing_s_a = []
        self_absorbing_state = set()
        for i, row in enumerate(transitions):
            state_i = get_state(i)
            # check for current state column for non-zero transition probability
            if (row[state_i] > 0):
                self_transition_state.add(state_i)
                self_transition_s_a.append(get_s_a(i))
                if (np.isclose(row[state_i], 1, 1e-4)):
                    self_absorbing_state.add(state_i)
                    self_absorbing_s_a.append(get_s_a(i))
                    
        print(f"There are {len(self_transition_state)} self-transition states, {len(self_transition_s_a)} (s, a) pairs.")
        print(f"There are {len(self_absorbing_state)} self-absorbing states {self_absorbing_state}, {len(self_absorbing_s_a)} (s, a) pairs.")
    self_transition(transitions)
                
    pass

'''
    To detect cycles:
    1. check for probability of self absorption: get current state, check for non-zero probability of self transition
    2. absorbing state: subcase of 1, check probability for self transition with probability of 1
    3. DFS: follow 1/2 for cases above for all paths
    
    Cases to be reported:
    1. self-absorbing state: given that positive terminal state is given 0 reward, check for such cases and report
    2. dead-end state: transition to a absorbing state with probability of 1
    
'''
if __name__ == "__main__":

    transitions = genfromtxt("../data/icu-sepsis-csv-tables/transitionFunction.csv", delimiter=",")
    detect_cycles(transitions=transitions)