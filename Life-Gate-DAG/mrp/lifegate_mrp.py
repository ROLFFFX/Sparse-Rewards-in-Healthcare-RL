import random
import numpy as np
from graphviz import Digraph

class LifeGateMRP:
    def __init__(self, max_num_states=20, p_terminal=0.2, max_branch=3, allow_cycles=False,  relapse_prob=0.0, seed=42):
        """
        Initializes the LifeGate MRP environment with a randomly generated DAG.

        Parameters:
        - max_num_states (int): Maximum number of states, including terminals.
        - p_terminal (float): Probability that a node becomes terminal (death/recovery).
        - max_branch (int): Maximum number of child nodes from any non-terminal node.
        - seed (int): Random seed for reproducibility.
        """
        assert max_num_states > 2, "Minimum 3 states required (1 start, 2 terminals)"
        self.max_num_states = max_num_states
        self.p_terminal = p_terminal
        self.max_branch = max_branch
        self.seed = seed
        self.transitions = {}
        self.state_roles = {}
        self.states = []
        self.state_idx = {}
        self.allow_cycles = allow_cycles
        self.relapse_prob = relapse_prob

        self.generate_dag()
        self.tagging_dag()
        

    def generate_dag(self):
        """
        Constructs a random DAG with specified terminal probability and branching factor.

        Populates:
        - self.transitions (dict): Maps each state to a dictionary of successors with transition probabilities.
        
        Example:
            {
              's0': {'s1': 0.6, 's2': 0.4},
              's1': {'death': 1.0},
              ...
            }
        """
        random.seed(self.seed)
        transitions = {}
        state_queue = ["s0"]
        all_states = ["s0"]
        next_state_id = 1

        # terminal states are defined to be absorbing states with 100% chance of transitioning to itself
        # modified ver.: if relapse_prob > 0, then recovery state is also absorbing with some chance of transitioning to itself
        transitions["death"] = {"death": 1.0}
        if self.relapse_prob > 0:
            transitions["recovery"] = {
                "recovery": 1.0 - self.relapse_prob,
                "s0": self.relapse_prob
            }
        else:
            transitions["recovery"] = {"recovery": 1.0}
        
        while len(all_states) + 2 < self.max_num_states and state_queue:
            current = state_queue.pop(0)
            transitions[current] = {}

            if random.random() < self.p_terminal:
                terminal = random.choice(["death", "recovery"])
                transitions[current][terminal] = 1.0
            else:
                num_children = random.randint(1, self.max_branch)
                remaining = self.max_num_states - len(all_states) - 2
                num_children = min(num_children, remaining)
                probs = [random.random() for _ in range(num_children)]
                probs = [p / sum(probs) for p in probs]

                for i in range(num_children):
                    if self.allow_cycles and random.random() < 0.3 and all_states:
                        child = random.choice(all_states)
                    else:
                        child = f"s{next_state_id}"
                        next_state_id += 1
                        all_states.append(child)
                        state_queue.append(child)
                    transitions[current][child] = probs[i]


        # resolve remaining queue nodes to terminal
        for current in state_queue:
            terminal = random.choice(["death", "recovery"])
            transitions[current] = {terminal: 1.0}

        self.transitions = transitions

    def tagging_dag(self):
        """
        Tagging special states for the generated DAG.

        Populates:
        - self.state_roles (dict): Mapping of each state to its role.
        - self.states (List[str]): Sorted list of all state names, including terminals.
        - self.state_idx (dict): Mapping from state name to index (represented in sparse matrix).
        
        How it works:
        """
        transitions = self.transitions
        terminal_states = {"death", "recovery"}
        state_roles = {}

        for term in terminal_states:    # terminal states are already defined from DAG
            state_roles[term] = "terminal"

        # detect immediate dead_end candidates
        dead_end = set()
        for state, dests in transitions.items():
            if state in terminal_states:
                continue
            if set(dests.keys()) == {"death"} and abs(dests["death"] - 1.0) < 1e-6:
                dead_end.add(state)

        # propagate upward (fixed point)
        changed = True
        while changed:
            changed = False
            for state, dests in transitions.items():
                if state in terminal_states or state in dead_end:
                    continue
                if all(dest in dead_end or dest == "death" for dest in dests):
                    dead_end.add(state)
                    changed = True

        # assign roles to each states
        # states that are not tagged yet are assigned to be rescue states.
        for state in transitions:
            if state in terminal_states:
                state_roles[state] = "terminal"
            elif state in dead_end:
                state_roles[state] = "dead_end"
            else:
                state_roles[state] = "rescue"

        all_states = set(transitions.keys())
        all_destinations = {s for dests in transitions.values() for s in dests}
        full_states = sorted(all_states.union(all_destinations))
        state_idx = {s: i for i, s in enumerate(full_states)}

        self.state_roles = state_roles
        self.states = full_states
        self.state_idx = state_idx

    def visualize_mrp(self, title="LifeGate-DAG MRP", fontname="Courier"):
        """
        Creates a Graphviz visualization of the MRP DAG.

        Returns:
        - dot (graphviz.Digraph): Graph object representing the MRP structure.
        """
        dot = Digraph(comment=title)
        dot.attr(rankdir='LR', fontname=fontname, size="10,10")
        dot.attr('node', fontname=fontname)
        dot.attr('edge', fontname=fontname)
    

        for state in self.transitions:
            role = self.state_roles.get(state, "rescue")
            if role == "terminal":
                color = "red" if state == "death" else "green"
            elif role == "dead_end":
                color = "gray"
            elif role == "rescue":
                color = "lightgreen"
            else:
                color = "purple"    # @NOTE: if you see purple state, there's something wrong with the generation/tagging
            dot.node(state, state, style="filled", fillcolor=color)

        for from_state, edges in self.transitions.items():
            for to_state, prob in edges.items():
                dot.edge(from_state, to_state, label=str(round(prob, 2)))
                
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label="Legend", style="solid")
            legend.attr(rank='same')
            legend.node("L_rescue", "Rescue State", style="filled", fillcolor="lightgreen")
            legend.node("L_deadend", "Dead-end State", style="filled", fillcolor="gray")
            legend.node("L_death", "Death (Terminal)", style="filled", fillcolor="red", fontcolor="white")
            legend.node("L_recovery", "Recovery (Terminal)", style="filled", fillcolor="green", fontcolor="white")
            legend.edge("L_rescue", "L_deadend", style="invis")
            legend.edge("L_deadend", "L_death", style="invis")
            legend.edge("L_death", "L_recovery", style="invis")

        return dot

    def build_mrp_matrices(self, version="recovery"):
        """
        Builds the transition matrix (P) and reward vector (R) for the MRP.

        Parameters:
        - version (str): Choose from:
            'recovery' => reward = 1 for transitions into 'recovery'; 0 otherwise.
            'death'    => reward = -1 for transitions into 'death'; 0 otherwise.

        Returns:
        - P (np.ndarray, shape (n, n)): Transition probability matrix where P[i, j] is
              the probability of transitioning from state i to state j. sparse, each row sum to 1.
        - R (np.ndarray, shape (n)): Reward vector where R[i] is the expected reward from state i.
        
        @NOTE: the first two entries for both P and R would be terminal states
        """
        n = len(self.states)
        P = np.zeros((n, n))
        R = np.zeros(n)

        recovery_idx = self.state_idx["recovery"]
        death_idx = self.state_idx["death"]

        for from_state, to_probs in self.transitions.items():
            i = self.state_idx[from_state]

            for to_state, prob in to_probs.items():
                j = self.state_idx[to_state]
                P[i, j] = prob

                if version == "recovery" and to_state == "recovery":
                    R[i] += prob * 1.0
                elif version == "death" and to_state == "death":
                    R[i] += prob * -1.0

        # terminal states are absorbing and have zero reward
        P[recovery_idx, :] = 0
        P[recovery_idx, recovery_idx] = 1.0
        R[recovery_idx] = 0.0

        P[death_idx, :] = 0
        P[death_idx, death_idx] = 1.0
        R[death_idx] = 0.0

        return P, R
    
    def build_mrp_matrices_partial_reward(self, version="recovery", dead_end_bonus=0.0):
        """
        Builds the transition matrix (P) and reward vector (R) for the MRP.

        Parameters:
        - version (str): 
            'recovery' => reward = 1 for transitions into 'recovery'; 0 otherwise.
            'death'    => reward = -1 for transitions into 'death'; 0 otherwise.
        - dead_end_bonus (float): Reward for entering dead-end states (default 0 = no shaping).

        Returns:
        - P (np.ndarray, shape (n, n)): Transition matrix.
        - R (np.ndarray, shape (n)): Reward vector.
        """
        n = len(self.states)
        P = np.zeros((n, n))
        R = np.zeros(n)

        recovery_idx = self.state_idx["recovery"]
        death_idx = self.state_idx["death"]

        for from_state, to_probs in self.transitions.items():
            i = self.state_idx[from_state]

            for to_state, prob in to_probs.items():
                j = self.state_idx[to_state]
                P[i, j] = prob

                # Terminal reward logic
                if version == "recovery" and to_state == "recovery":
                    R[i] += prob * 1.0
                elif version == "death" and to_state == "death":
                    R[i] += prob * -1.0

                # Dead-end bonus shaping
                elif self.state_roles.get(to_state) == "dead_end":
                    R[i] += prob * dead_end_bonus

        # Terminal states are absorbing
        P[recovery_idx, :] = 0
        P[recovery_idx, recovery_idx] = 1.0
        R[recovery_idx] = 0.0

        P[death_idx, :] = 0
        P[death_idx, death_idx] = 1.0
        R[death_idx] = 0.0

        return P, R

    def build_mrp_matrices_with_relapse(self, version="recovery", dead_end_bonus=0.0):
        """
        Version-aware MRP builder that respects relapse dynamics.

        If relapse_prob > 0, recovery is not absorbing.
        If relapse_prob = 0, recovery is absorbing.

        Returns:
        - P: np.ndarray (n x n) transition matrix
        - R: np.ndarray (n) reward vector
        """
        n = len(self.states)
        P = np.zeros((n, n))
        R = np.zeros(n)

        recovery_idx = self.state_idx["recovery"]
        death_idx = self.state_idx["death"]

        for from_state, to_probs in self.transitions.items():
            i = self.state_idx[from_state]
            for to_state, prob in to_probs.items():
                j = self.state_idx[to_state]
                P[i, j] = prob

                if version == "recovery" and to_state == "recovery":
                    R[i] += prob * 1.0
                elif version == "death" and to_state == "death":
                    R[i] += prob * -1.0
                elif self.state_roles.get(to_state) == "dead_end":
                    R[i] += prob * dead_end_bonus

        # Death is always absorbing
        P[death_idx, :] = 0
        P[death_idx, death_idx] = 1.0
        R[death_idx] = 0.0

        # Only force recovery to be absorbing if relapse_prob = 0
        if self.relapse_prob == 0.0:
            P[recovery_idx, :] = 0
            P[recovery_idx, recovery_idx] = 1.0
            R[recovery_idx] = 0.0

        return P, R



    
    def value_iter_mrp(self, P, R, gamma=1.0, epsilon=1e-6, max_iters=1000):
        """
        value iteration on MRP.

        Parameters:
        - P (np.ndarray, shape [n, n]): Transition matrix.
        - R (np.ndarray, shape [n]): Reward vector.
        - gamma (float): Discount factor (default to 1 for our purpose)
        - epsilon (float): small convergence threshold.
        - max_iters (int): maximum number of iterations.    @FIXME: is max_iters needed if no cycles?

        Returns:
        - V (np.ndarray, shape (n)): Estimated value function for each state. First 2 entries are terminal states.
        """
        n_states = P.shape[0]
        V = np.zeros(n_states)
        for _ in range(max_iters):
            V_next = R + gamma * P @ V
            if np.max(np.abs(V_next - V)) < epsilon:
                break
            V = V_next
        return V


    