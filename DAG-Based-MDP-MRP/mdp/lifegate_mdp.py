import random
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

class LifeGate_MDP:
    def __init__(self, max_num_states=20, p_terminal=0.2, max_actions=2, seed=42, mode="+-"):
        assert max_num_states > 2, "Minimum 3 states required (1 start, 2 terminals)"
        self.max_num_states = max_num_states
        self.p_terminal = p_terminal
        self.max_actions = max_actions
        self.seed = seed
        self.mode = mode

        self.transitions = {}
        self.state_roles = {}
        self.states = []
        self.state_idx = {}
        self.actions = [str(i) for i in range(max_actions)]

        self.generate_dag()
        self.tagging_dag()

    def reward_fn(self, terminal_state):
        if self.mode == "+":
            return 1.0 if terminal_state == "recovery" else 0.0
        elif self.mode == "-":
            return -1.0 if terminal_state == "death" else 0.0
        else:  # "+-" and default
            return 1.0 if terminal_state == "recovery" else -1.0

    def generate_dag(self):
        # setup seed and queues
        random.seed(self.seed)
        transitions = {}
        state_queue = ["s0"]
        all_states = ["s0"]
        next_state_id = 1

        # transitions for death and recovery - self absorbing
        transitions["death"] = {a: {"death": (1.0, 0.0)} for a in self.actions}
        transitions["recovery"] = {a: {"recovery": (1.0, 0.0)} for a in self.actions}

        # main loop - while the max hasnt been reached, stochastically generate the next state's children
        while len(all_states) + 2 < self.max_num_states and state_queue:
            current = state_queue.pop(0)
            transitions[current] = {}

            for a in self.actions:
                # either 1 or 2 children for each action
                num_children = random.randint(1, 2)
                remaining = self.max_num_states - len(all_states) - 2
                remaining = 1 if remaining == 0 else remaining
                num_children = min(num_children, remaining)

                # calculates and normalizes random probabilities for each child
                probs = [random.random() for _ in range(num_children)]
                probs = [p / sum(probs) for p in probs]

                # child map holds the probability of transitioning to each child, children are appended to appropriate data structures
                child_map = {}
                for i in range(num_children):
                    child = f"s{next_state_id}"
                    next_state_id += 1
                    all_states.append(child)
                    state_queue.append(child)
                    child_map[child] = probs[i]

                # stochastically decide if state transitions to terminal state
                for terminal in ["death", "recovery"]:
                    if random.random() < self.p_terminal:
                        child_map[terminal] = child_map.get(terminal, 0.0) + random.uniform(0.05, 0.3)

                # normalize the probabilities if any went above 1 (because of the terminal probabilities)
                total = sum(child_map.values())
                child_map = {k: v / total for k, v in child_map.items()}

                # add proper rewards based on the mode
                transitions[current][a] = {}
                for state, prob in child_map.items():
                    if state == "recovery":
                        reward = 1.0 if self.mode == "+-" or self.mode == "+" else 0
                    elif state == "death":
                        reward = -1.0 if self.mode == "+-" or self.mode  == "-" else 0
                    else:
                        reward = 0.0
                    transitions[current][a][state] = (prob, reward)

        # make sure that the rest of the queue is handled once we have reached the max
        for current in state_queue:
            transitions[current] = {}
            for a in self.actions:
                child_map = {}
                for terminal in ["death", "recovery"]:
                    if random.random() < self.p_terminal:
                        child_map[terminal] = random.uniform(0.2, 0.8)
                if not child_map:
                    child_map["death"] = 1.0
                else:
                    total = sum(child_map.values())
                    child_map = {k: v / total for k, v in child_map.items()}

                transitions[current][a] = {}
                for state, prob in child_map.items():
                    if state == "recovery":
                        reward = 1.0 if self.mode == "+-" or self.mode == "+" else 0
                    elif state == "death":
                        reward = -1.0 if self.mode == "+-" or self.mode == "-" else 0
                    else:
                        reward = 0.0
                    transitions[current][a][state] = (prob, reward)

        self.transitions = transitions

    # tagging function for graph depiction
    def tagging_dag(self):
        terminal_states = {"death", "recovery"}
        state_roles = {s: "terminal" for s in terminal_states}

        for state in self.transitions:
            if state not in terminal_states:
                state_roles[state] = "rescue"

        all_states = set(self.transitions.keys())
        all_destinations = {s for dests in self.transitions.values() for d in dests.values() for s in d}
        full_states = sorted(all_states.union(all_destinations))
        state_idx = {s: i for i, s in enumerate(full_states)}

        self.state_roles = state_roles
        self.states = full_states
        self.state_idx = state_idx

    def build_mdp_matrices(self):
        n = len(self.states)
        P_dict = {}
        R_dict = {}

        for a in self.actions:
            P = np.zeros((n, n))
            R = np.zeros(n)

            for from_state, action_map in self.transitions.items():
                if a not in action_map:
                    continue
                i = self.state_idx[from_state]
                for to_state, (prob, reward) in action_map[a].items():
                    j = self.state_idx[to_state]
                    P[i, j] = prob
                    R[i] = reward

            for term in ["death", "recovery"]:
                idx = self.state_idx[term]
                P[idx, :] = 0
                P[idx, idx] = 1.0
                R[idx] = 0.0

            P_dict[a] = P
            R_dict[a] = R

        return self.actions, P_dict, R_dict

    def visualize_mdp(self, title="LifeGate-MDP"):
        dot = Digraph(comment=title)
        dot.attr(rankdir='LR')

        for state in self.transitions:
            role = self.state_roles.get(state, "rescue")
            if role == "terminal":
                color = "red" if state == "death" else "green"
            elif role == "rescue":
                color = "lightgreen"
            else:
                color = "purple"
            dot.node(state, state, style="filled", fillcolor=color)

        for from_state, actions in self.transitions.items():
            for a, to_states in actions.items():
                for to_state, (prob, _) in to_states.items():
                    dot.edge(from_state, to_state, label=f"{a}: {prob:.2f}")

        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label="Legend", style="solid")
            legend.attr(rank='same')
            legend.node("L_rescue", "Rescue State", style="filled", fillcolor="lightgreen")
            legend.node("L_death", "Death (Terminal)", style="filled", fillcolor="red", fontcolor="white")
            legend.node("L_recovery", "Recovery (Terminal)", style="filled", fillcolor="green", fontcolor="white")
            legend.edge("L_rescue", "L_death", style="invis")
            legend.edge("L_death", "L_recovery", style="invis")

        return dot

    def visualize_rewards(self, title="LifeGate-MDP (Rewards)"):
        dot = Digraph(comment=title)
        dot.attr(rankdir='LR')

        for state in self.transitions:
            role = self.state_roles.get(state, "rescue")
            if role == "terminal":
                color = "red" if state == "death" else "green"
            elif role == "rescue":
                color = "lightgreen"
            else:
                color = "purple"
            dot.node(state, state, style="filled", fillcolor=color)

        for from_state, actions in self.transitions.items():
            for a, to_states in actions.items():
                for to_state, (prob, reward) in to_states.items():
                    dot.edge(from_state, to_state, label=f"{a}: R={reward:.2f}")

        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label="Legend", style="solid")
            legend.attr(rank='same')
            legend.node("L_rescue", "Rescue State", style="filled", fillcolor="lightgreen")
            legend.node("L_death", "Death (Terminal)", style="filled", fillcolor="red", fontcolor="white")
            legend.node("L_recovery", "Recovery (Terminal)", style="filled", fillcolor="green", fontcolor="white")
            legend.edge("L_rescue", "L_death", style="invis")
            legend.edge("L_death", "L_recovery", style="invis")

        return dot

    def policy_eval(self, policy=None, theta=0.0001, gamma=1.0):
        V = {s: 0.0 for s in self.states}

        while True:
            delta = 0.0
            for state in self.states:
                if state in ("death", "recovery"):
                    continue

                if policy:
                    action = policy.get(state)
                    if action is None or action not in self.transitions[state]:
                        continue
                    transitions = self.transitions[state][action]
                    v = sum(prob * (reward + gamma * V[next_state])
                            for next_state, (prob, reward) in transitions.items())
                else:
                    available_actions = self.transitions[state].keys()
                    num_actions = len(available_actions)
                    v = 0.0
                    for a in available_actions:
                        transitions = self.transitions[state][a]
                        for next_state, (prob, reward) in transitions.items():
                            v += (1 / num_actions) * prob * (reward + gamma * V[next_state])

                delta = max(delta, abs(v - V[state]))
                V[state] = v

            if delta < theta:
                break

        return V
    
    def policy_iter_deter(self, theta=1e-4, gamma=1.0):
        policy = {}
        final_policy = {}

        for state in self.states:
            if state in ("death", "recovery"):
                continue
            actions = sorted(self.transitions[state].keys())
            if actions:
                policy[state] = actions[0]
                final_policy[state] = [actions[0]]

        while True:
            V = self.policy_eval(policy, theta=theta, gamma=gamma)
            policy_stable = True

            for state in self.states:
                if state in ("death", "recovery"):
                    continue

                old_action = policy.get(state)
                best_value = float('-inf')
                best_actions = []

                for a in sorted(self.transitions[state].keys()):
                    q_value = sum(
                        prob * (reward + gamma * V[next_state])
                        for next_state, (prob, reward) in self.transitions[state][a].items()
                    )
                    if q_value > best_value + theta:
                        best_value = q_value
                        best_actions = [a]
                    elif abs(q_value - best_value) < theta:
                        best_actions.append(a)

                chosen_action = best_actions[0]
                if old_action != chosen_action:
                    policy_stable = False
                    policy[state] = chosen_action

                final_policy[state] = best_actions

            if policy_stable:
                break

        V = {k: round(v, 8) for k, v in V.items()}
        return V, final_policy



    def policy_iter(self, theta=0.0001, gamma=1.0):
        policy = {}
        for state in self.states:
            if state in ("death", "recovery"):
                continue
            actions = list(self.transitions[state].keys())
            if actions:
                policy[state] = actions[0]

        while True:
            V = self.policy_eval(policy, theta=theta, gamma=gamma)

            policy_stable = True
            for state in self.states:
                if state in ("death", "recovery"):
                    continue

                old_action = policy.get(state)
                best_action = None
                best_value = float('-inf')

                for a in self.transitions[state]:
                    q_value = 0.0
                    for next_state, (prob, reward) in self.transitions[state][a].items():
                        q_value += prob * (reward + gamma * V[next_state])
                    if q_value >= best_value:
                        best_value = q_value
                        best_action = a

                if old_action != best_action:
                    policy_stable = False
                    policy[state] = best_action

            if policy_stable:
                break

        V = {k: round(v, 8) for k, v in V.items()}
        return V, policy
    
    

    def compute_Qs(self, policy=None, gamma=1.0):
        """
        Compute Q(s, a) for all state-action pairs given a policy.

        Parameters:
            policy (dict): A dict mapping states to chosen actions
            gamma (float): Discount factor

        Returns:
            dict: Nested dict Q[state][action] = Q-value
        """
        V = self.policy_eval(policy, gamma=gamma)
        Q = {}

        for state in self.states:
            if state in ("death", "recovery"):
                continue

            Q[state] = {}
            for action in self.transitions[state]:
                q_value = 0.0
                for next_state, (prob, reward) in self.transitions[state][action].items():
                    q_value += prob * (reward + gamma * V[next_state])
                Q[state][action] = q_value

        return Q

    def get_terminal_probabilities(self, policy, state, max_depth=50, depth=0):
        """
        Returns the probabilities of transitioning to 'death' and 'recovery' 
        from the given state using the given policy.
        
        Parameters:
            transition_graph (dict): Nested dict representing the transitions.
            state (str): The current state.
            policy (dict): Maps each state to an action (as int or str).

        Returns:
            (float, float): Tuple of (death_probability, recovery_probability)
        """
        if depth > max_depth:
            return 0.0, 0.0

        action = policy[state]
        if action is None:
            raise ValueError(f"No action found in policy for state '{state}'")

        transitions = self.transitions.get(state, {}).get(str(action), {})

        death_prob = 0.0
        recovery_prob = 0.0

        for target_state in transitions.keys():
            prob, reward = transitions.get(target_state)
            if target_state == 'death':
                death_prob += prob
            elif target_state == 'recovery':
                recovery_prob += prob
            else:
                target_recovery, target_death = self.get_terminal_probabilities(policy, target_state, depth=depth+1)
                recovery_prob += prob * target_recovery
                death_prob += prob * target_death

        return recovery_prob, death_prob



def add_Qs(q1, q2):
    result = {}
    for state in q1:
        result[state] = {}
        for action in q1[state]:
            result[state][action] = q1[state][action] + q2[state][action]
    return result

def subtract_Qs(q1, q2):
    result = {}
    for state in q1:
        result[state] = {}
        for action in q1[state]:
            result[state][action] = q1[state][action] - q2[state][action]
    return result


def plot_Vs(v_plus, v_minus, title):
    sns.set(style="whitegrid")
    excluded = ['recovery', 'death']
    keys = list(k for k in v_plus.keys() & v_minus.keys() if k not in excluded)

    x = [v_minus[k] for k in keys]
    y = [v_plus[k] for k in keys]

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, color='dodgerblue', edgecolor='black', s=100, alpha=0.7)
    plt.plot([-1, 0], [0, 1], linestyle='--', color='grey', linewidth=1.5)


    plt.xlabel('V- (Negative Reward Value)', fontsize=14)
    plt.ylabel('V+ (Positive Reward Value)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_Qs(q_plus, q_minus, title, size=(3, 3), show=False):
    """
    Scatter plot comparing Q-values from two MDP variants (e.g., positive vs negative reward settings).
    
    Parameters:
        q_plus (dict): Q-values from the MDP with positive reward setup (e.g., mode="+")
        q_minus (dict): Q-values from the MDP with negative reward setup (e.g., mode="-")
        title (str): Plot title
    """
    sns.set(style="whitegrid")
    x, y, labels = [], [], []

    for state in q_plus:
        if state in ('death', 'recovery'):
            continue
        for action in q_plus[state]:
            if state in q_minus and action in q_minus[state]:
                x_val = q_minus[state][action]
                y_val = q_plus[state][action]
                x.append(x_val)
                y.append(y_val)
                labels.append(f"{state}, a={action}")

    plt.figure(figsize=size)
    plt.scatter(x, y, color='mediumseagreen', edgecolor='black', s=100, alpha=0.3)

    plt.plot([-1, 0], [0, 1], linestyle='--', color='grey', linewidth=1.5)

    plt.xlabel('$Q_-$', fontsize=11)
    plt.ylabel('$Q_+$', fontsize=11)
    # plt.title(title, fontsize=12, fontweight='bold')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/{title}.pdf", format='pdf', bbox_inches='tight')

    if show:
        plt.show()

def plot_mixed(q_mixed, q_pm, title, size=(3, 3), show=False):
    sns.set(style="whitegrid")
    x, y, labels = [], [], []

    for state in q_mixed:
        if state in ('death', 'recovery'):
            continue
        for action in q_pm[state]:
            if state in q_mixed and action in q_pm[state]:
                x_val = q_pm[state][action]
                y_val = q_mixed[state][action]
                x.append(x_val)
                y.append(y_val)
                labels.append(f"{state}, a={action}")

    plt.figure(figsize=size)
    plt.scatter(x, y, color='mediumseagreen', edgecolor='black', s=100, alpha=0.3)

    plt.plot([-1, 1], [-1, 1], linestyle='--', color='grey', linewidth=1.5)

    plt.xlabel('$Q_{\pm}$', fontsize=11)
    plt.ylabel('$Q_+ + Q_-$', fontsize=11)
    # plt.title(title, fontsize=12, fontweight='bold')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/{title}.pdf", format='pdf', bbox_inches='tight')

    if show:
        plt.show() 

def compare_pis(p1, p2, mdp1, mdp2):
    for state in p1:
        if p1.get(state) != p2.get(state):
            print()
            print("mismatch: ", state, " - action p1 ", p1.get(state), " action p2 ", p2.get(state))
            plus_survival_prob, plus_death_prob = mdp1.get_terminal_probabilities(p2, state)
            minus_survival_prob, minus_death_prob = mdp2.get_terminal_probabilities(p2, state)
            print("Transition probabilities for mdp_plus: survival - ", plus_survival_prob, " death - ", plus_death_prob)
            print("Transition probabilities for mdp_minus: survival - ", minus_survival_prob, " death - ", minus_death_prob)
            print()
            

    return sum(p1.get(state) != p2.get(state) for state in p1)

def run_lifegate(mdp_plus, mdp_minus, title_base):
    print(f"-----uniform random {title_base}-----")

    Q_plus = mdp_plus.compute_Qs()
    Q_minus = mdp_minus.compute_Qs()

    plot_Qs(Q_plus, Q_minus, f"uniform_random_{title_base}")

    print()
    print()

    print(f"-----policy iteration {title_base}-----")

    print()
    print()

    V_plus, pi_plus = mdp_plus.policy_iter()
    V_minus, pi_minus = mdp_minus.policy_iter()
    Q_plus = mdp_plus.compute_Qs(pi_plus)
    Q_minus = mdp_minus.compute_Qs(pi_minus)

    print("number of mismatching actions plus vs minus", title_base, ": ", compare_pis(pi_plus, pi_minus, mdp_plus, mdp_minus))

    plot_Qs(Q_plus, Q_minus, f"policy_iter_{title_base}")

def run_mixed(mdp_plus, mdp_minus, mdp_mixed, title_base):
    print(f"-----uniform random mixed-----")

    Q_plus = mdp_plus.compute_Qs()
    Q_minus = mdp_minus.compute_Qs()
    Q_mixed = mdp_mixed.compute_Qs()

    Q_ppm = add_Qs(Q_plus, Q_minus)

    plot_mixed(Q_mixed, Q_ppm, f"uniform_random_{title_base}")

    print()
    print()

    print(f"-----policy iteration mixed-----")

    print()
    print()

    V_plus, pi_plus = mdp_plus.policy_iter()
    V_minus, pi_minus = mdp_minus.policy_iter()
    V_mixed, pi_mixed = mdp_mixed.policy_iter()

    Q_plus = mdp_plus.compute_Qs(pi_plus)
    Q_minus = mdp_minus.compute_Qs(pi_minus)
    Q_mixed = mdp_mixed.compute_Qs(pi_mixed)

    Q_ppm = add_Qs(Q_plus, Q_minus)

    print("number of mismatching actions plus vs mixed", title_base, ": ", compare_pis(pi_plus, pi_mixed, mdp_plus, mdp_mixed))
    
    print("number of mismatching actions minus vs mixed", title_base, ": ", compare_pis(pi_minus, pi_mixed, mdp_plus, mdp_mixed))

    plot_mixed(Q_mixed, Q_ppm, f"policy_iter_{title_base}")