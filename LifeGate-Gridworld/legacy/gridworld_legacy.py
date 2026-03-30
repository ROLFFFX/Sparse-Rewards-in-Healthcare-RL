import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import itertools
import os

# Ensure plots directories exist
os.makedirs("plots", exist_ok=True)
os.makedirs("asym_plots", exist_ok=True)

# --- Helper Functions ---

def amax(arr):
    """
    Returns the list of indices where the value is close to the max.
    """
    if len(arr) == 0:
        return []
    max_value = np.max(arr)
    return [i for i, val in enumerate(arr) if np.isclose(val, max_value, rtol=0, atol=1e-12)]

def V_index(s, n):
    return (int(s/n), int(s%n))

def add_Vs(V1, V2):
    return [x + y for x, y in zip(V1, V2)]

def subtract_Vs(V1, V2):
    return [x - y for x, y in zip(V1, V2)]

def combine_Qs(q1, q2, operation="-"):
    """
    Combine two Q-value dictionaries elementwise using addition or subtraction.
    """
    if q1.keys() != q2.keys():
        raise ValueError("State sets of q1 and q2 must match")

    q_result = {}
    for s in q1:
        if len(q1[s]) != len(q2[s]):
            raise ValueError(f"Action counts differ for state {s}")
        q_result[s] = []
        for a in range(len(q1[s])):
            if operation == "+":
                q_result[s].append(q1[s][a] + q2[s][a])
            elif operation == "-":
                q_result[s].append(q1[s][a] - q2[s][a])
            else:
                raise ValueError("operation must be '+' or '-'")
    return q_result


# --- Core Logic: Environment ---

def gridworld(mode='mixed', slip_prob=0.0, pos_reward=1.0, neg_reward=-1.0, intermediate_reward=0.0):
    P = {}
    n = 10
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # north, east, south, west
    
    def state_index(x, y):
        return x * n + y

    def out_of_bounds(x, y):
        return x < 0 or x >= n or y < 0 or y >= n

    deadends_x = range(5, 10)
    deadends_y = range(5, 9)
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]

    for x in range(n):
        for y in range(n):
            state = state_index(x, y)
            P[state] = {}

            # 1. Handle "Terminal" or "Waiting" states
            if (x, y) in survivals + mortalities + invalids:
                # These states allow 'waiting' on all actions
                P[state] = {0 : [(1.0, state, 0)], 1 : [(1.0, state, 0)], 2 : [(1.0, state, 0)], 3 : [(1.0, state, 0)]}
                continue
            
            # 2. Iterate through potential actions
            for action in range(len(actions)):
                (dx, dy) = actions[action]
                
                target_x = x + dx
                target_y = y + dy
                target = (target_x, target_y)

                # --- VALIDITY CHECK (Shearing) ---
                is_deadend = x in deadends_x and y in deadends_y
                
                if not is_deadend:
                    if out_of_bounds(target_x, target_y) or target in invalids:
                        continue # Action sheared off
                # ---------------------------------

                # Initialize the list for this valid action
                P[state][action] = []
                
                # Logic for Deadends (Trap Zone)
                if is_deadend:
                    outcome = (0 if mode == 'survival' else neg_reward, state_index(9, 9))
                    reward = outcome[0]
                    next_state = outcome[1]
                    P[state][action].append((1.0, next_state, reward))
                    continue

                # Logic for Standard Moves
                next_state = state_index(target_x, target_y)
                
                reward = intermediate_reward 

                if target in survivals and mode != 'avoidance':
                    reward = pos_reward

                if target in mortalities and mode != 'survival':
                    reward = neg_reward

                move = (1.0-slip_prob, next_state, reward)
                P[state][action].append(move)

                # Logic for Slipping
                slip_coords = (x, y+1)
                
                slip_reward = intermediate_reward 

                if slip_coords in survivals and mode != 'avoidance':
                    slip_reward = pos_reward

                if slip_coords in mortalities and mode != 'survival':
                    slip_reward = neg_reward
                
                # If slip hits a wall/invalid, bounce (stay in current state)
                if out_of_bounds(x, y+1) or slip_coords in invalids:
                    slip_state = state
                else:
                    slip_state = state_index(x, y+1)
                
                slip = (slip_prob, slip_state, slip_reward)
                if slip_prob > 0.0:
                    P[state][action].append(slip)

    return P

def gridworld_ep(survival=True, slip_prob=0.2):
    P = gridworld(survival, slip_prob)
    A_prime = [(0.0, 21, 0)]
    B_prime = [(0.0, 13, 0)]

    for i in range(4):
        if i in P[21]: P[21][i] = A_prime
        if i in P[13]: P[13][i] = B_prime
    return P


# --- Core Logic: Solvers ---

def policy_eval(P, policy=None, theta=0.0001, gamma=0.9):
    n = 10
    V = np.zeros((n, n))

    if policy is None:
        policy = np.full((n*n, 4), 0.25)

    while True:
        delta = 0

        for s in range(n*n):
            x, y = V_index(s, n)
            v = V[x][y]
            new_v = 0

            # Only iterate through actions that exist in the sheared P[s]
            for a in range(len(policy[s])):
                if a not in P[s]:
                    continue

                action_prob = policy[s][a]
                state_expectation = 0
                for state_prob, next_state, reward in P[s][a]:
                    next_x, next_y = V_index(next_state, n)
                    state_expectation += state_prob * (reward + gamma * V[next_x][next_y])
                new_v += state_expectation * action_prob
            
            V[x][y] = new_v
            delta = max(delta, abs(v - V[x][y]))

        if delta < theta:
            break

    return V

def policy_iter(P, theta=0.0001, gamma=0.9):
    n = 10
    policy = np.full((n*n, 4), 0.25)
    
    while True:
        policy_stable = True
        V = policy_eval(P, policy, theta, gamma)

        for s in range(n*n):
            Q = np.full(4, -np.inf)
            
            if not P[s]: 
                continue

            for a in P[s]:
                Q[a] = 0
                for prob, next_state, reward in P[s][a]:
                    next_x, next_y = V_index(next_state, n)
                    Q[a] += prob * (reward + gamma * V[next_x][next_y])

            best_actions = amax(Q)
            new_policy = np.zeros(4)
            for a in best_actions:
                new_policy[a] = 1/len(best_actions)
            
            if not np.array_equal(policy[s], new_policy):
                policy_stable = False
            policy[s] = new_policy

        if policy_stable:
            break

    return V, policy

def value_iter(P, theta=0.0001, gamma=0.9):
    n = 10
    V = np.zeros((n, n))
    
    while True:
        delta = 0
        for s in range(n*n):
            x, y = V_index(s, n)
            v = V[x][y]
            
            Q = np.full(4, -np.inf)
            
            if not P[s]:
                V[x][y] = 0
                continue

            for a in P[s]:
                Q[a] = 0
                for prob, next_state, reward in P[s][a]:
                    next_x, next_y = V_index(next_state, n)
                    Q[a] += prob * (reward + gamma * V[next_x][next_y])
            
            V[x][y] = np.max(Q)
            delta = max(delta, abs(v - V[x][y]))

        if delta < theta:
            break

    # Extract Policy
    policy = np.zeros((n*n, 4))
    for s in range(n*n):
        Q = np.full(4, -np.inf)
        if not P[s]: continue

        for a in P[s]:
            Q[a] = 0
            for prob, next_state, reward in P[s][a]:
                next_x, next_y = V_index(next_state, n)
                Q[a] += prob * (reward + gamma * V[next_x][next_y])

        best_actions = amax(Q)
        new_policy = np.zeros(4)
        if len(best_actions) > 0:
            for a in best_actions:
                new_policy[a] = 1/len(best_actions)
        policy[s] = new_policy

    return V, policy

def compute_Q(policy, P, gamma=0.99, theta=1e-15):
    num_states = 100 # n*n
    # 1. Calculate V using policy evaluation logic inline
    V = np.zeros(num_states)
    while True:
        V_prev = V.copy()
        for s in range(num_states):
            V[s] = 0
            for a in range(4): # iterate full action space
                if a not in P[s]: continue
                
                action_value = 0
                for prob, next_s, reward in P[s][a]:
                    action_value += prob * (reward + gamma * V_prev[next_s])
                V[s] += policy[s, a] * action_value
        if np.max(np.abs(V - V_prev)) < theta:
            break

    # 2. Build Q dict for visualization
    q_dict = {}
    for s in range(num_states):
        q_dict[s] = []
        for a in range(4):
            # If action sheared, append -1.0 so it draws as Red (blocked) in plots
            if a not in P[s]:
                q_dict[s].append(-1.0)
                continue

            q_sa = 0
            for prob, next_s, reward in P[s][a]:
                q_sa += prob * (reward + gamma * V[next_s])
            q_dict[s].append(q_sa)

    return q_dict


# --- Legacy / unused Printing Functions ---
def policy_to_chars(P):
    return [] 

def print_policy(P):
    pass 

def print_V(V):
    n = len(V[0])
    print('\033[4m', end='')
    print("".join("      " for _ in range(n)))
    for row in V:
        print('|', end='')
        print("|".join(f"{val:5.2f}" for val in row), end='|\n')
    print('\033[0m', end='')


# --- Plotting Functions ---

def plot_value_heatmap(V, save_path=None, show=True):
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(V, cmap="coolwarm_r", center=0, vmin=-1, vmax=1,
                     square=True, linewidths=0.5, linecolor='gray',
                     cbar_kws={'label': 'Value'})
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_policy(prob_array, grid_shape=(10, 10), save_path="policy_vector_field.pdf"):
    dir_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    directions = ['N', 'E', 'S', 'W']
    rows, cols = grid_shape
    
    plt.figure(figsize=(3, 3))
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.xticks(ticks=np.arange(cols+1))
    plt.yticks(ticks=np.arange(rows+1))

    for r in range(rows):
        for c in range(cols):
            probs = prob_array[r * cols + c]
            if np.all(probs == 0):
                continue 
            for i, p in enumerate(probs):
                if p > 0:
                    dx, dy = dir_map[directions[i]]
                    plt.arrow(c + 0.5, r + 0.5, dx * 0.3, dy * 0.3,
                              head_width=0.15, head_length=0.1, fc='b', ec='b', length_includes_head=True)
    plt.gca().set_aspect('equal')
    if save_path:
        plt.savefig(save_path, format="pdf")
    plt.show()

def plot_highlighted_policies(prob_arrays, save_paths, grid_shape=(10, 10)):
    """
    Generates three separate plots.
    Highlighting Logic:
    - Calculates the SET of optimal actions for each policy (handling ties).
    - A policy is 'Safe' (no red) if its optimal action set is a SUBSET 
      of at least one other policy's optimal action set.
    - Otherwise, it is highlighted RED.
    """
    if len(prob_arrays) != 3 or len(save_paths) != 3:
        raise ValueError("Must provide exactly three arrays and three save paths.")

    dir_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    directions = ['N', 'E', 'S', 'W']
    rows, cols = grid_shape
    
    # --- Helper: Get set of best action indices ---
    def get_optimal_set(probs):
        # Handle terminal states (all zeros)
        if np.all(probs == 0):
            return {-1} 
        
        # Get max probability with small tolerance for float comparison
        max_p = np.max(probs)
        # Find indices where prob is close to max_p
        # We use a small epsilon for float comparison
        return set(np.where(np.isclose(probs, max_p))[0])

    # --- Pre-calculate optimal sets for all policies ---
    # Structure: all_optimal_sets[policy_idx][grid_cell_idx] = Set of actions
    all_optimal_sets = []
    for pol in prob_arrays:
        policy_sets = []
        for idx in range(rows * cols):
            policy_sets.append(get_optimal_set(pol[idx]))
        all_optimal_sets.append(policy_sets)

    # --- Generate One Plot Per Policy ---
    for current_idx in range(3):
        current_policy = prob_arrays[current_idx]
        save_path = save_paths[current_idx]
        
        plt.figure(figsize=(3, 3))
        plt.xlim(0, cols)
        plt.ylim(0, rows)
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.xticks(ticks=np.arange(cols+1))
        plt.yticks(ticks=np.arange(rows+1))

        # Identify indices of the other two policies
        other_indices = [x for x in range(3) if x != current_idx]
        idx1, idx2 = other_indices[0], other_indices[1]

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                
                # Retrieve sets
                set_me = all_optimal_sets[current_idx][idx]
                set_other1 = all_optimal_sets[idx1][idx]
                set_other2 = all_optimal_sets[idx2][idx]
                
                # --- Subset Logic ---
                # I am safe if my choices are fully valid in at least one other policy.
                # (i.e., I am a subset of Other1 OR I am a subset of Other2)
                matches_1 = set_me.issubset(set_other1)
                matches_2 = set_me.issubset(set_other2)
                
                # If I don't match EITHER, then I am doing something unique/disagreeable -> Red
                if not (matches_1 or matches_2):
                    rect = patches.Rectangle((c, r), 1, 1, linewidth=0, 
                                             facecolor='#ffcccb', alpha=0.6, zorder=1)
                    plt.gca().add_patch(rect)

                # --- Draw Arrows ---
                probs = current_policy[idx]
                if np.all(probs == 0):
                    continue
                    
                for k, p in enumerate(probs):
                    if p > 0:
                        dx, dy = dir_map[directions[k]]
                        # Scale arrow slightly; aesthetics preserved
                        plt.arrow(c + 0.5, r + 0.5, dx * 0.3, dy * 0.3,
                                  head_width=0.15, head_length=0.1, fc='b', ec='b', 
                                  length_includes_head=True, zorder=2)
        
        plt.gca().set_aspect('equal')
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

def visualize_gridworld(title):
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
    deadends_x = range(5, 9)
    deadends_y = range(5, 10)
    grid_size = (10, 10)

    grid_colors = np.full(grid_size, 'white', dtype=object)
    for r, c in invalids: grid_colors[r, c] = 'grey'
    for r, c in survivals: grid_colors[r, c] = 'blue'
    for r, c in mortalities: grid_colors[r, c] = 'red'
    for r in deadends_y:
        for c in deadends_x:
            grid_colors[r, c] = '#ffcccc'

    fig, ax = plt.subplots(figsize=(3, 3))
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=grid_colors[r, c], edgecolor='black')
            ax.add_patch(rect)
    
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    if title: plt.savefig(f"{title}.pdf", format="pdf")
    plt.close()

def plot_q_value_grid(q_dict, grid_shape=(10,10), save_path=None, show=True):
    rows, cols = grid_shape
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linewidth=0.5)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_green_blue", [(0.0, "red"), (0.5, "white"), (1.0, "blue")]
    )
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            if state not in q_dict: continue

            q_vals = q_dict[state]
            q_vals = np.clip(q_vals, -1, 1)

            x0, y0 = c, r
            x1, y1 = c + 1, r + 1
            cx, cy = c + 0.5, r + 0.5
            
            triangles = {
                0: [(x0, y0), (x1, y0), (cx, cy)], # N
                1: [(x1, y0), (x1, y1), (cx, cy)], # E
                2: [(x0, y1), (x1, y1), (cx, cy)], # S
                3: [(x0, y0), (x0, y1), (cx, cy)]  # W
            }

            for a, verts in triangles.items():
                color = cmap(norm(q_vals[a]))
                poly = plt.Polygon(verts, facecolor=color, edgecolor='gray', linewidth=0.2)
                ax.add_patch(poly)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Q-value', fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

def plot_policy_with_q(policy, q_dict, grid_shape=(10,10), save_path=None, show=True):
    rows, cols = grid_shape
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linewidth=0.5)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_green_blue", [(0.0, "red"), (0.5, "white"), (1.0, "blue")]
    )
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    dir_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    directions = ['N', 'E', 'S', 'W']

    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            if state not in q_dict: continue
            q_vals = np.clip(q_dict[state], -1, 1)

            x0, y0 = c, r
            x1, y1 = c + 1, r + 1
            cx, cy = c + 0.5, r + 0.5
            triangles = {
                0: [(x0, y0), (x1, y0), (cx, cy)], 
                1: [(x1, y0), (x1, y1), (cx, cy)], 
                2: [(x0, y1), (x1, y1), (cx, cy)], 
                3: [(x0, y0), (x0, y1), (cx, cy)] 
            }
            for a, verts in triangles.items():
                color = cmap(norm(q_vals[a]))
                poly = plt.Polygon(verts, facecolor=color, edgecolor='gray', linewidth=0.2)
                ax.add_patch(poly)

    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            if state >= len(policy): continue
            probs = policy[state]
            if np.all(probs == 0): continue

            for i, p in enumerate(probs):
                if p > 0:
                    dx, dy = dir_map[directions[i]]
                    ax.arrow(c + 0.5, r + 0.5, dx * 0.25, dy * 0.25,
                             head_width=0.12, head_length=0.08,
                             fc='black', ec='black', alpha=0.9, length_includes_head=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Q-value', fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if show: plt.show()
    else: plt.close()


def run_experiment(gamma, slip_prob, intermediate_reward, pos_reward=1.0, neg_reward=-1.0, exp_suffix=""):
    """
    Runs experiments and plots three highlighted policies using the subset-matching logic.
    """
    theta = 1e-15
    print("\n" + "*"*30)
    print(f"RUNNING EXPERIMENT: r={intermediate_reward}, gamma={gamma}, slip={slip_prob}, pos={pos_reward}, neg={neg_reward}")
    print("*"*30 + "\n")
    
    # Initialize environments
    survival = gridworld(mode="survival", slip_prob=slip_prob, pos_reward=pos_reward, neg_reward=neg_reward, intermediate_reward=intermediate_reward)
    avoidance = gridworld(mode="avoidance", slip_prob=slip_prob, pos_reward=pos_reward, neg_reward=neg_reward, intermediate_reward=intermediate_reward)
    mixed = gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=pos_reward, neg_reward=neg_reward, intermediate_reward=intermediate_reward)

    # --- Compute Policies ---
    print("Computing Policies...")
    s_V, s_pi = value_iter(survival, theta=theta, gamma=gamma)
    s_Q = compute_Q(s_pi, survival, gamma, theta)

    a_V, a_pi = value_iter(avoidance, theta=theta, gamma=gamma)
    a_Q = compute_Q(a_pi, avoidance, gamma, theta)

    m_V, m_pi = value_iter(mixed, theta=theta, gamma=gamma)
    m_Q = compute_Q(m_pi, mixed, gamma, theta)

    # --- Plot Highlighted Policies ---
    print("Generating Highlighted Policy Plots...")
    
    # Define save paths for the three individual plots
    path_s = f'plots/plus_policy_with_q_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf'
    path_a = f'plots/minus_policy_with_q_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf'
    path_m = f"plots/mixed_policy_with_q_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf"

    # Pass all three to the plotter
    plot_highlighted_policies(
        prob_arrays=[s_pi, a_pi, m_pi],
        save_paths=[path_s, path_a, path_m]
    )

    # --- Value Combinations ---
    print("Combining Values...")
    
    # Survival + Avoidance
    s_plus_a = add_Vs(s_V, a_V)
    sq_plus_aq = combine_Qs(s_Q, a_Q, "+")

    # Mixed - (Survival + Avoidance)
    subtracted = subtract_Vs(m_V, s_plus_a)
    subtracted_Qs = combine_Qs(sq_plus_aq, m_Q, "-")
    
    plot_value_heatmap(
        subtracted,
        save_path=f"plots/gridworld_add_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf",
        show=False
    )

    # Survival - Avoidance
    s_minus_a = subtract_Vs(s_V, a_V)
    sq_minus_aq = combine_Qs(s_Q, a_Q)
    
    plot_value_heatmap(
        s_minus_a,
        save_path=f"plots/gridworld_subtract_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf",
        show=False
    )
    
    print(f"Experiment {exp_suffix} Complete.\n")

def gridworld_break_asms():
    visualize_gridworld("plots/gridworld_layout")
    
    # Define the parameter spaces for the sweep
    gammas = [1.0, 0.9]
    slip_probs = [0.2, 0.0]
    intermediate_rewards = [-0.04, 0.00]
    
    # Generate all 8 combinations
    experiments = list(itertools.product(gammas, slip_probs, intermediate_rewards))
    print(f"Total Base Combinations to run: {len(experiments)}\n")
    
    for gamma, slip, reward in experiments:
        # Programmatically generate a clear suffix for the saved plots
        suffix = f"_g{gamma}_slip{slip}_step{reward}"
        
        run_experiment(
            gamma=gamma, 
            slip_prob=slip, 
            intermediate_reward=reward, 
            exp_suffix=suffix
        )

def gridworld_asym_reward():
    run_experiment(
        gamma=1.0, 
        slip_prob=0.2, 
        intermediate_reward=0.0, 
        pos_reward=1.0, 
        neg_reward=-0.5, 
        exp_suffix="_asym"
    )

def count_death_preferences(policy, n=10):
    """
    Counts the number of non-terminal, non-deadend states where following 
    the policy's preferred action(s) eventually transitions into a mortality 
    state without getting stuck in a cycle.
    """
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # north, east, south, west
    
    # Environment geometry
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    deadends_x = range(5, 10)
    deadends_y = range(5, 9)

    def dfs(s, visited):
        x, y = int(s / n), int(s % n)
        state_coord = (x, y)

        # --- Base Cases ---
        if state_coord in mortalities:
            return True
        if state_coord in survivals or state_coord in invalids:
            return False
        if s in visited:
            return False # Cycle detected, this path does not lead to death
            
        # Add current state to the path
        visited.add(s)

        # Handle Trap Zone (deadends immediately transition to state 99, a mortality state)
        if x in deadends_x and y in deadends_y:
            leads_to_death = dfs(99, visited)
            visited.remove(s)
            return leads_to_death

        # --- Branching & Traversal ---
        max_prob = np.max(policy[s])
        
        # Terminal/Waiting states might have 0 probability for movement
        if np.isclose(max_prob, 0):
            visited.remove(s)
            return False

        best_actions = [a for a, p in enumerate(policy[s]) if np.isclose(p, max_prob)]
        leads_to_death = False
        
        for a in best_actions:
            dx, dy = actions[a]
            target_x, target_y = x + dx, y + dy
            target_coord = (target_x, target_y)

            # Check if the action is sheared off (out of bounds or invalid)
            if target_x < 0 or target_x >= n or target_y < 0 or target_y >= n or target_coord in invalids:
                continue 
            
            target_s = target_x * n + target_y
            
            # If ANY optimal action leads down a path to death, flag it
            if dfs(target_s, visited):
                leads_to_death = True
                break 

        # Remove state from path as we backtrack up the recursion tree
        visited.remove(s)
        return leads_to_death


    # --- Evaluation Loop ---
    death_preference_count = 0

    for s in range(n * n):
        x, y = int(s / n), int(s % n)

        # Filter out terminal, invalid, and deadend states from being counted as the origin
        if (x, y) in invalids or (x, y) in survivals or (x, y) in mortalities:
            continue
        if x in deadends_x and y in deadends_y:
            continue

        # Start a fresh DFS from this state
        if dfs(s, set()):
            death_preference_count += 1

    return death_preference_count


def gridworld_scaling():
    # Define a vast range of gammas and small negative intermediate rewards
    gammas = np.linspace(0.5, 0.99, 20) 
    step_costs = np.linspace(-0.15, 0.0, 20) 
    
    # Matrices to store the results
    deaths_plus = np.zeros((len(gammas), len(step_costs)))
    deaths_minus = np.zeros((len(gammas), len(step_costs)))
    deaths_mixed = np.zeros((len(gammas), len(step_costs)))
    
    print("Running parameter sweep (this may take a few seconds)...")
    for i, g in enumerate(gammas):
        for j, r in enumerate(step_costs):
            # Generate environments
            env_plus = gridworld(mode="survival", slip_prob=0.2, intermediate_reward=r)
            env_minus = gridworld(mode="avoidance", slip_prob=0.2, intermediate_reward=r)
            env_mixed = gridworld(mode="mixed", slip_prob=0.2, intermediate_reward=r)
            
            # Get policies
            _, pi_plus = value_iter(env_plus, gamma=g)
            _, pi_minus = value_iter(env_minus, gamma=g)
            _, pi_mixed = value_iter(env_mixed, gamma=g)
            
            # Count preferences
            deaths_plus[i, j] = count_death_preferences(pi_plus)
            deaths_minus[i, j] = count_death_preferences(pi_minus)
            deaths_mixed[i, j] = count_death_preferences(pi_mixed)

    # Calculate total number of valid non-terminal, non-deadend states
    # 100 total - (9 invalids + 3 survivals + 11 mortalities + 20 deadends)
    total_valid_states = 57 
    
    # --- Plotting Helper ---
    def plot_heatmap(data, title, filename):
        plt.figure(figsize=(7, 5))
        
        # Format ticks for readability
        x_labels = [f"{val:.3f}" for val in step_costs]
        y_labels = [f"{val:.2f}" for val in gammas]
        
        # 1. Use fixed global min and max for the color scale across all plots
        min_val = 0
        max_val = total_valid_states
        
        # 2. Create a discrete colormap with exactly enough bins for our integers
        n_colors = max_val - min_val + 1
        cmap = plt.get_cmap("magma", n_colors)
        
        # 3. Define the boundaries so they fall exactly halfway between integers
        bounds = np.arange(min_val, max_val + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax = sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, 
                         cmap=cmap, norm=norm, annot=False)
        
        ax.invert_yaxis() # Put highest gamma at the top of the Y-axis
        
        # Keep tick labels from overlapping too much
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_xticklabels(x_labels[::2], rotation=45)
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_yticklabels(y_labels[::2], rotation=0)

        # 4. Force the colorbar to show integer ticks at reasonable intervals
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.arange(min_val, max_val + 1, 5))
        cbar.set_label('States Transitioning to Death')

        plt.title(title)
        plt.xlabel("Intermediate Reward (Step Cost)")
        plt.ylabel("Gamma")
        plt.tight_layout()
        plt.savefig(f"plots/{filename}.pdf")
        plt.close()

    # --- Generate the three plots ---
    print("Generating heatmaps...")
    plot_heatmap(deaths_plus, "Death Preference Count (Plus / Survival)", "scaling_deaths_plus")
    plot_heatmap(deaths_minus, "Death Preference Count (Minus / Avoidance)", "scaling_deaths_minus")
    plot_heatmap(deaths_mixed, "Death Preference Count (Mixed)", "scaling_deaths_mixed")


def calculate_policy_similarity(pi_current, pi_anchor, n=10):
    """
    Calculates the percentage of valid states where the optimal action sets 
    of the current policy exactly match the anchor policy.
    """
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
    deadends_x = range(5, 10)
    deadends_y = range(5, 9)

    matching_states = 0
    valid_state_count = 0

    for s in range(n * n):
        x, y = int(s / n), int(s % n)
        
        # Skip states where the agent doesn't actually make a choice
        if (x, y) in invalids or (x, y) in survivals or (x, y) in mortalities:
            continue
        if x in deadends_x and y in deadends_y:
            continue

        valid_state_count += 1

        # Extract max probabilities (handling floating point inaccuracies)
        max_p_current = np.max(pi_current[s])
        max_p_anchor = np.max(pi_anchor[s])
        
        # If both are terminal/sheared off (all zeros), they match
        if np.isclose(max_p_current, 0) and np.isclose(max_p_anchor, 0):
            matching_states += 1
            continue

        # Get the sets of actions that are tied for the highest probability
        set_current = set(np.where(np.isclose(pi_current[s], max_p_current))[0])
        set_anchor = set(np.where(np.isclose(pi_anchor[s], max_p_anchor))[0])

        if set_current == set_anchor:
            matching_states += 1

    return matching_states / valid_state_count if valid_state_count > 0 else 0


def plot_similarity_trajectory(similarities, steps, exp_suffix=""):
    """
    Plots the structural similarity of the policy over the interpolation trajectory.
    """
    plt.figure(figsize=(10, 6))
    
    x_axis = np.linspace(0, 100, steps + 1)
    
    plt.plot(x_axis, similarities['seek'], label='Similarity to (1, 0) [Survival]', color='blue', linewidth=2)
    plt.plot(x_axis, similarities['mixed'], label='Similarity to (1, -1) [Mixed]', color='purple', linewidth=2)
    plt.plot(x_axis, similarities['avoid'], label='Similarity to (0, -1) [Avoidance]', color='red', linewidth=2)

    # Transition line between Phase A and Phase B
    plt.axvline(x=50, color='gray', linestyle='--', label='Transition (1, -1)')
    
    plt.title("Policy Evolution under Asymmetric Rewards", fontsize=14)
    plt.xlabel("Interpolation Progress (%)", fontsize=12)
    plt.ylabel("Policy Similarity (Exact Action Match)", fontsize=12)
    
    # Custom X-ticks mapped to the actual reward coordinate states
    plt.xticks(
        ticks=[0, 25, 50, 75, 100], 
        labels=["(1, 0)", "(1, -0.5)", "(1, -1)", "(0.5, -1)", "(0, -1)"]
    )
    
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    save_path = f"asym_plots/asym_similarity_trajectory{exp_suffix}.pdf"
    plt.savefig(save_path, format='pdf')
    print(f"Saved trajectory plot to {save_path}")
    plt.close()

def run_asym(gamma=1.0, slip_prob=0.2, intermediate_reward=0.00, steps=100, exp_suffix=""):
    """
    Executes the asymmetric reward sweep from (1, 0) -> (1, -1) -> (0, -1).
    """
    print("\n" + "="*40)
    print(f"RUNNING ASYMMETRIC SWEEP: gamma={gamma}, slip={slip_prob}, step_cost={intermediate_reward}")
    print("="*40)

    theta = 1e-15

    # --- 1. Generate Anchor Policies ---
    print("Generating Anchor Policies...")
    # Anchor 1: Pure Seek (1, 0)
    env_seek = gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=1.0, neg_reward=0.0, intermediate_reward=intermediate_reward)
    _, pi_seek = value_iter(env_seek, theta=theta, gamma=gamma)

    # Anchor 2: Mixed (1, -1)
    env_mixed = gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=1.0, neg_reward=-1.0, intermediate_reward=intermediate_reward)
    _, pi_mixed = value_iter(env_mixed, theta=theta, gamma=gamma)

    # Anchor 3: Pure Avoid (0, -1)
    env_avoid = gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=0.0, neg_reward=-1.0, intermediate_reward=intermediate_reward)
    _, pi_avoid = value_iter(env_avoid, theta=theta, gamma=gamma)

    # --- 2. Initialize Tracking Variables ---
    similarities = {'seek': [], 'mixed': [], 'avoid': []}

    print(f"Running Interpolation Sweep over {steps} steps...")
    # --- 3. Run the Interpolation Loop ---
    for i in range(steps + 1):
        alpha = i / steps
        
        if alpha <= 0.5:
            # Phase A: (1, 0) -> (1, -1). Alpha goes 0->0.5, neg_r goes 0->-1
            pos_r = 1.0
            neg_r = -2.0 * alpha 
        else:
            # Phase B: (1, -1) -> (0, -1). Alpha goes 0.5->1.0, pos_r goes 1->0
            pos_r = 1.0 - 2.0 * (alpha - 0.5) 
            neg_r = -1.0

        # Generate environment for current reward coordinate
        env_current = gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=pos_r, neg_reward=neg_r, intermediate_reward=intermediate_reward)
        _, pi_current = value_iter(env_current, theta=theta, gamma=gamma)

        # Calculate similarities against the anchors
        similarities['seek'].append(calculate_policy_similarity(pi_current, pi_seek))
        similarities['mixed'].append(calculate_policy_similarity(pi_current, pi_mixed))
        similarities['avoid'].append(calculate_policy_similarity(pi_current, pi_avoid))

    # --- 4. Plot the Results ---
    print("Generating Visualization...")
    plot_similarity_trajectory(similarities, steps, exp_suffix)
    print("Sweep Complete.\n")

def run_all_asym_sweeps():
    """
    Runs the asymmetric reward trajectory across all four core configurations
    of the environment stochasticity and agent discount factor.
    """
    print("Starting Comprehensive Asymmetric Sweeps...\n")
    
    # 1. Baseline: High Horizon, Default Stochasticity
    run_asym(gamma=1.0, slip_prob=0.2, intermediate_reward=0.00, steps=100, exp_suffix="_g1.0_slip0.2")

    # 2. Deterministic: High Horizon, Zero Stochasticity
    run_asym(gamma=1.0, slip_prob=0.0, intermediate_reward=0.00, steps=100, exp_suffix="_g1.0_slip0.0")

    # 3. Short Horizon: Low Horizon, Default Stochasticity
    run_asym(gamma=0.9, slip_prob=0.2, intermediate_reward=0.000, steps=100, exp_suffix="_g0.9_slip0.2")

    # 4. Deterministic & Short Horizon: Low Horizon, Zero Stochasticity
    run_asym(gamma=0.9, slip_prob=0.0, intermediate_reward=0.00, steps=100, exp_suffix="_g0.9_slip0.0")
    
    print("All configurations complete! Check the 'asym_plots' folder.")

def main(): 
    # Run the base configuration sweeps
    print("=== STARTING BASE PARAMETER SWEEPS ===")
    gridworld_break_asms()
    
    # Run the asymmetric interpolation suite
    print("\n=== STARTING ASYMMETRIC REWARD SWEEPS ===")
    run_all_asym_sweeps()

if __name__ == "__main__":
    main()