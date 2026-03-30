import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# --- Helper Functions ---
def amax(arr):
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
def get_gridworld(mode='mixed', slip_prob=0.0, pos_reward=1.0, neg_reward=-1.0, intermediate_reward=0.0):
    P = {}
    n = 10
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # north, east, south, west
    
    def state_index(x, y): return x * n + y
    def out_of_bounds(x, y): return x < 0 or x >= n or y < 0 or y >= n

    deadends_x = range(5, 10)
    deadends_y = range(5, 9)
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]

    for x in range(n):
        for y in range(n):
            state = state_index(x, y)
            P[state] = {}

            if (x, y) in survivals + mortalities + invalids:
                P[state] = {0: [(1.0, state, 0)], 1: [(1.0, state, 0)], 2: [(1.0, state, 0)], 3: [(1.0, state, 0)]}
                continue
            
            for action in range(len(actions)):
                (dx, dy) = actions[action]
                target_x, target_y = x + dx, y + dy
                target = (target_x, target_y)

                is_deadend = x in deadends_x and y in deadends_y
                if not is_deadend and (out_of_bounds(target_x, target_y) or target in invalids):
                    continue

                P[state][action] = []
                
                if is_deadend:
                    outcome = (0 if mode == 'survival' else neg_reward, state_index(9, 9))
                    P[state][action].append((1.0, outcome[1], outcome[0]))
                    continue

                next_state = state_index(target_x, target_y)
                reward = intermediate_reward 
                if target in survivals and mode != 'avoidance': reward = pos_reward
                if target in mortalities and mode != 'survival': reward = neg_reward

                P[state][action].append((1.0-slip_prob, next_state, reward))

                slip_coords = (x, y+1)
                slip_reward = intermediate_reward 
                if slip_coords in survivals and mode != 'avoidance': slip_reward = pos_reward
                if slip_coords in mortalities and mode != 'survival': slip_reward = neg_reward
                
                if out_of_bounds(x, y+1) or slip_coords in invalids: slip_state = state
                else: slip_state = state_index(x, y+1)
                
                if slip_prob > 0.0: P[state][action].append((slip_prob, slip_state, slip_reward))

    return P

# --- Core Logic: Solvers ---
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
        if delta < theta: break

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
            for a in best_actions: new_policy[a] = 1/len(best_actions)
        policy[s] = new_policy
    return V, policy

def compute_Q(policy, P, gamma=0.99, theta=1e-15):
    num_states = 100
    V = np.zeros(num_states)
    while True:
        V_prev = V.copy()
        for s in range(num_states):
            V[s] = 0
            for a in range(4):
                if a not in P[s]: continue
                action_value = 0
                for prob, next_s, reward in P[s][a]:
                    action_value += prob * (reward + gamma * V_prev[next_s])
                V[s] += policy[s, a] * action_value
        if np.max(np.abs(V - V_prev)) < theta: break

    q_dict = {}
    for s in range(num_states):
        q_dict[s] = []
        for a in range(4):
            if a not in P[s]:
                q_dict[s].append(-1.0)
                continue
            q_sa = 0
            for prob, next_s, reward in P[s][a]:
                q_sa += prob * (reward + gamma * V[next_s])
            q_dict[s].append(q_sa)
    return q_dict

# --- Plotting Functions ---
def plot_value_heatmap(V, save_path=None, show=False):
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(V, cmap="coolwarm_r", center=0, vmin=-1, vmax=1,
                     square=True, linewidths=0.5, linecolor='gray',
                     cbar_kws={'label': 'Value'})
    plt.tight_layout()
    if save_path: plt.savefig(save_path, format='pdf', bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

def plot_highlighted_policies(prob_arrays, save_paths, grid_shape=(10, 10)):
    dir_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    directions = ['N', 'E', 'S', 'W']
    rows, cols = grid_shape
    
    def get_optimal_set(probs):
        if np.all(probs == 0): return {-1} 
        return set(np.where(np.isclose(probs, np.max(probs)))[0])

    all_optimal_sets = [[get_optimal_set(pol[idx]) for idx in range(rows * cols)] for pol in prob_arrays]

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

        other_indices = [x for x in range(3) if x != current_idx]
        idx1, idx2 = other_indices[0], other_indices[1]

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                set_me = all_optimal_sets[current_idx][idx]
                set_other1 = all_optimal_sets[idx1][idx]
                set_other2 = all_optimal_sets[idx2][idx]
                
                matches_1 = set_me.issubset(set_other1)
                matches_2 = set_me.issubset(set_other2)
                
                if not (matches_1 or matches_2):
                    rect = patches.Rectangle((c, r), 1, 1, linewidth=0, facecolor='#ffcccb', alpha=0.6, zorder=1)
                    plt.gca().add_patch(rect)

                probs = current_policy[idx]
                if np.all(probs == 0): continue
                    
                for k, p in enumerate(probs):
                    if p > 0:
                        dx, dy = dir_map[directions[k]]
                        plt.arrow(c + 0.5, r + 0.5, dx * 0.3, dy * 0.3, head_width=0.15, head_length=0.1, fc='b', ec='b', length_includes_head=True, zorder=2)
        
        plt.gca().set_aspect('equal')
        if save_path: plt.savefig(save_path, format="pdf", bbox_inches='tight')
        plt.close()

def visualize_gridworld(save_path):
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
        for c in deadends_x: grid_colors[r, c] = '#ffcccc'

    fig, ax = plt.subplots(figsize=(3, 3))
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=grid_colors[r, c], edgecolor='black')
            ax.add_patch(rect)
    
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    if save_path: plt.savefig(save_path, format="pdf")
    plt.close()