import numpy as np
import seaborn as sns
import pprint
import matplotlib.pyplot as plt

deadends_x = range(0,0)
deadends_y = range(0,0)
# invalids = [(0, 0), (2, 1), (2, 2)]
# survivals = [(0, 1), (0, 2)]
# mortalities = [(0, 3), (1, 3), (2, 3), (3, 3)]

# counter example: survival is unreachable (right column is death)
invalids = [(0,0), (2,0), (2,1), (2,2), (0,1), (0,2), (0,3), (3,3), (3,0), (3,1), (3,2), (2,3)]
survivals = [] 
mortalities = [(1,3)]


# counter example 2
# invalids = [(0,0),(0,1), (0,2), (3,0), (3,1), (3,2), (0,3),(3,3)]
# survivals = [] 
# mortalities = [ (1,3), (2,3)]


def amax(arr):
    if len(arr) == 0:
        return []
    max_value = np.max(arr)
    return [i for i, val in enumerate(arr) if np.isclose(val, max_value)]

def policy_to_chars(P):
    n = 4
    char_policy = []
    for i in range(n):
        char_policy.append([])
        for j in range(n):
            char_policy[i].append(amax(P[i*n+j]))
    for i in range(n):
        for j in range(n):
            chars = char_policy[i][j].copy()
            char_policy[i][j] = ''
            for char in chars:
                if char == 0:
                    char_policy[i][j] += 'n'
                elif char == 1:
                    char_policy[i][j] += 'e'
                elif char == 2:
                    char_policy[i][j] += 's'
                elif char == 3:
                    char_policy[i][j] += 'w'
    return char_policy

def print_policy(P):
    char_policy = policy_to_chars(P)
    n = 4

    print('\033[4m', end='')
    print("".join("     " for _ in range(n)))
    for row in char_policy:
        print('|', end='')
        print('|'.join(f"{action:<4}" for action in row), end = '|\n')
    print('\033[0m', end='')

def print_V(V):
    n = len(V[0])
    print('\033[4m', end='')
    print("".join("      " for _ in range(n)))
    for row in V:
        print('|', end='')
        print("|".join(f"{val:5.2f}" for val in row), end='|\n')
    print('\033[0m', end='')
            
def V_index(s, n):
    return (int(s/n), int(s%n))

def gridworld(mode='mixed', slip_prob=0.0):
    P = {}
    n = 4
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # north, east, south, west
    
    def state_index(x, y):
        return x * n + y

    def out_of_bounds(x, y):
        return x < 0 or x >= n or y < 0 or y >= n



    for x in range(n):
        for y in range(n):
            state = state_index(x, y)
            P[state] = {}

            if (x, y) in survivals + mortalities + invalids:
                P[state] = {0 : [(1.0, state, 0)], 1 : [(1.0, state, 0)], 2 : [(1.0, state, 0)], 3 : [(1.0, state, 0)]}
                continue
                
            
            for action in range(len(actions)):
                (dx, dy) = actions[action]
                P[state][action] = []
                
                if x in deadends_x and y in deadends_y:
                    outcome = (0 if mode == 'survival' else -1, state_index(9, 9))
                    reward = outcome[0]
                    next_state = outcome[1]
                    P[state][action].append((1.0, next_state, reward))
                    continue

                target_x = x+dx
                target_y = y+dy
                target = (target_x, target_y)

                if out_of_bounds(target_x, target_y) or (target_x, target_y) in invalids:
                    next_state = state
                    reward = -1
                    P[state][action].append((1.0, next_state, reward))
                    continue
                
                next_state = state_index(target_x, target_y)

                reward = 0

                if target in survivals and mode != 'avoidance':
                    reward = 1

                if target in mortalities and mode != 'survival':
                    reward = -1

                move = (1.0-slip_prob, next_state, reward)
                P[state][action].append(move)
                slip_state = state_index(x+1, y) if not out_of_bounds(x+1, y) else state
                # slip reward
                slip = (slip_prob, slip_state, 0)
                if slip_prob > 0.0:
                    P[state][action].append(slip)

    return P

def policy_eval(P, policy=None, theta=0.0001, gamma=0.9):
    n = 4
    V = np.zeros((n, n))

    if policy is None:
        policy = np.full((n*n, 4), 0.25)

    while True:
        delta = 0

        for s in range(n*n):
            x, y = V_index(s, n)
            v = V[x][y]
            new_v = 0

            for a in range(len(policy[s])):
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

def gridworld_ep(survival=True, slip_prob=0.2):
    P = gridworld(survival, slip_prob)
    A_prime = [(0.0, 21, 0)]
    B_prime = [(0.0, 13, 0)]

    for i in range(4):
        P[21][i] = A_prime
        P[13][i] = B_prime
    return P

def add_Vs(V1, V2):
    return [x + y for x, y in zip(V1, V2)]

def subtract_Vs(V1, V2):
    return [x - y for x, y in zip(V1, V2)]


def plot_value_heatmap(V, save_path=None, show=False):
    """
    Plots a heatmap of the value function.

    Parameters:
    - V: 2D numpy array of state values (shape: n x n)
    - title: Title of the plot
    - save_path: Optional file path to save the figure
    - show: Whether to display the plot
    """
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

def visualize_policy(prob_array, grid_shape=(4, 4), title="policy_vector_field"):
    # Directions mapping: N, E, S, W
    dir_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    directions = ['N', 'E', 'S', 'W']

    rows, cols = grid_shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    plt.figure(figsize=(3, 3))
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Show ticks and labels (numbers)
    plt.xticks(ticks=np.arange(cols+1))
    plt.yticks(ticks=np.arange(rows+1))

    for r in range(rows):
        for c in range(cols):
            probs = prob_array[r * cols + c]  # flatten index
            if np.all(probs == 0):
                continue  # skip states with zero probabilities
            for i, p in enumerate(probs):
                if p > 0:
                    dx, dy = dir_map[directions[i]]
                    plt.arrow(c + 0.5, r + 0.5, dx * 0.3, dy * 0.3,
                              head_width=0.15, head_length=0.1, fc='b', ec='b', length_includes_head=True)

    plt.gca().set_aspect('equal')
    plt.savefig(f"{title}.pdf", format="pdf")
    # plt.show()


def visualize_gridworld(title):
    # deadends_x = range(0,0)
    # deadends_y = range(0,0)
    # invalids = [(0, 0), (2, 1), (2, 2)]
    # survivals = [(0, 1), (0, 2)]
    # mortalities = [(0, 3), (1, 3), (2, 3), (3, 3)]

    grid_size = (4, 4)

    grid_colors = np.full(grid_size, 'white', dtype=object)
    
    for r, c in invalids:
        grid_colors[r, c] = 'grey'
    for r, c in survivals:
        grid_colors[r, c] = 'blue'
    for r, c in mortalities:
        grid_colors[r, c] = 'red'
    for r in deadends_y:
        for c in deadends_x:
            grid_colors[r, c] = '#ffcccc'  # light red

    fig, ax = plt.subplots(figsize=(3, 3))

    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=grid_colors[r, c], edgecolor='black')
            ax.add_patch(rect)
    
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(np.arange(grid_size[1]))
    ax.set_yticks(np.arange(grid_size[0]))
    ax.set_xticklabels(np.arange(grid_size[1]))
    ax.set_yticklabels(np.arange(grid_size[0]))
    ax.invert_yaxis()  # flip y so row 0 is on top

    ax.grid(which='both', color='black', linewidth=1)
    ax.set_aspect('equal')

    plt.savefig(f"{title}.pdf", format="pdf")
    # plt.show()

def value_iter_deter(P, theta=0.0001, gamma=0.9):
    n = 4
    V = np.zeros((n, n))
    
    while True:
        delta = 0
        for s in range(n*n):
            x, y = V_index(s)
            v = V[x][y]
            
            Q = np.zeros(4)
            for a in range(4):
                for prob, next_state, reward in P[s][a]:
                    next_x, next_y = V_index(next_state)
                    Q[a] += prob * (reward + gamma * V[next_x][next_y])

            # Deterministic: choose the first action in sorted order if ties
            max_q = np.max(Q)
            best_actions = amax(Q)
            chosen_action = min(best_actions)  # always pick the smallest index

            V[x][y] = Q[chosen_action]

            delta = max(delta, abs(v - V[x][y]))

        if delta < theta:
            break

    policy = np.zeros((n*n, 4))
    for s in range(n*n):
        Q = np.zeros(4)
        for a in range(4):
            for prob, next_state, reward in P[s][a]:
                next_x, next_y = V_index(next_state)
                Q[a] += prob * (reward + gamma * V[next_x][next_y])

        best_actions = amax(Q)
        chosen_action = min(best_actions)
        new_policy = np.zeros(4)
        new_policy[chosen_action] = 1.0
        policy[s] = new_policy

    return V, policy

def policy_iter_deter(P, theta=0.0001, gamma=0.9):
    n = 4
    policy = np.zeros((n*n, 4))
    final_policy = np.zeros((n*n, 4))  # <- to record all optimal actions

    # Deterministic: initialize policy to always choose the first action
    for s in range(n*n):
        policy[s][0] = 1.0

    while True:
        policy_stable = True
        V = policy_eval(P, policy, theta, gamma)

        for s in range(n*n):
            old_action = np.argmax(policy[s])
            Q = np.zeros(4)

            for a in range(4):
                for prob, next_state, reward in P[s][a]:
                    next_x, next_y = V_index(next_state, n)
                    Q[a] += prob * (reward + gamma * V[next_x][next_y])

            best_actions = amax(Q)

            # For improvement, pick a single consistent action
            chosen_action = min(best_actions)  # deterministic tie-breaking

            new_policy = np.zeros(4)
            new_policy[chosen_action] = 1.0

            if not np.array_equal(policy[s], new_policy):
                policy_stable = False

            policy[s] = new_policy

            # For final output: record all optimal actions
            final_policy[s] = np.zeros(4)
            for a in best_actions:
                final_policy[s][a] = 1.0

        if policy_stable:
            break

    return V, final_policy



def main():
        g = 0.99
        visualize_gridworld("gridworld")

        survival = gridworld(mode="survival")
        avoidance = gridworld(mode="avoidance")
        mixed = gridworld()

        survival_V, survival_pi = policy_iter_deter(survival, gamma=g)
        visualize_policy(survival_pi, title='plus_policy')

        avoidance_V,  avoidance_pi= policy_iter_deter(avoidance, gamma=g)
        visualize_policy(avoidance_pi, title='minus_policy')

        mixed_V, mixed_pi = policy_iter_deter(mixed, gamma=g)
        visualize_policy(mixed_pi, title="mixed_policy")

        


if __name__ == "__main__":
    main()
