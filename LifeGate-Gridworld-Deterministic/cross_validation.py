from gridworld import * 
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def policy_eval(MDP, policy, gamma, theta=1e-10):
    n_states = len(MDP)
    n_actions = policy.shape[1]
    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            
            v = 0
            for a in range(n_actions):
                action_prob = policy[s, a] / np.sum(policy[s])
                if action_prob == 0:
                    continue
                for prob, next_state, reward in MDP[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break

    return V


def plot_value_grid_with_special_states(V, save_path=None, show=True):
    """
    Plots the gridworld with special state colors and overlays state IDs and values.
    
    Parameters:
    - V: 2D numpy array of shape (n, n)
    - save_path: optional file path to save the figure
    - show: whether to display the plot
    """
    nrows, ncols = V.shape
    grid_size = (nrows, ncols)

    # Define your special states
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
    deadends_x = range(5, 9)
    deadends_y = range(5, 10)

    grid_colors = np.full(grid_size, 'white', dtype=object)

    for r, c in invalids:
        grid_colors[r, c] = 'grey'
    for r, c in survivals:
        grid_colors[r, c] = 'blue'
    for r, c in mortalities:
        grid_colors[r, c] = 'red'
    for y in deadends_y:
        for x in deadends_x:
            grid_colors[y, x] = '#ffcccc'  # light red

    fig, ax = plt.subplots(figsize=(5, 5))

    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=grid_colors[r, c], edgecolor='black')
            ax.add_patch(rect)
            
            # Add state ID and value
            state_id = r * ncols + c
            val = V[r, c]
            text = f"{state_id}\n{val:.2f}"
            ax.text(
                c + 0.5, r + 0.5, text,
                ha='center', va='center',
                color='black',
                fontsize=6
            )

    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(np.arange(grid_size[1]))
    ax.set_yticks(np.arange(grid_size[0]))
    ax.set_xticklabels(np.arange(grid_size[1]))
    ax.set_yticklabels(np.arange(grid_size[0]))
    ax.invert_yaxis()  # so row 0 is on top

    ax.grid(which='both', color='black', linewidth=1)
    ax.set_aspect('equal')

    plt.title("Gridworld Values with Special States")

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
    

def main():
    g = 0.99
    plus_gw = gridworld(mode="survival")
    minus_gw = gridworld(mode="avoidance")
    mixed_gw = gridworld()
    
    plus_V, plus_pi = policy_iter_deter(plus_gw, gamma=g)
    minus_v, minus_pi = policy_iter_deter(minus_gw, gamma=g)
    mixed_V, mixed_pi = policy_iter_deter(mixed_gw, gamma=g)
    
    # pprint.pprint(plus_pi)

# plus/mixed policy evaluated on three MDPs
    plus_pi_on_plus_gw = policy_eval(plus_gw, plus_pi, g)
    plot_value_grid_with_special_states(plus_pi_on_plus_gw.reshape(10, 10), save_path="cv/plus_pi_on_plus_gw.pdf", show=False)

    plus_pi_on_mixed_gw = policy_eval(mixed_gw, plus_pi, g)
    plot_value_grid_with_special_states(plus_pi_on_mixed_gw.reshape(10, 10), save_path="cv/plus_pi_on_mixed_gw.pdf", show=False)

    plus_pi_on_minus_gw = policy_eval(minus_gw, plus_pi, g)
    plot_value_grid_with_special_states(plus_pi_on_minus_gw.reshape(10, 10), save_path="cv/plus_pi_on_minus_gw.pdf", show=False)

# minus policy evaluated on three MDPs
    minus_pi_on_plus_gw = policy_eval(plus_gw, minus_pi, g)
    plot_value_grid_with_special_states(minus_pi_on_plus_gw.reshape(10, 10), save_path="cv/minus_pi_on_plus_gw.pdf", show=False)

    minus_pi_on_mixed_gw = policy_eval(mixed_gw, minus_pi, g)
    plot_value_grid_with_special_states(minus_pi_on_mixed_gw.reshape(10, 10), save_path="cv/minus_pi_on_mixed_gw.pdf", show=False)

    minus_pi_on_minus_gw = policy_eval(minus_gw, minus_pi, g)
    plot_value_grid_with_special_states(minus_pi_on_minus_gw.reshape(10, 10), save_path="cv/minus_pi_on_minus_gw.pdf", show=False)

if __name__ == "__main__":
    main()