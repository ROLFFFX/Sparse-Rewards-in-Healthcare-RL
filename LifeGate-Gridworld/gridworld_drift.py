"""LifeGate with the original dead-end dynamics of Fatemi et al. (2021).

In a dead-end state every action moves the agent one cell to the right with probability 0.7 and leaves it in
place with probability 0.3, until it reaches the death column (gridworld.get_gridworld instead sends every
dead-end state to death in one step).  All other dynamics are identical to gridworld.get_gridworld.

Cells are (x, y) = (row, column), as in gridworld.py; the layout constants below repeat the ones defined
locally in gridworld.get_gridworld.

`get_gridworld` and `plot_highlighted_policies` are drop-in replacements for the functions of the same names
in gridworld.py; run_drift.py hands them to the experiment scripts.
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import gridworld as gw

DRIFT_RIGHT = 0.7
INVALIDS = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
SURVIVALS = [(0, 5), (0, 6), (0, 7)]
MORTALITIES = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
DEADENDS_X, DEADENDS_Y = range(5, 10), range(5, 9)
NO_DECISION = set(INVALIDS) | set(SURVIVALS) | set(MORTALITIES)     # walls and terminal cells
CELL_COLORS = {**{p: '#d9d9d9' for p in INVALIDS}, **{p: '#9ecae1' for p in SURVIVALS},
               **{p: '#fcbba1' for p in MORTALITIES}}


def get_gridworld(mode='mixed', slip_prob=0.0, pos_reward=1.0, neg_reward=-1.0, intermediate_reward=0.0):
    """Transition model P[s][a] = [(prob, next_state, reward), ...] with the drifting dead-ends."""
    P = gw.get_gridworld(mode=mode, slip_prob=slip_prob, pos_reward=pos_reward,
                         neg_reward=neg_reward, intermediate_reward=intermediate_reward)
    n = 10
    for x in DEADENDS_X:
        for y in DEADENDS_Y:
            s = x * n + y
            if (x, y + 1) in MORTALITIES:
                r_right = 0.0 if mode == 'survival' else neg_reward
            else:
                r_right = intermediate_reward
            outcomes = [(DRIFT_RIGHT, s + 1, r_right), (1.0 - DRIFT_RIGHT, s, intermediate_reward)]
            P[s] = {a: list(outcomes) for a in range(4)}
    return P


def plot_highlighted_policies(prob_arrays, save_paths, titles=None, grid_shape=(10, 10)):
    """gridworld.plot_highlighted_policies with colored walls, terminal cells and dead-ends.

    Draws one PDF per policy in `prob_arrays` (three policies); cells where a policy's optimal actions are not
    a subset of either other policy's are shaded red.  Arrows are drawn only on non-terminal, in-bounds cells.
    """
    dir_map = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    directions = ['N', 'E', 'S', 'W']
    rows, cols = grid_shape

    def get_optimal_set(probs):
        if np.all(probs == 0):
            return {-1}
        return set(np.where(np.isclose(probs, np.max(probs)))[0])

    sets = [[get_optimal_set(pol[i]) for i in range(rows * cols)] for pol in prob_arrays]
    for k in range(3):
        pol = prob_arrays[k]
        plt.figure(figsize=(3, 3))
        plt.xlim(0, cols)
        plt.ylim(0, rows)
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.xticks(ticks=np.arange(cols + 1))
        plt.yticks(ticks=np.arange(rows + 1))
        if titles:
            plt.title(titles[k])
        other1, other2 = [x for x in range(3) if x != k]
        for r in range(rows):
            for c in range(cols):
                if (r, c) in NO_DECISION:
                    plt.gca().add_patch(patches.Rectangle((c, r), 1, 1, linewidth=0, facecolor=CELL_COLORS[(r, c)],
                                                          zorder=1))
                    continue
                idx = r * cols + c
                if r in DEADENDS_X and c in DEADENDS_Y:
                    plt.gca().add_patch(patches.Rectangle((c, r), 1, 1, linewidth=0, facecolor='#ffe5e5',
                                                          zorder=0.5))
                own = sets[k][idx]
                if not (own.issubset(sets[other1][idx]) or own.issubset(sets[other2][idx])):
                    plt.gca().add_patch(patches.Rectangle((c, r), 1, 1, linewidth=0, facecolor='#ffcccb', alpha=0.6,
                                                          zorder=1))
                probs = pol[idx]
                if np.all(probs == 0):
                    continue
                for a, p in enumerate(probs):
                    if p > 0:
                        dx, dy = dir_map[directions[a]]
                        plt.arrow(c + 0.5, r + 0.5, dx * 0.3, dy * 0.3, head_width=0.15, head_length=0.1,
                                  fc='b', ec='b', length_includes_head=True, zorder=2)
        plt.gca().set_aspect('equal')
        plt.savefig(save_paths[k], format='pdf', bbox_inches='tight')
        plt.close()
