"""Count risky states and policy disagreements in LifeGate, with collapsing and with drifting dead-ends.

For each dead-end model (`collapse`: gridworld.get_gridworld, `drift`: gridworld_drift.get_gridworld), discount
(0.9, 1.0), slip probability (0.2, 0.0) and step reward (-0.04, 0.0), the optimal policies of the survival
(`plus`), avoidance (`minus`) and mixed reward designs are computed by value iteration.  Recorded per setting:
  plus, minus, mixed   number of decision states (not a wall, terminal or dead-end) where some optimal action
                       moves directly into a death cell or a dead-end
  d_pm, d_ppm, d_mpm   number of non-wall, non-terminal states where the sets of optimal actions differ
                       (plus vs minus, plus vs mixed, minus vs mixed)
Output: drift_danger_counts.json (rewritten after every setting).

Usage:  python drift_danger_counts.py
"""
import itertools
import json

import numpy as np

import gridworld as gw
import gridworld_drift as gd

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]        # north, east, south, west, as in gridworld.get_gridworld
DEADLY = set(gd.MORTALITIES) | {(x, y) for x in gd.DEADENDS_X for y in gd.DEADENDS_Y}


def count_risky_states(pi):
    """Decision states where some optimal action (pi > 0) moves into a death cell or a dead-end."""
    count = 0
    for s in range(100):
        x, y = divmod(s, 10)
        if (x, y) in gd.NO_DECISION or (x, y) in DEADLY:
            continue
        best = np.flatnonzero(pi[s] > 0)
        if any((x + MOVES[a][0], y + MOVES[a][1]) in DEADLY for a in best):
            count += 1
    return count


def count_disagreements(pi_a, pi_b):
    """Non-wall, non-terminal states where the two policies have different sets of optimal actions."""
    return sum(1 for s in range(100)
               if divmod(s, 10) not in gd.NO_DECISION and not np.array_equal(pi_a[s] > 0, pi_b[s] > 0))


def main():
    out = {}
    for label, builder in [('collapse', gw.get_gridworld), ('drift', gd.get_gridworld)]:
        for gamma, slip, step in itertools.product([0.9, 1.0], [0.2, 0.0], [-0.04, 0.0]):
            res, policies = {}, {}
            for mode, name in [('survival', 'plus'), ('avoidance', 'minus'), ('mixed', 'mixed')]:
                env = builder(mode=mode, slip_prob=slip, intermediate_reward=step)
                _, pi = gw.value_iter(env, theta=1e-15, gamma=gamma)
                res[name] = count_risky_states(pi)
                policies[name] = pi
            res.update(d_pm=count_disagreements(policies['plus'], policies['minus']),
                       d_ppm=count_disagreements(policies['plus'], policies['mixed']),
                       d_mpm=count_disagreements(policies['minus'], policies['mixed']))
            out[f'{label}_g{gamma}_sp{slip}_step{step}'] = res
            print(label, gamma, slip, step, res, flush=True)
            with open('drift_danger_counts.json', 'w') as f:
                json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
