"""Check, before any learning, which behavior is optimal in Continuous LifeGate under each reward design.

Value iteration on a 0.1 grid of the arena, 24 directions plus no action, the noise integrated by a 3 x 3
Gauss-Hermite rule, bilinear interpolation; the greedy policy (ties drawn uniformly) is run in the exact
simulator from uniform starts on the free space.  Prints the outcome rates of every design and, for the
step-cost designs, the range of the starts that die.  Runs on the GPU when one is available.

Usage:  python plan_check.py
"""
import numpy as np
import torch

import lifegate_continuous as lifegate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = lifegate.Config()
GRID_STEP = 0.1
GRID = np.arange(0, 10 + GRID_STEP / 2, GRID_STEP)                # grid points per axis
DIRECTIONS = np.linspace(0, 2 * np.pi, 24, endpoint=False)
ACTIONS = np.vstack([np.c_[np.cos(DIRECTIONS), np.sin(DIRECTIONS)], [[0., 0.]]])
GAUSS_HERMITE = [(-np.sqrt(3), 1 / 6), (0., 2 / 3), (np.sqrt(3), 1 / 6)]   # (node, weight) for N(0, 1)


def as_tensor(a):
    return torch.as_tensor(a, dtype=torch.float64, device=DEVICE)


def interp(V, x, y):
    """Bilinear interpolation of the grid values V[row = y index, column = x index] at (x, y)."""
    fx = torch.clamp(x / GRID_STEP, 0, len(GRID) - 1 - 1e-9)
    fy = torch.clamp(y / GRID_STEP, 0, len(GRID) - 1 - 1e-9)
    i, j = fy.floor().long(), fx.floor().long()
    wy, wx = fy - i, fx - j
    return ((1 - wy) * (1 - wx) * V[i, j] + (1 - wy) * wx * V[i, j + 1]
            + wy * (1 - wx) * V[i + 1, j] + wy * wx * V[i + 1, j + 1])


def q_values(V, x, y, design):
    """Q-values (..., action) at positions (x, y) under `design` = (exit reward, death reward, step reward).

    The dynamics are those of lifegate_continuous.transition, written in torch for all actions and noise nodes
    at once; `blocked` and `regions` are shared with the simulator.
    """
    r_exit, r_death, step_reward = design
    a = as_tensor(ACTIONS)
    mx = x[..., None] + CFG.speed * a[:, 0] + CFG.drift
    my = y[..., None] + CFG.speed * a[:, 1]
    q = torch.zeros_like(mx)
    for zx, wx in GAUSS_HERMITE:
        for zy, wy in GAUSS_HERMITE:
            nx, ny = mx + CFG.noise * zx, my + CFG.noise * zy
            stay = lifegate.blocked(nx, ny)
            nx = torch.where(stay, x[..., None].expand_as(nx), nx)
            ny = torch.where(stay, y[..., None].expand_as(ny), ny)
            death, exit_ = lifegate.regions(nx, ny)
            cont = step_reward + CFG.gamma * interp(V, nx, ny)
            q += wx * wy * torch.where(death, torch.full_like(nx, r_death),
                                       torch.where(exit_, torch.full_like(nx, r_exit), cont))
    return q


def solve(design, tol=1e-9, max_iter=3000):
    """Value iteration on the grid; returns the values and the number of iterations."""
    gx, gy = np.meshgrid(GRID, GRID)
    x, y = as_tensor(gx), as_tensor(gy)
    V = torch.zeros_like(x)
    for it in range(max_iter):
        V_next = q_values(V, x, y, design).max(-1).values
        delta = float((V_next - V).abs().max())
        V = V_next
        if delta < tol:
            break
    return V, it + 1


def run(V, design, starts, seed=0):
    """Run the greedy policy of V in the exact simulator; returns the outcome of every start (+1, -1, 0)."""
    rng = np.random.default_rng(seed)
    xy = starts.copy()
    n = len(xy)
    outcome = np.zeros(n, int)
    for _ in range(CFG.horizon):
        live = outcome == 0
        if not live.any():
            break
        q = q_values(V, as_tensor(xy[:, 0]), as_tensor(xy[:, 1]), design).cpu().numpy()
        tied = q >= q.max(axis=1, keepdims=True) - 1e-9
        a = ACTIONS[np.argmax(np.where(tied, rng.random(q.shape), -1), axis=1)]
        nxt = lifegate.transition(xy, a, rng.standard_normal((n, 2)), CFG)
        xy = np.where(live[:, None], nxt, xy)
        ev = lifegate.terminal(xy[:, 0], xy[:, 1])
        hit = live & (ev != 0)
        outcome[hit] = ev[hit]
    return outcome


def main():
    starts = lifegate.initial_states(400, np.random.default_rng(1))
    designs = {'survival': (1., 0., 0.), 'death': (0., -1., 0.)}
    for c in (-0.02, -0.04, -0.06):
        designs[f'step {c}'] = (1., 0., c)
    for name, design in designs.items():
        V, iterations = solve(design)
        outcome = run(V, design, starts)
        print(f'{name:12s} exit {np.mean(outcome == 1):5.1%}  death {np.mean(outcome == -1):5.1%}  '
              f'unresolved {np.mean(outcome == 0):5.1%}  ({iterations} iterations)', flush=True)
        if name.startswith('step'):
            dead = starts[outcome == -1]
            if len(dead):
                print('   dying starts: x', np.round([dead[:, 0].min(), dead[:, 0].max()], 2),
                      'y', np.round([dead[:, 1].min(), dead[:, 1].max()], 2), flush=True)


if __name__ == '__main__':
    main()
