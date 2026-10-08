"""Continuous LifeGate: the 10 x 10 LifeGate gridworld (LifeGate-Gridworld/gridworld.py) in continuous space.

Coordinates: x to the right, y up, the arena [0, 10]^2; grid cell (row r, column c) is the unit square
x in [c, c+1], y in [9-r, 10-r].
  exit (survival)   x in [5, 8), y >= 9                      grid cells (0, 5)-(0, 7)
  death             x >= 9, or x >= 8 and y >= 9              grid column 9 and cells (0, 8), (0, 9)
  dead-ends         x in [5, 9), y < 5: death on entry        grid rows 5-9, columns 5-8
  walls             x < 5 and y >= 9; x in [1, 5) and 4 <= y < 5    grid cells (0, 0)-(0, 4), (5, 1)-(5, 4)
A step moves the patient by Config.speed * a (the action a is clipped to the unit disk), plus a rightward drift
(the slip of LifeGate) and Gaussian noise.  A move into a wall or out of the arena is not made.
Rewards are paid on entry into the exit or death; an optional step reward is paid on every other step.

`inside`, `blocked` and `regions` use only comparisons and boolean operators, so they accept NumPy arrays
and torch tensors alike (plan_check.py evaluates them on the GPU).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    """Dynamics and task constants: step length, rightward drift and noise scale per step, discount, horizon."""
    speed: float = 0.5
    drift: float = 0.1
    noise: float = 0.05
    gamma: float = 0.97
    horizon: int = 200


WALLS = [((0., 5.), (9., 10.)), ((1., 5.), (4., 5.))]           # (x range, y range)
EXIT = ((5., 8.), (9., 10.))
DEADENDS = ((5., 9.), (0., 5.))


def inside(x, y, box):
    """Whether (x, y) lies in the half-open box ((x0, x1), (y0, y1))."""
    (x0, x1), (y0, y1) = box
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


def blocked(x, y):
    """Whether (x, y) is outside the arena or in a wall."""
    out = (x < 0) | (x > 10) | (y < 0) | (y > 10)
    for box in WALLS:
        out = out | inside(x, y, box)
    return out


def regions(x, y):
    """Boolean masks (death, exit); death covers the death column, the top-right corner and the dead-ends."""
    death = (x >= 9) | ((x >= 8) & (y >= 9)) | inside(x, y, DEADENDS)
    exit_ = ~death & inside(x, y, EXIT)
    return death, exit_


def terminal(x, y):
    """+1 exit, -1 death (death column, top-right corner or a dead-end), 0 otherwise."""
    death, exit_ = regions(x, y)
    return np.where(death, -1, np.where(exit_, 1, 0))


def clip_action(a):
    """Project actions onto the unit disk (actions inside it are unchanged)."""
    a = np.asarray(a, dtype=float)
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(norm, 1.0)


def transition(xy, a, eps, cfg=Config()):
    """xy: (..., 2), a: (..., 2), eps: (..., 2) standard normal -> next positions."""
    move = cfg.speed * clip_action(a)
    nx = xy[..., 0] + move[..., 0] + cfg.drift + cfg.noise * eps[..., 0]
    ny = xy[..., 1] + move[..., 1] + cfg.noise * eps[..., 1]
    stay = blocked(nx, ny)
    return np.stack([np.where(stay, xy[..., 0], nx), np.where(stay, xy[..., 1], ny)], axis=-1)


def free(x, y):
    """Neither a wall nor a terminal region (exit, death or dead-end)."""
    return ~blocked(x, y) & (terminal(x, y) == 0)


def initial_states(n, rng):
    """`n` positions uniform on the free space (as LifeGate averages over all non-terminal, non-wall cells)."""
    out = np.empty((0, 2))
    while len(out) < n:
        xy = rng.uniform(0.25, 9.75, (2 * n, 2))
        out = np.vstack([out, xy[free(xy[:, 0], xy[:, 1])]])
    return out[:n]
