"""Continuous LifeGate figure: the environment and the PPO policies learned under three reward designs.

(a) The arena and the display starts.  (b)-(d) The policy of training fold SHOWN_FOLD learned under survival reward,
death penalty and survival reward with a step cost, run with deterministic actions (as in the test episodes)
under drift and noise: one episode from each display start, with the same starts and noise in every panel.
The display starts form a staggered grid on the free space (--under-wall: a denser grid under the wall only,
where the route to discharge is the long detour).  For display only, positions are smoothed by a centered moving
average over SMOOTH steps; episodes that neither exit nor die are smoothed by a Gaussian over STALL_SIGMA steps
and end in a dot at the horizon.  Paths are colored by outcome.
summary.json: outcome rates (mean and standard deviation over the 5 seeds) of the 5,000 deterministic test
episodes of every seed (train.py), for all test patients and for those starting under the wall.
Output: ../plots/<out>.{pdf,svg,png} (default out: continuous_lifegate), ../results/summary.json

Usage:  python plot_figure.py [--under-wall] [--out NAME]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pymupdf  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import lifegate_continuous as lifegate  # noqa: E402
import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SHOWN_FOLD = 2              # training fold shown in (b)-(d): the behavior of most seeds
SMOOTH = 5                  # moving-average window (steps)
STALL_SIGMA = 8             # Gaussian smoothing of unresolved episodes (steps)
ROLLOUT_SEED = 7            # noise of the displayed episodes
LINE_WIDTH = .55
SPACING, TOP, MARGIN = 1.25, 8.3, .3     # display starts: a staggered grid on the free space
UNDER_SPACING = .8                        # --under-wall: a denser grid under the wall only
# the look of the patient-navigation schematic: saturated blocks, cell lines, a black frame;
# paths that reach the exit are blue, paths that stall or die are purple
BLUE, PURPLE = '#1f5fbf', '#7b3294'
INK, MUTED = '#1a1a1a', '#555555'
FILL = {'wall': '#808080', 'exit': '#2ca25f', 'death': '#e3201c', 'deadend': '#f4a582'}
PANELS = [('survival', '(b) survival reward'), ('death', '(c) death penalty'), ('stepcost', '(d) step cost')]
COLOR = {1: BLUE, -1: PURPLE, 0: PURPLE}


def under_wall(xy):
    return (xy[:, 0] < 5) & (xy[:, 1] < 4)


def display_starts(under=False):
    """A staggered grid on the free space below y = TOP (under: under the wall only), each start at least MARGIN
    from walls and terminal regions."""
    pts = []
    spacing, top, right = (UNDER_SPACING, 4, 5) if under else (SPACING, TOP, 9)
    for j, y in enumerate(np.arange(.6, top, spacing)):
        for x in np.arange(.6 + spacing / 2 * (j % 2), right, spacing):
            near = np.array([(x + dx, y + dy) for dx in (-MARGIN, 0, MARGIN) for dy in (-MARGIN, 0, MARGIN)])
            if lifegate.free(near[:, 0], near[:, 1]).all():
                pts.append((x, y))
    return np.array(pts)


def style():
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'cm', 'font.size': 7.5,
                         'axes.titlesize': 9, 'text.color': INK, 'axes.edgecolor': '#9aa5ae', 'axes.linewidth': .6,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def arena(ax):
    """Draw the walls, exit, death column and dead-ends as grid cells, with a black frame."""
    cells = [((c, r), FILL['deadend']) for c in range(5, 9) for r in range(0, 5)]
    cells += [((9, r), FILL['death']) for r in range(10)] + [((8, 9), FILL['death'])]
    cells += [((c, 9), FILL['exit']) for c in range(5, 8)]
    cells += [((c, 9), FILL['wall']) for c in range(0, 5)] + [((c, 4), FILL['wall']) for c in range(1, 5)]
    for (c, r), color in cells:                         # one square per grid cell, with white cell lines
        ax.add_patch(Rectangle((c, r), 1, 1, fc=color, ec='white', lw=.5, zorder=1))
    ax.set(xlim=(0, 10), ylim=(0, 10), xticks=[], yticks=[], aspect='equal')
    for side in ax.spines.values():
        side.set_color(INK)
        side.set_linewidth(1.1)


def rollout(model, starts, cfg, seed):
    """Deterministic actions (as in the test episodes) with drift and noise.  One episode per start.

    Returns the paths (start, step, 2), the outcomes (+1, -1, 0) and the end steps.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xy, n = starts.copy(), len(starts)
    outcome, end = np.zeros(n, int), np.full(n, cfg.horizon)
    paths = [xy.copy()]
    for step in range(cfg.horizon):
        a, _ = model.predict(train.observation(xy, np.full(n, step), cfg), deterministic=True)
        live = outcome == 0
        xy = np.where(live[:, None], lifegate.transition(xy, a, rng.standard_normal((n, 2)), cfg), xy)
        ev = lifegate.terminal(xy[:, 0], xy[:, 1])
        hit = live & (ev != 0)
        outcome[hit], end[hit] = ev[hit], step + 1
        paths.append(xy.copy())
    return np.array(paths).transpose(1, 0, 2), outcome, end


def smooth(p, w=SMOOTH):
    """Centered moving average (display only); the end points are kept."""
    if len(p) <= w:
        return p
    k = np.ones(w) / w
    pad = np.vstack([np.repeat(p[:1], w // 2, 0), p, np.repeat(p[-1:], w // 2, 0)])
    q = np.c_[np.convolve(pad[:, 0], k, 'valid'), np.convolve(pad[:, 1], k, 'valid')]
    q[0], q[-1] = p[0], p[-1]
    return q


def smooth_stall(p, sigma=STALL_SIGMA):
    """Gaussian smoothing over time (sigma in steps) for episodes that neither exit nor die (display only)."""
    q = gaussian_filter1d(p, sigma, axis=0, mode='nearest')
    q[0] = p[0]
    return q


def rates(design):
    """Outcome rates over the seeds: all test patients, and those starting under the wall."""
    out = {}
    for name, part in (('all', None), ('under_wall', under_wall)):
        fractions = []
        for k in range(len(train.SEEDS)):
            o = np.load(RESULTS / f'{design}/fold{k}/evaluation.npz')['outcome']
            if part is not None:
                o = o[part(train.test_starts(train.N_TEST, train.TEST_SEEDS[k]))]
            fractions.append([np.mean(o == 1), np.mean(o == -1), np.mean(o == 0)])
        fractions = np.array(fractions)
        out[name] = {m: [float(fractions[:, i].mean()), float(fractions[:, i].std(ddof=1))]
                     for i, m in enumerate(('exit', 'death', 'unresolved'))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--under-wall', action='store_true', help='display starts under the wall only')
    parser.add_argument('--out', default='continuous_lifegate')
    args = parser.parse_args()
    style()
    cfg = lifegate.Config()
    starts = display_starts(args.under_wall)
    drift_y = 6.6 if args.under_wall else 7.47                  # clear of the starts
    fig, axs = plt.subplots(1, 4, figsize=(6.75, 1.95))
    fig.subplots_adjust(left=.01, right=.99, bottom=.13, top=.98, wspace=.08)
    for ax in axs:
        arena(ax)
    ax = axs[0]
    ax.text(6.5, 9.5, 'exit', ha='center', va='center', color='white', fontsize=7, zorder=3)
    ax.text(9.5, 5, 'death', ha='center', va='center', color='white', fontsize=7, rotation=90, zorder=3)
    ax.text(7, 2.5, 'dead-ends', ha='center', va='center', color=INK, fontsize=7, zorder=3)
    ax.annotate('', xy=(8.4, drift_y), xytext=(6.2, drift_y),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.8, mutation_scale=7))
    ax.text(6.05, drift_y, 'drift', ha='right', va='center', color=MUTED, fontsize=7)
    ax.scatter(*starts.T, s=5, facecolor='white', edgecolor=INK, lw=.6, zorder=6)
    ax.set_xlabel('(a) environment', labelpad=3, fontsize=8.5)
    summary = {}
    for ax, (design, title) in zip(axs[1:], PANELS):
        model = PPO.load(RESULTS / f'{design}/fold{SHOWN_FOLD}/model.zip', device='cpu')
        trace, outcomes, ends = rollout(model, starts, cfg, ROLLOUT_SEED)
        for p, outcome, end in zip(trace, outcomes, ends):
            p = p[:int(end) + 1]
            q = smooth(p) if outcome != 0 else smooth_stall(p)
            color = COLOR[int(outcome)]
            ax.plot(*q.T, color=color, lw=LINE_WIDTH, alpha=.8, zorder=4, solid_capstyle='round',
                    solid_joinstyle='round')
            if outcome != 0 and len(q) > 1:             # arrowhead where the episode exits or dies
                ax.annotate('', xy=q[-1], xytext=q[-min(4, len(q))], zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=LINE_WIDTH, mutation_scale=5))
            else:                                       # dot where an unresolved patient is at the horizon
                ax.scatter(*q[-1], s=4, color=color, lw=0, zorder=6)
        ax.set_xlabel(title, labelpad=3, fontsize=8.5)
        summary[design] = rates(design)
        print(design, 'displayed outcomes', outcomes.tolist(), flush=True)
    out = ROOT / 'plots' / args.out
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)
    with pymupdf.open(out.with_suffix('.pdf')) as doc:          # PNG preview rendered from the PDF
        doc[0].get_pixmap(dpi=300).save(out.with_suffix('.png'))
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2))
    for design, rate in summary.items():
        print(design, rate)
    print('saved', out)


if __name__ == '__main__':
    main()
