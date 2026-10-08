"""
Shared plotting style, colors, output paths and result I/O for the theory experiments.

Fonts match the other figures of the paper: STIXGeneral text and Computer Modern math. Sizes follow
LifeGate-Gridworld/gridworld_asym.py (18 pt text, 22 pt titles, 3 pt lines) on ~7 x 5.5 in panels, so text
and plot keep the proportions of the original ~10 x 6 in figures. Every figure is saved as PDF and SVG, with
a PNG preview rasterized from the PDF.

Importing this module selects the Agg backend and applies the style; the experiment scripts also import `np`
and `plt` from here.
"""
import json
import os

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))              # Theory-Experiments/code
ROOT = os.path.dirname(HERE)                                   # Theory-Experiments
REPO = os.path.dirname(ROOT)
PLOT_DIR = os.path.join(ROOT, 'plots')
RESULT_DIR = os.path.join(ROOT, 'results')
# per-state arrays derived from MIMIC stay out of the repository (mimic-sepsis/data is git-ignored)
MIMIC_RESULT_DIR = os.path.join(REPO, 'mimic-sepsis', 'data', 'theory_results')

# which part of the paper the running script belongs to: 'theorems' or 'propositions' (supplementary)
GROUP = 'theorems'

STYLE = {
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'cm',
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
PANEL = (7.0, 5.5)            # inches per subplot


def apply_style():
    """(Re)apply the style; LifeGate-Gridworld/gridworld.py changes rcParams when it is imported."""
    plt.rcParams.update(STYLE)


apply_style()

# Color roles of the original plots (survival blue, mixed purple, death red, reference gray), in the
# ColorBrewer RdBu / PRGn tones: same hues, print- and colorblind-friendly (blue vs. red).
C_SURV, C_DEATH, C_MIXED, C_NEVER, C_TERM = '#2166ac', '#b2182b', '#762a83', '#8c8c8c', '#252525'
C_LATE_SURV, C_LATE_DEATH = '#92c5de', '#f4a582'
C_POLICY = {'plus': C_SURV, 'mixed': C_MIXED, 'minus': C_DEATH, 'ref': C_NEVER}
L_POLICY = {'plus': r'$\pi_{+}^{*}$', 'mixed': r'$\pi_{\pm}^{*}$', 'minus': r'$\pi_{-}^{*}$'}
GRID_ALPHA = 0.3


def figsize(nrows=1, ncols=1, w=PANEL[0], h=PANEL[1]):
    """Figure size in inches for a grid of nrows x ncols panels of w x h inches."""
    return (w * ncols, h * nrows)


def set_group(group):
    """Scripts in code/supplementary call set_group('propositions'): results go to results/supplementary."""
    global GROUP, RESULT_DIR
    GROUP = group
    if group == 'propositions':
        RESULT_DIR = os.path.join(ROOT, 'results', 'supplementary')
    else:
        RESULT_DIR = os.path.join(ROOT, 'results')


def savefig(fig, name, main=False, **kwargs):
    """
    Save and close `fig`. PDF: plots/<name>.pdf for the main figure, plots/supplementary/<group>/<name>.pdf
    otherwise; SVG and PNG preview: plots/exports/{main,supplementary}/<name>.{svg,png}.
    Extra keyword arguments go to Figure.savefig (default bbox_inches='tight').
    """
    kwargs = {'bbox_inches': 'tight', **kwargs}
    pdf_dir = PLOT_DIR if main else os.path.join(PLOT_DIR, 'supplementary', GROUP)
    export = os.path.join(PLOT_DIR, 'exports', 'main' if main else 'supplementary')
    for d in (pdf_dir, export):
        os.makedirs(d, exist_ok=True)
    pdf = os.path.join(pdf_dir, name + '.pdf')
    fig.savefig(pdf, facecolor='white', **kwargs)
    fig.savefig(os.path.join(export, name + '.svg'), facecolor='white', **kwargs)
    plt.close(fig)
    rasterize(pdf, os.path.join(export, name + '.png'), dpi=400 if main else 150)
    print('saved', pdf)


def rasterize(pdf, png, dpi):
    """PNG from the PDF: matplotlib's Agg drops the Computer Modern minus in subscripts (pi_-^*, V_-)."""
    import pymupdf
    with pymupdf.open(pdf) as doc:
        doc[0].get_pixmap(dpi=dpi).save(png)


def result_path(name, mimic=False):
    """Path of a result file (directory created if needed); MIMIC per-state arrays go to MIMIC_RESULT_DIR."""
    d = MIMIC_RESULT_DIR if mimic else RESULT_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_json(obj, name):
    """Write `obj` to RESULT_DIR/<name>; numpy and torch scalars are converted with .item()."""
    path = result_path(name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print('saved', path)


def load_json(name):
    with open(result_path(name)) as f:
        return json.load(f)


def to_np(x):
    """Tensor or array-like -> numpy array (tensors are moved to the CPU)."""
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
