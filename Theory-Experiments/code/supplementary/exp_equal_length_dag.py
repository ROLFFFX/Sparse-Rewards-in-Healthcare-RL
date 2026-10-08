"""
Corollary "Exact invariance" (iii) and Corollary "Invariance beyond guaranteed absorption and no
discounting" (a), on random layered DAGs.

If the terminal defect Z^pi = 1 - V_+^pi + V_-^pi does not depend on the policy, every design has the
optimal set Pi_+^*, for any gamma.  Random layered DAGs with L layers of W states and A actions:
  equal-length:    only the last layer terminates, so tau = L - layer for every path and every policy
                   (P(tau = k) is policy-independent, case (a)), and Z^pi is policy-independent;
  variable-length: the same DAGs, but every action in earlier layers also ends the episode immediately
                   with probability q ~ U(0, 0.6) (survival w.p. v ~ U(0, 1)), so tau depends on the policy.
For each discount factor we compare the optimal-policy sets of all designs lambda in [0, 1] (including
survival-only and death-only) with Pi_+^*.  The theory predicts:
  equal-length:    identical for every gamma (discounting alone does not break the equivalence);
  variable-length: design-dependent for gamma < 1, identical again at gamma = 1
                   (Corollary "Zero defect and classical equivalence").
It also checks the premise directly: the spread of Z^pi across random policies.

Outputs: results/supplementary/equal_length_dag.json and plots/supplementary/propositions/equal_length_dag.pdf.

Usage:  python exp_equal_length_dag.py [--n 2000] [--layers 6] [--width 4] [--actions 3] [--seed 0] [--plot-only]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # library in code/

import numpy as np
import torch

import common
from batched import design_rewards, optimal_sets, value_iteration
from common import apply_style, C_NEVER, C_POLICY, figsize, GRID_ALPHA, load_json, plt, save_json, savefig
from tabular import DEVICE, DTYPE, normalize_Q

common.set_group('propositions')      # figures and results go to the supplementary folders

GAMMAS = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]


def make_dags(n, L, W, A, variable, seed):
    """
    n random layered DAGs with L layers of W states (state index layer * W + i) and A actions.

    Every action in layers 0..L-2 moves to two random states of the next layer with Dirichlet(1, 1)
    weights; every action in the last layer terminates, surviving with probability u ~ U(0, 1).  With
    variable=True, actions in earlier layers also terminate early as described in the module docstring.
    Returns P_cont (n, S, A, S), p_plus, p_minus (n, S, A) and an all-True action mask (n, S, A).
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    S = L * W
    P = torch.zeros(n, S, A, S, dtype=DTYPE)
    p_plus = torch.zeros(n, S, A, dtype=DTYPE)
    p_minus = torch.zeros(n, S, A, dtype=DTYPE)
    for layer in range(L):
        rows = slice(layer * W, (layer + 1) * W)
        if layer == L - 1:
            u = torch.rand(n, W, A, generator=g, dtype=DTYPE)
            p_plus[:, rows], p_minus[:, rows] = u, 1 - u
            continue
        succ = torch.stack([torch.randperm(W, generator=g)[:2] for _ in range(n * W * A)]).view(n, W, A, 2)
        wts = torch.distributions.Dirichlet(torch.ones(2, dtype=DTYPE)).sample((n, W, A))
        block = torch.zeros(n, W, A, W, dtype=DTYPE).scatter_add_(-1, succ, wts)
        if variable:
            q = 0.6 * torch.rand(n, W, A, generator=g, dtype=DTYPE)
            v = torch.rand(n, W, A, generator=g, dtype=DTYPE)
            p_plus[:, rows], p_minus[:, rows] = q * v, q * (1 - v)
            block = block * (1 - q)[..., None]
        P[:, rows, :, (layer + 1) * W:(layer + 2) * W] = block
    mask = torch.ones(n, S, A, dtype=torch.bool)
    return P.to(DEVICE), p_plus.to(DEVICE), p_minus.to(DEVICE), mask.to(DEVICE)


def premise_spread(P, p_plus, p_minus, gamma, L, n_policies, seed):
    """
    Max over states of (max - min) of V_+^pi - V_-^pi = 1 - Z^pi across random deterministic policies.
    Returns a tensor of shape (n,).
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    n, S, A, _ = P.shape
    act = torch.randint(0, A, (n, n_policies, S), generator=g).to(DEVICE)
    pi = torch.zeros(n, n_policies, S, A, dtype=DTYPE, device=DEVICE).scatter_(-1, act[..., None], 1.0)
    Ppi = torch.einsum('npsa,nsat->npst', pi, P)
    r = torch.einsum('npsa,nsa->nps', pi, p_plus + p_minus)       # reward of entering any terminal
    V = torch.zeros(n, n_policies, S, dtype=DTYPE, device=DEVICE)
    for _ in range(L + 1):
        V = r + gamma * torch.einsum('npst,npt->nps', Ppi, V)
    return (V.max(1).values - V.min(1).values).max(-1).values


def run(n, L, W, A, seed):
    """Compare the optimal sets of all designs with Pi_+^* for both DAG variants and every gamma in GAMMAS."""
    ratio = 10.0 ** np.linspace(-3, 3, 21)
    w = np.concatenate([[0.0], ratio / (1 + ratio), [1.0]])        # alpha; the last design (1, 0) is Pi_+^*
    rows = []
    for variant in ['equal-length', 'variable-length']:
        P, p_plus, p_minus, mask = make_dags(n, L, W, A, variant == 'variable-length', seed)
        r = design_rewards(p_plus, p_minus, w, 1 - w)
        for gamma in GAMMAS:
            Q, _ = value_iteration(P, r, gamma, mask, iters=L + 1)     # exact: the DAG has depth L
            sets = optimal_sets(normalize_Q(Q, (1, 2, 3)), mask, atol=1e-9)
            differs = (sets != sets[:, -1:]).any(-1)                    # (n, D, S) vs Pi_+^*
            spread = premise_spread(P, p_plus, p_minus, gamma, L, 32, seed + 1)
            rows.append({
                'variant': variant, 'gamma': gamma,
                'fraction of instances where the optimal set depends on the design':
                    float(differs.any(-1).any(-1).double().mean()),
                'fraction where Pi_+ != Pi_- (endpoints)': float(differs[:, 0].any(-1).double().mean()),
                'mean fraction of (design, state) pairs that differ': float(differs.double().mean()),
                'max over instances of spread of V+ - V- across policies': float(spread.max()),
                'mean spread of V+ - V- across policies': float(spread.mean()),
            })
            print(rows[-1])
    return rows


def plot(rows):
    """(a) share of instances with a design-dependent optimum; (b) largest spread of Z^pi; both against gamma."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(1, 2))
    styles = {'equal-length': (C_POLICY['plus'], 'o', '-'), 'variable-length': (C_POLICY['minus'], 's', '--')}
    xpos = np.arange(len(GAMMAS))
    for variant, (color, marker, ls) in styles.items():
        sub = [r for r in rows if r['variant'] == variant]
        dependent = [100 * r['fraction of instances where the optimal set depends on the design'] for r in sub]
        # floor exact zeros so that they appear on the log axis, inside the round-off band
        spread = [max(r['max over instances of spread of V+ - V- across policies'], 1e-17) for r in sub]
        axes[0].plot(xpos, dependent, color=color, marker=marker, linestyle=ls, label=variant)
        axes[1].plot(xpos, spread, color=color, marker=marker, linestyle=ls, label=variant)
    axes[1].set_yscale('log')
    axes[1].axhspan(1e-17, 1e-14, color=C_NEVER, alpha=0.2, zorder=0)
    axes[1].text(0.3, 2e-17, 'round-off', fontsize=14, va='bottom', transform=axes[1].get_yaxis_transform())
    axes[0].set_ylabel('Design-dependent optimum (%)')
    axes[0].set_title(r'(a) Optimal set depends on $\lambda$')
    axes[1].set_ylabel(r'Max spread of $Z^\pi$ over policies')
    axes[1].set_title(r'(b) Spread of the defect')
    for ax in axes:
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(g) for g in GAMMAS])
        ax.set_xlabel(r'Discount $\gamma$')
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(loc='center left')
    fig.tight_layout()
    savefig(fig, 'equal_length_dag')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2000, help='random DAGs per variant')
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--actions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    if args.plot_only:
        plot(load_json('equal_length_dag.json')['results'])
        return
    rows = run(args.n, args.layers, args.width, args.actions, args.seed)
    save_json({'n_instances': args.n, 'layers': args.layers, 'width': args.width, 'actions': args.actions,
               'results': rows}, 'equal_length_dag.json')
    plot(rows)


if __name__ == '__main__':
    main()
