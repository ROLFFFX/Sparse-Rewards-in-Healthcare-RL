"""
Corollary "Exact invariance" (i), (ii) as a diagnostic, on random MDPs (with cycles, gamma < 1).

For each random MDP we record
  A  : Pi_+^* and Pi_-^* intersect                       (per state: argmax sets intersect)
  B1 : one policy is optimal for every lambda in (0, 1) on the grid
  B2 : the optimal set is the same for every lambda in (0, 1) on the grid
  C  : (only when A) every interior design has optimal set exactly Pi_+^* cap Pi_-^*   (part (ii))
The corollary predicts A <=> B1 <=> B2 and A => C.  The grid is alpha/beta = 10^x, x in [-6, 6],
i.e. lambda = beta/(alpha+beta) from ~1e-6 to 1 - 1e-6.

Random MDPs have S non-terminal states, A actions and two absorbing terminals; each (s, a) moves to
1-3 random successors.  To obtain both outcomes of A in useful proportions, half of the instances
round transition probabilities to multiples of 0.1 and duplicate one action (structural ties).

Outputs: results/supplementary/random_mdp_diagnostic.json (gamma = 0.9 and 0.99) and
plots/supplementary/propositions/random_mdp_diagnostic.pdf.

Usage:  python exp_random_mdp_diagnostic.py [--n 20000] [--states 5] [--actions 3] [--chunk 2000] [--seed 0]
                                            [--plot-only]
"""
import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # library in code/

import numpy as np
import torch

import common
from batched import design_rewards, exact_optimal, optimal_sets
from common import apply_style, C_MIXED, C_SURV, figsize, GRID_ALPHA, load_json, plt, save_json, savefig
from tabular import DEVICE, DTYPE, normalize_Q

common.set_group('propositions')      # figures and results go to the supplementary folders


def random_mdps(n, S, A, seed):
    """
    n random MDPs.  Returns P_cont (n, S, A, S), p_plus, p_minus (n, S, A), the action mask (n, S, A) and
    a boolean (n,) marking the structured instances (rounded probabilities, possibly a duplicated action).
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    T = S + 2                                             # + survival, death
    k = torch.randint(1, 4, (n, S, A), generator=g)
    logits = torch.rand(n, S, A, T, generator=g, dtype=DTYPE)
    ranks = logits.argsort(-1).argsort(-1)                # random ranking of successors
    keep = ranks < k[..., None]
    wts = torch.distributions.Gamma(torch.ones(()), torch.ones(())).sample((n, S, A, T)).to(DTYPE) * keep
    probs = wts / wts.sum(-1, keepdim=True)
    structured = torch.rand(n, generator=g) < 0.5
    rounded = torch.round(probs * 10) / 10
    rounded = torch.where(rounded.sum(-1, keepdim=True) > 0, rounded, probs)
    rounded = rounded / rounded.sum(-1, keepdim=True)
    probs = torch.where(structured[:, None, None, None], rounded, probs)
    dup = structured & (torch.rand(n, generator=g) < 0.5)
    probs[dup, :, A - 1] = probs[dup, :, 0]               # an exact duplicate treatment
    P_cont = probs[..., :S]
    p_plus, p_minus = probs[..., S], probs[..., S + 1]
    mask = torch.ones(n, S, A, dtype=torch.bool)
    return P_cont.to(DEVICE), p_plus.to(DEVICE), p_minus.to(DEVICE), mask.to(DEVICE), structured


def run(n, S, A, gamma, seed, chunk):
    """Evaluate A, B1, B2 and C on n random MDPs (processed in chunks) and summarize their agreement."""
    ratio = 10.0 ** np.linspace(-6, 6, 49)
    w = np.concatenate([[1.0, 0.0], ratio / (1 + ratio)])  # index 0: +, 1: -, then interior designs
    counts = {}
    records = []
    worst_mismatch_gap = 0.0
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        P, pp, pm, mask, structured = random_mdps(m, S, A, seed + start)
        r = design_rewards(pp, pm, w, 1 - w)
        Q, _ = exact_optimal(P, r, gamma, mask)
        # Q* comes from exact linear solves (~1e-15); a looser tolerance would merge actions whose death-only
        # gap (>= 1.7e-5 here) is scaled below it by the extreme weights beta ~ 1e-6 of the grid
        sets = optimal_sets(normalize_Q(Q, (2, 3)), mask, atol=1e-12)                  # (m, D, S, A)
        plus, minus, interior = sets[:, 0], sets[:, 1], sets[:, 2:]
        cap = plus & minus
        A_ = cap.any(-1).all(-1)
        B1 = interior.all(1).any(-1).all(-1)
        B2 = (interior == interior[:, :1]).all(-1).all(-1).all(-1)
        C = (interior == cap[:, None]).all(-1).all(-1).all(-1)
        # size of the true weighted gap alpha*gap_+ + beta*gap_- of every action classified differently
        # from Pi_+ cap Pi_- (only in A-instances); values below atol are numerical-resolution cases
        gp = Q[:, 0].max(-1, keepdim=True).values - Q[:, 0]
        gm = Q[:, 1].max(-1, keepdim=True).values - Q[:, 1]
        wt = torch.as_tensor(w[2:], dtype=Q.dtype, device=Q.device)[None, :, None, None]
        weighted = wt * gp[:, None] + (1 - wt) * gm[:, None]
        wrong = (interior != cap[:, None]) & A_[:, None, None, None] & mask[:, None]
        if wrong.any():
            worst_mismatch_gap = max(worst_mismatch_gap, float(weighted[wrong].max()))
        for a, b1, b2, c, st in zip(A_.tolist(), B1.tolist(), B2.tolist(), C.tolist(), structured.tolist()):
            key = (a, b1, b2)
            counts[key] = counts.get(key, 0) + 1
            records.append((a, b1, b2, c, st))
    rec = np.array(records, dtype=bool)
    A_, B1, B2, C = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    return {
        'gamma': gamma, 'n': n, 'S': S, 'A': A, 'grid': 'alpha/beta = 10^x, x in linspace(-6, 6, 49)',
        'P(A)': float(A_.mean()),
        'A == B1 (fraction of instances)': float((A_ == B1).mean()),
        'A == B2 (fraction of instances)': float((A_ == B2).mean()),
        'A => C (fraction of A-instances with interior sets == Pi+ cap Pi-)':
            float(C[A_].mean()) if A_.any() else None,
        'contingency (A, B1, B2) -> count': {str(k): v for k, v in sorted(counts.items())},
        'tie tolerance (on Q / max|Q|)': 1e-12,
        'largest true weighted gap alpha*gap_+ + beta*gap_- among (A, not C) mismatches':
            worst_mismatch_gap,
        'P(A) on structured / generic instances': [float(A_[rec[:, 4]].mean()), float(A_[~rec[:, 4]].mean())],
    }


def plot(results):
    """Share of random MDPs in each cell of the (A, B1) contingency table, one bar group per gamma."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize(1, 1, w=8.0))
    width = 0.8 / len(results)
    for j, res in enumerate(results):
        c = {ast.literal_eval(k): v for k, v in res['contingency (A, B1, B2) -> count'].items()}
        tot = sum(c.values())
        vals = [sum(v for (a, b1, _), v in c.items() if a and b1),
                sum(v for (a, b1, _), v in c.items() if a and not b1),
                sum(v for (a, b1, _), v in c.items() if b1 and not a),
                sum(v for (a, b1, _), v in c.items() if not a and not b1)]
        pos = np.arange(4) + (j - (len(results) - 1) / 2) * width
        bars = ax.bar(pos, [100 * v / tot for v in vals], width=width * 0.92,
                      color=[C_SURV, C_MIXED][j % 2], label=f"$\\gamma={res['gamma']}$")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, str(v), ha='center', fontsize=14)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([r'$A\wedge B$', r'$A\wedge\neg B$', r'$\neg A\wedge B$', r'$\neg A\wedge\neg B$'])
    ax.set_ylim(0, 108)
    ax.set_ylabel('Random MDPs (%)')
    ax.set_title('Common optimum diagnostic')
    ax.grid(True, axis='y', alpha=GRID_ALPHA)
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, 'random_mdp_diagnostic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--states', type=int, default=5)
    parser.add_argument('--actions', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    if args.plot_only:
        plot(load_json('random_mdp_diagnostic.json'))
        return
    results = []
    for gamma in [0.9, 0.99]:
        res = run(args.n, args.states, args.actions, gamma, args.seed, args.chunk)
        print(res)
        results.append(res)
    save_json(results, 'random_mdp_diagnostic.json')
    plot(results)


if __name__ == '__main__':
    main()
