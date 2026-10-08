"""
Proposition "Terminal scale versus step cost".

Reward k * R_{alpha,beta} + c with a fixed per-step reward c on transitions between non-terminal
states, for k from 1 to 100 (the +-1 and +-100 conventions of the literature).  The proposition predicts,
for optimal policies pi_k, pi_k' with k < k' and at every state:
    V_{alpha,beta}^{pi_k'} >= V_{alpha,beta}^{pi_k}      (terminal part improves)
    V_c^{pi_k'}           <= V_c^{pi_k}                  (intermediate part is sacrificed)
and, with c = 0, identical optimal-policy sets for all k.  Checked for the designs (1,0), (1,1), (0,1)
at all pairs of scales, all states and three optimal tie-breaking rules.

Outputs: results/supplementary/terminal_scale_<env>.json, the curves at the initial states in
terminal_scale_<env>.npz (results/supplementary for LifeGate, mimic-sepsis/data/theory_results for MIMIC),
and the figure plots/supplementary/propositions/terminal_scale_<env>.pdf.

Usage:  python exp_terminal_scale.py --env lifegate|mimic [--gamma G] [--step-cost C] [--n-scales 81] [--plot-only]
        (defaults: gamma = 0.9, c = -0.04 for LifeGate; gamma = 0.99, c = -0.01 for MIMIC)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # library in code/

import numpy as np
import torch

import common
import envs
from common import apply_style, C_POLICY, figsize, GRID_ALPHA, plt, result_path, save_json, savefig, to_np
from exp_mass_transfer import tie_break
from tabular import (DTYPE, DEVICE, absorption_probabilities, at_start, evaluate_discounted, normalize_Q,
                     optimal_sets, solve_optimal)

common.set_group('propositions')      # figures and results go to the supplementary folders

DESIGNS = {'plus': (1.0, 0.0), 'mixed': (1.0, 1.0), 'minus': (0.0, 1.0)}


def pairwise_violation(q, k, increasing):
    """
    q: (K, S) for increasing scales k.  Monotonicity over all pairs k < k' and states: returns the max
    violation, the number of violations > 1e-9 and the number of comparisons.
    """
    diff = q[None, :, :] - q[:, None, :]                          # q[k'] - q[k] at [k, k']
    later = torch.as_tensor(k, device=q.device)[None, :] > torch.as_tensor(k, device=q.device)[:, None]
    viol = (-diff if increasing else diff)[later]
    return float(viol.clamp(min=0).max()), int((viol > 1e-9).sum()), int(viol.numel())


def run(env, gamma, step_cost, n_scales):
    """Check the monotonicity predictions for each design; save and return the curves at the initial states."""
    mdp = envs.lifegate(0.2) if env == 'lifegate' else envs.mimic()
    ks = np.logspace(0, 2, n_scales)
    p_plus, p_minus, P_cont = mdp.entry_probs()
    r_c = step_cost * P_cont.sum(-1)
    rep = mdp.report
    out = {'env': mdp.name, 'gamma': gamma, 'step_cost': step_cost, 'k': [float(ks[0]), float(ks[-1]), n_scales]}
    curves = {}
    for name, (alpha, beta) in DESIGNS.items():
        a = torch.full((n_scales,), alpha, dtype=DTYPE, device=DEVICE)
        b = torch.full((n_scales,), beta, dtype=DTYPE, device=DEVICE)
        scale = torch.as_tensor(ks, dtype=DTYPE, device=DEVICE)
        Q, _ = solve_optimal(mdp, mdp.rewards(a, b, step_cost=step_cost, scale=scale), gamma)
        sets = optimal_sets(mdp, normalize_Q(Q, (1, 2)), atol=1e-12)
        r_ab = alpha * p_plus - beta * p_minus
        res = {}
        for rule in ['uniform', 'first', 'last']:
            pi = tie_break(sets, rule)
            V_ab = evaluate_discounted(mdp, pi, r_ab, gamma)
            V_c = evaluate_discounted(mdp, pi, r_c, gamma)
            res[rule] = {'V_ab non-decreasing in k': pairwise_violation(V_ab[:, rep], ks, True),
                         'V_c non-increasing in k': pairwise_violation(V_c[:, rep], ks, False)}
            if rule == 'uniform':
                surv, death, never = absorption_probabilities(mdp, pi)
                changed = (sets != sets[:1]).any(-1)[:, rep].double().mean(-1)
                curves[name] = {'V_ab': to_np(at_start(mdp, V_ab)), 'V_c': to_np(at_start(mdp, V_c)),
                                'P_survival': to_np(at_start(mdp, surv)), 'P_death': to_np(at_start(mdp, death)),
                                'P_never': to_np(at_start(mdp, never)), 'frac_states_changed': to_np(changed)}
        # control: without the intermediate reward the optimal sets do not depend on k
        Q0, _ = solve_optimal(mdp, mdp.rewards(a, b, step_cost=0.0, scale=scale), gamma)
        sets0 = optimal_sets(mdp, normalize_Q(Q0, (1, 2)), atol=1e-12)
        out[name] = {'checks (max violation, #violations > 1e-9, #comparisons)': res,
                     'distinct optimal sets across k': len({tuple(to_np(s).ravel()) for s in sets}),
                     'control c=0: distinct optimal sets across k': len({tuple(to_np(s).ravel()) for s in sets0}),
                     'k=1 vs k=100 at d0': {q: [float(v[0]), float(v[-1])] for q, v in curves[name].items()}}
    save_json(out, f'terminal_scale_{env}.json')
    np.savez(result_path(f'terminal_scale_{env}.npz', mimic=(env == 'mimic')), k=ks,
             **{f'{d}_{q}': v for d, c in curves.items() for q, v in c.items()})
    return ks, curves, out


def plot(env, ks, curves):
    """Terminal part, intermediate part and mortality at d0 against the scale k, one line per design."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(1, 3))
    panels = [('V_ab', r'$V_{\alpha,\beta}^{\pi_k}-V_{\alpha,\beta}^{\pi_1}$', '(a) Terminal part'),
              ('V_c', r'$V_c^{\pi_k}$', '(b) Intermediate part'),
              ('P_death', r'$\mathbb{P}(\mathcal{E}_-)$', '(c) Mortality')]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name in DESIGNS:
            y = curves[name][key]
            if key == 'V_ab':
                y = y - y[0]                                       # designs have different ranges
            a, b = DESIGNS[name]
            ax.plot(ks, y, color=C_POLICY[name], label=f'$(\\alpha,\\beta)=({a:g},{b:g})$')
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.set_xlabel('Terminal reward scale $k$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=GRID_ALPHA)
    axes[0].legend(loc='best')
    fig.tight_layout()
    savefig(fig, f'terminal_scale_{env}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['lifegate', 'mimic'], required=True)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--step-cost', type=float, default=None)
    parser.add_argument('--n-scales', type=int, default=81)
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    if args.plot_only:
        d = np.load(result_path(f'terminal_scale_{args.env}.npz', mimic=(args.env == 'mimic')))
        curves = {name: {k[len(name) + 1:]: d[k] for k in d.files if k.startswith(name + '_')} for name in DESIGNS}
        plot(args.env, d['k'], curves)
        return
    gamma = args.gamma or (0.9 if args.env == 'lifegate' else 0.99)
    step_cost = args.step_cost if args.step_cost is not None else (-0.04 if args.env == 'lifegate' else -0.01)
    ks, curves, out = run(args.env, gamma, step_cost, args.n_scales)
    for name in DESIGNS:
        print(name, out[name])
    plot(args.env, ks, curves)


if __name__ == '__main__':
    main()
