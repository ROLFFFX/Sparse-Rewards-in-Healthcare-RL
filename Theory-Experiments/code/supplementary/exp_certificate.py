"""
Corollary "Robustness certificate".

For pi_0 in Pi_+^* and pi_lambda in Pi_lambda^*, at every non-terminal state:
    0 <= V_+^{pi_0} - V_+^{pi_lambda} <= lambda (Z^{pi_lambda} - Z^{pi_0}) <= lambda sup_mu Z^mu      (discounted)
    P^{pi_0}(E_+) - P^{pi_lambda}(E_+) <= lambda eps + (1 + lambda)(1 - gamma) T                       (undiscounted)
with eps = sup_mu P(tau = inf) and T = sup_mu E[(tau - 1) 1{tau < inf}].  sup_mu Z^mu, eps and T are
computed exactly as auxiliary control problems on the same model (tabular.sup_defect / sup_never /
sup_time); T is only finite when no policy can avoid termination (eps = 0).
The bounds are checked for a grid of lambda, all states and three optimal tie-breaking rules.  A second
part checks the undiscounted bound across discount factors on random bounded-horizon DAGs.

Outputs: results/supplementary/certificate_<env>.json, the curves at the initial states in
certificate_<env>.npz (results/supplementary for LifeGate, mimic-sepsis/data/theory_results for MIMIC),
and the figure plots/supplementary/propositions/certificate_<env>.pdf.

Usage:  python exp_certificate.py --env lifegate|mimic [--gamma G] [--plot-only]
        (default gamma: 0.9 for LifeGate, 0.99 for MIMIC)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # library in code/

import numpy as np
import torch

import common
import envs
from batched import design_rewards, optimal_sets as batched_optimal_sets, value_iteration
from common import (apply_style, C_NEVER, C_SURV, C_TERM, figsize, GRID_ALPHA, load_json, plt, result_path,
                    save_json, savefig, to_np)
from exp_equal_length_dag import make_dags
from exp_mass_transfer import lambda_axis, lambda_grid, style_lambda_axis, tie_break
from tabular import (DTYPE, DEVICE, at_start, normalize_Q, optimal_sets, solve_optimal, summarize, sup_defect,
                     sup_never, sup_time)

common.set_group('propositions')      # figures and results go to the supplementary folders


def run(env, gamma):
    """Check both certificates on LifeGate or MIMIC; save and return the curves at the initial states."""
    mdp = envs.lifegate(0.2) if env == 'lifegate' else envs.mimic()
    rep = mdp.report
    x, lam = lambda_grid()
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)
    Q, _ = solve_optimal(mdp, mdp.rewards(1 - lam_t, lam_t), gamma)
    sets = optimal_sets(mdp, normalize_Q(Q, (1, 2)), atol=1e-10)

    supZ = sup_defect(mdp, gamma)
    eps, safe = sup_never(mdp)
    T = sup_time(mdp)
    summary = {'env': mdp.name, 'gamma': gamma,
               'sup_mu Z (max over states, at d0)': [float(supZ[rep].max()), float(at_start(mdp, supZ[None]))],
               'eps = sup_mu P(tau=inf) (max over states, at d0)':
                   [float(eps[rep].max()), float(at_start(mdp, eps[None]))],
               'states that can avoid termination surely': int(safe.sum()),
               'T = sup_mu E[(tau-1) 1{tau<inf}] (max, at d0)':
                   None if T is None else [float(T[rep].max()), float(at_start(mdp, T[None]))],
               'checks': {}}
    curves = {}
    for rule in ['uniform', 'first', 'last']:
        q = summarize(mdp, tie_break(sets, rule), gamma)
        gap = q['V_plus'][:1] - q['V_plus']                     # pi_0 is the lambda = 0 entry
        thm_bound = lam_t[:, None] * (q['Z'] - q['Z'][:1])
        cert = lam_t[:, None] * supZ[None]
        gap_u = q['P_survival'][:1] - q['P_survival']
        checks = {
            'gap >= 0 (min)': float(gap[:, rep].min()),
            'gap <= lambda (Z_lambda - Z_0): max violation': float((gap - thm_bound)[:, rep].clamp(min=0).max()),
            'gap <= lambda sup Z: max violation': float((gap - cert)[:, rep].clamp(min=0).max()),
            'max over lambda, states of gap / (lambda sup Z)':
                float((gap[1:] / cert[1:].clamp(min=1e-12))[:, rep].max()),
        }
        if T is not None:
            cert_u = lam_t[:, None] * eps[None] + (1 + lam_t[:, None]) * (1 - gamma) * T[None]
            checks['undiscounted gap <= lambda eps + (1+lambda)(1-gamma)T: max violation'] = \
                float((gap_u - cert_u)[:, rep].clamp(min=0).max())
        summary['checks'][rule] = checks
        if rule == 'uniform':
            curves = {'gap': gap, 'thm_bound': thm_bound, 'cert': cert, 'gap_undiscounted': gap_u}
            if T is not None:
                curves['cert_undiscounted'] = cert_u
            curves = {k: to_np(at_start(mdp, v)) for k, v in curves.items()}
    save_json(summary, f'certificate_{env}.json')
    np.savez(result_path(f'certificate_{env}.npz', mimic=(env == 'mimic')), x=x, lam=lam, **curves)
    return x, curves, summary


def dag_sweep(gammas, n=1000, L=6, W=4, A=3, seed=0):
    """
    Bounded-horizon case: variable-length random DAGs (exp_equal_length_dag.make_dags), where every path
    terminates within L steps, so eps = 0 and T <= L - 1.  For each gamma: the largest undiscounted
    survival loss max_lambda [P^{pi_0}(E_+) - P^{pi_lambda}(E_+)] at the first-layer states, against the
    certificate lambda eps + (1 + lambda)(1 - gamma) T with the exact T of each instance.
    """
    P, pp, pm, mask = make_dags(n, L, W, A, True, seed)
    _, lam = lambda_grid(n=25, span=6.0)
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)[None, :, None]
    rows = []
    # T: maximal expected number of steps between non-terminal states (exact, depth L)
    T = torch.zeros(n, P.shape[1], dtype=DTYPE, device=DEVICE)
    for _ in range(L + 1):
        T = torch.einsum('nsat,nt->nsa', P, 1 + T).max(-1).values
    first = slice(0, W)
    for gamma in gammas:
        Q, _ = value_iteration(P, design_rewards(pp, pm, 1 - lam, lam), gamma, mask, iters=L + 1)
        sets = batched_optimal_sets(normalize_Q(Q, (2, 3)), mask, atol=1e-10)
        pi = sets.to(DTYPE) / sets.sum(-1, keepdim=True).clamp(min=1)
        Ppi = torch.einsum('ndsa,nsat->ndst', pi, P)
        surv = torch.zeros(pi.shape[:3], dtype=DTYPE, device=DEVICE)
        b = torch.einsum('ndsa,nsa->nds', pi, pp)
        for _ in range(L + 1):                                       # undiscounted P(E_+), exact on a DAG
            surv = b + torch.einsum('ndst,ndt->nds', Ppi, surv)
        gap = (surv[:, :1] - surv)[:, :, first]                      # (n, D, W)
        cert = (1 + lam_t) * (1 - gamma) * T[:, None, first]
        rows.append({'gamma': gamma, 'max undiscounted survival loss': float(gap.max()),
                     'mean over instances of max loss': float(gap.amax(dim=(1, 2)).mean()),
                     'max certificate': float(cert.max()),
                     'max violation': float((gap - cert).clamp(min=0).max()), 'max T': float(T[:, first].max())})
    return rows


def sci(v):
    """2.5 x 10^4 style mathtext."""
    e = int(np.floor(np.log10(abs(v))))
    return f'{v / 10 ** e:.1f}\\times10^{{{e}}}'


def plot(env, x, c, summary, dag):
    """(a) discounted bound and (b) undiscounted bound along lambda at d0; (c) the random-DAG sweep."""
    apply_style()
    xs, ticks, labels = lambda_axis(x)
    fig, axes = plt.subplots(1, 3, figsize=figsize(1, 3))
    ax = axes[0]
    ax.plot(xs, c['cert'], color=C_NEVER, linestyle='--', label=r'$\lambda\,\sup_\mu Z^\mu$')
    ax.plot(xs, c['thm_bound'], color=C_TERM, linestyle='-.', label=r"$\lambda(Z^{\pi_\lambda}-Z^{\pi_0})$")
    ax.plot(xs, c['gap'], color=C_SURV, label=r'$V_+^{\pi_0}-V_+^{\pi_\lambda}$')
    ax.set_ylabel('Discounted survival lost')
    ax.set_title('(a) Discounted bound')
    ax.legend(loc='upper left')
    ax = axes[1]
    ax.plot(xs, c['gap_undiscounted'], color=C_SURV,
            label=r'$\mathbb{P}^{\pi_0}(\mathcal{E}_+)-\mathbb{P}^{\pi_\lambda}(\mathcal{E}_+)$')
    T = summary['T = sup_mu E[(tau-1) 1{tau<inf}] (max, at d0)']
    # draw the undiscounted bound only where it is informative; otherwise state why it is absent
    if 'cert_undiscounted' in c and c['cert_undiscounted'].max() <= 2:
        ax.plot(xs, c['cert_undiscounted'], color=C_NEVER, linestyle='--',
                label=r'$\lambda\varepsilon+(1+\lambda)(1-\gamma)T$')
        note = ''
    elif 'cert_undiscounted' in c:
        note = f'bound $\\geq{sci(c["cert_undiscounted"].min())}$ (vacuous):\n$T={sci(T[1])}$ steps'
    else:
        note = 'no bound: some policy can\navoid termination ($T=\\infty$)'
    ax.text(0.04, 0.6, note, transform=ax.transAxes, ha='left', fontsize=14)
    ax.axhline(0, color=C_TERM, linewidth=1)
    ax.set_ylabel('Undiscounted survival lost')
    ax.set_title('(b) Undiscounted bound')
    ax.legend(loc='upper left')
    for ax in axes[:2]:
        style_lambda_axis(ax, xs, ticks, labels)

    ax = axes[2]
    g = [r['gamma'] for r in dag]
    ax.plot(g, [r['max certificate'] for r in dag], color=C_NEVER, linestyle='--', marker='s',
            label=r'bound $(1+\lambda)(1-\gamma)T$')
    ax.plot(g, [r['max undiscounted survival loss'] for r in dag], color=C_SURV, marker='o', label='loss, max')
    ax.plot(g, [r['mean over instances of max loss'] for r in dag], color=C_SURV, linestyle=':', marker='o',
            label='loss, mean')
    ax.set_xscale('function', functions=(lambda v: -np.log10(1 - np.minimum(v, 1 - 1e-9)),
                                         lambda u: 1 - 10 ** (-u)))
    ax.set_yscale('log')
    ax.set_xticks([0.5, 0.9, 0.99, 0.999])
    ax.set_xticklabels(['0.5', '0.9', '0.99', '0.999'])
    ax.set_xlabel(r'Discount $\gamma$')
    ax.set_ylabel('Undiscounted survival lost')
    ax.set_title('(c) Random DAGs')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc='lower left')
    fig.tight_layout()
    savefig(fig, f'certificate_{env}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['lifegate', 'mimic'], required=True)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    if args.plot_only:
        summary = load_json(f'certificate_{args.env}.json')
        d = np.load(result_path(f'certificate_{args.env}.npz', mimic=(args.env == 'mimic')))
        curves = {k: d[k] for k in d.files if k not in ('x', 'lam')}
        plot(args.env, d['x'], curves, summary, summary['bounded-horizon DAGs'])
        return
    gamma = args.gamma or (0.9 if args.env == 'lifegate' else 0.99)
    x, curves, summary = run(args.env, gamma)
    dag = dag_sweep([0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
    summary['bounded-horizon DAGs'] = dag
    save_json(summary, f'certificate_{args.env}.json')
    print({k: v for k, v in summary.items() if k != 'checks'})
    for rule, checks in summary['checks'].items():
        print(rule, checks)
    plot(args.env, x, curves, summary, dag)


if __name__ == '__main__':
    main()
