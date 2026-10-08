"""
Proposition "Value difference and terminal defect" and Theorem "Hidden incentives", checked on LifeGate and
MIMIC-III.

(a) Per state and policy: V_+ - V_- (value route, two exact policy evaluations) against
    sum_k gamma^(k-1) P_s(tau = k) (hitting-time route, forward propagation of P(tau = k)); equal by
    the proposition.  Policies: pi_+^*, pi_+-^*, pi_-^* and a reference policy (uniform random on
    LifeGate, the clinician policy pi_b on MIMIC-III).
(b) The conservation law of Theorem "Hidden incentives"(i) with the defect split as in the proposition
    and (iii):  1 = V_+ + (-V_-) + E[(1-gamma^(tau-1)) 1_{E+}] + E[(1-gamma^(tau-1)) 1_{E-}] + P(tau=inf),
    where the first and third parts add up to P(E_+) and the second and fourth to P(E_-).
(c) The objective V_+ + lambda Z of each design, decomposed as in (iii) into
    P(E_+), + lambda P(tau = inf), - (1 - lambda) late survival, + lambda late death,
    evaluated at the design's own optimal policy.

Outputs: results/terminal_defect_<env>.{json,npz} (the MIMIC npz goes to mimic-sepsis/data/theory_results)
and the figure plots/supplementary/theorems/terminal_defect_<env>.pdf.

Usage:  python exp_terminal_defect.py --env lifegate|mimic [--gamma 0.9] [--K 20000] [--plot-only]
"""
import argparse

import torch

import envs
from common import (apply_style, C_DEATH, C_LATE_DEATH, C_LATE_SURV, C_NEVER, C_POLICY, C_SURV, C_TERM, figsize,
                    GRID_ALPHA, L_POLICY, load_json, np, plt, result_path, save_json, savefig, to_np)
from exp_mass_transfer import TIE_ATOL
from tabular import (DTYPE, DEVICE, at_start, hitting_time_series, normalize_Q, optimal_sets, solve_optimal,
                     summarize, uniform_over)

POLICIES = ['plus', 'mixed', 'minus', 'ref']
MARKERS = 'os^D'                                    # one per policy, in the order of POLICIES
LAMBDA = {'plus': 0.0, 'mixed': 0.5, 'minus': 1.0}
# V_+ + (-V_-) + late survival + late death + P(tau = inf) = 1
PARTS = ['V_plus', 'minus_V_minus', 'late_survival', 'late_death', 'P_never']


def policies(mdp, gamma, env):
    """pi_+^*, pi_+-^*, pi_-^* (uniform over optimal actions) and the reference policy, stacked (4, S, A)."""
    lam = torch.tensor([LAMBDA[p] for p in ['plus', 'mixed', 'minus']], dtype=DTYPE, device=DEVICE)
    Q, _ = solve_optimal(mdp, mdp.rewards(1 - lam, lam), gamma)
    pis = uniform_over(optimal_sets(mdp, normalize_Q(Q, (1, 2)), atol=TIE_ATOL))
    if env == 'mimic':
        ref, ref_name = torch.as_tensor(mdp.meta['pi_b'], dtype=DTYPE, device=DEVICE), r'clinician $\pi_b$'
    else:
        ref, ref_name = uniform_over(mdp.mask[None])[0], 'uniform random'
    return torch.cat([pis, ref[None]]), ref_name


def run(env, gamma, K):
    """Evaluate the four policies by both routes, check the identities and save the per-state arrays."""
    mdp = envs.lifegate(0.2) if env == 'lifegate' else envs.mimic()
    pi, ref_name = policies(mdp, gamma, env)
    q = summarize(mdp, pi, gamma)
    # the discounted series converges geometrically (tail <= gamma^K); P(tau = k) itself may have a very
    # long tail (policies that postpone death), which is why P(tau < inf) comes from the absorption solve
    series, p_within_K, _ = hitting_time_series(mdp, pi, gamma, K)
    rep = mdp.report
    q['minus_V_minus'] = -q['V_minus']
    total = sum(q[p] for p in PARTS)
    summary = {'env': mdp.name, 'gamma': gamma, 'K': K, 'reference policy': ref_name, 'n_states': int(rep.sum())}
    for i, p in enumerate(POLICIES):
        summary[p] = {
            'max |(V+ - V-) - sum_k gamma^(k-1) P(tau=k)|': float((q['disc_term'] - series)[i, rep].abs().max()),
            'max |sum of the five parts - 1|': float((total - 1)[i, rep].abs().max()),
            'min of the five parts (>= 0)': float(min(q[k][i, rep].min() for k in PARTS)),
            'max P(tau > K, tau < inf)': float((q['P_survival'] + q['P_death'] - p_within_K)[i, rep].max()),
            'at d0': {k: float(at_start(mdp, q[k])[i]) for k in PARTS + ['Z', 'P_survival', 'P_death']},
            'states with P(tau=inf) > 1e-6': int((q['P_never'][i, rep] > 1e-6).sum()),
        }
    save_json(summary, f'terminal_defect_{env}.json')
    arrays = {k: to_np(q[k][:, rep]) for k in PARTS + ['disc_term']}
    arrays['series'] = to_np(series[:, rep])
    arrays['d0'] = to_np(mdp.d0[rep])
    np.savez(result_path(f'terminal_defect_{env}.npz', mimic=(env == 'mimic')), **arrays)
    return arrays, summary, ref_name


def plot(env, a, summary, ref_name):
    """Panels (a)-(c) of the module docstring; masses are averaged over the initial distribution."""
    apply_style()
    labels = [L_POLICY['plus'], L_POLICY['mixed'], L_POLICY['minus'], ref_name]
    d0 = a['d0'] / a['d0'].sum()
    m = {k: (a[k] * d0).sum(1) for k in PARTS}
    fig, axes = plt.subplots(1, 3, figsize=figsize(1, 3, h=6.0), gridspec_kw={'width_ratios': [1, 1.1, 1.25]})

    ax = axes[0]
    for i, p in enumerate(POLICIES):
        ax.scatter(a['series'][i], a['disc_term'][i], s=40, alpha=0.6, color=C_POLICY[p], label=labels[i],
                   edgecolor='none', marker=MARKERS[i])
    ax.plot([0, 1], [0, 1], color=C_NEVER, linestyle='--', zorder=0)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect='equal')
    ax.set_xlabel(r'$\sum_k \gamma^{k-1}\,\mathbb{P}_s(\tau=k)$')
    ax.set_ylabel(r'$V_+^\pi(s)-V_-^\pi(s)$')
    worst = max(summary[p]['max |(V+ - V-) - sum_k gamma^(k-1) P(tau=k)|'] for p in POLICIES)
    ax.set_title('(a) Value difference')
    ax.text(0.97, 0.04, f'max error {worst:.0e}', transform=ax.transAxes, ha='right')
    ax.legend(loc='upper left', markerscale=1.5, handletextpad=0.2)

    ax = axes[1]
    pos = np.arange(len(POLICIES))
    segments = [('V_plus', r'$V_+$', C_SURV), ('late_survival', 'late survival', C_LATE_SURV),
                ('minus_V_minus', r'$-V_-$', C_DEATH), ('late_death', 'late death', C_LATE_DEATH),
                ('P_never', r'$\mathbb{P}(\tau=\infty)$', C_NEVER)]
    bottom = np.zeros(len(POLICIES))
    for key, label, color in segments:
        ax.bar(pos, m[key], bottom=bottom, color=color, width=0.62, edgecolor='white', linewidth=1.5, label=label)
        bottom += m[key]
    for i in range(len(POLICIES)):
        Z = m['late_survival'][i] + m['late_death'][i] + m['P_never'][i]
        ax.text(pos[i], 1.02, f'$Z$={Z:.2f}', ha='center', fontsize=14)
    ax.set_xticks(pos)
    ax.set_xticklabels([lab.replace(' ', '\n', 1) if not lab.startswith('$') else lab for lab in labels])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Mass at initial states')
    ax.set_title('(b) Parts of the defect')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.0, handlelength=1.2)

    ax = axes[2]
    designs = ['plus', 'mixed', 'minus']
    # (label, term as a function of the design index i and its lambda, color)
    terms = [(r'$\mathbb{P}(\mathcal{E}_+)$', lambda i, lam: m['V_plus'][i] + m['late_survival'][i], C_SURV),
             (r'$+\lambda\,\mathbb{P}(\tau=\infty)$', lambda i, lam: lam * m['P_never'][i], C_NEVER),
             (r'$+\lambda\,$late death', lambda i, lam: lam * m['late_death'][i], C_LATE_DEATH),
             (r'$-(1-\lambda)\,$late survival', lambda i, lam: -(1 - lam) * m['late_survival'][i], C_LATE_SURV)]
    y = np.arange(len(designs))[::-1]
    for j, p in enumerate(designs):
        lam = LAMBDA[p]
        # positive terms stack to the right of 0, negative ones to the left
        left_pos, left_neg = 0.0, 0.0
        for label, f, color in terms:
            v = f(j, lam)
            left = left_pos if v >= 0 else left_neg
            ax.barh(y[j], v, left=left, color=color, height=0.55, edgecolor='white', linewidth=1.5,
                    label=label if j == 0 else None)
            if v >= 0:
                left_pos += v
            else:
                left_neg += v
        obj = sum(f(j, lam) for _, f, _ in terms)
        ax.plot([obj, obj], [y[j] - 0.33, y[j] + 0.33], color=C_TERM)
        ax.text(max(left_pos, obj) + 0.02, y[j], f'{obj:.2f}', va='center', fontsize=14)
    ax.axvline(0, color=C_TERM, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{L_POLICY[p]}\n($\\lambda={LAMBDA[p]:g}$)" for p in designs])
    ax.set_xlabel(r'$V_+^\pi+\lambda Z^\pi$ (black bar: total)')
    ax.set_title('(c) Hidden incentives')
    ax.set_xlim(min(-0.4, ax.get_xlim()[0]), 1.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handlelength=1.2)
    for ax in axes:
        ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    savefig(fig, f'terminal_defect_{env}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['lifegate', 'mimic'], required=True)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--K', type=int, default=20000, help='hitting-time steps propagated')
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    gamma = args.gamma or (0.9 if args.env == 'lifegate' else 0.99)
    if args.plot_only:
        summary = load_json(f'terminal_defect_{args.env}.json')
        arrays = dict(np.load(result_path(f'terminal_defect_{args.env}.npz', mimic=(args.env == 'mimic'))))
        plot(args.env, arrays, summary, summary['reference policy'])
        return
    arrays, summary, ref_name = run(args.env, gamma, args.K)
    for p in POLICIES:
        print(p, {k: v for k, v in summary[p].items() if k != 'at d0'})
        print('   at d0', {k: round(v, 4) for k, v in summary[p]['at d0'].items()})
    plot(args.env, arrays, summary, ref_name)


if __name__ == '__main__':
    main()
