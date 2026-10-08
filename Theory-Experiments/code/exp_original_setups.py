"""
Theorem "Hidden incentives" and Theorem "Mass transfer", checked on the original setups (the experimental
setups of the earlier experiments in this repository), and the main theory figure.

Setups (the original scripts, with their parameters):
  DAG MDPs   DAG-Based-MDP-MRP/mdp, 2 actions: toy (5 states, p_terminal 0.3, seed 28, gamma 1; run_toy_mdp),
             toy with the trap state (break_absorption, gamma 1), large (200 states, p_terminal 0.3, seed 28,
             gamma 1 and 0.9, 0.8, 0.7; run_large_mdp, break_gamma), the other seeds 20, 18, 97, 42, 1018 of
             compare_policy_across_gamma (gamma 1, 0.9, 0.8, 0.7), and invert_classes (200 states,
             p_terminal 0.01, seed 1, original and flipped, gamma 0.9, 0.8, 0.7)
  LifeGate   LifeGate-Gridworld: break_assumptions (gamma in {1, 0.9} x slip in {0.2, 0}) and
             gridworld_scaling (slip 0.2, 20 values of gamma in [0.5, 0.99]), with step reward 0
  MIMIC-III  mimic-sepsis/pipeline: gamma in {0.99, 1}, clinician action mask
Settings with a nonzero step reward (break_rewards, step cost -0.04 in LifeGate) are outside the theorems,
which concern terminal rewards only; they are covered by Proposition "Terminal scale versus step cost"
(supplementary/exp_terminal_scale.py).

Checks, at every non-terminal state:
  Hidden incentives (i)   V_+ + (-V_-) + Z = 1
                    (ii)  V_ab = (a+b)(V_+ + lambda Z) - b, with V_ab evaluated directly under R_ab, for
                          random designs and random stochastic policies
                    (iii) V_+ + lambda Z = P(E_+) + lambda P(tau=inf) - (1-lambda) late survival
                          + lambda late death, with P(E_+), P(E_-), P(tau=inf) from absorption probabilities
                          and V_+, V_- from discounted evaluation (two independent routes), for the optimal
                          policies of 147 designs lambda
  Mass transfer     (i)-(iii) for all pairs of the 147 designs and three optimal tie-breaking rules,
                    including the strictness clause; (iv) at gamma = 1
At gamma = 1, planning is exact when every policy terminates, and otherwise Blackwell-optimal policies are
approximated at gamma = 1 - 1e-10 (violations are then counted above 100 x that error scale).

Figure: (a) terminal mass at the initial states, split into V_+, -V_- and the parts of the defect Z, for
pi_+^*, pi_+-^*, pi_-^* in four showcase settings; (b) position of the exchange rate between neighbouring
designs within the transfer bound [lambda, lambda'], for every setting.

Outputs: results/original_setups.{json,md} and plots/original_setups_row.pdf.

Usage:  python exp_original_setups.py [--skip-mimic] [--no-plot | --plot-only]
"""
import argparse
import os
import time

import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
import envs
import exp_mass_transfer as mt
from common import (C_DEATH, C_LATE_DEATH, C_LATE_SURV, C_MIXED, C_SURV, load_json, plt, save_json, savefig,
                    to_np)
from tabular import (DTYPE, DEVICE, at_start, evaluate_discounted, normalize_Q, optimal_sets, solve_optimal,
                     solve_optimal_undiscounted, summarize, sup_never, uniform_over)

RULES = mt.TIE_RULES
TIE_ATOL = mt.TIE_ATOL
N_RANDOM_POLICIES = 8                       # random designs and stochastic policies for Hidden incentives (ii)
OUTCOMES = ['V_plus', 'V_minus', 'Z', 'P_survival', 'P_death', 'P_never']
NOT_OPTIMAL_PAIRS = "pairs with pi' not optimal for lambda"     # key of the strictness counts


def plan(mdp, lam, gamma):
    """Optimal action sets for the designs (1 - lambda, lambda), the violation tolerance and the method."""
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)
    if gamma < 1:
        Q, _ = solve_optimal(mdp, mdp.rewards(1 - lam_t, lam_t), gamma)
        return optimal_sets(mdp, normalize_Q(Q, (1, 2)), atol=TIE_ATOL), 1e-10, 'exact'
    sets, info = solve_optimal_undiscounted(mdp, 1 - lam_t, lam_t)
    return sets, max(1e-10, 100 * info.get('approximation error scale', 0.0)), info['method']


def masses(mdp, pi, gamma):
    """Outcome quantities; at gamma = 1, V_+ = P(E_+), V_- = -P(E_-), Z = P(tau = inf), no late terms."""
    q = summarize(mdp, pi, gamma if gamma < 1 else 1 - 1e-12)
    if gamma >= 1:
        q['V_plus'], q['V_minus'], q['Z'] = q['P_survival'], -q['P_death'], q['P_never']
        q['late_survival'] = torch.zeros_like(q['Z'])
        q['late_death'] = torch.zeros_like(q['Z'])
    return q


def check_hidden(mdp, gamma, lam, sets, seed):
    """Theorem "Hidden incentives" (i)-(iii); also returns the outcome curves along lambda at the initial states."""
    rep = mdp.report
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)[:, None]
    q = masses(mdp, uniform_over(sets), gamma)
    out = {}
    out['(i) max |V+ + (-V-) + Z - 1|'] = float((q['V_plus'] - q['V_minus'] + q['Z'] - 1)[:, rep].abs().max())
    # (iii) with P(E+/-), P(inf) from the absorption solve and V_+ from the discounted solve
    if gamma < 1:
        late_s = q['P_survival'] - q['V_plus']                   # E[(1 - gamma^(tau-1)) 1_{E+}]
        late_d = q['P_death'] + q['V_minus']                     # E[(1 - gamma^(tau-1)) 1_{E-}]
        Z_parts = q['P_never'] + late_s + late_d
        out['(iii) max |Z - (P(inf) + late survival + late death)|'] = float((q['Z'] - Z_parts)[:, rep].abs().max())
        out['(iii) min of late survival, late death (>= 0)'] = float(min(late_s[:, rep].min(), late_d[:, rep].min()))
        rhs = q['P_survival'] + lam_t * q['P_never'] - (1 - lam_t) * late_s + lam_t * late_d
    else:
        rhs = q['P_survival'] + lam_t * q['P_never']
    out['(iii) max |V+ + lambda Z - decomposition|'] = float(((q['V_plus'] + lam_t * q['Z']) - rhs)[:, rep].abs().max())
    # (ii) random designs and random stochastic policies, V_ab evaluated directly under R_ab
    p_plus, p_minus, _ = mdp.entry_probs()
    g = torch.Generator(device='cpu').manual_seed(seed)
    err = 0.0
    for _ in range(N_RANDOM_POLICIES):
        a, b = torch.rand(2, generator=g, dtype=DTYPE).tolist()
        w = torch.rand(mdp.S, mdp.A, generator=g, dtype=DTYPE).to(DEVICE) * mdp.mask
        pi = (w / w.sum(-1, keepdim=True).clamp(min=1e-300))[None]
        qq = masses(mdp, pi, gamma)
        V_ab = evaluate_discounted(mdp, pi, a * p_plus - b * p_minus, gamma) if gamma < 1 else \
            (a * qq['P_survival'] - b * qq['P_death'])
        err = max(err, float((V_ab - ((a + b) * (qq['V_plus'] + b / (a + b) * qq['Z']) - b))[:, rep].abs().max()))
    out['(ii) max |V_ab - ((a+b)(V+ + lambda Z) - b)|'] = err
    mid = len(lam) // 2
    out['at initial states (survival-only, mixed, death-only)'] = {
        k: [round(float(at_start(mdp, q[k][[0, mid, -1]])[i]), 4) for i in range(3)] for k in OUTCOMES}
    # for the figure: the whole lambda path at the initial states (uniform tie-breaking)
    out['curves at initial states'] = {k: to_np(at_start(mdp, q[k])).tolist() for k in OUTCOMES}
    return out


def check_mass_transfer(mdp, gamma, lam, sets, tol):
    """Theorem "Mass transfer" (i)-(iii) for every tie-breaking rule, and (iv) at gamma = 1."""
    mt.TOL = tol                                    # tolerance of mt.check_pairs
    per_rule, worst = {}, {}
    qs = {rule: masses(mdp, mt.tie_break(sets, rule), gamma) for rule in RULES}
    keep = mt.optimal_designs(mdp, lam, qs, tol) if gamma >= 1 else {r: None for r in RULES}
    for rule in RULES:
        q = qs[rule]
        lam_r = lam
        if keep[rule] is not None:                  # only designs whose policy is optimal (mt.optimal_designs)
            k = keep[rule]
            q = {name: v[k] for name, v in q.items()}
            lam_r = lam[k.cpu().numpy()]
        c, _ = mt.check_pairs(lam_r, q, mdp.report)
        if keep[rule] is not None:
            c['designs whose policy is optimal at gamma = 1 (others excluded)'] = f'{int(keep[rule].sum())}/{len(lam)}'
        if gamma >= 1:
            # (iv): P_never' - P_never = (P+ - P+') + (P- - P-') with both brackets >= 0
            P = {k: q[k][:, mdp.report] for k in ('P_survival', 'P_death', 'P_never')}
            lt = torch.as_tensor(lam_r, device=DEVICE)
            pair = lt[:, None] < lt[None, :]
            lhs = (P['P_never'][None] - P['P_never'][:, None])[pair]
            dp = (P['P_survival'][:, None] - P['P_survival'][None])[pair]
            dm = (P['P_death'][:, None] - P['P_death'][None])[pair]
            c['(iv) identity max |error|'] = float((lhs - dp - dm).abs().max())
            c['(iv) min survival drop (>= -tol)'] = float(dp.min())
            c['(iv) min death drop (>= -tol)'] = float(dm.min())
        c['(iii) strictness'] = mt.strictness_by_value(lam_r, q, mdp.report, tol)
        per_rule[rule] = c
        for k, v in c.items():
            if isinstance(v, dict) and 'violations > tol' in v:
                worst[k] = max(worst.get(k, 0), v['violations > tol'])
    strict = per_rule['uniform']['(iii) strictness']
    return {'distinct optimal sets': len({tuple(s.flatten().tolist()) for s in sets.cpu()}), 'tolerance': tol,
            'violations > tol (max over tie rules)': worst, 'strictness (uniform)': strict, 'per rule': per_rule}


def run_setup(label, mdp, gamma, seed=0):
    """Plan, run both checks and summarize them in a pass/fail verdict."""
    t0 = time.time()
    _, lam = mt.lambda_grid()
    _, safe = sup_never(mdp)
    sets, tol, method = plan(mdp, lam, gamma)
    e = {'setup': label, 'env': mdp.name, 'gamma': gamma, 'non-terminal states': int(mdp.report.sum()),
         'some policy can avoid termination': bool(safe.any()), 'planning': method,
         'hidden incentives': check_hidden(mdp, gamma, lam, sets, seed),
         'mass transfer': check_mass_transfer(mdp, gamma, lam, sets, tol)}
    h, m = e['hidden incentives'], e['mass transfer']
    tol_h = max(1e-9, tol)
    hidden_ok = (all(v < tol_h for k, v in h.items() if k.startswith(('(i)', '(ii)', '(iii) max')))
                 and h.get('(iii) min of late survival, late death (>= 0)', 0.0) > -1e-9)
    transfer_ok = (all(n == 0 for n in m['violations > tol (max over tie rules)'].values())
                   and (gamma < 1 or all(r['(iv) identity max |error|'] < 1e-9
                                         and r['(iv) min survival drop (>= -tol)'] > -tol
                                         and r['(iv) min death drop (>= -tol)'] > -tol
                                         for r in m['per rule'].values())))
    strict = m['strictness (uniform)']
    e['verdict'] = {
        'hidden incentives': hidden_ok,
        'mass transfer': transfer_ok,
        'strict pairs': f"{strict['of which strict at some state']}/{strict[NOT_OPTIMAL_PAIRS]}",
    }
    e['seconds'] = round(time.time() - t0, 1)
    print(f"[{e['seconds']:6.1f}s] {label}, gamma={gamma}: {e['verdict']}", flush=True)
    return e


def setups(skip_mimic):
    """Yield (label, MDP constructor, discount factors) for every original setup."""
    yield 'DAG toy (run_toy_mdp)', (lambda: envs.dag(5)), [1.0]
    yield 'DAG large (run_large_mdp; break_gamma)', (lambda: envs.dag(200)), [1.0, 0.9, 0.8, 0.7]
    for seed in [20, 18, 97, 42, 1018]:                            # seed 28 is the row above
        yield f'DAG large seed {seed} (compare_policy_across_gamma)', (lambda s=seed: envs.dag(200, seed=s)), \
            [1.0, 0.9, 0.8, 0.7]
    for flip in [False, True]:
        yield (f"DAG p_terminal 0.01 seed 1{', flipped' if flip else ''} (invert_classes)",
               (lambda f=flip: envs.dag(200, p_terminal=0.01, seed=1, flip=f)), [0.9, 0.8, 0.7])
    yield 'DAG toy + trap state (break_absorption)', (lambda: envs.dag(5, trap=True)), [1.0]
    for slip in [0.2, 0.0]:
        yield f'LifeGate slip {slip} (break_assumptions)', (lambda s=slip: envs.lifegate(s)), [1.0, 0.9]
    yield ('LifeGate slip 0.2, step reward 0 (gridworld_scaling)', (lambda: envs.lifegate(0.2)),
           [round(float(g), 4) for g in np.linspace(0.5, 0.99, 20)])
    if not skip_mimic:
        yield 'MIMIC-III sepsis (pipeline)', envs.mimic, [0.99, 1.0]


def write_table(entries):
    """Markdown table of the verdicts: results/original_setups.md."""
    lines = ['| setup | gamma | can avoid termination | distinct optimal policies | Hidden incentives | '
             'Mass transfer | strict pairs |', '|---|---|---|---|---|---|---|']
    for e in entries:
        v = e['verdict']
        lines.append(f"| {e['setup']} | {e['gamma']} | {'yes' if e['some policy can avoid termination'] else 'no'} | "
                     f"{e['mass transfer']['distinct optimal sets']} | "
                     f"{'pass' if v['hidden incentives'] else '**FAIL**'} | "
                     f"{'pass' if v['mass transfer'] else '**FAIL**'} | {v['strict pairs']} |")
    table = '\n'.join(lines)
    with open(os.path.join(common.RESULT_DIR, 'original_setups.md'), 'w', encoding='utf-8') as f:
        f.write(table + '\n')
    print(table)


# --- main figure (manuscript size, full text width) ---------------------------------------------

ENV_MARKERS = {'DAG': 'o', 'LifeGate': 's', 'MIMIC': '^'}

# settings shown in panel (a): label, setup prefix, gamma
SHOWCASE = [('DAG MDP', 'DAG large seed 20', 0.7), ('LifeGate', 'LifeGate slip 0.2 (', 0.9),
            ('MIMIC-III', 'MIMIC', 0.99), ('MIMIC-III', 'MIMIC', 1.0)]

INK = '#222222'
GRID = '#e6e6e6'
C_NEVER_DARK = '#737373'                     # never terminates: darker than the reference gray
ENV_COLORS = {'DAG': '#4d4d4d', 'LifeGate': '#1b7837', 'MIMIC': C_MIXED}
ENV_NAMES = {'DAG': 'DAG', 'LifeGate': 'LifeGate', 'MIMIC': 'MIMIC-Sepsis'}      # names used in the paper

PRINT_STYLE = {
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'axes.edgecolor': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def env_of(entry):
    """Environment family of a result entry: 'DAG', 'LifeGate' or 'MIMIC'."""
    return next(k for k in ENV_MARKERS if entry['setup'].startswith(k))


def curves(e):
    """Outcome curves along lambda at the initial states, as arrays."""
    return {k: np.asarray(v) for k, v in e['hidden incentives']['curves at initial states'].items()}


def panel_mass(ax, selected, lam):
    """(a) one stacked bar per setting and design; light segments are the parts of the defect Z."""
    parts = [('V_plus', C_SURV), ('minus_V_minus', C_DEATH),
             ('late_survival', C_LATE_SURV), ('late_death', C_LATE_DEATH), ('P_never', C_NEVER_DARK)]
    designs = [0, len(lam) // 2, len(lam) - 1]                    # lambda = 0, 1/2, 1
    labels = [r'$\pi_+^*$', r'$\pi_\pm^*$', r'$\pi_{\minus}^*$']
    step = 4                                                      # 3 bars + 1 gap per setting
    positions, ticklabels = [], []
    for g, e in enumerate(selected):
        c = curves(e)
        m = {'V_plus': c['V_plus'], 'minus_V_minus': -c['V_minus'], 'P_never': c['P_never'],
             'late_survival': c['P_survival'] - c['V_plus'], 'late_death': c['P_death'] + c['V_minus']}
        for j, i in enumerate(designs):
            values = np.array([m[k][i] for k, _ in parts])
            if values.min() < -1e-9 or abs(values.sum() - 1) > 1e-9:
                raise ValueError(f"mass budget violated in {e['setup']}: {values}")
            values = values.clip(min=0)
            x = g * step + j
            ax.bar(x, values, bottom=np.concatenate([[0], np.cumsum(values)[:-1]]), width=0.74,
                   color=[col for _, col in parts], edgecolor='white', linewidth=0.4, zorder=3)
            Z = values[2:].sum()
            assert abs(Z - c['Z'][i]) < 1e-9
            ax.text(x, 1.025, f'{Z:.2f}', ha='center', va='bottom', fontsize=6.5)
            positions.append(x)
            ticklabels.append(labels[j])
        ax.text(g * step + 1, -0.15, f"{ENV_NAMES[env_of(e)]}\n$\\gamma={e['gamma']:g}$",
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.5, linespacing=1.3)
    ax.text(-0.62, 1.025, '$Z$', ha='right', va='bottom', fontsize=6.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticklabels)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_xlim(-0.62, positions[-1] + 0.62)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.5, 1])
    ax.spines['left'].set_bounds(0, 1)
    ax.set_ylabel('Mass at initial states')
    ax.grid(axis='y', color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(color=C_SURV, label=r'Survival $V_+$'), Patch(color=C_DEATH, label=r'Death $-V_{\minus}$'),
               Patch(color=C_LATE_SURV, label='Late survival'), Patch(color=C_LATE_DEATH, label='Late death'),
               Patch(color=C_NEVER_DARK, label='Never terminates')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=False,
              handlelength=0.9, handleheight=0.9, handletextpad=0.35, columnspacing=0.9, borderaxespad=0)
    ax.set_title('(a) Terminal mass and defect $Z$', pad=24)


def panel_transfer(ax, entries, lam):
    """
    (b) where the exchange rate rho = (V_+ - V_+') / (Z' - Z) between neighbouring lambda falls inside the
    interval [lambda, lambda'] of Theorem "Mass transfer"(i): (rho - lambda) / (lambda' - lambda) in [0, 1].
    The interval is narrow, so plotted as rho itself every point sits on the diagonal; the normalized
    position spreads them out.  Horizontal axis: logit scale, matching the grid of lambda.
    Returns the number of intervals per environment and the largest excursion of rho outside the interval.
    """
    mid = 0.5 * (lam[:-1] + lam[1:])
    width = lam[1:] - lam[:-1]
    ax.axhspan(0, 1, color='#ececec', linewidth=0, zorder=0)
    for y in (0, 1):
        ax.axhline(y, color='#8c8c8c', linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    counts = {k: 0 for k in ENV_MARKERS}
    worst = 0.0
    for e in entries:
        c = curves(e)
        dz, dv = np.diff(c['Z']), -np.diff(c['V_plus'])
        ok = dz > 1e-9                                            # rho is defined where the defect moves
        rho = dv[ok] / dz[ok]
        env = env_of(e)
        counts[env] += int(ok.sum())
        worst = max(worst, float(np.max(np.r_[lam[:-1][ok] - rho, rho - lam[1:][ok], 0.0])))
        ax.scatter(mid[ok], (rho - lam[:-1][ok]) / width[ok], marker=ENV_MARKERS[env], s=8, facecolors='none',
                   edgecolors=ENV_COLORS[env], linewidths=0.55, alpha=0.85, zorder=3)
    ax.set_xscale('logit')
    ticks = [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['0.1', '0.5', '0.9', '', '0.999', ''])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(0.02, 0.99998)
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([r'$\lambda$', '', r'$\lambda^{\prime}$'])
    ax.set_xlabel(r'$(\lambda+\lambda^{\prime})/2$', labelpad=1)
    ax.set_ylabel(r'Position of $\rho$ in $[\lambda,\lambda^{\prime}]$', labelpad=2)
    ax.grid(axis='x', color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=ENV_MARKERS[k], ls='none', mfc='none', mec=ENV_COLORS[k], mew=0.7, ms=3.5,
                      label=ENV_NAMES[k]) for k in ENV_MARKERS]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.45, 1.07), ncol=3, frameon=False,
              handlelength=0.8, handletextpad=0.2, columnspacing=0.6, borderaxespad=0)
    ax.set_title('(b) Transfer bound', pad=24)
    return counts, worst


def plot(entries):
    """Main figure: panel (a) for the SHOWCASE settings, panel (b) for every setting."""
    _, lam = mt.lambda_grid()
    selected = [next(e for e in entries if e['setup'].startswith(prefix) and e['gamma'] == gamma)
                for _, prefix, gamma in SHOWCASE]
    with plt.rc_context(PRINT_STYLE):
        fig = plt.figure(figsize=(6.75, 2.25))
        ax_a = fig.add_axes([0.075, 0.25, 0.575, 0.55])
        ax_b = fig.add_axes([0.745, 0.25, 0.235, 0.55])
        panel_mass(ax_a, selected, lam)
        counts, worst = panel_transfer(ax_b, entries, lam)
        print(f'{len(entries)} settings, {sum(counts.values())} intervals {counts}; '
              f'largest excursion outside [lambda, lambda\']: {worst:.2g}')
        # exact print size (no tight bounding box): the axes positions above define the layout
        savefig(fig, 'original_setups_row', main=True, bbox_inches=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-mimic', action='store_true', help='leave out the MIMIC-III setup')
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from results/original_setups.json')
    parser.add_argument('--no-plot', action='store_true', help='run the checks without drawing the figure')
    args = parser.parse_args()
    if args.plot_only:
        entries = load_json('original_setups.json')
        write_table(entries)
        plot(entries)
        return
    entries = []
    for label, make, gammas in setups(args.skip_mimic):
        mdp = make()
        for g in gammas:
            entries.append(run_setup(label, mdp, g))
    save_json(entries, 'original_setups.json')
    write_table(entries)
    if not args.no_plot:
        plot(entries)


if __name__ == '__main__':
    main()
