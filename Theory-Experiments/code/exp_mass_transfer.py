"""
Theorem "Mass transfer" (and Theorem "Hidden incentives"(i)), checked on LifeGate and MIMIC-III.

Every design (alpha, beta) is the objective V_+ + lambda Z with lambda = beta / (alpha + beta), where
Z = 1 - V_+ + V_- is the terminal defect.  For a grid of lambda in [0, 1] (0 = survival-only,
1/2 = mixed, 1 = death-only) we take optimal policies pi_lambda and check, for every pair
lambda < lambda', every non-terminal state and three optimal tie-breaking rules:
  (i)   lambda (Z' - Z) <= V_+ - V_+' <= lambda' (Z' - Z)
  (ii)  (1 - lambda')(Z' - Z) <= V_-' - V_- <= (1 - lambda)(Z' - Z)
  (iii) Z' >= Z, V_+' <= V_+, V_-' >= V_-; strict somewhere when 0 < lambda < lambda' < 1 and pi' is
        not optimal for lambda
  (iv)  at gamma = 1: P(tau = inf) rises by exactly the drop of P(E_+) plus the drop of P(E_-), and both
        drops are >= 0.
The figure shows the conservation law V_+ + (-V_-) + Z = 1 along lambda, the exchange rate
(V_+ - V_+') / (Z' - Z) between consecutive grid points against the band [lambda, lambda'], and the
undiscounted outcome probabilities, at the initial states.

With --robustness, the transfer bound is checked beyond the exactly optimal policies of the main run:
exact rational examples where it is attained, an exhaustive policy enumeration on random small MDPs with
approximately optimal policies (error budgets eps, eps'), and every wide design pair of the original setups
(read from results/original_setups.json, written by exp_original_setups.py).

Outputs: results/mass_transfer_<env>[_gamma1].{json,npz} (the MIMIC npz goes to
mimic-sepsis/data/theory_results), results/mass_transfer_robustness.json, and the figures
plots/supplementary/theorems/mass_transfer_*.pdf.

Usage:  python exp_mass_transfer.py --env lifegate|mimic [--gamma 0.9 | --gamma 1] [--plot-only]
        python exp_mass_transfer.py --robustness [--plot-only]
"""
import argparse
import itertools
from fractions import Fraction

import torch

import common
import envs
from common import (apply_style, C_DEATH, C_NEVER, C_SURV, C_TERM, figsize, GRID_ALPHA, load_json, np, plt,
                    result_path, save_json, savefig, to_np)
from tabular import (DTYPE, DEVICE, at_start, normalize_Q, optimal_sets, solve_optimal,
                     solve_optimal_undiscounted, summarize, uniform_over)

# Violation tolerance used by check_pairs and strictness; set per run by run() (and by exp_original_setups.py).
TOL = 1e-10
# Optimal-action ties on the normalized Q of exact planning: true ties differ by round-off (< 1e-15) and true
# action gaps are > 1e-11 in every setup (LifeGate down to gamma = 0.5, MIMIC, DAGs), so 1e-13 separates them.
TIE_ATOL = 1e-13
TIE_RULES = ['uniform', 'first', 'last']

# Robustness audit: random small MDPs with three decision states, two actions and two terminal outcomes.
ROBUSTNESS_SEED = 20260929
ROBUSTNESS_GAMMAS = (.5, .9, .99, 1.)
MDPS_PER_GAMMA = 40
RANDOM_POLICIES_PER_MDP = 8


def robustness_audit():
    """Exact rational sharpness, exhaustive enumeration on small random MDPs, and wide design pairs."""
    # (1) Sharpness in rational arithmetic: one decision between terminating now and a nonterminating loop
    # (or, below, a guaranteed but delayed absorption).  The switch threshold r is the exchange rate, and
    # the error budgets eps, eps' of the two designs lo < hi make every inequality of the bound an equality.
    sharp = []
    lo, hi, p = Fraction(1, 5), Fraction(4, 5), Fraction(1, 2)
    for r in (lo, Fraction(1, 2), hi):
        for d in (Fraction(1, 100), Fraction(1, 10), Fraction(1, 2)):
            vp, z = [p, p - r * d], [Fraction(0), d]
            assert p - r * d >= 0 and 1 - p - (1 - r) * d >= 0
            assert vp[0] + lo * z[0] >= vp[1] + lo * z[1]
            assert vp[1] + hi * z[1] >= vp[0] + hi * z[0]
            assert (vp[0] - vp[1]) / (z[1] - z[0]) == r
            eps, eps2 = (r - lo) * d, (hi - r) * d
            assert -r * d == lo * (-d) - eps == hi * (-d) + eps2
            assert -(1 - r) * d == (1 - hi) * (-d) - eps2 == (1 - lo) * (-d) + eps
            assert -d == -(eps + eps2) / (hi - lo)
            sharp.append({'threshold': str(r), 'defect': str(d), 'epsilon': str(eps),
                          'epsilon_prime': str(eps2), 'all_equalities_exact': True})
    absorbing_sharp = []
    for gamma in (Fraction(1, 2), Fraction(9, 10), Fraction(99, 100)):
        for r in (lo, Fraction(1, 2), hi):
            d = 1 - gamma
            vp, z = [r, gamma * r], [Fraction(0), d]
            assert (vp[0] - vp[1]) / (z[1] - z[0]) == r
            eps, eps2 = (r - lo) * d, (hi - r) * d
            assert -r * d == lo * (-d) - eps == hi * (-d) + eps2
            assert -(1 - r) * d == (1 - hi) * (-d) - eps2 == (1 - lo) * (-d) + eps
            assert -d == -(eps + eps2) / (hi - lo)
            absorbing_sharp.append({'gamma': str(gamma), 'threshold': str(r),
                                    'all_equalities_exact': True, 'termination_within_steps': 2})

    # (2) Approximate optimality: for every policy, eps = optimum - objective (the optimum is exact, over all
    # deterministic policies); the bound with eps, eps' must hold for every policy pair, design pair and state.
    rng = np.random.default_rng(ROBUSTNESS_SEED)
    weights = np.array([0, .1, .25, .5, .75, .9, 1.])
    det = np.eye(2)[np.array(list(itertools.product(range(2), repeat=3)))]   # all 8 deterministic policies
    worst = np.zeros(5)
    comparisons = 0
    reversals = 0
    samples = []
    for gamma in ROBUSTNESS_GAMMAS:
        for _ in range(MDPS_PER_GAMMA):
            # A positive exit probability for every action keeps the gamma = 1 systems nonsingular.
            P = rng.dirichlet(np.ones(5), size=(3, 2))
            policies = np.concatenate([det, rng.dirichlet([1, 1], size=(RANDOM_POLICIES_PER_MDP, 3))])
            cont = np.einsum('psa,sat->pst', policies, P[:, :, :3])
            rhs = np.einsum('psa,sak->psk', policies, P[:, :, 3:])
            val = np.linalg.solve(np.eye(3)[None] - gamma * cont, rhs)
            vp, death = val[:, :, 0], val[:, :, 1]
            z = 1 - vp - death
            J = vp[None] + weights[:, None, None] * z[None]
            optimum = J[:, :len(det)].max(axis=1)
            eps = optimum[:, None] - J
            assert eps.min() >= -1e-12
            eps = np.maximum(0, eps)
            dz, dv, dd = z[None] - z[:, None], vp[:, None] - vp[None], death[:, None] - death[None]
            for i, j in itertools.combinations(range(len(weights)), 2):
                l, h = weights[i], weights[j]
                e, ep = eps[i, :, None], eps[j, None, :]
                budget = (e + ep) / (h - l)
                residuals = [l * dz - e - dv, dv - h * dz - ep,
                             (1 - h) * dz - ep - dd, dd - (1 - l) * dz - e, -budget - dz]
                worst = np.maximum(worst, [np.max(v) for v in residuals])
                comparisons += dz.size
                reversals += int((dz < -1e-10).sum())
                ok = (dz < -1e-10) & (budget <= 1)
                candidate = np.c_[budget[ok], -dz[ok]]
                if len(candidate):
                    # At most one seeded diagnostic point per MDP/weight pair for display.
                    samples.append(candidate[rng.integers(len(candidate))].tolist())
    assert worst.max() < 1e-10, worst

    # (3) Every eligible wide pair from the saved original setups, without rerunning planning.
    entries = load_json('original_setups.json')
    _, lam = lambda_grid()
    ii, jj = np.where(lam[None, :] - lam[:, None] >= .25)
    wide = {e: [] for e in ('DAG', 'LifeGate', 'MIMIC')}
    per_setup = []
    max_excursion = 0.
    for entry in entries:
        c = entry['hidden incentives']['curves at initial states']
        vp, z = np.array(c['V_plus']), np.array(c['Z'])
        dz = z[jj] - z[ii]
        ok = dz > 1e-9
        rho = (vp[ii[ok]] - vp[jj[ok]]) / dz[ok]
        position = (rho - lam[ii[ok]]) / (lam[jj[ok]] - lam[ii[ok]])
        violation = float(max(np.max(-position, initial=0), np.max(position - 1, initial=0)))
        max_excursion = max(max_excursion, violation)
        env = next(k for k in wide if entry['setup'].startswith(k))
        wide[env].extend(position.tolist())
        per_setup.append({'setup': entry['setup'], 'gamma': entry['gamma'], 'eligible_pairs': len(rho),
                          'max_normalized_excursion': violation})
    assert max_excursion < 1e-8

    enumeration = {
        'mdps': len(ROBUSTNESS_GAMMAS) * MDPS_PER_GAMMA, 'states': 3, 'actions': 2,
        'policies_per_mdp': len(det) + RANDOM_POLICIES_PER_MDP,
        'deterministic_policies_exhaustively_enumerated': len(det),
        'gammas': list(ROBUSTNESS_GAMMAS), 'weights': weights.tolist(),
        'policy_pair_state_weight_comparisons_per_inequality': comparisons,
        'max_violations': dict(zip(('survival_lower', 'survival_upper', 'death_lower', 'death_upper',
                                    'defect_lower'), worst.tolist())),
        'defect_reversals': reversals,
        'display_samples': samples,
        'note': 'Exact objective gaps from exhaustive deterministic-policy optimum; '
                'float64 linear solves; random policies are included. Display keeps one random reversal '
                'per MDP/weight pair with normalized budget <=1, not independent observations.'}
    wide_pairs = {
        'minimum_weight_gap': .25, 'minimum_defect_change': 1e-9,
        'max_normalized_excursion': max_excursion, 'per_setup': per_setup,
        'positions_by_environment': wide,
        'note': 'All eligible initial-distribution pairs under uniform tie-breaking; '
                'positions are (rho-lambda)/(lambda_prime-lambda). Pairs are not independent.'}
    result = {'sharpness': sharp, 'guaranteed_absorption_sharpness': absorbing_sharp,
              'enumeration': enumeration, 'wide_pairs': wide_pairs}
    save_json(result, 'mass_transfer_robustness.json')
    return result


def plot_robustness(result):
    """Three panels: exact sharpness, approximately optimal policies, wide design pairs."""
    with plt.rc_context({**common.STYLE, 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'lines.linewidth': 1.5, 'lines.markersize': 4}):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
        ax = axes[0]
        r = np.linspace(.2, .8, 61)
        ax.axhspan(.2, .8, color='#e8ebef')
        ax.plot(r, r, color=C_SURV)
        ax.scatter([.2, .8], [.2, .8], color=C_SURV, s=22, zorder=3)
        ax.set(title='(a) Exact sharpness', xlabel='Switch threshold $r$', ylabel=r'Exchange rate $\rho$',
               xlim=(.16, .84), ylim=(.16, .84), xticks=[.2, .5, .8], yticks=[.2, .5, .8])
        ax = axes[1]
        points = np.array(result['enumeration']['display_samples'])
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=.18, color='#838e99', rasterized=True)
        ax.plot([0, 1], [0, 1], '--', color='#555e66', label='Error bound')
        ax.plot([0, .5], [0, .5], color=C_SURV, label='Attained')
        ax.set(title='(b) Approximate policies', xlabel=r'$(\varepsilon+\varepsilon\prime)/(\lambda\prime-\lambda)$',
               ylabel=r'Defect decrease $-\Delta Z$', xlim=(0, 1), ylim=(0, 1),
               xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax.legend(frameon=False, loc='upper left', handlelength=1.5)
        ax = axes[2]
        wide = {k: v for k, v in result['wide_pairs']['positions_by_environment'].items() if v}
        positions = np.arange(1, len(wide) + 1)
        boxes = ax.boxplot(list(wide.values()), positions=positions, widths=.5,
                           whis=(0, 100), showfliers=False, patch_artist=True,
                           medianprops={'color': '#273646', 'linewidth': 1.3})
        for patch, color in zip(boxes['boxes'], ['#aec4d9', '#e2c4ad', '#c5bfd6']):
            patch.set_facecolor(color)
        ax.axhline(0, color='#9199a3', ls='--', lw=.6)
        ax.axhline(1, color='#9199a3', ls='--', lw=.6)
        ax.set(title='(c) Wide design pairs', ylabel='Position within bound', ylim=(-.04, 1.04),
               xticks=positions, xticklabels=list(wide), yticks=[0, .5, 1])
        for ax in axes:
            ax.grid(axis='y', color='#e1e5e9', lw=.5)
            ax.set_axisbelow(True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.24, top=.86, wspace=.57)
        savefig(fig, 'mass_transfer_robustness')


def lambda_grid(n=145, span=9.0):
    """lambda = sigmoid(x) for x in [-span, span], plus the endpoints 0 and 1."""
    x = np.linspace(-span, span, n)
    return x, np.concatenate([[0.0], 1 / (1 + np.exp(-x)), [1.0]])


def tie_break(sets, rule):
    """
    Policy from optimal action sets (L, S, A): 'uniform' over the set, or the deterministic choice of its
    'first' or 'last' action.  States without any action (terminal) get the zero row.
    """
    if rule == 'uniform':
        return uniform_over(sets)
    idx = torch.arange(sets.shape[-1], device=sets.device)
    # the argmax picks the largest key among the allowed actions; -10^6 rules out the others
    key = torch.where(sets, idx if rule == 'last' else -idx, torch.full_like(idx, -10 ** 6))
    pi = torch.zeros(sets.shape, dtype=DTYPE, device=sets.device)
    pi.scatter_(-1, key.argmax(-1, keepdim=True), 1.0)
    return pi * sets.any(-1, keepdim=True)


def check_pairs(lam, q, rep):
    """
    Inequalities (i)-(iii) of the theorem for all pairs lambda_i < lambda_j, at the states in `rep`.
    q: outcome quantities, each (L, S).  Returns the per-inequality violation counts (tolerance TOL) and the
    pair mask with the differences dZ, dV+, dV- (each (L, L, n_states)).
    """
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)
    li, lj = lam_t[:, None, None], lam_t[None, :, None]
    pair = (lam_t[:, None] < lam_t[None, :])                                   # i < j
    Vp, Vm, Z = [q[k][:, rep] for k in ('V_plus', 'V_minus', 'Z')]
    dZ = Z[None, :, :] - Z[:, None, :]                                          # Z_j - Z_i
    dVp = Vp[:, None, :] - Vp[None, :, :]                                       # V+_i - V+_j
    dVm = Vm[None, :, :] - Vm[:, None, :]                                       # V-_j - V-_i
    conds = {
        '(i) lower  lambda dZ <= dV+': li * dZ - dVp,
        '(i) upper  dV+ <= lambda\' dZ': dVp - lj * dZ,
        '(ii) lower (1-lambda\') dZ <= dV-': (1 - lj) * dZ - dVm,
        '(ii) upper dV- <= (1-lambda) dZ': dVm - (1 - li) * dZ,
        '(iii) Z non-decreasing': -dZ,
        '(iii) V+ non-increasing': -dVp,
        '(iii) V- non-decreasing': -dVm,
    }
    out = {}
    for name, v in conds.items():
        v = v[pair]
        out[name] = {'max violation': float(v.clamp(min=0).max()), 'violations > tol': int((v > TOL).sum()),
                     'comparisons': int(v.numel())}
    return out, (pair, dZ, dVp, dVm)


def strictness(lam, sets, q, rep):
    """
    Strict part of (iii): for 0 < lambda < lambda' < 1 with pi' not optimal for lambda, some state has
    Z' > Z, V+' < V+ and V-' > V- simultaneously.  Optimality is decided on the action sets (gamma < 1).
    """
    n_cases, n_ok = 0, 0
    Vp, Vm, Z = [q[k][:, rep] for k in ('V_plus', 'V_minus', 'Z')]
    interior = [i for i, l in enumerate(lam) if 0 < l < 1]
    for a in interior:
        for b in interior:
            if lam[a] >= lam[b]:
                continue
            # pi_b is optimal for lambda_a iff all its actions are optimal actions for lambda_a
            if bool(((sets[b] & ~sets[a]).any(-1) & rep).any()):
                n_cases += 1
                ok = (Z[b] > Z[a] + TOL) & (Vp[b] < Vp[a] - TOL) & (Vm[b] > Vm[a] + TOL)
                n_ok += int(ok.any())
    return {'pairs with pi\' not optimal for lambda': n_cases, 'of which strict at some state': n_ok}


def optimal_designs(mdp, lam, qs, tol):
    """
    At gamma = 1 with policies that can avoid termination, a policy that only uses optimal actions need
    not be optimal (it can cycle forever among optimal actions: conserving but not equalizing).  The
    theorem is about optimal policies, so for every tie-breaking rule we keep the designs whose policy
    attains the best objective (1 - lambda) P(E_+) - lambda P(E_-) found among the rules at every state.
    qs: rule -> outcome quantities.  Returns rule -> boolean mask over designs.
    """
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)[:, None]
    W = {r: (1 - lam_t) * q['P_survival'] - lam_t * q['P_death'] for r, q in qs.items()}
    best = torch.stack(list(W.values())).max(0).values
    return {r: ((w >= best - tol) | ~mdp.report[None]).all(-1) for r, w in W.items()}


def strictness_by_value(lam, q, rep, tol):
    """
    Strict part of (iii), with 'pi' not optimal for lambda' decided on values (needed at gamma = 1, where
    a policy using only optimal actions need not be optimal): its objective V_+ + lambda Z falls below
    that of pi (optimal for lambda) at some state.
    """
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)
    Vp, Vm, Z = q['V_plus'][:, rep], q['V_minus'][:, rep], q['Z'][:, rep]
    obj_own = Vp + lam_t[:, None] * Z                                    # objective of pi_i under lambda_i
    obj_cross = Vp[None, :, :] + lam_t[:, None, None] * Z[None, :, :]   # objective of pi_j under lambda_i
    interior = (lam_t > 0) & (lam_t < 1)
    pair = (lam_t[:, None] < lam_t[None, :]) & interior[:, None] & interior[None, :]
    not_opt = (obj_cross < obj_own[:, None, :] - tol).any(-1) & pair
    strict = ((Z[None] > Z[:, None] + tol) & (Vp[None] < Vp[:, None] - tol) & (Vm[None] > Vm[:, None] + tol)).any(-1)
    return {"pairs with pi' not optimal for lambda": int(not_opt.sum()),
            'of which strict at some state': int((not_opt & strict).sum())}


def run(env, gamma):
    """Plan for every lambda of the grid, check (i)-(iv) for each tie-breaking rule, and save the results."""
    global TOL
    mdp = envs.lifegate(0.2) if env == 'lifegate' else envs.mimic()
    x, lam = lambda_grid()
    lam_t = torch.as_tensor(lam, dtype=DTYPE, device=DEVICE)
    info = {}
    if gamma < 1:
        Q, _ = solve_optimal(mdp, mdp.rewards(1 - lam_t, lam_t), gamma)
        sets = optimal_sets(mdp, normalize_Q(Q, (1, 2)), atol=TIE_ATOL)
        eval_gamma = gamma
    else:
        sets, info = solve_optimal_undiscounted(mdp, 1 - lam_t, lam_t)
        eval_gamma = 1.0 - 1e-12        # discounted solve is singular at 1; V+ -> P(E+) etc. to ~1e-12 T
    # exact planning: round-off tolerance; Blackwell approximation at gamma = 1 - eps: its O(eps) error
    TOL = max(1e-10, 100 * info.get('approximation error scale', 0.0))
    rep = mdp.report
    summary = {'env': mdp.name, 'gamma': gamma, 'n_lambda': len(lam), 'violation tolerance': TOL,
               'distinct optimal sets': len({tuple(to_np(s).ravel()) for s in sets}), 'gamma=1 planning': info,
               'checks': {}}
    qs = {}
    for rule in TIE_RULES:
        q = summarize(mdp, tie_break(sets, rule), eval_gamma)
        if gamma >= 1:            # at gamma = 1 the theorem's masses are the absorption probabilities
            q['V_plus'], q['V_minus'], q['Z'] = q['P_survival'], -q['P_death'], q['P_never']
        qs[rule] = q
    curves = {k: to_np(at_start(mdp, v)) for k, v in qs['uniform'].items()}
    keep = optimal_designs(mdp, lam, qs, TOL) if gamma >= 1 else None
    for rule in TIE_RULES:
        q = qs[rule]
        lam_r = lam
        if keep is not None:      # only designs whose tie-broken policy is optimal (see optimal_designs)
            k = keep[rule]
            q, lam_r = {name: v[k] for name, v in q.items()}, lam[k.cpu().numpy()]
        checks, _ = check_pairs(lam_r, q, rep)
        if keep is not None:
            checks['designs whose policy is optimal at gamma = 1 (others excluded)'] = \
                f'{int(keep[rule].sum())}/{len(lam)}'
        if gamma >= 1:
            # (iv): P_never' - P_never = (P+ - P+') + (P- - P-') with both brackets >= 0
            P = {k: q[k][:, rep] for k in ('P_survival', 'P_death', 'P_never')}
            lam_rt = torch.as_tensor(lam_r, device=DEVICE)
            pair = lam_rt[:, None] < lam_rt[None, :]
            lhs = (P['P_never'][None] - P['P_never'][:, None])[pair]
            dp = (P['P_survival'][:, None] - P['P_survival'][None])[pair]
            dm = (P['P_death'][:, None] - P['P_death'][None])[pair]
            checks['(iv) identity |error|'] = float((lhs - dp - dm).abs().max())
            checks['(iv) survival drop >= 0: min'] = float(dp.min())
            checks['(iv) death drop >= 0: min'] = float(dm.min())
        checks['(iii) strictness'] = strictness(lam, sets, q, rep) if gamma < 1 else \
            strictness_by_value(lam_r, q, rep, TOL)
        summary['checks'][rule] = checks
    mid = 1 + len(x) // 2                                       # lambda = 1/2
    summary['at d0 (uniform tie-breaking)'] = {
        k: {'survival-only': float(v[0]), 'mixed': float(v[mid]), 'death-only': float(v[-1])}
        for k, v in curves.items()}
    tag = env + ('_gamma1' if gamma >= 1 else '')
    save_json(summary, f'mass_transfer_{tag}.json')
    np.savez(result_path(f'mass_transfer_{tag}.npz', mimic=(env == 'mimic')), x=x, lam=lam, **curves)
    return x, lam, curves, summary, tag


def lambda_axis(x):
    """
    Plot positions for the lambda grid: the logit x for 0 < lambda < 1, with lambda = 0 and 1 placed a
    short distance beyond the ends.  Returns positions, tick positions and tick labels.
    """
    dx = x[1] - x[0]
    xs = np.concatenate([[x[0] - 8 * dx], x, [x[-1] + 8 * dx]])
    ticks = [xs[0], np.log(1e-3 / (1 - 1e-3)), 0.0, np.log((1 - 1e-3) / 1e-3), xs[-1]]
    labels = ['0', '$10^{-3}$', '1/2', '$1-10^{-3}$', '1']
    return xs, ticks, labels


def style_lambda_axis(ax, xs, ticks, labels):
    """Ticks, limits, label and grid of a lambda axis built by lambda_axis."""
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(r'$\lambda=\beta/(\alpha+\beta)$')
    ax.grid(True, alpha=GRID_ALPHA)


def stack_masses(ax, xs, Vp, Vm, Z, legend=True):
    """V_+ + (-V_-) + Z = 1 as stacked areas."""
    ax.stackplot(xs, Vp, -Vm, Z, colors=[C_SURV, C_DEATH, C_NEVER], alpha=0.6,
                 labels=[r'$V_+$ (survival)', r'$-V_-$ (death)', r'$Z$ (defect)'])
    ax.set_ylim(0, 1)
    if legend:
        ax.legend(loc='lower left')


def plot(x, lam, c, tag):
    """Conservation, exchange rate and undiscounted outcomes along lambda, at the initial states."""
    apply_style()
    xs, ticks, labels = lambda_axis(x)
    fig, axes = plt.subplots(1, 3, figsize=figsize(1, 3))

    ax = axes[0]
    stack_masses(ax, xs, c['V_plus'], c['V_minus'], c['Z'])
    ax.set_ylabel('Mass at initial states')
    ax.set_title('(a) Conservation')

    ax = axes[1]
    Z, Vp = c['Z'], c['V_plus']
    dZ, dV = Z[1:] - Z[:-1], Vp[:-1] - Vp[1:]
    ok = dZ > 1e-9                      # the exchange rate is defined where the defect moves
    xm = 0.5 * (xs[:-1] + xs[1:])
    ax.fill_between(xm, lam[:-1], lam[1:], color=C_NEVER, alpha=0.3, linewidth=0, label=r"$[\lambda,\lambda']$")
    ax.plot(xs, lam, color=C_NEVER, linestyle='--', label=r'$\rho=\lambda$')
    ax.scatter(xm[ok], dV[ok] / dZ[ok], color=C_TERM, s=50, zorder=3, label=r"$\rho=\frac{V_+-V_+'}{Z'-Z}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r'Exchange rate $\rho$')
    ax.set_title('(b) Exchange rate')
    ax.legend(loc='upper left')

    ax = axes[2]
    for key, label, color, ls in [('P_survival', r'$\mathbb{P}(\mathcal{E}_+)$', C_SURV, '-'),
                                  ('P_death', r'$\mathbb{P}(\mathcal{E}_-)$', C_DEATH, '-'),
                                  ('P_never', r'$\mathbb{P}(\tau=\infty)$', C_NEVER, '--')]:
        ax.plot(xs, c[key], color=color, linestyle=ls, label=label)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel('Probability at initial states')
    ax.set_title('(c) Undiscounted outcomes')
    ax.legend(loc='center left')

    for ax in axes:
        style_lambda_axis(ax, xs, ticks, labels)
    fig.tight_layout()
    savefig(fig, f'mass_transfer_{tag}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['lifegate', 'mimic'])
    parser.add_argument('--robustness', action='store_true', help='approximate optimality, sharpness, wide pairs')
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    if args.robustness:
        result = load_json('mass_transfer_robustness.json') if args.plot_only else robustness_audit()
        plot_robustness(result)
        return
    if args.env is None:
        parser.error('--env is required unless --robustness is used')
    gamma = args.gamma or (0.9 if args.env == 'lifegate' else 0.99)
    if args.plot_only:
        tag = args.env + ('_gamma1' if gamma >= 1 else '')
        d = np.load(result_path(f'mass_transfer_{tag}.npz', mimic=(args.env == 'mimic')))
        curves = {k: d[k] for k in d.files if k not in ('x', 'lam')}
        plot(d['x'], d['lam'], curves, tag)
        return
    x, lam, curves, summary, tag = run(args.env, gamma)
    for rule, checks in summary['checks'].items():
        bad = {k: v for k, v in checks.items() if isinstance(v, dict) and v.get('violations > tol', 0)}
        print(rule, 'violations:', bad or 'none', '| strictness', checks['(iii) strictness'],
              {k: v for k, v in checks.items() if k.startswith('(iv)')})
    print('distinct optimal sets', summary['distinct optimal sets'], summary['gamma=1 planning'])
    plot(x, lam, curves, tag)


if __name__ == '__main__':
    main()
