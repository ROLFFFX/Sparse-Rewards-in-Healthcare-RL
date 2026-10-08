"""
Sensitivity of the MIMIC-III results to the estimated transition model.

The empirical MDP (750 states, 25 actions, survival and death terminals) is re-estimated from the training
trajectories under four perturbations, and for each model and gamma in {0.99, 1} the designs
lambda in {0, 1/2, 1} are solved, the mass-transfer inequalities are verified, and the change of the
terminal defect Z_1 - Z_0 at the initial states is recorded:
  support:   only actions observed at least 11, 21 or 51 times are allowed (the baseline uses the
             behavior-policy action mask, which on the training split is exactly count >= 6);
  leakage:   every non-terminal transition leaks a fraction delta in {1e-5, ..., 1e-2} into the terminals,
             split by the empirical terminal frequencies (a stress test of structural zeros);
  pooling:   250 or 500 states by Ward clustering of the 750 k-means centroids;
  bootstrap: trajectory reweighting with independent Exp(1) weights per trajectory, keeping the state
             partition, action mask and transition support fixed (positive weights preserve structural zeros).
For the baseline it also evaluates other optimal tie-breaking rules and a death-optimal policy refined by
a secondary survival objective, and records the closed classes reachable under the death-optimal policy.
All outputs are aggregate model diagnostics, not clinical estimates; the data and models are only read.

Outputs: results/supplementary/mimic_sensitivity.json (written after every model, so an interrupted run
resumes where it stopped; settings that differ from a saved study raise an error) and
plots/supplementary/propositions/mimic_sensitivity.pdf.

Usage:  python exp_mimic_sensitivity.py [--replicates 30] [--plot-only]
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # library in code/

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import common
import envs
import exp_mass_transfer as mt
from common import load_json, plt, save_json, savefig, to_np
from exp_original_setups import masses, plan
from tabular import TabularMDP, at_start, optimal_sets, policy_matrix, uniform_over

common.set_group('propositions')      # figures and results go to the supplementary folders

LAM = np.array([0., .5, 1.])
NAME = 'mimic_sensitivity'
GAMMAS = [.99, 1.]
THRESHOLDS = [6, 11, 21, 51]          # minimum action counts; 6 is the baseline
LEAKAGES = [0., 1e-5, 1e-4, 1e-3, .01]
POOLED_STATES = [250, 500, 750]       # 750 is the baseline
SEED = 20260930
N_MODELS = 10                         # baseline, 3 support thresholds, 4 leakage levels, 2 pooled resolutions


def model(counts, d0, threshold=6, fixed_mask=None, leak=0., terminal_prior=None):
    """
    Empirical MDP from transition counts (S + 2, A, S + 2), whose last two states are the survival and
    death terminals.  Allowed actions are those with at least `threshold` observations, or `fixed_mask`
    restricted to observed actions; a state left without any allowed action keeps its most frequent one.
    With leak > 0 a fraction `leak` of every non-terminal transition goes to the terminals, split by
    `terminal_prior`.  Returns the MDP and a dict of model diagnostics.
    """
    n_states = len(counts) - 2
    n = counts.sum(-1)
    observed = n[:n_states] > 0
    assert observed.any(1).all(), 'No unobserved-state zero rows may become artificial traps.'
    mask = np.zeros_like(n, dtype=bool)
    if fixed_mask is None:
        mask[:n_states] = observed & (n[:n_states] >= threshold)
    else:
        mask[:n_states] = fixed_mask[:n_states] & observed
    missing = ~mask[:n_states].any(1)
    mask[np.flatnonzero(missing), n[:n_states][missing].argmax(1)] = True
    P = np.divide(counts, n[..., None], out=np.zeros_like(counts), where=n[..., None] > 0)
    if leak:
        P[:n_states] *= 1 - leak
        P[:n_states, :, n_states:] += leak * np.asarray(terminal_prior)[None, None]
    kind = np.zeros(n_states + 2, dtype=int)
    kind[-2:] = [1, -1]
    mdp = TabularMDP.from_numpy(P, mask, kind, d0=d0, name='MIMIC sensitivity')
    assert np.allclose(P[mask].sum(-1), 1)
    return mdp, {'states': n_states, 'allowed_actions': int(mask.sum()),
                 'modal_fallback_states': int(missing.sum()),
                 'modal_fallback_initial_mass': float(d0[:n_states][missing].sum()),
                 'supported_count_quantiles': np.quantile(n[mask], [0, .1, .5, .9, 1]).tolist()}


def closed_classes(mdp, pi, counts):
    """Reachable closed SCCs of the selected policy; no cluster identifiers exported."""
    n_states = mdp.S - 2
    T = to_np(policy_matrix(mdp, pi[None]))[0, :n_states, :n_states]
    _, labels = connected_components(csr_matrix(T > 0), directed=True, connection='strong')
    reachable = to_np(mdp.d0)[:n_states] > 0
    while True:
        new = reachable | (T[reachable].sum(0) > 0)
        if np.array_equal(new, reachable):
            break
        reachable = new
    closed = []
    closed_states = np.zeros(n_states, dtype=bool)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if reachable[idx].any() and np.all(np.abs(T[np.ix_(idx, idx)].sum(1) - 1) < 1e-12):
            closed.append(len(idx))
            closed_states[idx] = True
    selected = to_np(pi)[:n_states] > 0
    supports = counts.sum(-1)[:n_states][selected & closed_states[:, None]]
    return {'reachable_closed_classes': len(closed), 'class_sizes': sorted(closed, reverse=True),
            'closed_states': int(closed_states.sum()),
            'selected_action_counts_in_closed_classes':
                [] if not len(supports) else np.quantile(supports, [0, .5, 1]).tolist(),
            'selected_actions_below_20_in_closed_classes': int((supports < 20).sum()),
            'selected_actions_in_closed_classes': len(supports)}


def evaluate(mdp, counts, gamma, label, family, parameter, diagnostics, tie_study=False):
    """
    Solve the designs LAM on one model, verify the mass-transfer inequalities (raise on a violation) and
    record the outcome curves at d0.  With tie_study, also evaluate the 'first'/'last' tie-breaking rules
    and a death-optimal policy refined by a secondary survival objective.
    """
    t0 = time.time()
    sets, tol, method = plan(mdp, LAM, gamma)
    pi = uniform_over(sets)
    q = masses(mdp, pi, gamma)
    curves = {k: to_np(at_start(mdp, v)).tolist() for k, v in q.items()}
    checks, _ = mt.check_pairs(LAM, q, mdp.report)
    maximum = max(v['max violation'] for v in checks.values())
    # All reported policies are evaluated, even when a tie rule loses optimality at gamma=1.
    result = {'label': label, 'family': family, 'parameter': parameter, 'gamma': gamma,
              'planning': method, 'lambdas': LAM.tolist(), 'diagnostics': diagnostics,
              'curves': curves, 'max_transfer_violation': maximum,
              'verification_tolerance': max(tol, 1e-8),
              'death_policy_topology': closed_classes(mdp, pi[-1], counts),
              'endpoint_difference': {k: v[-1] - v[0] for k, v in curves.items()}}
    if maximum > max(tol, 1e-8):
        raise AssertionError(f'{label} gamma={gamma}: transfer violation {maximum}')
    if tie_study:
        lam_t = torch.as_tensor(LAM, device=pi.device)[:, None]
        variants = {}
        for rule in ['first', 'last']:
            v = masses(mdp, mt.tie_break(sets, rule), gamma)
            gap = (1 - lam_t) * (q['V_plus'] - v['V_plus']) + lam_t * (q['V_minus'] - v['V_minus'])
            variants[rule] = {
                'curves': {k: to_np(at_start(mdp, x)).tolist() for k, x in v.items()},
                'max_objective_loss_vs_uniform_per_design': to_np(gap[:, mdp.report].max(1).values).tolist(),
                'note': 'Positive losses identify non-equivalent tie policies; no optimality claim for those.'}
        # A secondary recovery objective within death-optimal actions. At gamma=1 use the
        # complete Bellman action set of the evaluated minimum-death value, rather than its
        # near-one-discount refinement; verify the resulting death value explicitly.
        allowed = sets[-1]
        if gamma == 1:
            _, p_minus, P_cont = mdp.entry_probs()
            Q_death = -p_minus + torch.einsum('sat,t->sa', P_cont, q['V_minus'][-1])
            Q_death = Q_death.masked_fill(~mdp.mask, -torch.inf)
            allowed = optimal_sets(mdp, Q_death[None], atol=1e-12)[0]
        assert (allowed.any(1) | ~mdp.report).all(), 'Secondary objective must preserve an action at every state.'
        restricted = TabularMDP(mdp.P, mdp.mask & allowed, mdp.kind, mdp.d0, mdp.report, mdp.name, mdp.meta)
        secondary_sets, _, _ = plan(restricted, np.array([0.]), gamma)
        sec = masses(mdp, uniform_over(secondary_sets), gamma)
        death_loss = (q['V_minus'][-1] - sec['V_minus'][0])[mdp.report]
        variants['death_with_recovery_secondary'] = {
            'curves': {k: float(at_start(mdp, v)[0]) for k, v in sec.items()},
            'max_death_objective_loss_vs_uniform': float(death_loss.max()),
            'verified_primary_objective_max_abs_error': float(death_loss.abs().max()),
            'note': 'Secondary optimization over death-optimal actions, not a different reward scalarization.'}
        assert variants['death_with_recovery_secondary']['verified_primary_objective_max_abs_error'] < 1e-7
        result['tie_rules'] = variants
    print(f'{label:22s} gamma={gamma:g} deltaZ={result["endpoint_difference"]["Z"]:.4f} '
          f'never(death)={curves["P_never"][-1]:.4f} [{time.time() - t0:.1f}s]', flush=True)
    return result


def run(replicates):
    """Build every perturbed model, evaluate it at each gamma, and save the study after every evaluation."""
    root = Path(envs.MIMIC_DATA)
    source = root / 'traj_shifted_train.csv'
    mask_file = root / 'behavior_policy/SA_mask_shifted.pkl'
    centroid_file = root / 'kmeans_750.joblib'
    df = pd.read_csv(source).sort_values(['traj', 'step'])
    n_states, n_actions = 750, 25
    s, a, nxt = [df[k].to_numpy(dtype=int) for k in ['s:state', 'a:action', 's:next_state']]
    key = (s * n_actions + a) * (n_states + 2) + nxt                   # flat index into the count array
    shape = (n_states + 2, n_actions, n_states + 2)
    counts = np.bincount(key, minlength=np.prod(shape)).reshape(shape).astype(float)
    first = df.groupby('traj', sort=False)['s:state'].first().to_numpy(dtype=int)
    d0 = np.bincount(first, minlength=n_states + 2).astype(float)
    d0 /= d0.sum()
    behavior_mask = np.zeros((n_states + 2, n_actions), dtype=bool)
    behavior_mask[:n_states] = np.asarray(pd.read_pickle(mask_file), bool)[:n_states]
    prior = counts[:, :, -2:].sum((0, 1))
    prior /= prior.sum()
    base, diag = model(counts, d0, fixed_mask=behavior_mask)
    assert np.array_equal(to_np(base.mask), behavior_mask)
    manifest = {
        'trajectory_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'mask_sha256': hashlib.sha256(mask_file.read_bytes()).hexdigest(),
        'centroids_sha256': hashlib.sha256(centroid_file.read_bytes()).hexdigest(),
        'replicates': replicates, 'seed': SEED, 'lambdas': LAM.tolist(), 'gammas': GAMMAS,
        'thresholds': THRESHOLDS, 'leakages': LEAKAGES,
        'pooled_states': POOLED_STATES, 'trajectories': len(first), 'transitions': len(df),
        'terminal_prior': prior.tolist(),
        'note': 'Aggregate empirical-MDP sensitivity, not clinical efficacy. Bootstrap uses independent '
                'Exp(1) weights per trajectory, shared across all its transitions, normalized to sum N; '
                'fixed initial distribution, mask and state partition. Leakage is an imposed stress test. '
                'State pooling uses Ward clustering of training KMeans centroids, not fresh feature clustering.'}
    path = Path(common.result_path(NAME + '.json'))
    results = []
    if path.exists():
        saved = load_json(NAME + '.json')
        if saved['manifest'] != manifest:
            raise ValueError('Saved study settings differ; preserve it before starting another configuration.')
        results = saved['results']

    def task(mdp, c, info, label, family, parameter, ties=False):
        """Evaluate one model at every gamma not yet in the saved results, saving after each."""
        for gamma in GAMMAS:
            if any(r['label'] == label and r['gamma'] == gamma for r in results):
                continue
            results.append(evaluate(mdp, c, gamma, label, family, parameter, info, ties))
            save_json({'manifest': manifest, 'results': results, 'complete': False}, NAME + '.json')

    task(base, counts, diag, 'Baseline', 'baseline', 0, True)
    for threshold in THRESHOLDS[1:]:
        m, info = model(counts, d0, threshold=threshold)
        task(m, counts, info, f'Count >= {threshold}', 'support', threshold)
    for leak in LEAKAGES[1:]:
        m, info = model(counts, d0, fixed_mask=behavior_mask, leak=leak, terminal_prior=prior)
        task(m, counts, info, f'Leak {leak:g}', 'leakage', leak)
    centers = joblib.load(centroid_file).cluster_centers_
    tree = linkage(centers, method='ward')
    for k in POOLED_STATES[:-1]:
        mapping = np.r_[fcluster(tree, k, criterion='maxclust') - 1, k, k + 1]      # terminals keep their role
        pooled_key = (mapping[s] * n_actions + a) * (k + 2) + mapping[nxt]
        pooled_counts = np.bincount(pooled_key, minlength=(k + 2) * n_actions * (k + 2))
        pooled_counts = pooled_counts.reshape(k + 2, n_actions, k + 2).astype(float)
        pooled_d0 = np.bincount(mapping[first], minlength=k + 2).astype(float)
        pooled_d0 /= pooled_d0.sum()
        m, info = model(pooled_counts, pooled_d0)
        task(m, pooled_counts, info, f'{k} pooled states', 'pooling', k)
    traj_index = pd.factorize(df['traj'], sort=False)[0]
    rng = np.random.default_rng(manifest['seed'])
    for b in range(replicates):
        weights = rng.exponential(size=len(first))
        weights *= len(first) / weights.sum()
        boot_counts = np.bincount(key, weights=weights[traj_index], minlength=np.prod(shape)).reshape(shape)
        m, info = model(boot_counts, d0, fixed_mask=behavior_mask)
        task(m, boot_counts, info, f'Bootstrap {b:02d}', 'bootstrap', b)
    study = {'manifest': manifest, 'results': results, 'complete': True}
    assert len(results) == len(GAMMAS) * (N_MODELS + replicates)
    save_json(study, NAME + '.json')
    return study


def plot(study):
    """Z_1 - Z_0 at d0 under (a) support, (b) leakage, (c) pooling and (d) bootstrap perturbations."""
    assert study['complete'], 'Do not present incomplete sensitivity studies.'
    rows = study['results']
    assert len(rows) == len(GAMMAS) * (N_MODELS + study['manifest']['replicates'])
    assert len({(r['label'], r['gamma']) for r in rows}) == len(rows)
    with plt.rc_context({**common.STYLE, 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'lines.linewidth': 1.5, 'lines.markersize': 4}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
        colors = {.99: '#2166ac', 1.: '#b2182b'}
        specs = [('support', '(a) Action support', 'Minimum count', 6),
                 ('leakage', '(b) Terminal leakage', r'Per-step leakage $\delta$', 0),
                 ('pooling', '(c) State resolution', 'Number of states', 750)]
        for ax, (family, title, xlabel, baseline_x) in zip(axes.ravel(), specs):
            for gamma, color in colors.items():
                rs = [r for r in rows if r['gamma'] == gamma and r['family'] in (family, 'baseline')]
                xy = sorted([(baseline_x if r['family'] == 'baseline' else r['parameter'],
                              r['endpoint_difference']['Z']) for r in rs])
                x, y = np.array(xy).T
                if family == 'leakage':
                    x = np.arange(len(x))
                    ax.set_xticks(x, ['0', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])
                else:
                    ax.set_xticks(x)
                ax.plot(x, y, marker='o' if gamma < 1 else 's', color=color, label=rf'$\gamma={gamma:g}$')
            ax.set(title=title, xlabel=xlabel, ylabel=r'Defect change $Z_1-Z_0$')
        ax = axes[1, 1]
        for i, (gamma, color) in enumerate(colors.items()):
            vals = np.array([r['endpoint_difference']['Z'] for r in rows
                             if r['family'] == 'bootstrap' and r['gamma'] == gamma])
            jitter = np.linspace(-.12, .12, len(vals))
            ax.scatter(i + jitter, vals, color=color, s=14, alpha=.65)
            ax.plot([i - .16, i + .16], [np.median(vals)] * 2, color=color, lw=2)
            baseline = next(r['endpoint_difference']['Z'] for r in rows
                            if r['family'] == 'baseline' and r['gamma'] == gamma)
            ax.scatter(i, baseline, marker='D', facecolor='white', edgecolor='black', s=35, zorder=4)
        ax.set(title='(d) Trajectory reweighting', xticks=[0, 1], xticklabels=[r'$\gamma=0.99$', r'$\gamma=1$'],
               ylabel=r'Defect change $Z_1-Z_0$', xlabel='Fixed partition and action mask')
        ymax = max(r['endpoint_difference']['Z'] for r in rows)
        for ax in axes.ravel():
            ax.axhline(0, color='#959da5', ls='--', lw=.7)
            ax.set_ylim(-.02, max(.4, ymax + .04))
            ax.grid(axis='y', color='#e1e5e9', lw=.6)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1))
        fig.subplots_adjust(left=.09, right=.98, bottom=.1, top=.88, wspace=.3, hspace=.5)
        savefig(fig, NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicates', type=int, default=30, help='trajectory-bootstrap replicates')
    parser.add_argument('--plot-only', action='store_true', help='redraw the figure from saved results')
    args = parser.parse_args()
    plot(load_json(NAME + '.json') if args.plot_only else run(args.replicates))


if __name__ == '__main__':
    main()
