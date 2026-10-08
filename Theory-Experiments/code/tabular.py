"""
Exact tabular planning and policy evaluation for the theory experiments (PyTorch, float64, GPU if available).

An MDP has non-terminal states and two absorbing terminal sets S_+ (survival) and S_- (death). The reward
designs are the terminal rewards
    R_{alpha,beta}(s,a,s') = alpha 1{s not terminal, s' in S_+} - beta 1{s not terminal, s' in S_-},
optionally plus a step reward c on transitions between non-terminal states (Proposition "Terminal scale
versus step cost"). Terminal states have value 0, so only the non-terminal block of the kernel is needed.

Shapes: S states, A actions, B reward designs or policies handled in one batch. Inadmissible actions have
Q = -inf; states without admissible actions (terminal or unreachable) have V = 0.

Main entry points
    TabularMDP                       the MDP container (built by `envs`)
    solve_optimal                    exact Q* for a batch of rewards, gamma < 1
    solve_optimal_undiscounted       optimal action sets at gamma = 1
    optimal_sets, uniform_over       the tie rule: optimal actions within a tolerance, uniform over them
    evaluate_discounted, summarize   exact evaluation of fixed policies and their outcome quantities
    absorption_probabilities         undiscounted survival / death / never-terminate probabilities
    sup_defect, sup_never, sup_time  auxiliary control problems of Corollary "Robustness certificate"
"""
from dataclasses import dataclass, field

import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64

TRANSIENT, SURVIVAL, DEATH = 0, 1, -1


@dataclass
class TabularMDP:
    """
    A finite MDP with absorbing survival and death states (all tensors on DEVICE).

    P      (S, A, S) transition probabilities; rows of inadmissible actions are ignored
    mask   (S, A)    admissible actions (always False on terminal states)
    kind   (S,)      TRANSIENT, SURVIVAL or DEATH
    d0     (S,)      initial distribution (used for aggregate numbers)
    report (S,)      states included in per-state plots (e.g. excludes unreachable walls)
    name             display name
    meta             environment-specific extras
    """
    P: torch.Tensor
    mask: torch.Tensor
    kind: torch.Tensor
    d0: torch.Tensor
    report: torch.Tensor
    name: str = ''
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_numpy(cls, P, mask, kind, d0=None, report=None, name='', meta=None):
        """
        Build from array-likes (d0 and report may also be tensors). Actions of terminal states are masked out.
        d0 defaults to uniform over the non-terminal states with an admissible action and is normalized;
        report defaults to the same states.
        """
        P = torch.as_tensor(np.asarray(P), dtype=DTYPE, device=DEVICE)
        mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=DEVICE)
        kind = torch.as_tensor(np.asarray(kind), dtype=torch.int64, device=DEVICE)
        nonterm = kind == TRANSIENT
        mask = mask & nonterm[:, None]
        if d0 is None:
            d0 = (nonterm & mask.any(1)).to(DTYPE)
        d0 = torch.as_tensor(np.asarray(d0) if not torch.is_tensor(d0) else d0, dtype=DTYPE, device=DEVICE)
        d0 = d0 / d0.sum()
        if report is None:
            report = nonterm & mask.any(1)
        report = torch.as_tensor(np.asarray(report) if not torch.is_tensor(report) else report,
                                 dtype=torch.bool, device=DEVICE)
        return cls(P, mask, kind, d0, report, name, meta or {})

    # --- derived quantities -------------------------------------------------
    @property
    def S(self):
        return self.P.shape[0]

    @property
    def A(self):
        return self.P.shape[1]

    @property
    def nonterm(self):
        return self.kind == TRANSIENT

    def entry_probs(self):
        """
        (p_+, p_-, P_cont): probabilities (S, A) of entering S_+ / S_- in one step, and the transition kernel
        (S, A, S) between non-terminal states (all other entries zeroed).
        """
        nt = self.nonterm.to(DTYPE)
        p_plus = (self.P * (self.kind == SURVIVAL).to(DTYPE)).sum(-1) * nt[:, None]
        p_minus = (self.P * (self.kind == DEATH).to(DTYPE)).sum(-1) * nt[:, None]
        P_cont = self.P * nt[None, None, :] * nt[:, None, None]
        return p_plus, p_minus, P_cont

    def rewards(self, alpha, beta, step_cost=0.0, scale=1.0):
        """
        Expected one-step reward r(s,a) of scale * R_{alpha,beta} + c, batched over designs.
        alpha, beta, scale: floats or tensors broadcastable to (B,); step_cost: the float c. Returns (B, S, A).
        """
        p_plus, p_minus, P_cont = self.entry_probs()
        alpha, beta, scale = [torch.as_tensor(x, dtype=DTYPE, device=DEVICE).reshape(-1)
                              for x in (alpha, beta, scale)]
        r = scale[:, None, None] * (alpha[:, None, None] * p_plus - beta[:, None, None] * p_minus)
        if step_cost:
            r = r + step_cost * P_cont.sum(-1)[None]
        return r


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def _bellman_Q(mdp, r, V, gamma, P_cont):
    """Q(s,a) = r(s,a) + gamma sum_t P_cont(s,a,t) V(t); r (B, S, A), V (B, S) -> (B, S, A)."""
    Q = r + gamma * torch.einsum('sat,bt->bsa', P_cont, V)
    return Q.masked_fill(~mdp.mask[None], -torch.inf)


def _V_from_Q(mdp, Q):
    """Greedy value max_a Q(s,a), set to 0 on states without admissible actions."""
    V = Q.max(-1).values
    return torch.where(mdp.mask.any(1)[None], V, torch.zeros_like(V))


def policy_matrix(mdp, pi, P_cont=None):
    """P_pi restricted to non-terminal states. pi: (B, S, A) -> (B, S, S)."""
    if P_cont is None:
        P_cont = mdp.entry_probs()[2]
    return torch.einsum('bsa,sat->bst', pi, P_cont)


def evaluate_discounted(mdp, pi, r, gamma):
    """Exact V^pi (B, S) of policies pi (B, S, A) for reward r (B, S, A) or (S, A), gamma < 1 (linear solve)."""
    P_cont = mdp.entry_probs()[2]
    Ppi = policy_matrix(mdp, pi, P_cont)
    if r.dim() == 2:
        r = r.expand(pi.shape[0], -1, -1)
    r_pi = (pi * r).sum(-1)
    I = torch.eye(mdp.S, dtype=DTYPE, device=DEVICE).expand_as(Ppi)
    return torch.linalg.solve(I - gamma * Ppi, r_pi.unsqueeze(-1)).squeeze(-1)


def greedy_deterministic(mdp, Q):
    """Deterministic greedy policy (B, S, A) of Q (first maximizer); zero rows on states without actions."""
    a = Q.argmax(-1)
    pi = torch.zeros_like(Q)
    pi.scatter_(-1, a.unsqueeze(-1), 1.0)
    return pi * mdp.mask.any(1)[None, :, None]


def solve_optimal(mdp, r, gamma, vi_tol=1e-10, max_vi=100000, max_pi=100):
    """
    Optimal (Q*, V*), shapes (B, S, A) and (B, S), for a batch of reward functions r (B, S, A), gamma < 1.
    Value iteration to vi_tol, then policy iteration with exact evaluation until the greedy
    policy is stable, so that Q* is exact up to linear-solve round-off (~1e-14).
    """
    assert gamma < 1, 'use gamma < 1 for planning'
    P_cont = mdp.entry_probs()[2]
    V = torch.zeros(r.shape[0], mdp.S, dtype=DTYPE, device=DEVICE)
    if gamma > 0.999:
        # value iteration contracts by gamma per sweep, so near 1 it only warm-starts policy iteration,
        # whose number of iterations does not depend on gamma
        max_vi = min(max_vi, 500)
    for _ in range(max_vi):
        V_new = _V_from_Q(mdp, _bellman_Q(mdp, r, V, gamma, P_cont))
        if (V_new - V).abs().max() < vi_tol:
            V = V_new
            break
        V = V_new
    pi = greedy_deterministic(mdp, _bellman_Q(mdp, r, V, gamma, P_cont))
    for _ in range(max_pi):
        V = evaluate_discounted(mdp, pi, r, gamma)
        Q = _bellman_Q(mdp, r, V, gamma, P_cont)
        # keep the current action unless another one is strictly better (avoids tie cycling)
        current = (pi * Q.nan_to_num(neginf=0.0)).sum(-1)
        best = Q.max(-1).values
        improve = (best > current + 1e-12) & mdp.mask.any(1)[None]
        if not improve.any():
            return Q, V
        pi = torch.where(improve[..., None], greedy_deterministic(mdp, Q), pi)
    raise RuntimeError('policy iteration did not converge')


def normalize_Q(Q, dims):
    """Q divided by the largest finite |Q| over the dimensions `dims` (inadmissible actions stay -inf)."""
    scale = Q.nan_to_num(neginf=0.0).abs().amax(dim=dims, keepdim=True).clamp(min=1e-300)
    return Q / scale


def optimal_sets(mdp, Q, atol=1e-9):
    """Q (B, S, A) -> (B, S, A) bool: admissible actions within atol of the best action of their state."""
    best = Q.max(-1, keepdim=True).values
    return (Q >= best - atol) & mdp.mask[None]


def uniform_over(sets):
    """Uniform stochastic policy over a set of actions (B, S, A) bool -> (B, S, A) float."""
    indicator = sets.to(DTYPE)
    n = indicator.sum(-1, keepdim=True)
    return torch.where(n > 0, indicator / n.clamp(min=1), torch.zeros_like(indicator))


# ---------------------------------------------------------------------------
# Outcome quantities of a fixed policy
# ---------------------------------------------------------------------------
def discounted_quantities(mdp, pi, gamma):
    """
    (V_+, V_-), each (B, S): values of policies pi (B, S, A) for the rewards 1{s' in S_+} and -1{s' in S_-},
    i.e. V_+ = E[gamma^(tau-1) 1_{E_+}] and V_- = -E[gamma^(tau-1) 1_{E_-}].
    """
    p_plus, p_minus, _ = mdp.entry_probs()
    Vp = evaluate_discounted(mdp, pi, p_plus, gamma)
    Vm = evaluate_discounted(mdp, pi, -p_minus, gamma)
    return Vp, Vm


def absorption_probabilities(mdp, pi):
    """
    Undiscounted P(tau<inf, S_tau in S_+), P(tau<inf, S_tau in S_-) and P(tau=inf), each (B, S), for
    policies pi (B, S, A). Solved on the states from which the terminal set is reachable under pi; all other
    states never terminate. (On that set I - P_pi is non-singular, so the solve is exact.)
    """
    p_plus, p_minus, P_cont = mdp.entry_probs()
    Ppi = policy_matrix(mdp, pi, P_cont)
    b_plus = (pi * p_plus[None]).sum(-1)
    b_minus = (pi * p_minus[None]).sum(-1)
    B, S = b_plus.shape
    surv = torch.zeros(B, S, dtype=DTYPE, device=DEVICE)
    death = torch.zeros(B, S, dtype=DTYPE, device=DEVICE)
    for b in range(B):
        reach = (b_plus[b] + b_minus[b]) > 0
        edges = Ppi[b] > 0
        while True:     # backward closure: add states with an edge into `reach`
            grown = reach | (edges & reach[None, :]).any(1)
            if torch.equal(grown, reach):
                break
            reach = grown
        idx = torch.nonzero(reach).squeeze(-1)
        if idx.numel() == 0:
            continue
        M = torch.eye(idx.numel(), dtype=DTYPE, device=DEVICE) - Ppi[b][idx][:, idx]
        sol = torch.linalg.solve(M, torch.stack([b_plus[b][idx], b_minus[b][idx]], -1))
        surv[b, idx], death[b, idx] = sol[:, 0], sol[:, 1]
    return surv, death, 1.0 - surv - death


def hitting_time_series(mdp, pi, gamma, K):
    """
    Hitting-time route to Proposition "Value difference and terminal defect", independent of the linear
    solves: propagate the hitting-time distribution P(tau=k), k=1..K, of policies pi (B, S, A) and return
    (sum_k gamma^(k-1) P(tau=k), sum_k P(tau=k), sum_k (1-gamma^(k-1)) P(tau=k)), each (B, S).
    """
    p_plus, p_minus, P_cont = mdp.entry_probs()
    Ppi = policy_matrix(mdp, pi, P_cont)
    w = (pi * (p_plus + p_minus)[None]).sum(-1)         # P(tau = 1 | s)
    discounted = torch.zeros_like(w)
    total = torch.zeros_like(w)
    discount = 1.0
    for _ in range(K):
        discounted += discount * w
        total += w
        w = torch.einsum('bst,bt->bs', Ppi, w)
        discount *= gamma
    return discounted, total, total - discounted


def summarize(mdp, pi, gamma):
    """
    Per-state outcome quantities of policies pi (B, S, A), as a dict of (B, S) tensors:
      V_plus, V_minus, Z = 1 - V_plus + V_minus (terminal defect), disc_term = V_plus - V_minus,
      P_survival = P(E_+), P_death = P(E_-), P_never = P(tau = inf)          (exact absorption solve)
      late_survival = E[(1 - gamma^(tau-1)) 1_{E_+}] = P(E_+) - V_plus        (>= 0)
      late_death    = E[(1 - gamma^(tau-1)) 1_{E_-}] = P(E_-) + V_minus       (>= 0)
    so that V_plus + (-V_minus) + late_survival + late_death + P_never = 1.
    """
    Vp, Vm = discounted_quantities(mdp, pi, gamma)
    surv, death, never = absorption_probabilities(mdp, pi)
    return {'V_plus': Vp, 'V_minus': Vm, 'disc_term': Vp - Vm, 'Z': 1 - Vp + Vm,
            'P_survival': surv, 'P_death': death, 'P_never': never,
            'late_survival': surv - Vp, 'late_death': death + Vm}


# ---------------------------------------------------------------------------
# gamma = 1 planning and the auxiliary control problems of the robustness certificate
# ---------------------------------------------------------------------------
def terminal_value_undiscounted(mdp, pi, alpha, beta):
    """W^pi = alpha P(E_+) - beta P(E_-) (value of R_{alpha,beta} at gamma = 1), (B, S) for pi (B, S, A)."""
    surv, death, _ = absorption_probabilities(mdp, pi)
    a = torch.as_tensor(alpha, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    b = torch.as_tensor(beta, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    return a * surv - b * death


def solve_optimal_undiscounted(mdp, alpha, beta, one_minus_gamma=(1e-9, 1e-10), atol=1e-9):
    """
    Optimal action sets (B, S, A) bool for R_{alpha,beta} at gamma = 1, and a dict describing the method
    and its residual. alpha, beta: floats or tensors of shape (B,).

    If every policy terminates almost surely, the gamma = 1 Bellman equation is solved exactly.
    Otherwise (some policy can avoid termination forever) Blackwell-optimal policies are approximated by
    exact planning at gamma = 1 - eps for decreasing eps: a finite MDP has policies that are optimal for
    all gamma close to 1, and, every return being bounded by one terminal reward, they are optimal at
    gamma = 1. Ties are resolved at a tolerance of 1e-3 eps, and the O(eps) residual is reported.
    """
    alpha = torch.as_tensor(alpha, dtype=DTYPE, device=DEVICE).reshape(-1)
    beta = torch.as_tensor(beta, dtype=DTYPE, device=DEVICE).reshape(-1)
    r = mdp.rewards(alpha, beta)
    P_cont = mdp.entry_probs()[2]
    _, safe = sup_never(mdp)
    if not safe.any():
        # every policy terminates almost surely: the gamma = 1 Bellman equation has a unique solution,
        # so solve it exactly (policy iteration) and take the greedy action sets at gamma = 1
        nt = mdp.nonterm & mdp.mask.any(1)
        V = torch.stack([_policy_iteration_total(mdp, r[b], nt, maximize=True) for b in range(len(alpha))])
        Q1 = _bellman_Q(mdp, r, V, 1.0, P_cont)
        sets = optimal_sets(mdp, Q1, atol=atol)
        return sets, {'method': 'exact gamma = 1 (all policies proper)',
                      'max improvement at gamma=1': float((Q1.max(-1).values - V).masked_fill(
                          ~mdp.mask.any(1)[None], 0.0).max())}
    sets = []
    for eps in one_minus_gamma:
        Q, _ = solve_optimal(mdp, r, 1.0 - eps)
        # ties at gamma -> 1 differ by O(eps); compare at a tolerance well below eps but above round-off
        sets.append(optimal_sets(mdp, Q, atol=min(atol, 1e-3 * eps)))
    agree = bool(torch.equal(sets[0], sets[1]))
    pi = uniform_over(sets[-1])
    W = terminal_value_undiscounted(mdp, pi, alpha, beta)
    Q1 = _bellman_Q(mdp, r, W, 1.0, P_cont)
    residual = (Q1.max(-1).values - W).masked_fill(~mdp.mask.any(1)[None], 0.0).max().item()
    return sets[-1], {'method': f'Blackwell-optimal approximated at gamma = 1 - {one_minus_gamma[-1]:g} '
                                f'(some policy can avoid termination)',
                      'sets agree at both gammas': agree, 'max improvement at gamma=1': residual,
                      'approximation error scale': one_minus_gamma[-1]}


def sup_defect(mdp, gamma):
    """sup_mu Z^mu(s) = 1 - min_mu E[gamma^(tau-1) 1{tau<inf}], shape (S,): an auxiliary control problem."""
    p_plus, p_minus, _ = mdp.entry_probs()
    _, V = solve_optimal(mdp, -(p_plus + p_minus)[None], gamma)
    return 1.0 + V[0]


def _policy_iteration_total(mdp, r, rows, maximize, max_iter=1000):
    """
    Exact undiscounted policy iteration on the state subset `rows` for x = opt_a [r(s,a) + P_cont x],
    with x = 0 outside `rows`; r (S, A), rows (S,) bool, returns x (S,). Valid when every stationary
    policy leaves `rows` almost surely (then I - P_pi restricted to `rows` is non-singular), as guaranteed
    by the callers.
    """
    _, _, P_cont = mdp.entry_probs()
    idx = torch.nonzero(rows).squeeze(-1)
    P_sub = P_cont[idx][:, :, idx]                           # (n, A, n)
    r_sub, mask_sub = r[idx], mdp.mask[idx]
    fill = -torch.inf if maximize else torch.inf
    sign = 1.0 if maximize else -1.0
    act = torch.where(mask_sub, sign * r_sub, torch.full_like(r_sub, -torch.inf)).argmax(1)
    I = torch.eye(len(idx), dtype=DTYPE, device=DEVICE)
    for _ in range(max_iter):
        Ppi = P_sub[torch.arange(len(idx)), act]
        x = torch.linalg.solve(I - Ppi, r_sub[torch.arange(len(idx)), act])
        q = torch.where(mask_sub, r_sub + torch.einsum('sat,t->sa', P_sub, x), torch.full_like(r_sub, fill))
        current = q[torch.arange(len(idx)), act]
        best = q.max(1).values if maximize else q.min(1).values
        better = (sign * (best - current)) > 1e-12 * (1 + current.abs())
        if not better.any():
            out = torch.zeros(mdp.S, dtype=DTYPE, device=DEVICE)
            out[idx] = x
            return out
        arg = q.argmax(1) if maximize else q.argmin(1)
        act = torch.where(better, arg, act)
    raise RuntimeError('policy iteration did not converge')


def sup_never(mdp):
    """
    (epsilon, safe), each (S,). epsilon(s) = sup_mu P_s(tau = inf) = 1 - (minimal probability of reaching
    the terminal set). `safe` = states that can avoid the terminal set surely (greatest fixed point); they
    have epsilon = 1. On the other states the minimal reach probability is the unique solution of its
    Bellman equation with zero on `safe` (Baier & Katoen, Principles of Model Checking, Sec. 10.6), solved
    exactly.
    """
    p_plus, p_minus, P_cont = mdp.entry_probs()
    nt = mdp.nonterm & mdp.mask.any(1)
    safe = nt.clone()
    while True:   # keep states with an action that stays inside `safe` with probability one
        stay = (P_cont * safe.to(DTYPE)[None, None, :]).sum(-1)
        shrunk = safe & ((stay > 1 - 1e-12) & mdp.mask).any(1)
        if torch.equal(shrunk, safe):
            break
        safe = shrunk
    reach = _policy_iteration_total(mdp, p_plus + p_minus, nt & ~safe, maximize=False)
    return torch.where(nt, 1.0 - reach, torch.zeros_like(reach)), safe


def sup_time(mdp):
    """
    T(s) = sup_mu E[(tau - 1) 1{tau < inf}], shape (S,), when every policy terminates almost surely (no
    `safe` states): the maximal expected number of transitions between non-terminal states before
    absorption, solved exactly by policy iteration at gamma = 1. Returns None if some policy may never
    terminate.
    """
    _, safe = sup_never(mdp)
    if safe.any():
        return None
    _, _, P_cont = mdp.entry_probs()
    nt = mdp.nonterm & mdp.mask.any(1)
    return _policy_iteration_total(mdp, P_cont.sum(-1), nt, maximize=True)


def at_start(mdp, x):
    """Average a per-state quantity (B, S) under the initial distribution d0 -> (B,)."""
    return (x * mdp.d0[None]).sum(-1)
