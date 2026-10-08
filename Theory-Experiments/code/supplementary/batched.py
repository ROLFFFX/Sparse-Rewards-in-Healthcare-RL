"""
Batched exact planning for many small MDPs at once (used by the DAG and random-MDP experiments).

Shapes: N instances, D reward designs, S states, A actions.
    P_cont (N, S, A, S)  transitions between non-terminal states (terminal columns removed/zeroed)
    p_plus, p_minus (N, S, A)  probabilities of entering S_+ / S_-
    mask   (N, S, A)  admissible actions (all False on terminal / padding states)
    r      (N, D, S, A)  expected one-step rewards;  Q (N, D, S, A),  V (N, D, S)
"""
import torch

from tabular import DEVICE, DTYPE


def design_rewards(p_plus, p_minus, alpha, beta):
    """(N, D, S, A) expected reward of R_{alpha,beta} for designs alpha, beta of shape (D,)."""
    a = torch.as_tensor(alpha, dtype=DTYPE, device=DEVICE)[None, :, None, None]
    b = torch.as_tensor(beta, dtype=DTYPE, device=DEVICE)[None, :, None, None]
    return a * p_plus[:, None] - b * p_minus[:, None]


def _bellman_q(P_cont, r, V, gamma, mask):
    """Q = r + gamma P_cont V, with -inf on inadmissible actions."""
    Q = r + gamma * torch.einsum('nsat,ndt->ndsa', P_cont, V)
    return Q.masked_fill(~mask[:, None], -torch.inf)


def _greedy_value(Q, mask):
    """max_a Q, set to 0 on states without admissible actions."""
    V = Q.max(-1).values
    return torch.where(mask.any(-1)[:, None], V, torch.zeros_like(V))


def value_iteration(P_cont, r, gamma, mask, iters=None, tol=1e-12, max_iter=200000):
    """
    Plain value iteration, returning (Q, V). With `iters` it runs a fixed number of sweeps (exact for DAGs of
    that depth, for any gamma <= 1); otherwise it runs to `tol` (gamma < 1).
    """
    V = torch.zeros(r.shape[:3], dtype=DTYPE, device=DEVICE)
    n_sweeps = iters if iters is not None else max_iter
    for _ in range(n_sweeps):
        V_new = _greedy_value(_bellman_q(P_cont, r, V, gamma, mask), mask)
        if iters is None and (V_new - V).abs().max() < tol:
            V = V_new
            break
        V = V_new
    return _bellman_q(P_cont, r, V, gamma, mask), V


def exact_optimal(P_cont, r, gamma, mask, max_pi=200):
    """(Q*, V*) for gamma < 1: value iteration, then policy iteration with exact (batched) evaluation."""
    Q, V = value_iteration(P_cont, r, gamma, mask, tol=1e-10)
    S = Q.shape[2]
    has_action = mask.any(-1)[:, None]                             # (N, 1, S)
    act = Q.argmax(-1)
    I = torch.eye(S, dtype=DTYPE, device=DEVICE)
    for _ in range(max_pi):
        pi = torch.zeros_like(Q).scatter_(-1, act.unsqueeze(-1), 1.0) * has_action[..., None]
        Ppi = torch.einsum('ndsa,nsat->ndst', pi, P_cont)
        r_pi = (pi * r.nan_to_num(neginf=0.0)).sum(-1)
        V = torch.linalg.solve(I - gamma * Ppi, r_pi.unsqueeze(-1)).squeeze(-1)
        Q = _bellman_q(P_cont, r, V, gamma, mask)
        current = Q.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        best, arg = Q.max(-1)
        # keep the current action unless another one is strictly better (avoids tie cycling)
        improve = (best > current + 1e-12) & has_action
        if not improve.any():
            return Q, V
        act = torch.where(improve, arg, act)
    raise RuntimeError('policy iteration did not converge')


def optimal_sets(Q, mask, atol=1e-9):
    """Q (N, D, S, A) -> (N, D, S, A) bool: admissible actions within atol of the best action of their state."""
    best = Q.max(-1, keepdim=True).values
    return (Q >= best - atol) & mask[:, None]
