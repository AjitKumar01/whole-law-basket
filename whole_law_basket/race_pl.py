"""Vectorized Differentiable Exponential-Race Plackett-Luce (PL) Quadrature.

Replaces the P<=8 exact 2^P subset enumeration bottleneck with an O(P * M)
rational Gauss-Legendre quadrature over the unit-rescaled exponential-race integral:
    P_PL(A | u) = int_0^infty exp(-s) * prod_{i in A} (1 - exp(-(w_i / w_out) * s)) ds
with substitution s = (t / (1 - t))^2 on t in (0, 1).
"""

import math
import time
from numpy.polynomial.legendre import leggauss
import torch


_GRID_CACHE = {}


def _quadrature_grid(n_nodes: int = 64, device="cpu", dtype=torch.float64):
    nodes, weights = leggauss(n_nodes)
    t = torch.tensor((nodes + 1.0) / 2.0, device=device, dtype=dtype)
    wq = torch.tensor(weights / 2.0, device=device, dtype=dtype)
    s = (t / (1.0 - t)).square()
    log_jac = (2.0 * t / (1.0 - t).pow(3) * wq).log() - s
    return s, log_jac


def get_grid(n_nodes: int = 64, device="cpu", dtype=torch.float64):
    key = (n_nodes, str(device), dtype)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = _quadrature_grid(n_nodes, device, dtype)
    return _GRID_CACHE[key]


def subset_log_probabilities(u: torch.Tensor) -> torch.Tensor:
    """Exact unordered PL subset log-probabilities via 2^P dynamic programming (for P<=8 reference)."""
    p = u.shape[-1]
    probabilities = [torch.zeros_like(u[..., 0])]
    for mask in range(1, (1 << p) - 1):
        terms = []
        for j in range(p):
            if (mask >> j) & 1:
                parent = mask ^ (1 << j)
                remaining = [i for i in range(p) if not (parent >> i) & 1]
                terms.append(
                    probabilities[parent]
                    + u[..., j]
                    - torch.logsumexp(u[..., remaining], -1)
                )
        probabilities.append(torch.logsumexp(torch.stack(terms), 0))
    probabilities.append(torch.zeros_like(u[..., 0]))
    return torch.stack(probabilities, -1)


def race_subset_log_probabilities(u: torch.Tensor, n_nodes: int = 64) -> torch.Tensor:
    """Differentiable exponential-race Plackett-Luce log-probabilities for all 2^P subsets."""
    p = u.shape[-1]
    masks = torch.tensor(
        [[int((m >> j) & 1) for j in range(p)] for m in range(1 << p)],
        device=u.device,
        dtype=u.dtype,
    )
    w = (u - torch.logsumexp(u, dim=-1, keepdim=True)).exp()
    w_unsel = w.unsqueeze(-2) * (1.0 - masks)
    w_out = w_unsel.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    ratio = w.unsqueeze(-2) / w_out
    s, log_jac = get_grid(n_nodes, u.device, u.dtype)
    term = -torch.expm1(-ratio.unsqueeze(-1) * s)
    log_term = torch.where(
        masks.unsqueeze(-1) > 0.5, term.clamp(min=1e-300).log(), 0.0
    )
    log_prod = log_term.sum(dim=-2)
    integral = torch.logsumexp(log_prod + log_jac, dim=-1)
    full_or_empty = (masks.sum(dim=-1) == 0) | (masks.sum(dim=-1) == p)
    return torch.where(full_or_empty, 0.0, integral)


def race_observed_log_probability(
    u: torch.Tensor, y_mask: torch.Tensor, n_nodes: int = 64
) -> torch.Tensor:
    """O(B * C * P * Q) differentiable race PL log-probability for arbitrary category width P."""
    p = u.shape[-1]
    w = (u - torch.logsumexp(u, dim=-1, keepdim=True)).exp()
    w_unsel = w * (1.0 - y_mask)
    w_out = w_unsel.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    ratio = w / w_out
    s, log_jac = get_grid(n_nodes, u.device, u.dtype)
    term = -torch.expm1(-ratio.unsqueeze(-1) * s)
    log_term = torch.where(
        y_mask.unsqueeze(-1) > 0.5, term.clamp(min=1e-300).log(), 0.0
    )
    log_prod = log_term.sum(dim=-2)
    integral = torch.logsumexp(log_prod + log_jac, dim=-1)
    k = y_mask.sum(dim=-1)
    full_or_empty = (k == 0) | (k == p)
    return torch.where(full_or_empty, torch.zeros_like(integral), integral)


def verify_race_pl_quadrature() -> dict:
    """Verify machine-precision agreement and gradient accuracy at P=4, 16, 64."""
    torch.manual_seed(42)
    u4 = torch.randn(8, 3, 4, dtype=torch.float64, requires_grad=True)
    exact = subset_log_probabilities(u4)
    approx = race_subset_log_probabilities(u4, n_nodes=64)
    max_val_err = float((exact - approx).abs().max().detach())

    w_rand = torch.randn_like(exact)
    grad_exact = torch.autograd.grad((exact * w_rand).sum(), u4, retain_graph=True)[0]
    grad_approx = torch.autograd.grad((approx * w_rand).sum(), u4)[0]
    max_grad_err = float((grad_exact - grad_approx).abs().max().detach())

    large_checks = {}
    for P in (16, 64):
        u_eq = torch.zeros((4, 16, P), dtype=torch.float64, requires_grad=True)
        y_large = torch.zeros((4, 16, P), dtype=torch.float64)
        y_large[:, :, :3] = 1.0
        t0 = time.perf_counter()
        lp_race = race_observed_log_probability(u_eq, y_large, n_nodes=64)
        lp_race.sum().backward()
        elapsed = time.perf_counter() - t0

        expected_lp = -math.log(math.comb(P, 3))
        err_k3 = float((lp_race.detach() - expected_lp).abs().max())
        shift_invariance_err = float(u_eq.grad.sum(dim=-1).abs().max())
        large_checks[f"P_{P}"] = {
            "selected_k": 3,
            "max_log_prob_error_vs_exact_combinatorial": err_k3,
            "shift_invariance_gradient_sum_error": shift_invariance_err,
            "forward_backward_seconds_batch4x16": elapsed,
        }

    return {
        "P4_max_log_prob_error_all_subsets": max_val_err,
        "P4_max_autograd_error": max_grad_err,
        "large_category_checks": large_checks,
    }
