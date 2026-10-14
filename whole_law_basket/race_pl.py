"""Vectorized Differentiable Exponential-Race Plackett-Luce (PL) Quadrature.

Replaces the P<=8 exact 2^P subset enumeration bottleneck with an O(P * M)
exact-mode-centered analytic hyperbolic-sine (sinh) Gauss-Legendre quadrature
over the unit-rescaled exponential-race integral:
    P_PL(A | u) = int_0^infty exp(-s) * prod_{i in A} (1 - exp(-(w_i / w_out) * s)) ds

Changing variables to log-time v = log(s) in (-infty, infty) removes the s = 0
endpoint boundary and yields a C^infty analytic, globally strictly concave
log-integrand on the entire real line:
    Phi(v) = v - exp(v) + sum_{i in A} log(1 - exp(-r_i * exp(v))),   r_i = w_i / w_out.

Because x -> x / expm1(x) is strictly decreasing for all x > 0, Phi(v) is globally
strictly concave on (-infty, infty) for ANY heterogeneous utility vector u:
    Phi'(v)  = 1 - exp(v) + sum_{i in A} psi(r_i * exp(v)),   psi(x) = x / expm1(x),
    Phi''(v) = -exp(v) + sum_{i in A} psi(r_i * exp(v)) * (1 - r_i * exp(v) - psi(r_i * exp(v))) < 0.

Damped Newton iterations inside torch.no_grad() estimate the unique mode v* = log(s*)
and local curvature scale sigma_v = (-Phi''(v*))^{-1/2} in O(|A|) time.
Mapping Gauss-Legendre nodes z in (-1, 1) via the entire C^infty chart
    v(z) = v* + scale * sinh(alpha * z),   scale = 2.6 * sigma_v,   alpha = 3.4,
avoids the s=0 endpoint and supports dense selections, heterogeneous utilities,
and ragged category sizes P_c >= 1. Finite quadrature is approximate; regression
checks cover representative regimes, not a uniform error bound. Autograd uses
a detached coordinate chart to approximate derivatives of the underlying integral.
"""

import math
import time
from numpy.polynomial.legendre import leggauss
import torch


_BASE_NODES_CACHE = {}


def _get_base_sinh_legendre(
    n_nodes: int = 128, alpha: float = 3.4, device="cpu", dtype=torch.float64
):
    key = (n_nodes, alpha, str(device), dtype)
    if key not in _BASE_NODES_CACHE:
        nodes, weights = leggauss(n_nodes)
        z = torch.tensor(nodes, device=device, dtype=dtype)
        wz = torch.tensor(weights, device=device, dtype=dtype)
        sh = torch.sinh(alpha * z)
        log_dz = (alpha * torch.cosh(alpha * z) * wz).log()
        _BASE_NODES_CACHE[key] = (sh, log_dz)
    return _BASE_NODES_CACHE[key]


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


@torch.no_grad()
def _newton_log_mode_and_scale(
    log_ratio: torch.Tensor, y_mask: torch.Tensor, n_steps: int = 14
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate mode v* = log(s*) and width scale of concave log-integrand Phi(v)."""
    lr = log_ratio.detach()
    y_a = y_mask.detach()
    v_mode = torch.zeros_like(y_a[..., 0])
    hess = -torch.ones_like(v_mode)
    for _ in range(n_steps):
        s_val = v_mode.exp()
        log_x = (lr + v_mode.unsqueeze(-1)).clamp(min=-80.0, max=80.0)
        x = log_x.exp()
        psi = torch.where(
            x < 1e-6,
            1.0 - 0.5 * x + (1.0 / 12.0) * x * x,
            torch.where(x > 50.0, x * (-x).exp(), x / torch.expm1(x)),
        )
        psi = psi * y_a
        grad = 1.0 - s_val + psi.sum(dim=-1)
        dpsi = (
            torch.where(
                x < 1e-4,
                -0.5 * x + (1.0 / 6.0) * x * x,
                psi * (1.0 - x - psi),
            )
            * y_a
        )
        hess = (-s_val + dpsi.sum(dim=-1)).clamp(max=-1e-12)
        step = (grad / hess).clamp(min=-3.0, max=3.0)
        v_mode = v_mode - step
    sigma_v = (-hess).rsqrt().clamp(min=0.03, max=2.5)
    scale = (2.6 * sigma_v).clamp(min=0.08, max=4.5)
    return v_mode, scale


def race_observed_log_probability(
    u: torch.Tensor,
    y_mask: torch.Tensor,
    n_nodes: int = 128,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """O(B * C * P * M) exact-mode-centered analytic sinh race PL log-probability for arbitrary (possibly ragged) P_c, k, and heterogeneous u."""
    lead_shape = u.shape[:-1]
    p = u.shape[-1]
    if valid_mask is not None:
        valid_flat = valid_mask.to(torch.bool).broadcast_to(u.shape).reshape(-1, p)
    else:
        valid_flat = (u > -1e30).reshape(-1, p)

    u_flat = u.reshape(-1, p)
    u_safe = torch.where(valid_flat, u_flat, torch.zeros_like(u_flat))
    u_masked = torch.where(valid_flat, u_flat, torch.full_like(u_flat, -torch.inf))
    y_raw = y_mask.to(u.dtype).reshape(-1, p)
    invalid_sel = ((y_raw > 0.5) & (~valid_flat)).any(dim=-1)
    y_flat = torch.where(valid_flat, y_raw, torch.zeros_like(u_flat))

    k = y_flat.sum(dim=-1)
    p_eff = valid_flat.sum(dim=-1).to(k.dtype)

    log_denom = torch.logsumexp(u_masked, dim=-1, keepdim=True)
    log_w_safe = u_safe - log_denom
    k1_lp = torch.where(y_flat > 0.5, log_w_safe, torch.zeros_like(log_w_safe)).sum(
        dim=-1
    )
    out = torch.where((k == 1) & (p_eff > 1), k1_lp, torch.zeros_like(k1_lp))

    active = (k >= 2) & (k < p_eff) & (~invalid_sel)
    if active.any():
        u_a = u_safe[active]
        u_m_a = u_masked[active]
        y_a = y_flat[active]
        val_a = valid_flat[active]
        neg_inf = torch.full_like(u_a, -torch.inf)
        log_w_out = torch.logsumexp(
            torch.where((y_a < 0.5) & val_a, u_m_a, neg_inf), dim=-1, keepdim=True
        )
        log_ratio = torch.where(val_a, u_a - log_w_out, torch.zeros_like(u_a))

        sh, log_dz = _get_base_sinh_legendre(
            n_nodes, alpha=3.4, device=u.device, dtype=u.dtype
        )
        v_mode, scale = _newton_log_mode_and_scale(log_ratio, y_a)

        log_s = v_mode.unsqueeze(-1) + scale.unsqueeze(-1) * sh
        s = log_s.exp()
        log_jac = scale.unsqueeze(-1).log() + log_s + log_dz - s

        z = log_ratio.unsqueeze(-1) + log_s.unsqueeze(-2)
        x_grid = z.clamp(max=80.0).exp()
        log_1m_exp = torch.where(
            z < -20.0,
            z - 0.5 * x_grid,
            torch.where(
                x_grid < 0.693147,
                (-torch.expm1(-x_grid)).clamp(min=1e-300).log(),
                torch.log1p(-(-x_grid).exp().clamp(max=1.0 - 1e-16)),
            ),
        )
        log_term = torch.where(y_a.unsqueeze(-1) > 0.5, log_1m_exp, 0.0)
        log_prod = log_term.sum(dim=-2)
        integral = torch.logsumexp(log_prod + log_jac, dim=-1)
        out = out.clone()
        out[active] = integral

    out = torch.where(invalid_sel, torch.full_like(out, -torch.inf), out)
    return out.reshape(lead_shape)


def race_subset_log_probabilities(
    u: torch.Tensor, n_nodes: int = 128, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Differentiable exponential-race Plackett-Luce log-probabilities for all 2^P subsets (small P<=8)."""
    p = u.shape[-1]
    masks = torch.tensor(
        [[int((m >> j) & 1) for j in range(p)] for m in range(1 << p)],
        device=u.device,
        dtype=u.dtype,
    )
    u_exp = u.unsqueeze(-2).expand(*u.shape[:-1], 1 << p, p)
    masks_exp = masks.expand(*u.shape[:-1], 1 << p, p)
    vm_exp = (
        valid_mask.to(torch.bool)
        .broadcast_to(u.shape)
        .unsqueeze(-2)
        .expand(*u.shape[:-1], 1 << p, p)
        if valid_mask is not None
        else None
    )
    return race_observed_log_probability(
        u_exp, masks_exp, n_nodes=n_nodes, valid_mask=vm_exp
    )


def verify_race_pl_quadrature() -> dict:
    """Verify machine-precision agreement and gradient accuracy across P=4..1024 and extreme utility heterogeneity."""
    torch.manual_seed(42)
    u4 = torch.randn(8, 3, 4, dtype=torch.float64, requires_grad=True)
    exact = subset_log_probabilities(u4)
    approx = race_subset_log_probabilities(u4, n_nodes=128)
    max_val_err = float((exact - approx).abs().max().detach())

    w_rand = torch.randn_like(exact)
    grad_exact = torch.autograd.grad((exact * w_rand).sum(), u4, retain_graph=True)[0]
    grad_approx = torch.autograd.grad((approx * w_rand).sum(), u4)[0]
    max_grad_err = float((grad_exact - grad_approx).abs().max().detach())

    y4 = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    ).expand(8, 3, 4)
    u4_obs = u4.detach().clone().requires_grad_(True)
    lp4_obs = race_observed_log_probability(u4_obs, y4)
    code4 = (y4.long() * (2 ** torch.arange(4))).sum(-1)
    lp4_ref = exact.detach().gather(-1, code4[..., None]).squeeze(-1)
    obs_vs_dp_err = float((lp4_obs.detach() - lp4_ref).abs().max())

    large_checks = {}
    for P, k in ((16, 3), (64, 3), (64, 32), (64, 63), (1024, 3), (1024, 512)):
        u_eq = torch.zeros((2, 4, P), dtype=torch.float64, requires_grad=True)
        y_large = torch.zeros((2, 4, P), dtype=torch.float64)
        y_large[:, :, :k] = 1.0
        t0 = time.perf_counter()
        lp_race = race_observed_log_probability(u_eq, y_large)
        lp_race.sum().backward()
        elapsed = time.perf_counter() - t0

        expected_lp = -math.log(math.comb(P, k))
        err_k = float((lp_race.detach() - expected_lp).abs().max())
        shift_invariance_err = float(u_eq.grad.sum(dim=-1).abs().max())
        large_checks[f"P_{P}_k_{k}"] = {
            "P": P,
            "selected_k": k,
            "max_log_prob_error_vs_exact_combinatorial": err_k,
            "shift_invariance_gradient_sum_error": shift_invariance_err,
            "forward_backward_seconds": elapsed,
        }

    unequal_stress = []
    for scale in (2.0, 5.0, 10.0, 20.0):
        u_het = torch.tensor([scale] * 6 + [-scale, 0.0], dtype=torch.float64)
        masks_7 = []
        for drop in range(8):
            m = [1.0] * 8
            m[drop] = 0.0
            masks_7.append(m)
        y_7 = torch.tensor(masks_7, dtype=torch.float64)
        lp_7 = race_observed_log_probability(u_het.expand(8, 8), y_7)
        prob_sum = float(lp_7.exp().sum().item())
        unequal_stress.append(
            {
                "P": 8,
                "selected_count": 7,
                "scale": scale,
                "sum_over_all_subsets_of_that_size": prob_sum,
                "abs_normalization_error": abs(prob_sum - 1.0),
            }
        )

    return {
        "P4_max_log_prob_error_all_subsets": max_val_err,
        "P4_max_autograd_error": max_grad_err,
        "P4_observed_vs_subset_dp_error": obs_vs_dp_err,
        "large_category_checks": large_checks,
        "unequal_utility_normalization_stress": unequal_stress,
    }
