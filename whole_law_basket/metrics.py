"""Unified Evaluation Metrics and L-BFGS-B Fitting Harness for Whole-Law Basket Models."""

import time
import numpy as np
from scipy.optimize import minimize
import torch


def subset(ctx: dict, idx) -> dict:
    return {k: v[idx] for k, v in ctx.items()}


def gradient_norm(x: np.ndarray, grad: np.ndarray, bounds: list[tuple[float, float]]) -> float:
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    projected = np.where((x <= low + 1e-12) & (grad > 0), 0.0, grad)
    projected = np.where((x >= high - 1e-12) & (projected < 0), 0.0, projected)
    return float(np.max(np.abs(projected))) if len(projected) else 0.0


def fit_lbfgsb(
    model: torch.nn.Module,
    train: tuple[dict, torch.Tensor],
    validation: tuple[dict, torch.Tensor],
    ridge: float = 10.0,
    maxiter: int = 50,
    dense_scale: float = 15.0,
) -> dict:
    """Fit any baseline or proposed model with preconditioned L-BFGS-B and validation checkpoint selection."""
    t0 = time.perf_counter()
    ctx_tr, y_tr = train
    ctx_va, y_va = validation
    n = len(y_tr)
    users = int(model.config.get("users", int(ctx_tr["house"].max().item()) + 1))
    frequencies = (
        torch.bincount(ctx_tr["house"], minlength=users).double() / n
    )
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params:
        with torch.no_grad():
            va_ll = float(model.log_probability(ctx_va, y_va).mean().item())
        return {
            "best_validation_ll": va_ll,
            "final_raw_projected_gradient": 0.0,
            "iterations": 0,
            "seconds": time.perf_counter() - t0,
        }

    scale_parts = []
    for name, p in params:
        if p.ndim >= 1 and p.shape[0] == users and ("house" in name or "need" in name or "weights" in name or "user" in name):
            sh = (users,) + (1,) * (p.ndim - 1)
            s = (0.15 * frequencies.reshape(sh) + ridge / n).rsqrt().expand_as(p)
        else:
            s = torch.ones_like(p) * dense_scale
        scale_parts.append(s.numpy().ravel())
    scales = np.concatenate(scale_parts)

    raw_bounds = []
    for name, p in params:
        low, high = (0.0, 8.0) if "price" in name and "cross" not in name and "slope" not in name and "mix" not in name and "cat" not in name else (-15.0, 15.0)
        raw_bounds.extend([(low, high)] * p.numel())
    bounds = [(b[0] / s, b[1] / s) for b, s in zip(raw_bounds, scales)]

    def assign(v):
        raw = v * scales
        off = 0
        with torch.no_grad():
            for _, p in params:
                num = p.numel()
                p.copy_(torch.from_numpy(raw[off : off + num].reshape(p.shape)))
                off += num

    def objective(v):
        assign(v)
        model.zero_grad(set_to_none=True)
        ll = model.log_probability(ctx_tr, y_tr)
        pen = sum(
            p.square().sum()
            for name, p in params
            if not any(k in name for k in ("lam", "kappa", "size_lam", "node_bias"))
        )
        loss = -ll.mean() + (ridge * pen / (2.0 * n) if isinstance(pen, torch.Tensor) else 0.0)
        loss.backward()
        raw_grad = np.concatenate([p.grad.detach().numpy().ravel() for _, p in params])
        return float(loss.detach()), raw_grad * scales

    v0 = np.concatenate([p.detach().numpy().ravel() for _, p in params]) / scales
    v0 = np.clip(
        v0,
        [b[0] for b in bounds],
        [b[1] for b in bounds],
    )
    init_loss, _ = objective(v0)
    with torch.no_grad():
        best_val_ll = float(model.log_probability(ctx_va, y_va).mean().item())
    best_v = v0.copy()

    def callback(v):
        nonlocal best_val_ll, best_v
        assign(v)
        with torch.no_grad():
            va_ll = float(model.log_probability(ctx_va, y_va).mean().item())
        if va_ll > best_val_ll:
            best_val_ll = va_ll
            best_v = v.copy()

    res = minimize(
        objective,
        v0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": maxiter, "maxfun": maxiter * 2, "maxcor": 15, "ftol": 1e-13, "gtol": 1e-7},
    )
    final_loss, final_scaled_grad = objective(res.x)
    raw_grad = final_scaled_grad / scales
    raw_norm = gradient_norm(res.x * scales, raw_grad, raw_bounds)
    assign(best_v)
    return {
        "initial_objective": init_loss,
        "final_objective": final_loss,
        "best_validation_ll": best_val_ll,
        "final_raw_projected_gradient": raw_norm,
        "iterations": int(res.nit),
        "seconds": time.perf_counter() - t0,
    }


@torch.no_grad()
def draws(model, ctx: dict, n_draws: int, seed: int, batch: int = 128) -> np.ndarray:
    n = len(ctx["house"])
    C, P = int(model.C), int(model.P)
    out = np.empty((n, n_draws, C, P), dtype=np.uint8)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sub = subset(ctx, slice(start, end))
        for d in range(n_draws):
            out[start:end, d] = (
                model.sample(sub, seed + d * 997 + start).cpu().numpy().astype(np.uint8)
            )
    return out


def size_distribution(baskets: np.ndarray, max_size: int) -> np.ndarray:
    sizes = baskets.reshape(-1, baskets.shape[-2] * baskets.shape[-1]).sum(-1)
    counts = np.bincount(sizes.astype(int), minlength=max_size + 1)[1 : max_size + 1]
    return counts / max(counts.sum(), 1)


def generation_metrics(
    generated: np.ndarray, observed: np.ndarray, households: np.ndarray
) -> dict:
    n, m, C, P = generated.shape
    J = C * P
    gen_flat = generated.reshape(n * m, J).astype(float)
    obs_flat = observed.reshape(n, J).astype(float)
    gen_size = size_distribution(generated, J)
    obs_size = size_distribution(observed[:, None], J)
    gen_cat = (generated.sum(-1) > 0).reshape(n * m, C).astype(float)
    obs_cat = (observed.sum(-1) > 0).reshape(n, C).astype(float)
    gen_pairs = (gen_cat.T @ gen_cat) / len(gen_cat)
    obs_pairs = (obs_cat.T @ obs_cat) / len(obs_cat)
    upper = np.triu_indices(C, 1)
    gen_mult = (generated.sum(-1) >= 2).reshape(n * m, C).mean(0)
    obs_mult = (observed.sum(-1) >= 2).reshape(n, C).mean(0)

    gen_row_sizes = generated.reshape(n, m, J).sum(-1).mean(-1)
    obs_row_sizes = obs_flat.sum(-1)
    uniq = np.unique(households)
    hh_errors = [
        abs(gen_row_sizes[households == h].mean() - obs_row_sizes[households == h].mean())
        for h in uniq
    ]
    return {
        "product_frequency_MAE": float(np.abs(gen_flat.mean(0) - obs_flat.mean(0)).mean()),
        "size_TV": float(0.5 * np.abs(gen_size - obs_size).sum()),
        "mean_size": float(gen_flat.sum(-1).mean()),
        "observed_mean_size": float(obs_flat.sum(-1).mean()),
        "category_pair_MAE": float(np.abs(gen_pairs[upper] - obs_pairs[upper]).mean()) if C > 1 else 0.0,
        "within_category_multiple_MAE": float(np.abs(gen_mult - obs_mult).mean()),
        "shopper_mean_size_MAE": float(np.mean(hh_errors)),
    }


def predictive_metrics(generated: np.ndarray, observed: np.ndarray) -> dict:
    n, m, C, P = generated.shape
    J = C * P
    probs = generated.reshape(n, m, J).mean(1)
    y = observed.reshape(n, J).astype(float)
    top5 = np.argsort(-probs, axis=1, kind="stable")[:, :5]
    hits5 = np.take_along_axis(y, top5, axis=1).sum(1)
    recall5 = float((hits5 / np.maximum(y.sum(1), 1.0)).mean())
    mc_var = probs * (1.0 - probs) / max(m - 1, 1)
    brier = float(((probs - y) ** 2 - mc_var).mean())
    return {
        "context_recall_5": recall5,
        "Brier_MC_corrected": brier,
    }


def cluster_interval(
    values: np.ndarray, clusters: np.ndarray, draws_count: int = 1000, seed: int = 942981
) -> dict:
    uniq, inv = np.unique(clusters, return_inverse=True)
    totals = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws_count, dtype=float)
    for b in range(draws_count):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        boot[b] = totals[idx].sum() / counts[idx].sum()
    return {
        "mean_gain": float(values.mean()),
        "cluster_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "cluster_count": int(len(uniq)),
    }
