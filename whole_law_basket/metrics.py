"""Fitting, final-weight convergence certificates, and basket evaluation metrics."""

import time
import numpy as np
from scipy.optimize import minimize
import torch
from .layout import CategoryLayout

STATIONARITY_TOL = 2e-4


def subset(ctx: dict, idx) -> dict:
    return {k: v[idx] for k, v in ctx.items()}


def gradient_norm(x, grad, bounds) -> float:
    """Infinity norm of the gradient projected onto the feasible box."""
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    projected = np.where((x <= low + 1e-12) & (grad > 0), 0.0, grad)
    projected = np.where((x >= high - 1e-12) & (projected < 0), 0.0, projected)
    return float(np.max(np.abs(projected))) if len(projected) else 0.0


class FittingObjective:
    """One definition of the loss, bounds, and scaling for fitting AND certification.

    Constructing or certifying this object never changes model weights. Frozen
    parameters are excluded from stationarity checks and listed in the record.
    """

    def __init__(self, model, train, validation, ridge=15.0, dense_scale=1.5):
        self.model = model
        self.train, self.validation = train, validation
        self.ridge, self.dense_scale = ridge, dense_scale
        ctx, y = train
        self.n = len(y)
        if not self.n:
            raise ValueError("Training data must contain at least one basket")
        if hasattr(model, "layout"):
            for name, (_, baskets) in zip(
                ("training", "validation"), (train, validation)
            ):
                if not model.layout.valid_baskets(baskets).all():
                    raise ValueError(
                        f"Invalid {name} baskets: require nonempty binary active products and zero padding"
                    )
        self.params = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        mission = (
            model.config.get("model_family") == "joint_pretrip_mission_count_race_PL"
        )
        self.exempt = (
            {"lam", "need", "count_need", "count_offsets"} if mission else {"lam"}
        )
        d = ctx["dlp"].double()
        if mission:
            # Use the declared layout even for packed uniform/singleton catalogues.
            d_cp = model.layout.padded(d, finite=True)
            index = d_cp.sum(-1) / model.cat_sizes_f
            price_mom = d_cp.square().mean(0)
            price_mom = torch.where(
                model.valid_mask, price_mom, torch.ones_like(price_mom)
            )
            self.moments = {
                "price": price_mom,
                "category_price": index.square().mean(0),
                "category_cross": index.square().mean(0).roll(1),
                "category_hinge": index.clamp(min=0).square().mean(0),
                "count_price": index.square().mean(0),
            }
        else:
            if hasattr(model, "layout"):
                d = model.layout.flatten(d, finite=True)
            elif hasattr(model, "_flatten_active"):
                d = model._flatten_active(d)
            elif hasattr(model, "re_model") and hasattr(
                model.re_model, "_flatten_active"
            ):
                d = model.re_model._flatten_active(d)
            else:
                d = d.reshape(self.n, -1)
            self.moments = {
                "price": d.square().mean(0),
                "price_below": d.clamp(max=0).square().mean(0),
                "price_above": d.clamp(min=0).square().mean(0),
            }
        users = int(model.config.get("users", int(ctx["house"].max()) + 1))
        frequency = torch.bincount(ctx["house"], minlength=users).double() / self.n
        information = (
            model.fit_information(ctx) if hasattr(model, "fit_information") else {}
        )
        scales = []
        for name, p in self.params:
            if name in information:
                scale = (information[name] + ridge / self.n).clamp(min=1e-8).rsqrt()
            elif (
                p.ndim
                and p.shape[0] == users
                and any(token in name for token in ("house", "need", "weights", "user"))
            ):
                shape = (users,) + (1,) * (p.ndim - 1)
                scale = (0.15 * frequency.reshape(shape) + ridge / self.n).rsqrt()
            elif mission and name in self.moments:
                moment = self.moments[name]
                scale = (moment * (0.15 + ridge / self.n)).clamp(min=1e-5).rsqrt()
            elif name in self.moments and p.shape == self.moments[name].shape:
                scale = (
                    (0.15 * self.moments[name] + ridge / self.n).clamp(min=1e-5).rsqrt()
                )
            else:
                scale = torch.ones_like(p) * dense_scale
            scales.append(scale.expand_as(p).detach().cpu().numpy().ravel())
        self.scales = np.concatenate(scales) if scales else np.empty(0)
        if hasattr(model, "parameter_bounds"):
            all_bounds = list(model.parameter_bounds())
            self.raw_bounds = []
            offset = 0
            for _, p in model.named_parameters():
                if p.requires_grad:
                    self.raw_bounds.extend(all_bounds[offset : offset + p.numel()])
                offset += p.numel()
            if offset != len(all_bounds):
                raise ValueError("parameter_bounds does not match the model parameters")
        else:
            self.raw_bounds = []
            for name, p in self.params:
                positive = "price" in name and not any(
                    token in name for token in ("cross", "slope", "mix", "cat")
                )
                bound = (0.0, 8.0) if positive else (-15.0, 15.0)
                if name == "count_offsets":
                    bound = (-30.0, 15.0)
                self.raw_bounds.extend([bound] * p.numel())
        self.bounds = [
            (lo / s, hi / s) for (lo, hi), s in zip(self.raw_bounds, self.scales)
        ]
        self.cached = hasattr(model, "precompute_split") and hasattr(
            model, "forward_cached"
        )
        self.caches = (
            [model.precompute_split(*split) for split in (train, validation)]
            if self.cached
            else [None, None]
        )

    def vector(self):
        if not self.params:
            return np.empty(0)
        return (
            np.concatenate([p.detach().cpu().numpy().ravel() for _, p in self.params])
            / self.scales
        )

    def assign(self, vector):
        raw, offset = vector * self.scales, 0
        with torch.no_grad():
            for _, p in self.params:
                size = p.numel()
                p.copy_(
                    torch.as_tensor(
                        raw[offset : offset + size].reshape(p.shape), device=p.device
                    )
                )
                offset += size

    def log_probability(self, validation=False):
        index = int(validation)
        ctx, y = self.validation if validation else self.train
        if self.cached:
            return self.model.forward_cached(ctx, self.caches[index])
        return self.model.log_probability(ctx, y)

    def loss(self):
        custom = hasattr(self.model, "regularization_loss")
        price_names = {
            "price",
            "category_price",
            "category_cross",
            "category_hinge",
            "count_price",
        }
        penalty = self.train[1].new_zeros(())
        for name, p in self.params:
            if name in self.exempt or (custom and name in price_names):
                continue
            term = (
                (p.square() * self.moments[name]).sum()
                if name in self.moments
                else p.square().sum()
            )
            penalty = penalty + term
        if custom:
            penalty = penalty + self.model.regularization_loss(self.moments)
        return -self.log_probability().mean() + self.ridge * penalty / (2.0 * self.n)

    def evaluate(self, vector):
        self.assign(vector)
        self.model.zero_grad(set_to_none=True)
        loss = self.loss()
        loss.backward()
        gradient = np.concatenate(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p))
                .detach()
                .cpu()
                .numpy()
                .ravel()
                for _, p in self.params
            ]
        )
        if not np.isfinite(float(loss.detach())) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite training objective or gradient")
        scaled_gradient = gradient * self.scales
        self.last_vector = vector.copy()
        self.last_scaled_gradient = scaled_gradient.copy()
        return float(loss.detach()), scaled_gradient

    def certificate(self):
        """Check precisely the current weights; do not assign, clip, or optimize."""
        self.model.zero_grad(set_to_none=True)
        loss = self.loss()
        if self.params:
            loss.backward()
            raw = np.concatenate(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p))
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()
                    for _, p in self.params
                ]
            )
            vector = self.vector()
            raw_norm = gradient_norm(vector * self.scales, raw, self.raw_bounds)
            scaled_norm = gradient_norm(vector, raw * self.scales, self.bounds)
            feasible = all(
                lo - 1e-10 <= value <= hi + 1e-10
                for value, (lo, hi) in zip(vector * self.scales, self.raw_bounds)
            )
        else:
            raw_norm = scaled_norm = 0.0
            feasible = True
        with torch.no_grad():
            val = float(self.log_probability(validation=True).mean())
        return {
            "final_objective": float(loss.detach()),
            "final_validation_ll": val,
            "final_raw_projected_gradient": raw_norm,
            "final_scaled_projected_gradient": scaled_norm,
            "active_parameter_count": sum(p.numel() for _, p in self.params),
            "total_parameter_count": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameter_names": [name for name, _ in self.params],
            "frozen_parameter_names": [
                name for name, p in self.model.named_parameters() if not p.requires_grad
            ],
            "feasible": feasible,
            "stationarity_passed": bool(
                feasible
                and np.isfinite(float(loss.detach()))
                and np.isfinite(val)
                and raw_norm <= STATIONARITY_TOL
                and scaled_norm <= STATIONARITY_TOL
            ),
            "stationarity_threshold": STATIONARITY_TOL,
            "certificate_scope": "current_saved_weights_active_parameters",
        }


def certify_fit(model, train, validation, ridge=15.0, dense_scale=1.5):
    """Independently certify saved or polished weights without training."""
    return FittingObjective(model, train, validation, ridge, dense_scale).certificate()


def fit_lbfgsb(
    model,
    train,
    validation,
    ridge=15.0,
    maxiter=400,
    dense_scale=1.5,
    use_final_stationary=True,
):
    """Fit under a fixed iteration limit; report failure instead of implying convergence.

    The same dual projected-gradient threshold is checked on the actual returned
    weights, including when validation selects an earlier iterate.
    """
    started = time.perf_counter()
    objective = FittingObjective(model, train, validation, ridge, dense_scale)
    vector = objective.vector()
    best_vector, accepted = vector.copy(), vector.copy()
    with torch.no_grad():
        best_val = float(objective.log_probability(validation=True).mean())
    initial = objective.certificate()
    iterations = 0
    message = "initial weights already stationary"
    last_loss = initial["final_objective"]

    def callback(candidate):
        nonlocal accepted, best_vector, best_val, iterations
        accepted = candidate.copy()
        iterations += 1
        if not np.array_equal(candidate, getattr(objective, "last_vector", None)):
            objective.evaluate(candidate)
        gradient = objective.last_scaled_gradient
        raw = gradient_norm(
            candidate * objective.scales,
            gradient / objective.scales,
            objective.raw_bounds,
        )
        scaled = gradient_norm(candidate, gradient, objective.bounds)
        if iterations % 5 == 0 or max(raw, scaled) <= STATIONARITY_TOL:
            with torch.no_grad():
                value = float(objective.log_probability(validation=True).mean())
            if value > best_val:
                best_val, best_vector = value, candidate.copy()
        if max(raw, scaled) <= 1.9e-4:
            raise StopIteration

    if objective.params and not initial["stationarity_passed"] and maxiter > 0:
        vector = np.clip(
            vector, [b[0] for b in objective.bounds], [b[1] for b in objective.bounds]
        )
        try:
            result = minimize(
                objective.evaluate,
                vector,
                jac=True,
                method="L-BFGS-B",
                bounds=objective.bounds,
                callback=callback,
                options={
                    "maxiter": maxiter,
                    "maxfun": maxiter * 3,
                    "maxcor": 20,
                    "ftol": 1e-15,
                    "gtol": 1e-8,
                    "maxls": 30,
                },
            )
            accepted = result.x.copy()
            iterations = int(result.nit)
            message = str(result.message)
        except StopIteration:
            message = "CONVERGENCE: DUAL_PROJECTED_GRADIENT_LE_1.9E-4"
        objective.assign(accepted)
        with torch.no_grad():
            final_val = float(objective.log_probability(validation=True).mean())
        if final_val > best_val:
            best_val, best_vector = final_val, accepted.copy()
    elif not initial["stationarity_passed"]:
        message = "iteration budget is zero; certification only"

    objective.assign(accepted if use_final_stationary else best_vector)
    certificate = objective.certificate()
    return {
        **certificate,
        "initial_objective": last_loss,
        "best_validation_ll": best_val,
        "returned_state": "final" if use_final_stationary else "best_validation",
        "optimizer_message": message,
        "iterations": iterations,
        "iteration_limit": maxiter,
        "ridge": ridge,
        "dense_scale": dense_scale,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def draws(model, ctx: dict, n_draws: int, seed: int, batch: int = 2048) -> np.ndarray:
    """Sample in the model's input layout: packed active products or padded categories."""
    n = len(ctx["house"])
    layout = model.layout if hasattr(model, "layout") else CategoryLayout(model.config)
    packed = layout.ragged and ctx["dlp"].ndim == 2 and ctx["dlp"].shape[1] == layout.J
    shape = (layout.J,) if packed else (layout.C, layout.P)
    out = np.empty((n, n_draws, *shape), dtype=np.uint8)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sub = subset(ctx, slice(start, end))
        for d in range(n_draws):
            sample = model.sample(sub, seed + d * 997 + start)
            if not layout.valid_baskets(sample).all():
                raise ValueError(
                    "Sampler returned a basket outside active nonempty support"
                )
            out[start:end, d] = sample.cpu().numpy().astype(np.uint8)
    return out


def _evaluation_arrays(generated, observed, category_sizes=None):
    generated, observed = np.asarray(generated), np.asarray(observed)
    if category_sizes is None:
        if observed.ndim != 3:
            raise ValueError("Packed evaluation requires category_sizes")
        category_sizes = [observed.shape[2]] * observed.shape[1]
    layout = CategoryLayout({"category_sizes": category_sizes})
    n, m = generated.shape[:2]
    if n != len(observed) or m < 1:
        raise ValueError(
            "Generated and observed contexts must match, with at least one draw"
        )
    observed_t = torch.as_tensor(observed)
    generated_t = torch.as_tensor(generated.reshape(n * m, *generated.shape[2:]))
    if (
        not layout.valid_baskets(observed_t).all()
        or not layout.valid_baskets(generated_t).all()
    ):
        raise ValueError(
            "Evaluation baskets must be binary, nonempty, and free of padded selections"
        )
    obs = layout.flatten(observed_t).numpy().astype(float)
    gen = layout.flatten(generated_t).numpy().reshape(n, m, layout.J).astype(float)
    return layout, gen, obs


def size_distribution(
    baskets: np.ndarray, max_size: int, category_sizes=None
) -> np.ndarray:
    if category_sizes is None:
        sizes = baskets.reshape(-1, baskets.shape[-2] * baskets.shape[-1]).sum(-1)
    else:
        layout = CategoryLayout({"category_sizes": category_sizes})
        array = np.asarray(baskets)
        trailing = 2 if array.shape[-2:] == (layout.C, layout.P) else 1
        flat = torch.as_tensor(array.reshape(-1, *array.shape[-trailing:]))
        sizes = layout.flatten(flat).numpy().sum(-1)
    counts = np.bincount(sizes.astype(int), minlength=max_size + 1)[1 : max_size + 1]
    return counts / max(counts.sum(), 1)


def generation_metrics(generated, observed, households, category_sizes=None) -> dict:
    """Compare active products only, with the same results for packed/padded data."""
    layout, gen, obs = _evaluation_arrays(generated, observed, category_sizes)
    n, m, J = gen.shape
    gen_flat = gen.reshape(n * m, J)
    gen_counts = np.stack(
        [
            gen[..., layout.offsets[c] : layout.offsets[c + 1]].sum(-1)
            for c in range(layout.C)
        ],
        -1,
    )
    obs_counts = np.stack(
        [
            obs[..., layout.offsets[c] : layout.offsets[c + 1]].sum(-1)
            for c in range(layout.C)
        ],
        -1,
    )
    gen_cat, obs_cat = (gen_counts > 0).reshape(-1, layout.C).astype(float), (
        obs_counts > 0
    ).astype(float)
    gen_pairs, obs_pairs = gen_cat.T @ gen_cat / (n * m), obs_cat.T @ obs_cat / n
    upper = np.triu_indices(layout.C, 1)
    gen_size = np.bincount(gen_flat.sum(-1).astype(int), minlength=J + 1)[1:] / (n * m)
    obs_size = np.bincount(obs.sum(-1).astype(int), minlength=J + 1)[1:] / n
    gen_row_sizes, obs_row_sizes = gen.sum(-1).mean(-1), obs.sum(-1)
    households = np.asarray(households)
    hh_errors = [
        abs(
            gen_row_sizes[households == h].mean()
            - obs_row_sizes[households == h].mean()
        )
        for h in np.unique(households)
    ]
    return {
        "product_frequency_MAE": float(np.abs(gen_flat.mean(0) - obs.mean(0)).mean()),
        "size_TV": float(0.5 * np.abs(gen_size - obs_size).sum()),
        "mean_size": float(gen_flat.sum(-1).mean()),
        "observed_mean_size": float(obs.sum(-1).mean()),
        "category_pair_MAE": (
            float(np.abs(gen_pairs[upper] - obs_pairs[upper]).mean())
            if layout.C > 1
            else 0.0
        ),
        "within_category_multiple_MAE": float(
            np.abs((gen_counts >= 2).mean((0, 1)) - (obs_counts >= 2).mean(0)).mean()
        ),
        "shopper_mean_size_MAE": float(np.mean(hh_errors)),
    }


def predictive_metrics(generated, observed, category_sizes=None) -> dict:
    _, gen, y = _evaluation_arrays(generated, observed, category_sizes)
    n, m, J = gen.shape
    probs = gen.mean(1)
    top5 = np.argsort(-probs, axis=-1, kind="stable")[:, : min(5, J)]
    hits = np.take_along_axis(y, top5, axis=1).sum(-1)
    # Unbiased correction for Monte Carlo noise in predicted marginals, m > 1.
    correction = probs * (1.0 - probs) / (m - 1) if m > 1 else np.zeros_like(probs)
    return {
        "context_recall_5": float((hits / np.maximum(y.sum(1), 1.0)).mean()),
        "Brier_MC_corrected": float(((probs - y) ** 2 - correction).mean()),
    }


def cluster_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    draws_count: int = 1000,
    seed: int = 942981,
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
        "cluster_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "cluster_count": int(len(uniq)),
    }
