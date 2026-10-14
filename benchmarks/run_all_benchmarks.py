"""Fresh conditional-control benchmarks and content-verified checkpoint evaluation.

Fresh fits never load reference initialization weights or historical result rows.
Real-data preference controls and screened graph topology are declared shared
inputs; their historical preparation cost is separate from correction fitting.
Resume verifies inputs, code, configuration, fitting policy, and weights, then
recomputes evaluation. Outputs go to a new directory by default so historical
results are preserved.
"""

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from whole_law_basket import (
    verify_race_pl_quadrature,
    StructuredEnergy,
    CompositeWholeLawEnergy,
    CalibratedMissionLaw,
    PooledPopularity,
    NoPriceIndependent,
    PersonalizedIndependent,
    SizeAwareESP,
    IndependentCategoriesESP,
    LatentMissionESP,
    PairwiseBinaryAR,
    LowRankBilinearAR,
    GaussianFactorMixture,
    FastNeuralNADE,
    fit_lbfgsb,
    draws,
    generation_metrics,
    predictive_metrics,
    cluster_interval,
    subset,
    World,
    SETS,
    moments,
    actions,
    response_score,
    generation,
    global_promotion,
    generate_synthetic_splits,
    load_real_dataset,
)
from whole_law_basket.metrics import certify_fit
from whole_law_basket.proposed_model import log_esp
from whole_law_basket.provenance import (
    array_hash,
    code_hash,
    dataset_manifest,
    file_hash,
    json_hash,
    model_manifest,
    restore_checkpoint,
    save_checkpoint,
    state_hash,
)

SEED = 942981
MAXITER = 400
REAL_GROUPS = {
    "g1": [
        "popularity",
        "no_price",
        "independent_parent",
        "size_aware_esp",
        "independent_categories_esp",
        "mission_esp",
    ],
    "g2": ["pairwise_ar", "low_rank_bilinear_ar", "gaussian_factor_mixture"],
    "g3": ["structured_energy", "neural_nade", "proposed_whole_law"],
}
REAL_MODELS = [name for group in REAL_GROUPS.values() for name in group]
SYNTHETIC_GROUPS = {
    "g1": REAL_GROUPS["g1"],
    "g2": [
        "pairwise_binary_ar",
        "low_rank_bilinear_ar",
        "gaussian_factor_mixture",
        "neural_nade",
    ],
    "g3": ["proposed_calibrated_mission_law", "uncalibrated_mission_law"],
}
SYNTHETIC_MODELS = [name for group in SYNTHETIC_GROUPS.values() for name in group]


def _load_synthetic_split_pair(name: str, family: str, seed: int):
    src = REPO_ROOT / "data/synthetic" / name / "data"
    if (src / "config.json").exists():
        cfg = json.loads((src / "config.json").read_text())
        splits = {}
        test_house = None
        for sp in ("train", "validation", "test"):
            with np.load(src / f"{sp}.npz", allow_pickle=False) as d:
                raw = {k: d[k] for k in d.files}
            ctx = {
                k: torch.tensor(
                    raw[k],
                    dtype=torch.long if k in ("house", "store") else torch.float64,
                )
                for k in ("house", "store", "week", "dlp", "disp", "mail")
            }
            splits[sp] = (ctx, torch.tensor(raw["member"], dtype=torch.float64))
            if sp == "test":
                test_house = raw["house"]
        return cfg, splits["train"], splits["validation"], splits["test"], test_house

    cfg, tr, va, te = generate_synthetic_splits(family, seed)
    return cfg, (tr[0], tr[1]), (va[0], va[1]), (te[0], te[1]), te[2]


@torch.no_grad()
def _compute_exact_mass(model, ctx, sets_t):
    """Evaluate an explicit set list on the same support as direct likelihood."""
    safe, valid = model.layout.safe_baskets(sets_t)
    mass = _compute_exact_mass_valid(model, ctx, safe)
    mass[:, ~valid.cpu().numpy()] = 0.0
    return mass


@torch.no_grad()
def _compute_exact_mass_valid(
    model: torch.nn.Module, ctx: dict, sets_t: torch.Tensor
) -> np.ndarray:
    if hasattr(model, "exact_mass"):
        return model.exact_mass(ctx, sets_t).cpu().numpy()
    if isinstance(
        model, (PooledPopularity, NoPriceIndependent, PersonalizedIndependent)
    ):
        u = model.utilities(ctx)
        a = F.softplus(u).sum(-1)
        norm = a + torch.log((-torch.expm1(-a)).clamp(min=1e-30))
        y_sets = model.layout.flatten(sets_t).double()
        return (u @ y_sets.T - norm[:, None]).exp().cpu().numpy()
    if isinstance(model, SizeAwareESP):
        u, q = model.terms(ctx)
        e = log_esp(u)
        y_sets = model.layout.flatten(sets_t).double()
        n_sets = y_sets.sum(-1).long()
        return (u @ y_sets.T - e[:, n_sets] + q[:, n_sets - 1]).exp().cpu().numpy()
    if isinstance(model, IndependentCategoriesESP):
        u, esp, count_lp, eta = model.terms(ctx)
        y_cat = model.layout.padded(sets_t).double()
        n = y_cat.sum(-1).long()
        x = (n > 0).double()
        cat_ll = (
            F.logsigmoid(eta) @ x.T
            + F.logsigmoid(-eta) @ (1.0 - x).T
            - torch.log((-torch.expm1(F.logsigmoid(-eta).sum(-1))).clamp(min=1e-30))[
                :, None
            ]
        )
        within = u.reshape(len(u), -1) @ y_cat.reshape(len(y_cat), -1).T
        for c in range(model.C):
            nc = n[:, c]
            xc = x[:, c]
            within = (
                within
                + (-esp[:, c, nc] + count_lp[:, c, (nc - 1).clamp(min=0)]) * xc[None, :]
            )
        return (cat_ll + within).exp().cpu().numpy()
    if isinstance(model, LatentMissionESP):
        y_cat = model.layout.padded(sets_t).double()
        n = y_cat.sum(-1).long()
        x = (n > 0).double()
        w = F.log_softmax(model.weights[ctx["house"]], dim=-1)
        comps = []
        for m in range(model.K):
            u, esp, count_lp, eta = model.terms(ctx, m)
            cat_ll = (
                F.logsigmoid(eta) @ x.T
                + F.logsigmoid(-eta) @ (1.0 - x).T
                - torch.log(
                    (-torch.expm1(F.logsigmoid(-eta).sum(-1))).clamp(min=1e-30)
                )[:, None]
            )
            within = u.reshape(len(u), -1) @ y_cat.reshape(len(y_cat), -1).T
            for c in range(model.C):
                nc = n[:, c]
                xc = x[:, c]
                within = (
                    within
                    + (-esp[:, c, nc] + count_lp[:, c, (nc - 1).clamp(min=0)])
                    * xc[None, :]
                )
            comps.append(cat_ll + within)
        return (
            torch.logsumexp(torch.stack(comps, dim=1) + w[:, :, None], dim=1)
            .exp()
            .cpu()
            .numpy()
        )
    B = len(ctx["house"])
    S = len(sets_t)
    sets_cp = model.layout.format(model.layout.flatten(sets_t), ctx["dlp"])
    out = np.empty((B, S), dtype=float)
    chunk = 16
    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        b_len = end - start
        rep_ctx = {
            k: v[start:end, None]
            .expand(b_len, S, *([-1] * (v.ndim - 1)))
            .reshape(b_len * S, *v.shape[1:])
            for k, v in ctx.items()
        }
        rep_y = (
            sets_cp.unsqueeze(0)
            .expand(b_len, S, *sets_cp.shape[1:])
            .reshape(b_len * S, *sets_cp.shape[1:])
        )
        lp = model.log_probability(rep_ctx, rep_y).reshape(b_len, S)
        out[start:end] = lp.exp().cpu().numpy()
    return out


def build_composite_model(
    dataset: str,
    config: dict,
    re_model: StructuredEnergy,
    train: tuple[dict, torch.Tensor],
) -> CompositeWholeLawEnergy:
    """Build CompositeWholeLawEnergy with price-invariant pre-trip mission prior (`allow_price_mix=False`)."""
    y_tr = train[1]
    C, J = re_model.C, re_model.J
    y_flat = re_model.layout.flatten(y_tr).double()
    pres = (re_model._category_counts(y_flat) > 0).double()

    p_centered = pres - pres.mean(0, keepdim=True)
    _, _, Vt_cat = torch.linalg.svd(p_centered[:25000], full_matrices=False)
    n_svd_c = min(6, C, len(p_centered))
    K_c = 4 + 2 * n_svd_c
    M_cat_init = torch.zeros((K_c, C), dtype=torch.float64)
    M_cat_init[:4, :] = torch.linspace(-1.2, 1.8, 4, dtype=torch.float64)[:, None]
    M_cat_init[4 : 4 + n_svd_c, :] = 2.0 * Vt_cat[:n_svd_c] + 0.3
    M_cat_init[4 + n_svd_c :, :] = -2.0 * Vt_cat[:n_svd_c] + 0.3

    y_centered = y_flat - y_flat.mean(0, keepdim=True)
    _, _, Vt_prod = torch.linalg.svd(y_centered[:25000], full_matrices=False)
    n_svd_p = 8
    M_prod_fixed = torch.cat(
        [1.8 * Vt_prod[:n_svd_p] + 0.2, -1.8 * Vt_prod[:n_svd_p] + 0.2], dim=0
    )

    model = CompositeWholeLawEnergy(
        config,
        re_model,
        M_cat_init,
        M_prod_fixed,
        freeze_house_1=(dataset == "dunnhumby"),
        allow_price_mix=False,
    )

    return model


def _targets(group, groups):
    names = [name for members in groups.values() for name in members]
    requested = names if group == "all" else groups.get(group, group.split(","))
    requested = [name.strip() for name in requested]
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return [name for name in names if name in requested]


def _write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_evaluation(
    directory, name, model, manifest, fit_record, summary, ll, arrays, diagnostics=None
):
    """Save enough evaluation output to audit scores without retraining."""
    ll_path = directory / f"{name}_test_ll.npy"
    generation_path = directory / f"{name}_evaluation.npz"
    np.save(ll_path, ll)
    np.savez_compressed(generation_path, **arrays)
    evaluation_files = [ll_path, generation_path]
    if diagnostics is not None:
        diagnostics_path = directory / f"{name}_diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        evaluation_files.append(diagnostics_path)
    save_checkpoint(
        directory,
        name,
        model,
        manifest,
        fit_record,
        summary,
        evaluation_files=evaluation_files,
    )


def _fit_or_restore(model, directory, name, manifest, train, validation, mode, policy):
    """Reuse only certified matching weights, and recertify them before use."""
    record = (
        restore_checkpoint(directory, name, model, manifest)
        if mode == "resume"
        else None
    )
    if record is not None:
        certificate = certify_fit(
            model,
            train,
            validation,
            ridge=policy["ridge"],
            dense_scale=policy["dense_scale"],
        )
        if certificate["stationarity_passed"]:
            return {**record["fit_record"], **certificate}, True
        print(
            f"  {name}: saved weights fail convergence; continuing the same objective",
            flush=True,
        )
    fit = fit_lbfgsb(
        model,
        train,
        validation,
        ridge=policy["ridge"],
        dense_scale=policy["dense_scale"],
        maxiter=policy["maxiter"],
    )
    if record is not None:
        fit["previous_attempt_seconds"] = record["fit_record"]["seconds"]
        fit["seconds"] += record["fit_record"]["seconds"]
    save_checkpoint(directory, name, model, manifest, fit)
    return fit, False


def _shared_real_inputs(dataset):
    """Declare fitted nuisance controls and graph screening, rather than hide them."""
    parent_dir = REPO_ROOT / "data" / f"{dataset}_parent"
    paths = [p for p in parent_dir.iterdir() if p.suffix in (".json", ".pt")]
    screen = REPO_ROOT / "data" / f"{dataset}_ref" / "screen.json"
    fit_path = parent_dir / "fit.json"
    historical = json.loads(fit_path.read_text()) if fit_path.exists() else {}
    return {
        "scope": "correction_models_conditional_on_bundled_preference_controls_and_screened_graph",
        "files": {
            str(p.relative_to(REPO_ROOT)): file_hash(p)
            for p in sorted(paths + [screen])
        },
        "historical_parent_fit_seconds": historical.get("seconds"),
        "historical_parent_cost_complete": False,
        "note": "Historical parent cost excludes unrecorded tuning/preparation. Not raw-data end-to-end training.",
    }


def evaluate_real_dataset(dataset, results_dir, model_group="all", mode="fresh"):
    results_dir = Path(results_dir)
    directory = results_dir / "checkpoints" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    config, price, train, validation, test, clusters = load_real_dataset(dataset)
    shared = _shared_real_inputs(dataset)
    screen = json.loads((REPO_ROOT / f"data/{dataset}_ref/screen.json").read_text())
    data = dataset_manifest(config, (train, validation, test), clusters)
    source_hash = code_hash(REPO_ROOT)
    rows_gen = np.random.default_rng(SEED + 10).choice(
        len(test[1]), min(2048, len(test[1])), replace=False
    )
    ctx_gen, observed = subset(test[0], rows_gen), test[1][rows_gen].numpy()
    evaluation_policy = {
        "method": "sample",
        "draws": 48,
        "replicas": 3,
        "contexts_hash": array_hash(rows_gen),
        "seed": SEED + 1000,
    }
    targets = _targets(model_group, REAL_GROUPS)
    # A composite's frozen base must be fitted/verified first in this protocol.
    if "proposed_whole_law" in targets and "structured_energy" not in targets:
        targets.insert(0, "structured_energy")
    energy = None
    energy_fit = None
    energy_dependency = None
    summaries, likelihoods = {}, {}

    def make_model(name):
        torch.manual_seed(SEED)
        if name == "popularity":
            return PooledPopularity(config)
        if name == "no_price":
            return NoPriceIndependent(config, has_baseline=True)
        if name == "independent_parent":
            model = PersonalizedIndependent(config, price=price)
            model.house.requires_grad_(
                False
            )  # Shared preference controls are already in baseline.
            return model
        if name == "size_aware_esp":
            return SizeAwareESP(config, price=price, train_y=train[1])
        if name == "independent_categories_esp":
            model = IndependentCategoriesESP(config, price=price)
            model.house_cat.requires_grad_(False)
            return model
        if name == "mission_esp":
            return LatentMissionESP(config, price=price, K=2, seed=SEED)
        if name == "pairwise_ar":
            return PairwiseBinaryAR(config, price=price)
        if name == "low_rank_bilinear_ar":
            return LowRankBilinearAR(config, price=price, rank=8, seed=SEED)
        if name == "gaussian_factor_mixture":
            return GaussianFactorMixture(
                config, price=price, n_factors=6, n_nodes=12, seed=SEED
            )
        if name == "neural_nade":
            model = FastNeuralNADE(config, price=price, hidden=16, seed=972301)
            model.house.requires_grad_(False)
            return model
        if name == "structured_energy":
            model = StructuredEnergy(config, screen["edges"])
            with torch.no_grad():
                model.price_below.copy_(price)
                model.price_above.copy_(price)
            model.house.requires_grad_(False)
            return model
        if name == "proposed_whole_law":
            if energy is None or not energy_fit["stationarity_passed"]:
                raise RuntimeError(
                    "Composite fitting blocked: its energy dependency is not converged"
                )
            # Copy protects the baseline model and prevents nested state loading
            # from mutating a previously fitted dependency.
            return build_composite_model(dataset, config, copy.deepcopy(energy), train)
        raise ValueError(name)

    for name in targets:
        build_started = time.perf_counter()
        model = make_model(name)
        build_seconds = time.perf_counter() - build_started
        policy = {
            "ridge": (
                10.0
                if name in ("structured_energy", "neural_nade")
                else (20.0 if dataset == "dunnhumby" else 15.0)
            ),
            "dense_scale": (
                1.0
                if name in ("proposed_whole_law", "structured_energy", "neural_nade")
                else 1.5
            ),
            "maxiter": MAXITER,
            "stationarity_threshold": 2e-4,
            "initialization": "constructor_and_training_data_only",
            "evaluation": evaluation_policy,
        }
        dependencies = {"shared_controls": shared}
        if name == "proposed_whole_law":
            dependencies["structured_energy"] = energy_dependency
        manifest = model_manifest(model, data, source_hash, policy, SEED, dependencies)
        fit, reused = _fit_or_restore(
            model, directory, name, manifest, train, validation, mode, policy
        )
        if not reused:
            fit["model_build_seconds"] = build_seconds
            fit["seconds"] += build_seconds
        if name == "structured_energy":
            energy, energy_fit = model, fit
            energy_dependency = {
                "state_hash": state_hash(model),
                "manifest_hash": json_hash(manifest),
            }
        dependency_seconds = (
            energy_fit["seconds"] if name == "proposed_whole_law" else 0.0
        )
        with torch.no_grad():
            ll = model.log_probability(*test).numpy()
            val = float(model.log_probability(*validation).mean())
        started = time.perf_counter()
        samples = np.concatenate(
            [draws(model, ctx_gen, 16, SEED + 1000 + rep * 10000) for rep in range(3)],
            axis=1,
        )
        generation_seconds = time.perf_counter() - started
        gm = generation_metrics(
            samples,
            observed,
            clusters["house"][rows_gen],
            category_sizes=model.layout.sizes,
        )
        pm = predictive_metrics(samples, observed, category_sizes=model.layout.sizes)
        summary = {
            "model": name,
            "family": model.config.get(
                "benchmark_baseline", model.config.get("model_family", name)
            ),
            "train_baskets": len(train[1]),
            "generation_draws": 48,
            "test_ll": float(ll.mean()),
            "validation_ll": val,
            "size_TV": gm["size_TV"],
            "frequency_MAE": gm["product_frequency_MAE"],
            "category_pair_MAE": gm["category_pair_MAE"],
            "within_category_multiple_MAE": gm["within_category_multiple_MAE"],
            "recall5": pm["context_recall_5"],
            "Brier": pm["Brier_MC_corrected"],
            "raw_projected_gradient": fit["final_raw_projected_gradient"],
            "scaled_projected_gradient": fit["final_scaled_projected_gradient"],
            "stationarity_passed": fit["stationarity_passed"],
            "generation_seconds": generation_seconds,
            "fit_seconds": fit["seconds"],
            "dependency_fit_seconds": dependency_seconds,
            "pipeline_fit_seconds": fit["seconds"] + dependency_seconds,
            "historical_parent_fit_seconds": shared["historical_parent_fit_seconds"],
            "end_to_end_cost_complete": False,
            "checkpoint_reused": reused,
            "checkpoint_path": str(directory / f"{name}.pt"),
        }
        _save_evaluation(
            directory,
            name,
            model,
            manifest,
            fit,
            summary,
            ll,
            {
                "samples": samples,
                "context_rows": rows_gen,
                "observed": observed,
                "house": clusters["house"][rows_gen],
            },
        )
        summaries[name], likelihoods[name] = summary, ll
        print(
            f"[{dataset}] {name}: ll={ll.mean():.6f}; converged={fit['stationarity_passed']}; reused={reused}",
            flush=True,
        )

    paired = {}
    if "proposed_whole_law" in likelihoods:
        for name, ll in likelihoods.items():
            if name == "proposed_whole_law":
                continue
            comparison = cluster_interval(
                likelihoods["proposed_whole_law"] - ll, clusters["house"]
            )
            comparison["both_models_converged"] = (
                summaries[name]["stationarity_passed"]
                and summaries["proposed_whole_law"]["stationarity_passed"]
            )
            paired[name] = comparison
            summaries[name].update(
                proposed_gain_over_model=comparison["mean_gain"],
                proposed_gain_95_ci_low=comparison["cluster_95_ci"][0],
                proposed_gain_95_ci_high=comparison["cluster_95_ci"][1],
                comparison_certified=comparison["both_models_converged"],
            )
    rows = sorted(summaries.values(), key=lambda row: row["test_ll"], reverse=True)
    output = {
        "dataset": dataset,
        "mode": mode,
        "code_hash": source_hash,
        "data_manifest": data,
        "shared_inputs": shared,
        "train_baskets": len(train[1]),
        "validation_baskets": len(validation[1]),
        "test_baskets": len(test[1]),
        "generation_contexts": len(rows_gen),
        "generation_draws_per_context": 48,
        "suite_stationarity_passed": all(row["stationarity_passed"] for row in rows),
        "summary_table": rows,
        "paired_household_bootstrap_vs_proposed": paired,
    }
    _write_table(results_dir / f"{dataset}_comparison.csv", rows)
    (results_dir / f"{dataset}_comparison.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return output


def evaluate_synthetic_benchmarks(
    results_dir, target_world="all", model_group="all", mode="fresh"
):
    results_dir = Path(results_dir)
    requested = _targets(model_group, SYNTHETIC_GROUPS)
    sets_t = torch.tensor(SETS, dtype=torch.float64)
    source_hash = code_hash(REPO_ROOT)
    rows_by_world, selections = {}, {}
    for world_name, family, seed in [
        ("missions_a", "latent_missions_gumbel", 941000),
        ("missions_b", "latent_missions_gumbel", 941001),
        ("zero_price", "latent_missions_gumbel_zero_price", 943000),
    ]:
        if target_world != "all" and target_world != world_name:
            continue
        cfg, train, validation, test, houses = _load_synthetic_split_pair(
            world_name, family, seed
        )
        data = dataset_manifest(cfg, (train, validation, test), {"house": houses})
        directory = results_dir / "checkpoints" / f"synthetic_{world_name}"
        directory.mkdir(parents=True, exist_ok=True)
        targets = list(requested)
        dependent_names = {
            "size_aware_esp",
            "independent_categories_esp",
            "mission_esp",
            "pairwise_binary_ar",
            "low_rank_bilinear_ar",
            "gaussian_factor_mixture",
        }
        if (
            dependent_names.intersection(targets)
            and "independent_parent" not in targets
        ):
            targets.insert(0, "independent_parent")
        policy = {
            "ridge": 10.0,
            "dense_scale": 1.5,
            "maxiter": MAXITER,
            "stationarity_threshold": 2e-4,
            "world": world_name,
            "family": family,
            "evaluation": {
                "method": "exact_enumeration",
                "contexts": 1024,
                "scenario_seed": seed + 499,
                "actions": "actions()",
            },
        }
        world = World({"family": family, "seed": seed})
        gen_ctx = subset(test[0], slice(0, 1024))
        truth = world.probability(gen_ctx["house"].numpy(), gen_ctx["dlp"].numpy())
        rng = np.random.default_rng(seed + 499)
        prices, scenario_houses = rng.uniform(-0.1, 0.1, (64, 3, 4)), rng.integers(
            1, 17, 64
        )
        acts = actions()
        shifts = np.stack([prices + a["shift"].reshape(3, 4) for a in acts]).reshape(
            -1, 3, 4
        )
        scenario_ctx = {
            "house": torch.tensor(np.tile(scenario_houses, len(acts))),
            "store": torch.zeros(len(shifts), dtype=torch.long),
            "week": torch.full((len(shifts),), 20.0, dtype=torch.float64),
            "disp": torch.zeros((len(shifts), 3, 4), dtype=torch.float64),
            "mail": torch.zeros((len(shifts), 3, 4), dtype=torch.float64),
            "dlp": torch.tensor(shifts, dtype=torch.float64),
        }
        truth_scenarios = world.probability(
            scenario_ctx["house"].numpy(), shifts
        ).reshape(len(acts), 64, -1)
        actual = [moments(q) for q in truth_scenarios]
        parent = None
        parent_fit = None
        parent_dependency = None
        rows = []

        def evaluate(name, model, fit, manifest, reused, dependency_seconds=0.0):
            started = time.perf_counter()
            with torch.no_grad():
                ll = model.log_probability(*test).numpy()
                val_ll = float(model.log_probability(*validation).mean())
                mass = _compute_exact_mass(model, gen_ctx, sets_t)
                gen_metrics = generation(mass, truth, gen_ctx["house"].numpy())
                scenario_mass = _compute_exact_mass(
                    model, scenario_ctx, sets_t
                ).reshape(len(acts), 64, -1)
                predictions = [moments(q) for q in scenario_mass]
                responses = response_score(predictions, actual, acts)
                promotion = global_promotion(predictions, actual, prices, acts)
            summary = {
                "model": name,
                "train_baskets": len(train[1]),
                "test_ll": float(ll.mean()),
                "validation_ll": val_ll,
                "size_tv": gen_metrics["size_tv"],
                "own_price_rel_mae": responses["own"]["relative_mae"],
                "same_category_rel_mae": responses["same_category"]["relative_mae"],
                "cross_category_rel_mae": responses["cross_category"]["relative_mae"],
                "zero_price_mae": (
                    responses["own"]["mae"] if "zero_price" in world_name else None
                ),
                "optimal_plans_of_6": sum(r["optimal"] for r in promotion),
                "raw_projected_gradient": fit["final_raw_projected_gradient"],
                "scaled_projected_gradient": fit["final_scaled_projected_gradient"],
                "stationarity_passed": fit["stationarity_passed"],
                "fit_seconds": fit["seconds"],
                "dependency_fit_seconds": dependency_seconds,
                "pipeline_fit_seconds": fit["seconds"] + dependency_seconds,
                "eval_seconds": time.perf_counter() - started,
                "checkpoint_reused": reused,
                "checkpoint_path": str(directory / f"{name}.pt"),
            }
            # Moments and mean basket mass suffice for the current exact-generation
            # metrics, avoiding a large context-by-all-baskets artifact per model.
            arrays = {
                "mean_basket_mass": mass.mean(0),
                "generation_house": gen_ctx["house"].numpy(),
                "generation_mean_size": moments(mass)["basket_size"],
                "scenario_prices": prices,
                "scenario_house": scenario_houses,
                **{
                    f"scenario_{key}": np.stack([q[key] for q in predictions])
                    for key in predictions[0]
                },
            }
            _save_evaluation(
                directory,
                name,
                model,
                manifest,
                fit,
                summary,
                ll,
                arrays,
                diagnostics={
                    "generation": gen_metrics,
                    "response": responses,
                    "promotion": promotion,
                },
            )
            rows.append(summary)
            print(
                f"[synthetic:{world_name}] {name}: ll={ll.mean():.6f}; converged={fit['stationarity_passed']}; reused={reused}",
                flush=True,
            )

        def make_baseline(name):
            torch.manual_seed(seed)
            dependency = copy.deepcopy(parent) if name in dependent_names else None
            if name == "popularity":
                return PooledPopularity(cfg)
            if name == "no_price":
                return NoPriceIndependent(cfg)
            if name == "independent_parent":
                return PersonalizedIndependent(cfg)
            if name == "size_aware_esp":
                return SizeAwareESP(cfg, train_y=train[1], parent=dependency)
            if name == "independent_categories_esp":
                return IndependentCategoriesESP(cfg, parent=dependency)
            if name == "mission_esp":
                return LatentMissionESP(cfg, K=2, seed=seed, parent=dependency)
            if name == "pairwise_binary_ar":
                return PairwiseBinaryAR(cfg, parent=dependency)
            if name == "low_rank_bilinear_ar":
                return LowRankBilinearAR(cfg, rank=6, seed=seed, parent=dependency)
            if name == "gaussian_factor_mixture":
                return GaussianFactorMixture(
                    cfg, n_factors=4, n_nodes=12, seed=seed, parent=dependency
                )
            if name == "neural_nade":
                return FastNeuralNADE(cfg, hidden=16, seed=seed)
            raise ValueError(name)

        for name in targets:
            if name in SYNTHETIC_GROUPS["g3"]:
                continue
            if name in dependent_names and (
                parent_fit is None or not parent_fit["stationarity_passed"]
            ):
                raise RuntimeError(
                    "Dependent fitting blocked: independent parent failed convergence"
                )
            model = make_baseline(name)
            deps = (
                {"independent_parent": parent_dependency}
                if name in dependent_names
                else {}
            )
            manifest = model_manifest(model, data, source_hash, policy, seed, deps)
            fit, reused = _fit_or_restore(
                model, directory, name, manifest, train, validation, mode, policy
            )
            if name == "independent_parent":
                parent, parent_fit = model, fit
                parent_dependency = {
                    "state_hash": state_hash(model),
                    "manifest_hash": json_hash(manifest),
                }
            evaluate(
                name,
                model,
                fit,
                manifest,
                reused,
                parent_fit["seconds"] if name in dependent_names else 0.0,
            )

        mission_targets = [name for name in targets if name in SYNTHETIC_GROUPS["g3"]]
        if mission_targets:
            models = {name: CalibratedMissionLaw(cfg) for name in mission_targets}
            selection_policy = {
                **policy,
                "selection": "default_hierarchical_candidates_validation_ll_final_stationary",
                "base_maxiter": MAXITER,
                "reg_maxiter": MAXITER,
            }
            manifests = {
                name: model_manifest(model, data, source_hash, selection_policy, seed)
                for name, model in models.items()
            }
            records = {
                name: (
                    restore_checkpoint(directory, name, model, manifests[name])
                    if mode == "resume"
                    else None
                )
                for name, model in models.items()
            }
            valid = bool(records) and all(records.values())
            if valid:
                for name, model in models.items():
                    certificate = certify_fit(model, train, validation, ridge=10.0)
                    valid = valid and certificate["stationarity_passed"]
                    records[name]["fit_record"].update(certificate)
            if valid:
                selections[world_name] = {
                    **next(iter(records.values()))["fit_record"].get(
                        "validation_selection", {}
                    ),
                    "checkpoint_reused": True,
                }
                for name, model in models.items():
                    evaluate(
                        name, model, records[name]["fit_record"], manifests[name], True
                    )
            else:
                model = CalibratedMissionLaw(cfg)
                selection = model.fit_with_validation_selection(
                    train,
                    validation,
                    ridge=10.0,
                    base_maxiter=MAXITER,
                    reg_maxiter=MAXITER,
                )
                selections[world_name] = {
                    key: value
                    for key, value in selection.items()
                    if key != "unregularized_state_dict"
                }
                selection_summary = {
                    key: value
                    for key, value in selection.items()
                    if key
                    not in (
                        "unregularized_state_dict",
                        "selected_fit_record",
                        "unregularized_fit_record",
                    )
                }
                for name in mission_targets:
                    if name == "proposed_calibrated_mission_law":
                        fitted, fit = model, selection["selected_fit_record"]
                    else:
                        fitted = CalibratedMissionLaw(cfg)
                        fitted.load_state_dict(selection["unregularized_state_dict"])
                        fit = selection["unregularized_fit_record"]
                    fit["validation_selection"] = selection_summary
                    evaluate(name, fitted, fit, manifests[name], False)
        rows.sort(key=lambda row: row["test_ll"], reverse=True)
        rows_by_world[world_name] = rows
        _write_table(results_dir / f"synthetic_{world_name}_comparison.csv", rows)
    if not rows_by_world:
        raise ValueError(f"Unknown synthetic world: {target_world}")
    output = {
        "code_hash": source_hash,
        "worlds": rows_by_world,
        "calibration_records": selections,
        "suite_stationarity_passed": all(
            row["stationarity_passed"]
            for rows in rows_by_world.values()
            for row in rows
        ),
    }
    (results_dir / "synthetic_comparison.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("quadrature", "synthetic", "erim", "dunnhumby", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--world", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "verified_v2"
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("quadrature", "all"):
        (results_dir / "quadrature_verification.json").write_text(
            json.dumps(verify_race_pl_quadrature(), indent=2) + "\n"
        )
    if args.stage in ("synthetic", "all"):
        evaluate_synthetic_benchmarks(results_dir, args.world, args.models, args.mode)
    for dataset in ("erim", "dunnhumby"):
        if args.stage in (dataset, "all"):
            evaluate_real_dataset(dataset, results_dir, args.models, args.mode)
    if args.stage in ("finalize", "all"):
        files = {
            "synthetic": "synthetic_comparison.json",
            "erim": "erim_comparison.json",
            "dunnhumby": "dunnhumby_comparison.json",
        }
        stages = {
            name: json.loads((results_dir / filename).read_text())
            for name, filename in files.items()
        }
        current = code_hash(REPO_ROOT)
        if any(stage.get("code_hash") != current for stage in stages.values()):
            raise RuntimeError(
                "Cannot finalize mixed or legacy code versions; evaluate matching checkpoints first"
            )
        (results_dir / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "code_hash": current,
                    "invocation_elapsed_seconds": time.perf_counter() - started,
                    "invocation_elapsed_is_total_training_cost": False,
                    **stages,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Completed invocation in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
