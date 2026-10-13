"""Comprehensive Benchmark Runner: Proposed Whole-Law Model vs. 11 Baselines.

Evaluates all models on the exact same train/validation/test splits across:
  1. Synthetic `missions_a` (seed=941000, C=3, P=4, J=12)
  2. Synthetic `zero_price` (seed=943000, C=3, P=4, J=12)
  3. Real ERIM (C=7, P=4, J=28 products)
  4. Real Dunnhumby (C=16, P=4, J=64 products)

Writes structured CSV and JSON tables into `results/`.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[0]
REAL_DIR = PROJECT_ROOT / "artifacts/research_whole_law_real_20261013"
SOL_DIR = PROJECT_ROOT / "artifacts/research_whole_law_solution_20261014"
TEST_DIR = PROJECT_ROOT / "artifacts/research_whole_law_test_20261013"
CONF_DIR = PROJECT_ROOT / "artifacts/research_product_interactions_20261013/confirmation_v2"
PUBLIC = Path("/private/tmp/basket-readable-code-20261013")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REAL_DIR))
sys.path.insert(0, str(PUBLIC))
sys.path.insert(0, str(TEST_DIR))

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
    ShopperFactorAR,
    LogisticNormalFactorMixture,
    FastNeuralNADE,
    fit_lbfgsb,
    draws,
    generation_metrics,
    predictive_metrics,
    cluster_interval,
    subset,
)
from run import load_parent, select_catalogue, prepare_split, SEED


def _subsample_train(train: tuple[dict, torch.Tensor], max_rows: int = 8000):
    ctx, y = train
    if len(y) <= max_rows:
        return train
    idx = np.random.default_rng(SEED).choice(len(y), max_rows, replace=False)
    return subset(ctx, idx), y[idx]


@torch.no_grad()
def _compute_exact_mass(model: torch.nn.Module, ctx: dict, sets_t: torch.Tensor) -> np.ndarray:
    if hasattr(model, "exact_mass"):
        return model.exact_mass(ctx, sets_t).cpu().numpy()
    B = len(ctx["house"])
    S = len(sets_t)
    sets_cp = sets_t.reshape(S, model.C, model.P)
    out = np.empty((B, S), dtype=float)
    chunk = 8
    for start in range(0, B, chunk):
        end = min(start + chunk, B)
        b_len = end - start
        rep_ctx = {
            k: v[start:end, None].expand(b_len, S, *([-1] * (v.ndim - 1))).reshape(b_len * S, *v.shape[1:])
            for k, v in ctx.items()
        }
        rep_y = sets_cp[None, :, :, :].expand(b_len, S, model.C, model.P).reshape(b_len * S, model.C, model.P)
        lp = model.log_probability(rep_ctx, rep_y).reshape(b_len, S)
        out[start:end] = lp.exp().cpu().numpy()
    return out


def evaluate_real_dataset(dataset: str, results_dir: Path) -> dict:
    print(f"=== Running Real-Data Multi-Baseline Benchmark: {dataset} ===", flush=True)
    parent, _ = load_parent(dataset)
    config, items, source = select_catalogue(dataset)
    price = parent.choice_price[items].detach().clone()
    train, _, _ = prepare_split(source, "train", config, items, parent)
    validation, _, _ = prepare_split(source, "validation", config, items, parent)
    test, _, clusters = prepare_split(source, "test", config, items, parent)

    train_sub = _subsample_train(train, max_rows=8000)

    old_ds_dir = REAL_DIR / dataset
    sol_ds_dir = SOL_DIR / dataset
    sol_res = json.loads((sol_ds_dir / "result.json").read_text())

    new_baselines = {
        "popularity": (PooledPopularity(config), 20),
        "no_price": (NoPriceIndependent(config), 20),
        "size_aware_esp": (SizeAwareESP(config, price=price, train_y=train[1]), 25),
        "independent_categories_esp": (IndependentCategoriesESP(config, price=price), 22),
        "mission_esp": (LatentMissionESP(config, price=price, K=2, seed=SEED), 22),
        "pairwise_ar": (PairwiseBinaryAR(config, price=price), 22),
        "shopper_factor_ar": (ShopperFactorAR(config, price=price, rank=8, seed=SEED), 22),
        "logistic_normal_vae": (
            LogisticNormalFactorMixture(config, price=price, n_factors=6, n_nodes=12, seed=SEED),
            22,
        ),
    }

    rows_gen = np.random.default_rng(SEED + 10).choice(
        len(test[1]), min(2048, len(test[1])), replace=False
    )
    ctx_gen = subset(test[0], rows_gen)
    observed_gen = test[1][rows_gen].numpy()
    hh_gen = clusters["house"][rows_gen]

    all_test_lls = {
        "independent_parent": np.load(old_ds_dir / "independent_parent_test_ll.npy"),
        "structured_energy": np.load(old_ds_dir / "energy_test_ll.npy"),
        "neural_nade": np.load(old_ds_dir / "neural_test_ll.npy"),
        "proposed_whole_law": np.load(sol_ds_dir / "composite_energy_test_ll.npy"),
    }

    summary_rows = []

    for b_name, (b_model, maxiter) in new_baselines.items():
        fit_rec = fit_lbfgsb(
            b_model,
            train_sub if b_name not in ("popularity", "size_aware_esp") else train,
            validation,
            ridge=15.0,
            maxiter=maxiter,
            dense_scale=10.0,
        )
        with torch.no_grad():
            te_ll = b_model.log_probability(test[0], test[1]).numpy()
            va_ll = float(b_model.log_probability(validation[0], validation[1]).mean().item())
        all_test_lls[b_name] = te_ll

        t_gen = time.perf_counter()
        replicas = [
            draws(b_model, ctx_gen, 8, SEED + 1000 + rep * 10000, batch=512)
            for rep in range(2)
        ]
        gen_sec = time.perf_counter() - t_gen
        comb = np.concatenate(replicas, axis=1)
        gen_m = generation_metrics(comb, observed_gen, hh_gen)
        pred_m = predictive_metrics(comb, observed_gen)

        summary_rows.append(
            {
                "model": b_name,
                "family": b_model.config["benchmark_baseline"],
                "test_ll": float(te_ll.mean()),
                "validation_ll": va_ll,
                "size_TV": gen_m["size_TV"],
                "frequency_MAE": gen_m["product_frequency_MAE"],
                "category_pair_MAE": gen_m["category_pair_MAE"],
                "within_category_multiple_MAE": gen_m["within_category_multiple_MAE"],
                "recall5": pred_m["context_recall_5"],
                "Brier": pred_m["Brier_MC_corrected"],
                "generation_seconds": gen_sec,
                "fit_seconds": fit_rec["seconds"],
            }
        )
        print(
            f"  [{dataset}] {b_name}: test_ll={te_ll.mean():.5f}, val_ll={va_ll:.5f}, size_TV={gen_m['size_TV']:.5f}",
            flush=True,
        )

    name_map = {
        "independent_parent": ("independent_parent", "mvp_independent_logit"),
        "old_energy": ("structured_energy", "category_frontier_dp_energy"),
        "neural": ("neural_nade", "conditional_NADE_H16_crossprice_rank8"),
        "composite_whole_law_energy": (
            "proposed_whole_law",
            "composite_whole_law_nonparametric_count_and_missions",
        ),
    }
    for row in sol_res["summary_table"]:
        m_key, fam = name_map[row["model"]]
        summary_rows.append(
            {
                "model": m_key,
                "family": fam,
                "test_ll": row["test_ll"],
                "validation_ll": row["validation_ll"],
                "size_TV": row["size_TV"],
                "frequency_MAE": row["frequency_MAE"],
                "category_pair_MAE": row["category_pair_MAE"],
                "within_category_multiple_MAE": row["within_category_multiple_MAE"],
                "recall5": row["recall5"],
                "Brier": row["Brier"],
                "generation_seconds": row["generation_seconds"],
                "fit_seconds": sol_res["fit_record"]["seconds"] if m_key == "proposed_whole_law" else 0.0,
            }
        )

    summary_rows.sort(key=lambda r: r["test_ll"], reverse=True)

    prop_ll = all_test_lls["proposed_whole_law"]
    paired_vs_proposed = {}
    for m_key, b_ll in all_test_lls.items():
        if m_key == "proposed_whole_law":
            continue
        paired_vs_proposed[m_key] = cluster_interval(prop_ll - b_ll, clusters["house"])
        for r in summary_rows:
            if r["model"] == m_key:
                r["proposed_gain_over_model"] = paired_vs_proposed[m_key]["mean_gain"]
                r["proposed_gain_95_ci_low"] = paired_vs_proposed[m_key]["cluster_95_ci"][0]
                r["proposed_gain_95_ci_high"] = paired_vs_proposed[m_key]["cluster_95_ci"][1]
    for r in summary_rows:
        if r["model"] == "proposed_whole_law":
            r["proposed_gain_over_model"] = 0.0
            r["proposed_gain_95_ci_low"] = 0.0
            r["proposed_gain_95_ci_high"] = 0.0

    csv_path = results_dir / f"{dataset}_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    out = {
        "dataset": dataset,
        "summary_table": summary_rows,
        "paired_household_bootstrap_vs_proposed": paired_vs_proposed,
    }
    (results_dir / f"{dataset}_comparison.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    return out


def evaluate_synthetic_benchmarks(results_dir: Path) -> dict:
    print("=== Running Multi-Baseline Synthetic Comparison (missions_a & zero_price) ===", flush=True)
    sol_synth = json.loads((SOL_DIR / "synthetic_and_race_summary.json").read_text())
    old_synth = json.loads((TEST_DIR / "summary.json").read_text())

    from experiments import common as exp_common
    from experiments.synthetic.worlds import World, moments, actions, response_score, SETS
    from experiments.synthetic.metrics import generation, global_promotion

    sets_t = torch.tensor(SETS, dtype=torch.float64)
    rows_by_world = {"missions_a": [], "zero_price": []}

    for w_name, family, seed in [
        ("missions_a", "latent_missions_gumbel", 941000),
        ("zero_price", "latent_missions_gumbel_zero_price", 943000),
    ]:
        exp_common.OUT = CONF_DIR / w_name
        tr_ctx, tr_y = exp_common.Data("train").pair()
        va_ctx, va_y = exp_common.Data("validation").pair()
        te_data = exp_common.Data("test")
        te_ctx, te_y = te_data.pair()
        cfg = json.loads((CONF_DIR / w_name / "data/config.json").read_text())

        ind_parent = PersonalizedIndependent(cfg)
        fit_lbfgsb(ind_parent, (tr_ctx, tr_y), (va_ctx, va_y), ridge=10.0, maxiter=25)

        synth_baselines = {
            "popularity": PooledPopularity(cfg),
            "no_price": NoPriceIndependent(cfg),
            "independent_parent": ind_parent,
            "size_aware_esp": SizeAwareESP(cfg, train_y=tr_y, parent=ind_parent),
            "independent_categories_esp": IndependentCategoriesESP(cfg, parent=ind_parent),
            "mission_esp": LatentMissionESP(cfg, K=2, seed=seed, parent=ind_parent),
            "pairwise_binary_ar": PairwiseBinaryAR(cfg, parent=ind_parent),
            "shopper_factor_ar": ShopperFactorAR(cfg, rank=6, seed=seed, parent=ind_parent),
            "logistic_normal_vae": LogisticNormalFactorMixture(
                cfg, n_factors=4, n_nodes=12, seed=seed, parent=ind_parent
            ),
        }

        world = World({"family": family, "seed": seed})
        ctx_1024, _ = te_data.batch(slice(0, 1024))
        truth = world.probability(ctx_1024["house"].numpy(), ctx_1024["dlp"].numpy())
        rng = np.random.default_rng(seed + 499)
        price_sc, house_sc = rng.uniform(-0.1, 0.1, (64, 3, 4)), rng.integers(1, 17, 64)
        scenarios = {
            "house": torch.tensor(house_sc),
            "store": torch.zeros(64, dtype=torch.long),
            "week": torch.full((64,), 20.0, dtype=torch.float64),
            "disp": torch.zeros((64, 3, 4), dtype=torch.float64),
            "mail": torch.zeros((64, 3, 4), dtype=torch.float64),
        }
        acts = actions()
        actual = [
            moments(world.probability(house_sc, price_sc + a["shift"].reshape(3, 4)))
            for a in acts
        ]

        for m_name, m_obj in synth_baselines.items():
            if m_name != "independent_parent":
                fit_lbfgsb(m_obj, (tr_ctx, tr_y), (va_ctx, va_y), ridge=10.0, maxiter=25)
            with torch.no_grad():
                te_ll = float(m_obj.log_probability(te_ctx, te_y).mean().item())
                mass = _compute_exact_mass(m_obj, ctx_1024, sets_t)
                gen_m = generation(mass, truth, ctx_1024["house"].numpy())

                preds = []
                for a in acts:
                    c_shift = {
                        **scenarios,
                        "dlp": torch.tensor(price_sc + a["shift"].reshape(3, 4)),
                    }
                    preds.append(moments(_compute_exact_mass(m_obj, c_shift, sets_t)))
                what_if = response_score(preds, actual, acts)
                promo = global_promotion(preds, actual, price_sc, acts)

            rows_by_world[w_name].append(
                {
                    "model": m_name,
                    "test_ll": te_ll,
                    "size_tv": float(gen_m["size_tv"]),
                    "own_price_rel_mae": what_if["own"]["relative_mae"],
                    "same_category_rel_mae": what_if["same_category"]["relative_mae"],
                    "cross_category_rel_mae": what_if["cross_category"]["relative_mae"],
                    "zero_price_mae": what_if["own"]["mae"] if "zero_price" in w_name else None,
                    "optimal_plans_of_6": int(sum(r["optimal"] for r in promo)),
                }
            )
            print(f"  [synthetic:{w_name}] {m_name}: test_ll={te_ll:.5f}, size_tv={gen_m['size_tv']:.5f}", flush=True)

        w_sol = sol_synth["worlds"][w_name]
        w_old = old_synth["worlds"][w_name]
        rows_by_world[w_name].append(
            {
                "model": "neural_nade",
                "test_ll": float(w_sol["neural_test_ll"]),
                "size_tv": float(w_old["neural_generation"]["size_tv"]),
                "own_price_rel_mae": w_old["neural_price"]["own"]["relative_mae"],
                "same_category_rel_mae": w_old["neural_price"]["same_category"]["relative_mae"],
                "cross_category_rel_mae": w_old["neural_price"]["cross_category"]["relative_mae"],
                "zero_price_mae": w_old["neural_price"]["own"]["mae"] if "zero_price" in w_name else None,
                "optimal_plans_of_6": int(w_old["promotion_optimal"]["neural"]),
            }
        )
        rows_by_world[w_name].append(
            {
                "model": "uncalibrated_mission_law",
                "test_ll": float(w_old["replacement_test_ll"]),
                "size_tv": float(w_old["replacement_generation"]["size_tv"]),
                "own_price_rel_mae": w_old["replacement_price"]["own"]["relative_mae"],
                "same_category_rel_mae": w_old["replacement_price"]["same_category"]["relative_mae"],
                "cross_category_rel_mae": w_old["replacement_price"]["cross_category"]["relative_mae"],
                "zero_price_mae": w_old["replacement_price"]["own"]["mae"] if "zero_price" in w_name else None,
                "optimal_plans_of_6": int(w_old["promotion_optimal"]["replacement"]),
            }
        )
        rows_by_world[w_name].append(
            {
                "model": "proposed_calibrated_mission_law",
                "test_ll": float(w_sol["replacement_test_ll"]),
                "size_tv": float(w_sol["generation"]["size_tv"]),
                "own_price_rel_mae": w_sol["price_response"]["own"]["relative_mae"],
                "same_category_rel_mae": w_sol["price_response"]["same_category"]["relative_mae"],
                "cross_category_rel_mae": w_sol["price_response"]["cross_category"]["relative_mae"],
                "zero_price_mae": w_sol["price_response"]["own"]["mae"] if "zero_price" in w_name else None,
                "optimal_plans_of_6": int(w_sol["promotion_optimal_plans_of_6"]),
            }
        )
        rows_by_world[w_name].sort(key=lambda r: r["test_ll"], reverse=True)
        with (results_dir / f"synthetic_{w_name}_comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_by_world[w_name][0].keys()))
            writer.writeheader()
            writer.writerows(rows_by_world[w_name])

    (results_dir / "synthetic_comparison.json").write_text(
        json.dumps(rows_by_world, indent=2) + "\n"
    )
    return rows_by_world


def main():
    t0 = time.perf_counter()
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    quad_res = verify_race_pl_quadrature()
    synth_res = evaluate_synthetic_benchmarks(results_dir)
    erim_res = evaluate_real_dataset("erim", results_dir)
    dunn_res = evaluate_real_dataset("dunnhumby", results_dir)

    summary = {
        "elapsed_seconds": time.perf_counter() - t0,
        "differentiable_race_pl_verification": quad_res,
        "synthetic_benchmarks": synth_res,
        "real_benchmarks": {
            "erim": erim_res,
            "dunnhumby": dunn_res,
        },
    }
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"=== Completed All Benchmarks in {summary['elapsed_seconds']:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
