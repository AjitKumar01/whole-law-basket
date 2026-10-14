"""Ragged support must hold through likelihoods, fitting, sampling and evaluation."""

import copy
import importlib.util
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import binom

from whole_law_basket import (
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
    StructuredEnergy,
    CompositeWholeLawEnergy,
    CalibratedMissionLaw,
    fit_lbfgsb,
    certify_fit,
    draws,
    generation_metrics,
    predictive_metrics,
)
from whole_law_basket.layout import CategoryLayout
from whole_law_basket.metrics import FittingObjective

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ragged_runner", ROOT / "benchmarks/run_all_benchmarks.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
torch.set_num_threads(1)


def models(sizes):
    cfg = {"category_sizes": sizes, "users": 3, "stores": 1}
    base = StructuredEnergy(cfg, [])
    model_list = [
        PooledPopularity(cfg),
        NoPriceIndependent(cfg),
        PersonalizedIndependent(cfg),
        SizeAwareESP(cfg),
        IndependentCategoriesESP(cfg),
        LatentMissionESP(cfg),
        PairwiseBinaryAR(cfg),
        LowRankBilinearAR(cfg, rank=2),
        GaussianFactorMixture(cfg, n_factors=2, n_nodes=4),
        FastNeuralNADE(cfg, hidden=4),
        base,
        CompositeWholeLawEnergy(
            cfg,
            copy.deepcopy(base),
            torch.zeros(2, len(sizes)),
            torch.zeros(2, sum(sizes)),
        ),
        CalibratedMissionLaw(cfg),
    ]
    return model_list


def baskets(layout):
    packed = torch.tensor(
        list(itertools.product((0.0, 1.0), repeat=layout.J)), dtype=torch.float64
    )
    return packed[packed.sum(-1) > 0]


@pytest.mark.parametrize("sizes", [[1, 4, 2], [1, 1, 1]])
def test_all_baselines_ragged_support_normalization_and_gradients(sizes):
    for model in models(sizes):
        layout = model.layout
        sets = baskets(layout)
        generator = torch.Generator().manual_seed(12)
        d = 0.1 * torch.randn(1, layout.J, generator=generator, dtype=torch.float64)
        ctx = {
            "house": torch.zeros(len(sets), dtype=torch.long),
            "dlp": d.expand(len(sets), -1),
        }
        padded_ctx = {**ctx, "dlp": layout.padded(ctx["dlp"])}
        if layout.ragged:
            padded_ctx["dlp"][:, ~layout.valid_mask] = float("nan")
        lp = model.log_probability(ctx, sets)
        lp_pad = model.log_probability(padded_ctx, layout.padded(sets))
        assert torch.allclose(lp, lp_pad, atol=1e-10), type(model).__name__
        assert abs(lp.exp().sum().item() - 1.0) < 1e-9, type(model).__name__
        mass = runner._compute_exact_mass(
            model, {k: v[:1] for k, v in ctx.items()}, sets
        )
        assert np.allclose(mass[0], lp.detach().exp().numpy(), atol=1e-10), type(
            model
        ).__name__
        loss = FittingObjective(model, (ctx, sets), (ctx, sets), ridge=10.0).loss()
        loss.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        ), type(model).__name__
        rejected = torch.zeros(3, layout.J, dtype=torch.float64)
        rejected[1, 0] = 0.5
        rejected[2, 0] = float("nan")
        repeated_ctx = {
            "house": torch.zeros(3, dtype=torch.long),
            "dlp": torch.zeros(3, layout.J, dtype=torch.float64),
        }
        assert torch.isneginf(model.log_probability(repeated_ctx, rejected)).all()
        assert (
            runner._compute_exact_mass(
                model, {k: v[:1] for k, v in ctx.items()}, rejected
            )
            == 0.0
        ).all()
        if layout.ragged:
            bad = layout.padded(sets[:1])
            bad[:, 0, 1] = 1
            assert torch.isneginf(
                model.log_probability({k: v[:1] for k, v in padded_ctx.items()}, bad)
            ).all()
            assert (
                runner._compute_exact_mass(
                    model, {k: v[:1] for k, v in padded_ctx.items()}, bad
                )
                == 0.0
            ).all()
        samples = model.sample(
            {
                "house": torch.zeros(6, dtype=torch.long),
                "dlp": torch.zeros(6, layout.J, dtype=torch.float64),
            },
            7,
        )
        assert layout.valid_baskets(samples).all()


def test_cached_training_rejects_the_same_invalid_support():
    model = models([1, 4, 2])[-2]
    ctx = {"house": torch.tensor([0]), "dlp": torch.zeros(1, 3, 4, dtype=torch.float64)}
    bad = torch.zeros(1, 3, 4, dtype=torch.float64)
    bad[:, 0, :2] = 1
    assert torch.isneginf(model.log_probability(ctx, bad)).all()
    assert torch.isneginf(
        model.forward_cached(ctx, model.precompute_split(ctx, bad))
    ).all()
    with pytest.raises(ValueError, match="Invalid training baskets"):
        fit_lbfgsb(model, (ctx, bad), (ctx, bad))


def test_ragged_draws_and_all_metrics_match_packed_and_padded():
    for model in models([1, 4, 2]):
        layout = model.layout
        ctx = {
            "house": torch.arange(6) % 3,
            "dlp": torch.zeros(6, 7, dtype=torch.float64),
        }
        packed = draws(model, ctx, 6, 41)
        pad_ctx = {**ctx, "dlp": layout.padded(ctx["dlp"])}
        padded = draws(model, pad_ctx, 6, 41)
        assert np.array_equal(
            packed,
            layout.flatten(torch.tensor(padded.reshape(-1, 3, 4)))
            .numpy()
            .reshape(6, 6, 7),
        )
        observed = packed[:, 0]
        observed_pad = layout.padded(torch.tensor(observed)).numpy()
        kwargs = {"category_sizes": layout.sizes}
        assert generation_metrics(
            packed, observed, ctx["house"].numpy(), **kwargs
        ) == generation_metrics(padded, observed_pad, ctx["house"].numpy(), **kwargs)
        assert predictive_metrics(packed, observed, **kwargs) == predictive_metrics(
            padded, observed_pad, **kwargs
        )


def test_ragged_sampler_matches_likelihood_distribution():
    for model in models([1, 2]):
        layout = model.layout
        sets = baskets(layout)
        n = 8192
        ctx = {
            "house": torch.zeros(len(sets), dtype=torch.long),
            "dlp": torch.zeros(len(sets), 3, dtype=torch.float64),
        }
        probability = model.log_probability(ctx, sets).detach().exp().numpy()
        sampled = model.sample(
            {
                "house": torch.zeros(n, dtype=torch.long),
                "dlp": torch.zeros(n, 3, dtype=torch.float64),
            },
            seed=123,
        )
        codes = (
            (layout.flatten(sampled).numpy() * (2 ** np.arange(3))).sum(-1).astype(int)
        )
        set_codes = (sets.numpy() * (2 ** np.arange(3))).sum(-1).astype(int)
        count = np.bincount(codes, minlength=8)[set_codes]
        assert (
            (count >= binom.ppf(1e-6, n, probability))
            & (count <= binom.ppf(1 - 1e-6, n, probability))
        ).all(), type(model).__name__


def test_penalty_uses_documented_normalized_category_weights():
    m = CalibratedMissionLaw({"category_sizes": [1, 4, 2], "users": 1})
    m.set_regularization_config(
        {"w_pr_within": 1.0, "w_pr_between": 1.0, "l2_price": 0.0}
    )
    with torch.no_grad():
        m.price.zero_()
        m.price[0, 0] = 1
        m.price[1, :] = torch.tensor([1, 2, 3, 4])
        m.price[2, :2] = torch.tensor([1, 2])
    moments = {
        name: torch.ones(3, dtype=torch.float64)
        for name in [
            "category_price",
            "category_hinge",
            "category_cross",
            "count_price",
        ]
    }
    moments["price"] = torch.ones(3, 4, dtype=torch.float64)
    expected = 5.5 + 2.5 / (7 / 3)
    assert m.regularization_loss(moments).item() == pytest.approx(expected)


def test_actual_final_weight_convergence_after_ragged_selection():
    cfg = {"category_sizes": [1, 2, 3], "users": 3, "stores": 1}
    teacher = CalibratedMissionLaw(cfg)
    gen = torch.Generator().manual_seed(21)
    ctx = {
        "house": torch.arange(256) % 3,
        "dlp": 0.2 * torch.randn(256, 6, generator=gen, dtype=torch.float64),
    }
    y = teacher.sample(ctx, seed=71)
    model = CalibratedMissionLaw(cfg)
    record = model.fit_with_validation_selection(
        (ctx, y), (ctx, y), ridge=10.0, candidates=[("unregularized", None)]
    )
    actual = certify_fit(model, (ctx, y), (ctx, y), ridge=10.0)
    assert actual["stationarity_passed"]
    assert (
        record["selected_fit_record"]["final_raw_projected_gradient"]
        == actual["final_raw_projected_gradient"]
    )
    assert (
        record["selected_fit_record"]["seconds"]
        >= record["selected_fit_record"]["selected_candidate_seconds"]
    )


def test_real_loader_projects_explicit_ragged_layout(tmp_path, monkeypatch):
    """Use actual ERIM controls and source rows without changing bundled data."""
    from whole_law_basket import real_data

    source = tmp_path / "data/erim"
    source.mkdir(parents=True)
    parent = tmp_path / "data/erim_parent"
    parent.mkdir()
    for name in ("config.json", "best.pt"):
        shutil.copyfile(ROOT / "data/erim_parent" / name, parent / name)
    config = json.loads((ROOT / "data/erim/config.json").read_text())
    items = [0, 4, 5, 6, 7, 24, 25]
    config.update(C=3, P=4, category_sizes=[1, 4, 2], source_product_indices=items)
    (source / "config.json").write_text(json.dumps(config))
    originals = {}
    for split in ("train", "validation", "test"):
        with np.load(ROOT / "data/erim" / f"{split}.npz") as archive:
            raw = {k: archive[k][:128] for k in archive.files}
        originals[split] = raw
        np.savez(source / f"{split}.npz", **raw)
    monkeypatch.setattr(real_data, "REPO_ROOT", tmp_path)
    cfg, price, train, val, test, clusters = real_data.load_real_dataset("erim")
    assert cfg["category_sizes"] == [1, 4, 2]
    assert price.shape == (7,)
    for name, (ctx, y) in zip(("train", "validation", "test"), (train, val, test)):
        raw = originals[name]
        selected = raw["member"].reshape(128, -1)[:, items]
        keep = selected.any(-1)
        assert np.array_equal(y.numpy(), selected[keep])
        assert np.array_equal(
            ctx["dlp"].numpy(), raw["dlp"].reshape(128, -1)[keep][:, items]
        )
        assert ctx["baseline"].shape == y.shape
        assert torch.isfinite(ctx["baseline"]).all()
    assert np.array_equal(clusters["source_rows"], np.flatnonzero(selected.any(-1)))


def test_ragged_runner_fresh_fit_evaluation_and_resume(tmp_path, monkeypatch):
    """Exercise the real runner contract on a small known-law fixture, not a retail claim."""
    cfg = {"category_sizes": [1, 2, 3], "users": 3, "stores": 1}
    teacher = CalibratedMissionLaw(cfg)
    generator = torch.Generator().manual_seed(51)
    splits = []
    for i, n in enumerate((256, 64, 64)):
        ctx = {
            "house": torch.arange(n) % 3,
            "dlp": 0.2 * torch.randn(n, 6, generator=generator, dtype=torch.float64),
            "baseline": torch.zeros(n, 6, dtype=torch.float64),
        }
        splits.append((ctx, teacher.sample(ctx, seed=61 + i)))
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    ref = tmp_path / "data/erim_ref"
    ref.mkdir(parents=True)
    (ref / "screen.json").write_text(json.dumps({"edges": []}))
    monkeypatch.setattr(
        runner,
        "_shared_real_inputs",
        lambda dataset: {"historical_parent_fit_seconds": None},
    )
    monkeypatch.setattr(
        runner,
        "load_real_dataset",
        lambda dataset: (
            cfg,
            torch.ones(6, dtype=torch.float64),
            *splits,
            {"house": splits[-1][0]["house"].numpy()},
        ),
    )
    names = "size_aware_esp,structured_energy,proposed_whole_law"
    output = tmp_path / "out"
    first = runner.evaluate_real_dataset("erim", output, names, "fresh")
    second = runner.evaluate_real_dataset("erim", output, names, "resume")
    assert first["suite_stationarity_passed"]
    assert second["suite_stationarity_passed"]
    before = {row["model"]: row for row in first["summary_table"]}
    for row in second["summary_table"]:
        assert row["checkpoint_reused"]
        assert row["test_ll"] == before[row["model"]]["test_ll"]
        arrays = np.load(output / "checkpoints/erim" / f"{row['model']}_evaluation.npz")
        assert arrays["samples"].shape == (64, 48, 6)
        assert (
            CategoryLayout(cfg)
            .valid_baskets(torch.tensor(arrays["samples"].reshape(-1, 6)))
            .all()
        )
