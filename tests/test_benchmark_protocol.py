"""Regression checks for checkpoint identity, fresh/resume, and final-weight certificates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from whole_law_basket import PooledPopularity, SizeAwareESP, StructuredEnergy
from whole_law_basket.metrics import FittingObjective, certify_fit, fit_lbfgsb
from whole_law_basket.provenance import (
    array_hash,
    dataset_manifest,
    model_manifest,
    restore_checkpoint,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "benchmark_protocol_runner", ROOT / "benchmarks/run_all_benchmarks.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def tiny_data():
    generator = torch.Generator().manual_seed(41)
    cfg = {"C": 2, "P": 2, "R": 2, "users": 3, "stores": 1}
    ctx = {
        "house": torch.arange(48) % 3,
        "dlp": 0.2 * torch.randn((48, 2, 2), generator=generator, dtype=torch.float64),
        "baseline": torch.zeros((48, 2, 2), dtype=torch.float64),
    }
    y = (torch.rand((48, 2, 2), generator=generator) > 0.65).double()
    y[:, 0, 0] = 1.0
    return cfg, (ctx, y)


def manifest_for(model, split):
    data = dataset_manifest(model.config, (split, split, split))
    return model_manifest(model, data, "code-v1", {"ridge": 15.0}, 1)


@pytest.mark.parametrize(
    "field", ["data_hash", "code_hash", "policy", "seed", "dependencies"]
)
def test_resume_rejects_changed_protocol_without_mutating_weights(tmp_path, field):
    cfg, split = tiny_data()
    model = PooledPopularity(cfg)
    manifest = manifest_for(model, split)
    save_checkpoint(tmp_path, "model", model, manifest, {"seconds": 1.0})
    modified = copy.deepcopy(manifest)
    modified[field] = {"changed": True}
    with torch.no_grad():
        model.lam.add_(0.1)
    before = model.lam.detach().clone()
    assert restore_checkpoint(tmp_path, "model", model, modified) is None
    assert torch.equal(model.lam, before)


def test_resume_rejects_corrupt_or_legacy_checkpoints(tmp_path):
    cfg, split = tiny_data()
    model = PooledPopularity(cfg)
    manifest = manifest_for(model, split)
    save_checkpoint(tmp_path, "model", model, manifest, {"seconds": 1.0})
    assert restore_checkpoint(tmp_path, "model", model, manifest) is not None
    with (tmp_path / "model.pt").open("ab") as stream:
        stream.write(b"corrupted")
    assert restore_checkpoint(tmp_path, "model", model, manifest) is None
    (tmp_path / "model_record.json").write_text(
        json.dumps({"summary": {"train_baskets": 48}})
    )
    assert restore_checkpoint(tmp_path, "model", model, manifest) is None


def test_data_hash_detects_prices_labels_and_row_order():
    cfg, split = tiny_data()
    baseline = dataset_manifest(cfg, (split, split, split))
    for location in ("dlp", "y", "order"):
        ctx, y = copy.deepcopy(split)
        if location == "dlp":
            ctx["dlp"][0, 0, 0] += 0.01
        elif location == "y":
            y[0, 0, 1] = 1 - y[0, 0, 1]
        else:
            ctx = {k: v.flip(0) for k, v in ctx.items()}
            y = y.flip(0)
        assert dataset_manifest(cfg, ((ctx, y), split, split)) != baseline
    assert array_hash(np.empty((0, 2))) == array_hash(np.empty((0, 2)))


def test_certificate_describes_current_weights_and_does_not_train():
    cfg, split = tiny_data()
    model = PooledPopularity(cfg)
    fitted = fit_lbfgsb(model, split, split)
    assert fitted["stationarity_passed"]
    with torch.no_grad():
        model.lam[1] += 1.0
    before = {k: v.clone() for k, v in model.state_dict().items()}
    certificate = certify_fit(model, split, split)
    assert not certificate["stationarity_passed"]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    returned = fit_lbfgsb(model, split, split, maxiter=0)
    assert not returned["stationarity_passed"]
    assert returned["iterations"] == 0
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_validation_selected_iterate_has_its_own_certificate():
    cfg, train = tiny_data()
    ctx, y = copy.deepcopy(train)
    y[:, 0, 1] = 1 - y[:, 0, 1]
    model = PooledPopularity(cfg)
    result = fit_lbfgsb(model, train, (ctx, y), use_final_stationary=False)
    actual = certify_fit(model, train, (ctx, y))
    for key in (
        "final_objective",
        "final_validation_ll",
        "final_raw_projected_gradient",
        "stationarity_passed",
    ):
        assert result[key] == actual[key]


def test_size_fisher_matches_likelihood_hessian():
    cfg, split = tiny_data()
    model = SizeAwareESP(cfg, train_y=split[1])
    info = model.fit_information(split[0])
    loss = -model.log_probability(*split).mean()
    for name in ("size_lam", "need", "price_slope"):
        parameter = getattr(model, name)
        grad = torch.autograd.grad(
            loss, parameter, create_graph=True, retain_graph=True
        )[0]
        diagonal = torch.stack(
            [
                torch.autograd.grad(grad.flatten()[i], parameter, retain_graph=True)[
                    0
                ].flatten()[i]
                for i in range(parameter.numel())
            ]
        ).reshape(parameter.shape)
        assert torch.allclose(diagonal, info[name], atol=1e-12, rtol=1e-10)
    fitted = fit_lbfgsb(model, split, split)
    assert fitted["stationarity_passed"]


def test_composite_builder_never_loads_reference_weights(tmp_path, monkeypatch):
    cfg, split = tiny_data()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    reference = tmp_path / "data/erim_ref"
    reference.mkdir(parents=True)
    torch.save(
        {"invalid_reference": torch.tensor(9.0)}, reference / "composite_init.pt"
    )
    energy = StructuredEnergy(cfg, [])
    with torch.no_grad():
        energy.lam.fill_(0.314)
    before = copy.deepcopy(energy.state_dict())
    model = runner.build_composite_model("erim", cfg, energy, split)
    assert all(
        torch.equal(v, before[k]) for k, v in model.re_model.state_dict().items()
    )
    assert model.M_cat.requires_grad
    assert model.house_mix.requires_grad
    assert model.price_cat.requires_grad


def test_real_fresh_resume_recompute_and_data_invalidation(tmp_path, monkeypatch):
    cfg, split = tiny_data()
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
            torch.ones(4),
            split,
            split,
            split,
            {"house": split[0]["house"].numpy()},
        ),
    )
    real_fit, real_draws = runner.fit_lbfgsb, runner.draws
    calls = {"fit": 0, "draws": 0}

    def count_fit(*args, **kwargs):
        calls["fit"] += 1
        return real_fit(*args, **kwargs)

    def count_draws(*args, **kwargs):
        calls["draws"] += 1
        return real_draws(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_lbfgsb", count_fit)
    monkeypatch.setattr(runner, "draws", count_draws)
    output = tmp_path / "out"
    first = runner.evaluate_real_dataset("erim", output, "popularity", "fresh")
    second = runner.evaluate_real_dataset("erim", output, "popularity", "resume")
    assert calls == {"fit": 1, "draws": 6}
    assert second["summary_table"][0]["checkpoint_reused"]
    assert second["summary_table"][0]["test_ll"] == first["summary_table"][0]["test_ll"]
    runner.evaluate_real_dataset("erim", output, "popularity", "fresh")
    assert calls["fit"] == 2
    split[0]["dlp"][0, 0, 0] += 0.01
    runner.evaluate_real_dataset("erim", output, "popularity", "resume")
    assert calls["fit"] == 3
    arrays = np.load(output / "checkpoints/erim/popularity_evaluation.npz")
    assert arrays["samples"].shape == (48, 48, 2, 2)


def test_selected_regularization_survives_checkpoint_roundtrip(tmp_path):
    from whole_law_basket import CalibratedMissionLaw

    cfg, split = tiny_data()
    model = CalibratedMissionLaw(cfg)
    selected = {"w_cross_var": 80.0, "w_pr_within": 4.0}
    model.set_regularization_config(selected)
    manifest = manifest_for(model, split)
    save_checkpoint(tmp_path, "model", model, manifest, {"seconds": 1.0})
    restored = CalibratedMissionLaw(cfg)
    assert restore_checkpoint(tmp_path, "model", restored, manifest) is not None
    assert restored._reg_config == selected
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
    )


def test_failed_energy_blocks_composite_training(tmp_path, monkeypatch):
    cfg, split = tiny_data()
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
            torch.ones(4),
            split,
            split,
            split,
            {"house": split[0]["house"].numpy()},
        ),
    )
    calls = []

    def failed_fit(model, train, validation, **kwargs):
        calls.append(type(model).__name__)
        record = certify_fit(
            model, train, validation, kwargs["ridge"], kwargs["dense_scale"]
        )
        return {**record, "stationarity_passed": False, "seconds": 0.0}

    monkeypatch.setattr(runner, "fit_lbfgsb", failed_fit)
    with pytest.raises(RuntimeError, match="dependency is not converged"):
        runner.evaluate_real_dataset("erim", tmp_path / "out", "proposed_whole_law")
    assert calls == ["StructuredEnergy"]
    assert (tmp_path / "out/checkpoints/erim/structured_energy_record.json").exists()


def test_synthetic_resume_evaluates_without_refitting(tmp_path, monkeypatch):
    cfg, train, validation, test = runner.generate_synthetic_splits(
        "latent_missions_gumbel", 941001, sizes=(80, 32, 32)
    )
    monkeypatch.setattr(
        runner,
        "_load_synthetic_split_pair",
        lambda *args: (cfg, train[:2], validation[:2], test[:2], test[2]),
    )
    acts = runner.actions()
    short_actions = [
        next(action for action in acts if action["kind"] == kind)
        for kind in ("none", "single", "increase", "paired")
    ]
    monkeypatch.setattr(runner, "actions", lambda: short_actions)
    fit_impl = runner.fit_lbfgsb
    calls = []

    def count_fit(*args, **kwargs):
        calls.append(True)
        return fit_impl(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_lbfgsb", count_fit)
    first = runner.evaluate_synthetic_benchmarks(
        tmp_path, "missions_b", "popularity", "fresh"
    )
    second = runner.evaluate_synthetic_benchmarks(
        tmp_path, "missions_b", "popularity", "resume"
    )
    assert len(calls) == 1
    old, new = first["worlds"]["missions_b"][0], second["worlds"]["missions_b"][0]
    assert new["checkpoint_reused"]
    assert old["test_ll"] == new["test_ll"]
    assert old["same_category_rel_mae"] == new["same_category_rel_mae"]
