"""Content-addressed benchmark inputs and checkpoints.

Legacy result tables are evidence from their original runs, never a resume
cache. Reuse requires this schema and matching input, code, and fit manifests.
Evaluation is always performed again from restored weights.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

SCHEMA_VERSION = 2


def json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def state_hash(model) -> str:
    return json_hash({k: array_hash(v) for k, v in model.state_dict().items()})


def code_hash(root: Path) -> str:
    paths = sorted((root / "whole_law_basket").glob("*.py"))
    paths += sorted((root / "benchmarks").glob("*.py"))
    return json_hash({str(p.relative_to(root)): file_hash(p) for p in paths})


def dataset_manifest(config, splits, clusters=None) -> dict:
    return {
        "config": config,
        "splits": {
            name: {
                "rows": len(y),
                "baskets": array_hash(y),
                "context": {k: array_hash(v) for k, v in sorted(ctx.items())},
            }
            for name, (ctx, y) in zip(("train", "validation", "test"), splits)
        },
        "clusters": {k: array_hash(v) for k, v in sorted((clusters or {}).items())},
    }


def model_manifest(model, data, code, policy, seed, dependencies=None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "data_hash": json_hash(data),
        "code_hash": code,
        "policy": policy,
        "seed": seed,
        "model_config": model.config,
        "parameters": {
            k: {
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "trainable": p.requires_grad,
            }
            for k, p in model.named_parameters()
        },
        "fixed_buffers": {k: array_hash(v) for k, v in model.named_buffers()},
        "frozen_parameters": {
            k: array_hash(p) for k, p in model.named_parameters() if not p.requires_grad
        },
        "dependencies": dependencies or {},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": str(torch.__version__),
        },
    }


def save_checkpoint(
    directory, name, model, manifest, fit_record, summary=None, evaluation_files=()
) -> dict:
    """Save complete nested weights, trainability, selected prior, and checksums."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "config": model.config,
            "regularization_config": getattr(model, "_reg_config", None),
            "trainable_parameter_names": [
                k for k, p in model.named_parameters() if p.requires_grad
            ],
        },
        temporary,
    )
    temporary.replace(path)
    record = {
        "schema_version": SCHEMA_VERSION,
        "model": name,
        "manifest": manifest,
        "manifest_hash": json_hash(manifest),
        "checkpoint_sha256": file_hash(path),
        "fit_record": fit_record,
        "summary": summary,
        "evaluation_files": {
            Path(p).name: file_hash(Path(p)) for p in evaluation_files
        },
    }
    record_path = directory / f"{name}_record.json"
    temporary_record = record_path.with_suffix(".json.tmp")
    temporary_record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary_record.replace(record_path)
    return record


def restore_checkpoint(directory, name, model, expected_manifest) -> dict | None:
    """Reject legacy, changed-data, changed-code, or corrupted checkpoints.

    A rejected checkpoint does not modify the model. No saved evaluation result
    is reused: the caller evaluates the restored model on current inputs.
    """
    directory = Path(directory)
    path, record_path = directory / f"{name}.pt", directory / f"{name}_record.json"
    if not path.exists() or not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        if record.get("schema_version") != SCHEMA_VERSION:
            return None
        expected_hash = json_hash(expected_manifest)
        if (
            record.get("manifest_hash") != expected_hash
            or json_hash(record["manifest"]) != expected_hash
        ):
            return None
        if file_hash(path) != record.get("checkpoint_sha256"):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None
        expected_names = [k for k, p in model.named_parameters() if p.requires_grad]
        if (
            payload["trainable_parameter_names"] != expected_names
            or payload["config"] != model.config
        ):
            return None
        state = payload["state_dict"]
        current = model.state_dict()
        if state.keys() != current.keys() or any(
            state[k].shape != current[k].shape or state[k].dtype != current[k].dtype
            for k in current
        ):
            return None
    except (OSError, ValueError, KeyError, RuntimeError, EOFError):
        return None
    model.load_state_dict(state, strict=True)
    if hasattr(model, "set_regularization_config"):
        model.set_regularization_config(payload["regularization_config"])
    return record
