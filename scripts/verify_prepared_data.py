"""Verify bundled prepared inputs; optionally exercise every dataset loader."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def verify_files(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = data_dir / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if (
            path.stat().st_size != entry["bytes"]
            or digest.hexdigest() != entry["sha256"]
        ):
            failures.append(f"checksum/size mismatch: {entry['path']}")
    if failures:
        raise ValueError("Prepared data verification failed:\n" + "\n".join(failures))
    if sum(entry["bytes"] for entry in manifest["files"]) != manifest["total_bytes"]:
        raise ValueError("Manifest total does not match file sizes")
    print(
        f"Verified {len(manifest['files'])} files ({manifest['total_bytes'] / 1024**2:.2f} MiB)"
    )
    return manifest


def load_datasets(manifest: dict):
    import importlib.util
    import torch
    from whole_law_basket import load_real_dataset

    torch.set_num_threads(1)
    for dataset in ("erim", "dunnhumby"):
        config, _, train, validation, test, _ = load_real_dataset(dataset)
        expected = manifest["datasets"][dataset]
        assert config["C"] * config["P"] == expected["products"], dataset
        counts = [len(split[1]) for split in (train, validation, test)]
        assert counts == [
            expected[f"{split}_baskets"] for split in ("train", "validation", "test")
        ], dataset
        print(
            f"{dataset}: {expected['products']} products; train/validation/test = {counts}"
        )

    spec = importlib.util.spec_from_file_location(
        "basket_benchmark", REPO_ROOT / "benchmarks/run_all_benchmarks.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    expected = manifest["datasets"]["synthetic"]
    for name in expected["worlds"]:
        zero_price = name == "zero_price"
        family = (
            "latent_missions_gumbel_zero_price"
            if zero_price
            else "latent_missions_gumbel"
        )
        seed = {"missions_a": 941000, "missions_b": 941001, "zero_price": 943000}[name]
        config, train, validation, test, _ = benchmark._load_synthetic_split_pair(
            name, family, seed
        )
        assert config["C"] * config["P"] == expected["products_per_world"], name
        counts = [len(split[1]) for split in (train, validation, test)]
        assert counts == [
            expected[f"{split}_baskets_per_world"]
            for split in ("train", "validation", "test")
        ], name
        print(
            f"{name}: {expected['products_per_world']} products; train/validation/test = {counts}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Also check model-ready tensors and split counts",
    )
    args = parser.parse_args()
    manifest = verify_files(REPO_ROOT / "data")
    if args.load:
        load_datasets(manifest)


if __name__ == "__main__":
    main()
