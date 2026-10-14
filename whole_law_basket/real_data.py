"""Self-contained Real-Data Loader and Parent Utility Evaluator for ERIM and Dunnhumby.

Operates strictly from bundled repository paths under `data/` with zero external
directory or package dependencies.
"""

import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]


class ParentControls(nn.Module):
    """Standalone preference parent evaluator (replaces external TrainableBasketModel)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.C, self.P, self.R = (int(config[k]) for k in ("C", "P", "R"))
        J = self.C * self.P
        users, stores = int(config["users"]), int(config["stores"])
        taste, store_rank, hidden, cross = (
            int(config.get(k, d))
            for k, d in (
                ("taste_rank", 4),
                ("store_rank", 2),
                ("hidden", 8),
                ("cross_rank", min(self.C, 4)),
            )
        )
        chain_rank = int(config.get("chain_rank", 4 if self.C > 10 else 0))

        def zero(*shape):
            return nn.Parameter(torch.zeros(*shape, dtype=torch.float64))

        self.lam = zero(J)
        self.alpha = zero(J, taste)
        self.theta = zero(users, taste)
        self.zeta = zero(J, store_rank)
        self.xi = zero(stores, store_rank)
        self.mu = zero(J, 4)
        self.w_dsp = zero(J)
        self.w_mlr = zero(J)
        self.rho = zero(self.C)
        if config.get("learn_count_offset", False):
            self.count_offset = zero(self.C)
        self.choice_price = zero(J)
        self.category_price = zero(self.C, self.P)
        self.count_price_scale = zero(self.C)
        self.kappa = zero(self.C)
        self.tau = zero(self.C)
        self.cross_p = zero(self.C, cross)
        self.cross_q = zero(self.C, cross)
        self.V = zero(self.C, self.C) if not chain_rank else zero(0, 0)
        self.W = zero(self.C, hidden)
        self.h0 = zero(hidden)
        self.omega = zero(self.C, hidden)
        self.U = zero(self.C, taste)
        mode = config.get("preference_mode", "factors")
        if mode == "direct":
            self.house_effect = zero(users, J)
            self.store_effect = zero(stores, J)
        if config.get("learn_inclusive_price", False):
            self.inclusive_price_weight = zero(self.C)
        if config.get("category_anchor", "legacy") == "independent":
            self.register_buffer("independent_category_anchor", torch.tensor(True))
        if chain_rank:
            self.linear_p = zero(self.C, chain_rank)
            self.linear_q = zero(self.C, chain_rank)
        self.register_buffer("psi0", torch.zeros(self.C + 1, dtype=torch.float64))
        self.register_buffer("psi1", torch.zeros(self.C, dtype=torch.float64))
        self.register_buffer("layout", torch.arange(J).reshape(self.C, self.P))
        self.register_buffer(
            "order",
            torch.tensor(config.get("order", list(range(self.C))), dtype=torch.long),
        )
        self.register_buffer(
            "calendar_support",
            torch.tensor(config["calendar_support"], dtype=torch.float64),
        )
        self.register_buffer("separate_category_price", torch.tensor(True))

    def baseline(self, ctx: dict) -> torch.Tensor:
        week = ctx["week"].clamp(self.calendar_support[0], self.calendar_support[1])
        phase = (week - 1.0) * 2.0 * math.pi / 52.0
        cal = torch.stack(
            (phase.sin(), phase.cos(), (2.0 * phase).sin(), (2.0 * phase).cos()), -1
        )
        if hasattr(self, "house_effect"):
            b = (
                self.lam
                + self.house_effect[ctx["house"]]
                + self.store_effect[ctx["store"]]
                + cal @ self.mu.T
            )
        else:
            b = (
                self.lam
                + self.theta[ctx["house"]] @ self.alpha.T
                + self.xi[ctx["store"]] @ self.zeta.T
                + cal @ self.mu.T
            )
        b = b.reshape(-1, self.C, self.P)
        return (
            b
            + self.w_dsp.reshape(self.C, self.P) * ctx["disp"]
            + self.w_mlr.reshape(self.C, self.P) * ctx["mail"]
        )


def load_real_dataset(
    dataset: str,
) -> tuple[dict, torch.Tensor, tuple, tuple, tuple, dict]:
    """Load (config, price, train, validation, test, clusters) for 'erim' or 'dunnhumby' from bundled `data/`."""
    if dataset not in ("erim", "dunnhumby"):
        raise ValueError(f"Unknown real dataset: {dataset}")
    source = REPO_ROOT / "data" / dataset
    parent_dir = REPO_ROOT / "data" / f"{dataset}_parent"
    if not (source / "config.json").exists():
        raise FileNotFoundError(
            f"Missing bundled dataset config at {source / 'config.json'}"
        )
    if not (parent_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Missing bundled parent config at {parent_dir / 'config.json'}"
        )

    ckpt_name = "best.pt" if (parent_dir / "best.pt").exists() else "final.pt"
    parent_cfg = json.loads((parent_dir / "config.json").read_text())
    parent = ParentControls(parent_cfg)
    parent.load_state_dict(
        torch.load(parent_dir / ckpt_name, map_location="cpu", weights_only=True)
    )
    parent.eval()
    for p in parent.parameters():
        p.requires_grad_(False)

    config = json.loads((source / "config.json").read_text())
    if dataset == "erim":
        items = np.arange(28)
    else:
        with np.load(source / "train.npz", allow_pickle=False) as data:
            y_all = np.unpackbits(data["packed_member"], axis=1).reshape(-1, 64, 16)
        categories = np.sort(np.argsort(-y_all.sum((0, 2)), kind="stable")[:16])
        items = (16 * categories[:, None] + np.arange(4)).ravel()
        config = {
            **config,
            "C": 16,
            "P": 4,
            "R": 4,
            "products": [config["products"][i] for i in items],
            "categories": [config["categories"][i] for i in categories],
        }

    price = parent.choice_price[items].detach().clone()
    splits = {}
    clusters = {}
    shared = (
        {
            k: np.load(source / f"{k}.npy", mmap_mode="r")
            for k in ("dlp", "disp", "mail")
        }
        if (source / "dlp.npy").exists()
        else None
    )
    for split_name in ("train", "validation", "test"):
        C, P = config["C"], config["P"]
        with np.load(source / f"{split_name}.npz", allow_pickle=False) as archive:
            raw = {k: archive[k] for k in archive.files}
        if "packed_member" in raw:
            full_y = np.unpackbits(raw["packed_member"], axis=1)
            projected = full_y[:, items].reshape(-1, C, P)
            keep = np.flatnonzero(projected.any((1, 2)))
            indices = raw["context"][keep]
        else:
            projected = raw["member"]
            keep = np.arange(len(projected))
            indices = None
        ctx = {
            k: torch.tensor(
                raw[k][keep],
                dtype=torch.long if k in ("house", "store") else torch.float64,
            )
            for k in ("house", "store", "week")
        }
        n = len(keep)
        for k in ("dlp", "disp", "mail", "baseline"):
            ctx[k] = torch.empty((n, C, P), dtype=torch.float64)
        with torch.no_grad():
            for start in range(0, n, 8192):
                end = min(start + 8192, n)
                batch = {k: ctx[k][start:end] for k in ("house", "store", "week")}
                for k in ("dlp", "disp", "mail"):
                    values = (
                        raw[k][keep[start:end]]
                        if shared is None
                        else np.array(shared[k][indices[start:end]], copy=True)
                    )
                    batch[k] = torch.tensor(values, dtype=torch.float64)
                    ctx[k][start:end] = (
                        batch[k].reshape(end - start, -1)[:, items].reshape(-1, C, P)
                    )
                ctx["baseline"][start:end] = (
                    parent.baseline(batch)
                    .reshape(end - start, -1)[:, items]
                    .reshape(-1, C, P)
                )
        y = torch.tensor(projected[keep], dtype=torch.float64)
        splits[split_name] = (ctx, y)
        if split_name == "test":
            clusters = {
                "house": raw.get("raw_household", raw["house"])[keep],
                "store": raw.get("raw_store", raw["store"])[keep],
                "week": raw["week"][keep],
                "source_rows": keep,
            }
    return (
        config,
        price,
        splits["train"],
        splits["validation"],
        splits["test"],
        clusters,
    )
