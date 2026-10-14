"""Self-contained synthetic basket worlds, data generators, and exact oracle evaluators.

Vendors the synthetic data-generating mechanisms (`latent_missions_gumbel`,
`latent_missions_gumbel_zero_price`, `product_ising`) and oracle evaluation
metrics so `whole_law_basket_repo` operates with zero external code dependencies.
"""

import itertools
from pathlib import Path
import numpy as np
from scipy.special import expit, softmax
import torch

J = 12
C = 3
P = 4
BITS = ((np.arange(1 << J)[:, None] >> np.arange(J)) & 1).astype(float)
SETS = BITS[1:]
SIZE = SETS.sum(1).astype(int)
CAT = (SETS.reshape(-1, C, P).sum(-1) > 0).astype(float)
LOCAL = ((np.arange(16)[:, None] >> np.arange(P)) & 1).astype(float)
LOCALSIZE = LOCAL.sum(1).astype(int)
CODE = (SETS.reshape(-1, C, P) @ (2 ** np.arange(P))).astype(int)
PATS = np.array(list(itertools.product((0.0, 1.0), repeat=C)))
CATCODE = (CAT @ np.array([4, 2, 1])).astype(int)


def _normalized(p: np.ndarray) -> np.ndarray:
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.max(np.abs(p.sum(-1) - 1.0)) < 2e-9
    return p


def world_parts(w: dict, house: np.ndarray, price: np.ndarray, mission: int):
    index = price.mean(-1)
    eta = np.array([-0.5, -0.8, -0.6]) + w["house_need"][house] + w["missions"][mission]
    eta = eta - index * np.array([1.0, 1.6, 2.2]) + 0.25 * np.roll(index, 1, axis=-1)
    eta -= 0.35 * np.maximum(index, 0.0)
    q = expit(eta)
    patterns = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
    mass = np.prod(np.where(patterns[None] > 0, q[:, None], 1.0 - q[:, None]), axis=-1)
    mass[:, 0] = 0.0
    mass /= mass.sum(-1, keepdims=True)
    extra = expit(
        np.array([-0.8, -1.1, -0.6])
        + 0.15 * w["missions"][mission]
        - index * np.array([0.8, 1.2, 1.6])
    )
    count = np.stack(
        [(1.0 - extra) ** 2, 2.0 * extra * (1.0 - extra), extra**2], axis=-1
    )
    utility = (
        w["quality"]
        + w["taste"][house]
        + w["mission_taste"][mission]
        - w["beta"] * price
    )
    return patterns, mass, count, utility


def synthetic_sample(w: dict, N: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 16, N)
    price = rng.uniform(-0.35, 0.35, (N, 3, 4))
    y = np.zeros_like(price)
    mission = rng.choice(2, N, p=[0.55, 0.45])
    for m in range(2):
        rows = np.flatnonzero(mission == m)
        patterns, mass, count, u = world_parts(w, h[rows], price[rows], m)
        chosen = (rng.random((len(rows), 1)) > mass.cumsum(-1)).sum(-1).clip(max=7)
        present = patterns[chosen]
        size = 1 + (rng.random((len(rows), 3, 1)) > count.cumsum(-1)).sum(-1).clip(
            max=2
        )
        order = np.argsort(-(u + rng.gumbel(size=u.shape)), axis=-1)
        for c in range(3):
            for k in range(3):
                take = (present[:, c] > 0) & (size[:, c] > k)
                y[rows[take], c, order[take, c, k]] = 1.0
    return {
        "house": h + 1,
        "store": np.zeros(N, int),
        "week": rng.integers(1, 53, N),
        "dlp": price,
        "disp": np.zeros_like(price),
        "mail": np.zeros_like(price),
        "member": y,
    }


class World:
    def __init__(self, row: dict):
        self.row = row
        self.seed = int(row["seed"])
        self.family = str(row["family"])
        r = np.random.default_rng(self.seed)
        self.w = {
            "taste": r.normal(0.0, 0.5, (16, C, P)),
            "quality": r.normal(0.0, 0.4, (C, P)),
            "beta": r.uniform(0.8, 1.8, (C, P)),
            "house_need": r.normal(0.0, 0.25, (16, C)),
            "missions": np.array([[0.7, -0.4, 0.3], [-0.5, 0.6, -0.2]]),
            "mission_taste": r.normal(0.0, 0.15, (2, C, P)),
        }
        if self.family == "product_ising":
            self.taste = r.normal(0.0, 0.45, (16, J))
            self.quality = r.normal(-1.8, 0.25, J)
            self.beta = r.uniform(1.0, 2.5, J)
            self.pair = np.zeros((J, J))
            for c in range(C):
                for a, b in itertools.combinations(range(c * P, (c + 1) * P), 2):
                    self.pair[a, b] = -0.7
            for a, b in ((0, 4), (1, 8), (5, 9), (3, 11)):
                self.pair[a, b] = 1.5
            self.energy = (SETS @ self.pair * SETS).sum(1) - 0.08 * (SIZE - 1) ** 2

    def probability(self, house: np.ndarray, price: np.ndarray) -> np.ndarray:
        if self.family == "product_ising":
            d = price.reshape(-1, J)
            u = (
                self.quality
                + self.taste[house - 1]
                - self.beta * d
                - 0.15 * np.maximum(d, 0.0)
            )
            return _normalized(softmax(u @ SETS.T + self.energy, axis=-1))
        if "zero_price" in self.family:
            price = np.zeros_like(price)
        out = np.zeros((len(house), len(SETS)))
        for m, weight in enumerate((0.55, 0.45)):
            _, mass, count, u = world_parts(self.w, house - 1, price, m)
            within = np.zeros((len(house), C, 16))
            within[:, :, 0] = 1.0
            for code in range(1, 16):
                items = np.flatnonzero(LOCAL[code])
                n = len(items)
                if n > 3:
                    continue
                prob = np.zeros((len(house), C))
                for order in itertools.permutations(items):
                    logits = u.copy()
                    q = np.ones((len(house), C))
                    for j in order:
                        q *= softmax(logits, axis=-1)[..., j]
                        logits[..., j] = -np.inf
                    prob += q
                within[:, :, code] = count[:, :, n - 1] * prob
            joint = mass[:, CATCODE]
            for c in range(C):
                joint = joint * within[:, c, CODE[:, c]]
            out += weight * joint
        return _normalized(out)

    def sample(self, N: int, seed: int) -> dict:
        if self.family != "product_ising":
            if "zero_price" not in self.family:
                return synthetic_sample(self.w, N, seed)
            r = np.random.default_rng(seed)
            h = r.integers(0, 16, N)
            price = r.uniform(-0.35, 0.35, (N, C, P))
            y = np.zeros_like(price)
            mission = r.choice(2, N, p=[0.55, 0.45])
            for m in range(2):
                rows = np.flatnonzero(mission == m)
                pats, mass, count, u = world_parts(
                    self.w, h[rows], np.zeros_like(price[rows]), m
                )
                chosen = (
                    (r.random((len(rows), 1)) > mass.cumsum(-1)).sum(-1).clip(max=7)
                )
                present = pats[chosen]
                size = 1 + (r.random((len(rows), C, 1)) > count.cumsum(-1)).sum(
                    -1
                ).clip(max=2)
                order = np.argsort(-(u + r.gumbel(size=u.shape)), axis=-1)
                for c in range(C):
                    for k in range(3):
                        take = (present[:, c] > 0) & (size[:, c] > k)
                        y[rows[take], c, order[take, c, k]] = 1.0
            return {
                "house": h + 1,
                "store": np.zeros(N, int),
                "week": r.integers(1, 53, N),
                "dlp": price,
                "disp": np.zeros_like(price),
                "mail": np.zeros_like(price),
                "member": y,
            }
        r = np.random.default_rng(seed)
        h = r.integers(1, 17, N)
        price = r.uniform(-0.35, 0.35, (N, C, P))
        y = []
        for k in range(0, N, 256):
            q = self.probability(h[k : k + 256], price[k : k + 256])
            cdf = q.cumsum(-1)
            cdf[:, -1] = 1.0
            code = (r.random((len(q), 1)) > cdf).sum(-1)
            y.append(SETS[code].reshape(-1, C, P))
        return {
            "house": h,
            "store": np.zeros(N, int),
            "week": r.integers(1, 53, N),
            "dlp": price,
            "disp": np.zeros_like(price),
            "mail": np.zeros_like(price),
            "member": np.concatenate(y),
        }


def moments(p: np.ndarray) -> dict:
    return {
        "incidence": (p @ SETS).reshape(-1, C, P),
        "presence": p @ CAT,
        "basket_size": p @ SIZE,
    }


def actions() -> list[dict]:
    out = [{"name": "no_promotion", "shift": np.zeros(J), "items": [], "kind": "none"}]
    for j in range(J):
        for frac in (0.05, 0.1, 0.2):
            d = np.zeros(J)
            d[j] = np.log(1.0 - frac)
            out.append(
                {
                    "name": f"discount_{j}_{frac}",
                    "shift": d,
                    "items": [j],
                    "kind": "single",
                }
            )
        d = np.zeros(J)
        d[j] = np.log(1.1)
        out.append(
            {"name": f"increase_{j}_0.1", "shift": d, "items": [j], "kind": "increase"}
        )
    for a, b in ((0, 4), (1, 5), (2, 8), (3, 9), (6, 10), (7, 11)):
        d = np.zeros(J)
        d[[a, b]] = np.log(0.9)
        out.append(
            {"name": f"pair_{a}_{b}", "shift": d, "items": [a, b], "kind": "paired"}
        )
    return out


def response_score(pred: list[dict], truth: list[dict], acts: list[dict]) -> dict:
    result = {}
    for kind in ("single", "increase", "paired"):
        ids = [i for i, a in enumerate(acts) if a["kind"] == kind]
        row = {}
        for key in ("incidence", "presence", "basket_size"):
            t = np.stack([truth[i][key] - truth[0][key] for i in ids])
            p = np.stack([pred[i][key] - pred[0][key] for i in ids])
            mae = float(np.abs(p - t).mean())
            zero = float(np.abs(t).mean())
            den = float((t * t).sum())
            row[key] = {
                "mae": mae,
                "zero_effect_mae": zero,
                "relative_mae": mae / zero if zero > 1e-12 else None,
                "slope": float((p * t).sum() / den) if den > 1e-18 else None,
                "false_effect_fraction_above_0.005": (
                    float((np.abs(p) > 0.005).mean()) if zero < 1e-12 else None
                ),
            }
            row[key]["passed"] = bool(
                mae <= 0.005
                if zero < 1e-12
                else (mae / zero <= 0.25 and 0.75 <= row[key]["slope"] <= 1.25)
            )
        result[kind] = row
    for group in ("own", "same_category", "cross_category"):
        ps, ts = [], []
        for i, a in enumerate(acts):
            if a["kind"] != "single":
                continue
            j = a["items"][0]
            idx = (
                [j]
                if group == "own"
                else [
                    k
                    for k in range(J)
                    if k != j
                    and (
                        k // P == j // P
                        if group == "same_category"
                        else k // P != j // P
                    )
                ]
            )
            ps.append(
                (pred[i]["incidence"] - pred[0]["incidence"])
                .reshape(-1, J)[:, idx]
                .ravel()
            )
            ts.append(
                (truth[i]["incidence"] - truth[0]["incidence"])
                .reshape(-1, J)[:, idx]
                .ravel()
            )
        p = np.concatenate(ps)
        t = np.concatenate(ts)
        z = float(np.abs(t).mean())
        den = float(t @ t)
        result[group] = {
            "mae": float(np.abs(p - t).mean()),
            "relative_mae": float(np.abs(p - t).mean() / z) if z > 1e-12 else None,
            "slope": float(p @ t / den) if den > 1e-18 else None,
        }
    return result


def generation(prob: np.ndarray, oracle: np.ndarray, house: np.ndarray) -> dict:
    p = moments(prob)
    t = moments(oracle)
    dist = lambda q, z, n: np.bincount(z, weights=q.mean(0), minlength=n)
    pairs = np.stack([CAT[:, i] * CAT[:, j] for i, j in ((0, 1), (0, 2), (1, 2))], -1)
    multiple = (SETS.reshape(-1, 3, 4).sum(-1) > 1).mean(-1)
    return {
        "size_tv": float(
            0.5 * np.abs(dist(prob, SIZE, 13) - dist(oracle, SIZE, 13)).sum()
        ),
        "product_frequency_mae": float(
            np.abs(p["incidence"].mean(0) - t["incidence"].mean(0)).mean()
        ),
        "category_pair_mae": float(
            np.abs(prob.mean(0) @ pairs - oracle.mean(0) @ pairs).mean()
        ),
        "multiple_rate_error": float(
            np.abs(prob.mean(0) @ multiple - oracle.mean(0) @ multiple)
        ),
        "shopper_mean_size_mae": float(
            np.mean(
                [
                    np.abs(
                        p["basket_size"][house == h].mean()
                        - t["basket_size"][house == h].mean()
                    )
                    for h in np.unique(house)
                ]
            )
        ),
        "mean_size": float(p["basket_size"].mean()),
        "oracle_mean_size": float(t["basket_size"].mean()),
    }


def global_promotion(
    pred: list[dict], truth: list[dict], price: np.ndarray, acts: list[dict]
) -> list[dict]:
    ids = [i for i, a in enumerate(acts) if a["kind"] != "increase"]
    rows = []
    for cost in (0.3, 0.6, 0.85):
        for funding in (0.0, 0.1):
            p, t = [], []
            for i in ids:
                action = acts[i]
                margin = np.exp(price.reshape(-1, 12) + action["shift"]) - cost
                margin[:, action["items"]] += funding
                p.append((pred[i]["incidence"].reshape(-1, 12) * margin).sum(-1))
                t.append((truth[i]["incidence"].reshape(-1, 12) * margin).sum(-1))
            p = np.stack(p)
            t = np.stack(t)
            selected = int(p.mean(-1).argmax())
            best = int(t.mean(-1).argmax())
            rows.append(
                {
                    "cost": cost,
                    "funding": funding,
                    "selected": acts[ids[selected]]["name"],
                    "oracle_selected": acts[ids[best]]["name"],
                    "achieved_margin": float(t[selected].mean()),
                    "best_margin": float(t[best].mean()),
                    "regret": float(t[best].mean() - t[selected].mean()),
                    "optimal": selected == best,
                }
            )
    return rows


def generate_synthetic_splits(
    family: str, seed: int, sizes=(20000, 5000, 5000)
) -> tuple:
    """Deterministically generate train, validation, and test splits in-memory (no external files required)."""
    w = World({"family": family, "seed": seed})
    cfg = {
        "C": 3,
        "P": 4,
        "R": 4,
        "users": 17,
        "stores": 1,
        "calendar_support": [1, 52],
    }
    splits = []
    for offset, N in zip((101, 102, 103), sizes):
        raw = w.sample(N, seed + offset)
        ctx = {
            "house": torch.tensor(raw["house"], dtype=torch.long),
            "store": torch.tensor(raw["store"], dtype=torch.long),
            "week": torch.tensor(raw["week"], dtype=torch.float64),
            "dlp": torch.tensor(raw["dlp"], dtype=torch.float64),
            "disp": torch.tensor(raw["disp"], dtype=torch.float64),
            "mail": torch.tensor(raw["mail"], dtype=torch.float64),
        }
        y = torch.tensor(raw["member"], dtype=torch.float64)
        splits.append((ctx, y, raw["house"]))
    return cfg, splits[0], splits[1], splits[2]
