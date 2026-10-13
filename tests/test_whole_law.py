"""Unit tests verifying exact nonempty partition normalization, sampler validity, and race-PL quadrature."""

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

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
    ShopperFactorAR,
    LogisticNormalFactorMixture,
    FastNeuralNADE,
)


def _all_nonempty_baskets(C: int, P: int) -> torch.Tensor:
    J = C * P
    rows = [list(bits) for bits in itertools.product((0.0, 1.0), repeat=J) if any(bits)]
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, C, P)


def test_race_pl_quadrature():
    res = verify_race_pl_quadrature()
    assert res["P4_max_log_prob_error_all_subsets"] < 1e-11
    assert res["P4_max_autograd_error"] < 1e-11
    assert res["large_category_checks"]["P_16"]["max_log_prob_error_vs_exact_combinatorial"] < 1e-11
    assert res["large_category_checks"]["P_64"]["max_log_prob_error_vs_exact_combinatorial"] < 1e-11


def test_all_models_exact_normalization_and_sampling():
    C, P = 2, 4
    J = C * P
    cfg = {"C": C, "P": P, "R": P, "users": 4, "stores": 2}
    gen = torch.Generator().manual_seed(42)
    B = 3
    ctx = {
        "house": torch.tensor([0, 1, 2], dtype=torch.long),
        "dlp": 0.2 * torch.randn((B, C, P), generator=gen, dtype=torch.float64),
        "baseline": 0.3 * torch.randn((B, C, P), generator=gen, dtype=torch.float64) - 0.8,
    }
    all_baskets = _all_nonempty_baskets(C, P)
    n_sets = len(all_baskets)

    re_base = StructuredEnergy(cfg, [(0, 4), (1, 5)])
    with torch.no_grad():
        re_base.lam.copy_(0.2 * torch.randn(J, generator=gen, dtype=torch.float64))
        re_base.category_count.copy_(torch.tensor([-0.4, -0.3], dtype=torch.float64))
        re_base.global_count.fill_(-0.1)
        re_base.link.copy_(torch.tensor([0.3, -0.2], dtype=torch.float64))

    M_cat_init = 0.2 * torch.randn((4, C), generator=gen, dtype=torch.float64)
    M_prod_fixed = 0.2 * torch.randn((6, J), generator=gen, dtype=torch.float64)

    models = {
        "popularity": PooledPopularity(cfg),
        "no_price": NoPriceIndependent(cfg),
        "independent_parent": PersonalizedIndependent(cfg),
        "size_aware_esp": SizeAwareESP(cfg),
        "independent_categories_esp": IndependentCategoriesESP(cfg),
        "mission_esp": LatentMissionESP(cfg, K=2, seed=42),
        "pairwise_ar": PairwiseBinaryAR(cfg),
        "shopper_factor_ar": ShopperFactorAR(cfg, rank=4, seed=42),
        "logistic_normal_vae": LogisticNormalFactorMixture(cfg, n_factors=4, n_nodes=8, seed=42),
        "neural_nade": FastNeuralNADE(cfg, hidden=8, seed=42),
        "structured_energy": re_base,
        "composite_whole_law": CompositeWholeLawEnergy(cfg, re_base, M_cat_init, M_prod_fixed),
        "calibrated_mission_law": CalibratedMissionLaw(cfg),
    }

    for name, model in models.items():
        # 1. Check exact normalization over all 2^8 - 1 = 255 nonempty baskets
        for b_idx in range(B):
            sub_ctx = {
                k: v[b_idx : b_idx + 1].expand(n_sets, *([-1] * (v.ndim - 1)))
                for k, v in ctx.items()
            }
            with torch.no_grad():
                lp = model.log_probability(sub_ctx, all_baskets)
            total_prob = float(torch.logsumexp(lp, dim=0).exp().item())
            assert abs(total_prob - 1.0) < 1e-9, f"{name} failed exact normalization: sum={total_prob}"

        # 2. Check sampler returns valid nonempty binary baskets of shape (B, C, P)
        samples = model.sample(ctx, seed=123)
        assert samples.shape == (B, C, P), f"{name} bad sample shape {samples.shape}"
        assert ((samples == 0) | (samples == 1)).all(), f"{name} non-binary samples"
        assert samples.sum(dim=(1, 2)).min().item() >= 1, f"{name} generated empty basket"


if __name__ == "__main__":
    test_race_pl_quadrature()
    test_all_models_exact_normalization_and_sampling()
    print("ALL 13 MODELS & QUADRATURE PASSED EXACT NORMALIZATION AND SAMPLING TESTS!")
