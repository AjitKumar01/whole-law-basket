"""Unit tests verifying:
1. Mode-centered exponential-race PL quadrature across (P, k) in {(4, all), (16, 3), (64, 3), (64, 32), (64, 63), (1024, 3), (1024, 512)}.
2. End-to-end large-category training (log_probability + backward) and generation (sample) at P=5, 16, 64 for BOTH CalibratedMissionLaw and CompositeWholeLawEnergy.
3. Exact input-price autograd derivative d log P / d dlp matching finite differences for CompositeWholeLawEnergy.
4. Exact category-permutation invariance of CalibratedMissionLaw's hierarchical price penalty (`regularization_loss`).
5. Exact 2^J - 1 nonempty partition normalization and sampler distributional agreement across all 13 models.
6. Unequal (ragged) category sizes P_1, ..., P_C >= 1 across quadrature, DP, exact normalization, packed/padded equivalence, gradients, and L-BFGS-B fitting.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from whole_law_basket import (
    verify_race_pl_quadrature,
    race_observed_log_probability,
    race_subset_log_probabilities,
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
)


def _all_nonempty_baskets(C: int, P: int) -> torch.Tensor:
    J = C * P
    rows = [list(bits) for bits in itertools.product((0.0, 1.0), repeat=J) if any(bits)]
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, C, P)


def _all_nonempty_ragged_baskets(
    category_sizes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (packed_baskets (N, J_active), padded_baskets (N, C, P_max)) for all 2^{J_active}-1 valid nonempty baskets."""
    C = len(category_sizes)
    P_max = max(category_sizes)
    J_active = sum(category_sizes)
    rows = [
        list(bits)
        for bits in itertools.product((0.0, 1.0), repeat=J_active)
        if any(bits)
    ]
    packed = torch.tensor(rows, dtype=torch.float64)
    padded = torch.zeros((len(rows), C, P_max), dtype=torch.float64)
    offset = 0
    for c, p_c in enumerate(category_sizes):
        padded[:, c, :p_c] = packed[:, offset : offset + p_c]
        offset += p_c
    return packed, padded


def test_race_pl_quadrature_all_regimes():
    res = verify_race_pl_quadrature()
    assert res["P4_max_log_prob_error_all_subsets"] < 1e-11
    assert res["P4_max_autograd_error"] < 1e-11
    assert res["P4_observed_vs_subset_dp_error"] < 1e-11
    for p, k in ((64, 3), (64, 32), (64, 63), (1024, 3), (1024, 512)):
        u = torch.zeros((1, p), dtype=torch.float64, requires_grad=True)
        y = torch.zeros((1, p), dtype=torch.float64)
        y[:, :k] = 1.0
        val = race_observed_log_probability(u, y)
        val.sum().backward()
        exact = -math.log(math.comb(p, k))
        err = abs(float(val.detach().item()) - exact)
        assert err < 1e-7, f"Quadrature failed at P={p}, k={k}: err={err}"
        assert float(u.grad.sum().abs().item()) < 1e-12

    # Unequal-utility normalization stress test: P=8, k=7 with utilities [s, s, s, s, s, s, -s, 0]
    for scale in (2.0, 5.0, 10.0, 20.0):
        u_het = torch.tensor([scale] * 6 + [-scale, 0.0], dtype=torch.float64)
        y_7 = torch.ones((8, 8), dtype=torch.float64) - torch.eye(
            8, dtype=torch.float64
        )
        lp_7 = race_observed_log_probability(u_het.expand(8, 8), y_7)
        prob_sum = float(lp_7.exp().sum().item())
        assert (
            abs(prob_sum - 1.0) < 1e-9
        ), f"Unequal utility normalization failed at scale={scale}: sum={prob_sum}"


def test_category_permutation_invariance_of_regularization():
    """Verify that CalibratedMissionLaw.regularization_loss and its gradients are strictly equivariant to category renumbering."""
    C, P = 4, 3
    cfg = {"C": C, "P": P, "R": P, "users": 4, "stores": 1}
    reg_cfg = {
        "w_hinge_var": 1000.0,
        "w_cross_var": 1500.0,
        "w_cat_var": 90.0,
        "w_coherence": 200.0,
        "w_pr_within": 25.0,
        "w_pr_between": 10.0,
        "l2_price": 0.4,
        "l2_hinge_mult": 3.0,
        "l2_cross_mult": 250.0,
    }
    gen = torch.Generator().manual_seed(12345)
    m1 = CalibratedMissionLaw(cfg)
    m1.set_regularization_config(reg_cfg)
    with torch.no_grad():
        m1.price.copy_(torch.rand((C, P), generator=gen, dtype=torch.float64) + 0.5)
        m1.category_price.copy_(torch.rand(C, generator=gen, dtype=torch.float64) + 0.8)
        m1.category_hinge.copy_(0.4 * torch.rand(C, generator=gen, dtype=torch.float64))
        m1.count_price.copy_(torch.rand(C, generator=gen, dtype=torch.float64) + 0.4)
        m1.category_cross.copy_(
            0.3 * torch.randn(C, generator=gen, dtype=torch.float64)
        )

    d_tr = 0.2 * torch.randn((128, C, P), generator=gen, dtype=torch.float64)
    idx_m = d_tr.mean(-1).square().mean(0)
    moments_1 = {
        "price": d_tr.square().mean(0),
        "category_price": idx_m,
        "category_cross": idx_m.roll(1),
        "category_hinge": d_tr.mean(-1).clamp(min=0).square().mean(0),
        "count_price": idx_m,
    }
    loss1 = m1.regularization_loss(moments_1)
    loss1.backward()

    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    m2 = CalibratedMissionLaw(cfg)
    m2.set_regularization_config(reg_cfg)
    with torch.no_grad():
        m2.price.copy_(m1.price.detach()[perm])
        m2.category_price.copy_(m1.category_price.detach()[perm])
        m2.category_hinge.copy_(m1.category_hinge.detach()[perm])
        m2.count_price.copy_(m1.count_price.detach()[perm])
        m2.category_cross.copy_(m1.category_cross.detach()[perm])
    moments_2 = {k: v[perm] for k, v in moments_1.items()}
    loss2 = m2.regularization_loss(moments_2)
    loss2.backward()

    assert abs(float(loss1.item()) - float(loss2.item())) < 1e-12
    for name in (
        "price",
        "category_price",
        "category_hinge",
        "count_price",
        "category_cross",
    ):
        g1 = getattr(m1, name).grad[perm]
        g2 = getattr(m2, name).grad
        assert (
            float((g1 - g2).abs().max().item()) < 1e-12
        ), f"Gradient not permutation-equivariant for {name}"


def test_large_category_end_to_end_both_models():
    gen = torch.Generator().manual_seed(99)
    for P in (5, 16, 64):
        cfg = {"C": 2, "P": P, "R": P, "users": 3, "stores": 1}
        B = 4
        y = torch.zeros((B, 2, P), dtype=torch.float64)
        y[:, 0, : min(P, 5)] = 1.0
        y[1:, 1, : min(P, 3)] = 1.0
        dlp = (
            0.1 * torch.randn((B, 2, P), generator=gen, dtype=torch.float64)
        ).requires_grad_(True)
        ctx = {
            "house": torch.tensor([0, 1, 2, 1], dtype=torch.long),
            "dlp": dlp,
            "baseline": -1.5 * torch.ones((B, 2, P), dtype=torch.float64),
        }

        # 1. CalibratedMissionLaw at P in {5, 16, 64}:
        mission_model = CalibratedMissionLaw(cfg)
        lp_m = mission_model.log_probability(ctx, y)
        assert torch.isfinite(lp_m).all(), f"CalibratedMissionLaw nonfinite at P={P}"
        (-lp_m.mean()).backward()
        assert dlp.grad is not None and torch.isfinite(dlp.grad).all()
        samples_m = mission_model.sample(ctx, seed=7)
        assert samples_m.shape == (B, 2, P) and samples_m.sum((1, 2)).min().item() >= 1

        # 2. CompositeWholeLawEnergy at P in {5, 16, 64} (with intra- and cross-category edges!):
        dlp2 = (
            0.1 * torch.randn((B, 2, P), generator=gen, dtype=torch.float64)
        ).requires_grad_(True)
        ctx2 = {**ctx, "dlp": dlp2}
        re_base = StructuredEnergy(cfg, [(0, 1), (0, P)])
        M_cat = 0.2 * torch.randn((3, 2), generator=gen, dtype=torch.float64)
        M_prod = 0.2 * torch.randn((4, 2 * P), generator=gen, dtype=torch.float64)
        comp_model = CompositeWholeLawEnergy(cfg, re_base, M_cat, M_prod)
        lp_c = comp_model.log_probability(ctx2, y)
        assert torch.isfinite(lp_c).all(), f"CompositeWholeLawEnergy nonfinite at P={P}"
        (-lp_c.mean()).backward()
        assert dlp2.grad is not None and torch.isfinite(dlp2.grad).all()
        samples_c = comp_model.sample(ctx2, seed=11)
        assert samples_c.shape == (B, 2, P) and samples_c.sum((1, 2)).min().item() >= 1


def test_composite_input_price_gradient_and_binomial_sampler():
    cfg = {"C": 2, "P": 4, "R": 4, "users": 2, "stores": 1}
    base = StructuredEnergy(cfg, [(0, 4), (1, 2)])
    with torch.no_grad():
        base.category_count.fill_(-0.4)
        base.link.copy_(torch.tensor([0.5, -0.3], dtype=torch.float64))
    category = torch.tensor([[-0.5, 0.5], [0.5, -0.5]], dtype=torch.float64)
    product = torch.tensor(
        [[0.3, -0.3, 0.3, -0.3, -0.3, 0.3, -0.3, 0.3]], dtype=torch.float64
    )
    composite = CompositeWholeLawEnergy(cfg, base, category, product)
    with torch.no_grad():
        composite.dg[1:] = -0.15 * torch.arange(
            1, len(composite.dg), dtype=torch.float64
        )
        composite.delta_1.copy_(torch.linspace(-0.3, 0.3, 8))
        composite.mix_logits.copy_(torch.tensor([0.2, -0.2, 0.3, -0.1]))
    ctx = {
        "house": torch.tensor([1]),
        "dlp": torch.full((1, 2, 4), 0.1, dtype=torch.float64),
        "baseline": torch.full((1, 2, 4), -1.2, dtype=torch.float64),
    }

    # Exact input price derivative check:
    d = ctx["dlp"].clone().requires_grad_(True)
    y = torch.zeros((1, 2, 4), dtype=torch.float64)
    y[0, 0, :2] = 1.0
    lp = composite.log_probability({**ctx, "dlp": d}, y)
    derivative = torch.autograd.grad(lp.sum(), d)[0]
    eps = 1e-6
    for c_i in range(2):
        for p_i in range(4):
            plus, minus = ctx["dlp"].clone(), ctx["dlp"].clone()
            plus[0, c_i, p_i] += eps
            minus[0, c_i, p_i] -= eps
            with torch.no_grad():
                fd = float(
                    (
                        composite.log_probability({**ctx, "dlp": plus}, y)
                        - composite.log_probability({**ctx, "dlp": minus}, y)
                    )
                    / (2.0 * eps)
                )
            err = abs(float(derivative[0, c_i, p_i].item()) - fd)
            assert (
                err < 1e-8
            ), f"Input price derivative mismatch at ({c_i},{p_i}): autograd={derivative[0,c_i,p_i].item()}, fd={fd}"

    # 16,384-draw binomial interval check across all 255 nonempty baskets:
    sets = torch.tensor(
        (np.arange(1, 256)[:, None] >> np.arange(8)) & 1, dtype=torch.float64
    ).reshape(-1, 2, 4)
    with torch.no_grad():
        mass = (
            composite.log_probability(
                {k: v.expand(255, *v.shape[1:]) for k, v in ctx.items()}, sets
            )
            .exp()
            .numpy()
        )
        assert abs(float(mass.sum()) - 1.0) < 1e-12
        n = 16384
        samples = (
            composite.sample(
                {k: v.expand(n, *v.shape[1:]) for k, v in ctx.items()}, seed=29183
            )
            .numpy()
            .reshape(n, 8)
        )
    codes = (samples * (2 ** np.arange(8))).sum(-1).astype(int)
    frequencies = np.bincount(codes, minlength=256)[1:]
    lo, hi = binom.ppf(1e-5, n, mass), binom.ppf(1.0 - 1e-5, n, mass)
    assert ((frequencies >= lo) & (frequencies <= hi)).all()


def test_all_models_exact_normalization_and_sampling():
    C, P = 2, 4
    J = C * P
    cfg = {"C": C, "P": P, "R": P, "users": 4, "stores": 2}
    gen = torch.Generator().manual_seed(42)
    B = 3
    ctx = {
        "house": torch.tensor([0, 1, 2], dtype=torch.long),
        "dlp": 0.2 * torch.randn((B, C, P), generator=gen, dtype=torch.float64),
        "baseline": 0.3 * torch.randn((B, C, P), generator=gen, dtype=torch.float64)
        - 0.8,
    }
    all_baskets = _all_nonempty_baskets(C, P)
    n_sets = len(all_baskets)

    re_base = StructuredEnergy(cfg, [(0, 4), (1, 5), (0, 1)])
    with torch.no_grad():
        re_base.lam.copy_(0.2 * torch.randn(J, generator=gen, dtype=torch.float64))
        re_base.category_count.copy_(torch.tensor([-0.4, -0.3], dtype=torch.float64))
        re_base.global_count.fill_(-0.1)
        re_base.link.copy_(torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64))

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
        "low_rank_bilinear_ar": ShopperFactorAR(cfg, rank=4, seed=42),
        "gaussian_factor_mixture": LogisticNormalFactorMixture(
            cfg, n_factors=4, n_nodes=8, seed=42
        ),
        "neural_nade": FastNeuralNADE(cfg, hidden=8, seed=42),
        "structured_energy": re_base,
        "composite_whole_law": CompositeWholeLawEnergy(
            cfg, re_base, M_cat_init, M_prod_fixed
        ),
        "calibrated_mission_law": CalibratedMissionLaw(cfg),
    }

    for name, model in models.items():
        for b_idx in range(B):
            sub_ctx = {
                k: v[b_idx : b_idx + 1].expand(n_sets, *([-1] * (v.ndim - 1)))
                for k, v in ctx.items()
            }
            with torch.no_grad():
                lp = model.log_probability(sub_ctx, all_baskets)
            total_prob = float(torch.logsumexp(lp, dim=0).exp().item())
            assert (
                abs(total_prob - 1.0) < 1e-9
            ), f"{name} failed exact normalization: sum={total_prob}"

        samples = model.sample(ctx, seed=123)
        assert samples.shape == (B, C, P), f"{name} bad sample shape {samples.shape}"
        assert ((samples == 0) | (samples == 1)).all(), f"{name} non-binary samples"
        assert (
            samples.sum(dim=(1, 2)).min().item() >= 1
        ), f"{name} generated empty basket"


def test_ragged_category_sizes():
    """Verify unequal category sizes P_1, ..., P_C >= 1 across quadrature, DP, exact normalization, packed/padded equivalence, gradients, and L-BFGS-B fitting."""
    gen = torch.Generator().manual_seed(2026)

    # 1. Ragged exponential-race PL quadrature vs unpadded per-category ground truth (both P_max=4 DP and P_max=6 quadrature):
    for cat_sizes in ([1, 4, 2], [1, 6, 3]):
        C = len(cat_sizes)
        P_max = max(cat_sizes)
        valid_mask = torch.zeros((C, P_max), dtype=torch.bool)
        u_padded = torch.zeros((C, P_max), dtype=torch.float64)
        y_padded = torch.zeros((C, P_max), dtype=torch.float64)
        exact_lps = []
        for c, p_c in enumerate(cat_sizes):
            valid_mask[c, :p_c] = True
            u_c = 0.5 * torch.randn(p_c, generator=gen, dtype=torch.float64)
            u_padded[c, :p_c] = u_c
            k_c = min(p_c, c + 1)
            y_c = torch.zeros(p_c, dtype=torch.float64)
            y_c[:k_c] = 1.0
            y_padded[c, :p_c] = y_c
            exact_lps.append(
                float(race_observed_log_probability(u_c[None], y_c[None]).item())
            )

        u_req = u_padded.clone().requires_grad_(True)
        lp_ragged = race_observed_log_probability(
            u_req, y_padded, valid_mask=valid_mask
        )
        lp_ragged.sum().backward()
        for c in range(C):
            err = abs(float(lp_ragged[c].item()) - exact_lps[c])
            assert (
                err < 1e-11
            ), f"Ragged race_observed_log_probability mismatch at cat {c} (sizes={cat_sizes}): err={err}"
            if cat_sizes[c] < P_max:
                assert (
                    float(u_req.grad[c, cat_sizes[c] :].abs().max().item()) == 0.0
                ), "Non-zero gradient on padded utility slots!"

        if P_max <= 4:
            sub_lp = race_subset_log_probabilities(u_padded, valid_mask=valid_mask)
            for c, p_c in enumerate(cat_sizes):
                sub_lp_unpadded = race_subset_log_probabilities(
                    u_padded[c : c + 1, :p_c]
                )[0]
                assert (
                    float((sub_lp[c, : 1 << p_c] - sub_lp_unpadded).abs().max().item())
                    < 1e-12
                )
                if p_c < P_max:
                    assert torch.isneginf(sub_lp[c, 1 << p_c :]).all()

    # 2. Exact 2^{J_active} - 1 partition normalization, packed vs padded equivalence, and sampling for ragged models:
    cat_sizes = [1, 4, 2]
    C = len(cat_sizes)
    P_max = max(cat_sizes)
    J_active = sum(cat_sizes)
    cfg = {"category_sizes": cat_sizes, "users": 3, "stores": 1}
    packed_baskets, padded_baskets = _all_nonempty_ragged_baskets(cat_sizes)
    n_sets = len(packed_baskets)
    assert n_sets == (1 << J_active) - 1 == 127

    B = 2
    dlp_padded = 0.2 * torch.randn((B, C, P_max), generator=gen, dtype=torch.float64)
    base_padded = (
        0.3 * torch.randn((B, C, P_max), generator=gen, dtype=torch.float64) - 0.9
    )
    # Put arbitrary garbage in padded slots of dlp_padded and base_padded to verify strict masking invariance:
    dlp_padded[:, 0, 1:] = 999.0
    dlp_padded[:, 2, 2:] = -999.0
    base_padded[:, 0, 1:] = 777.0
    base_padded[:, 2, 2:] = -777.0

    ctx_padded = {
        "house": torch.tensor([0, 2], dtype=torch.long),
        "dlp": dlp_padded,
        "baseline": base_padded,
    }
    re_base = StructuredEnergy(cfg, [(0, 1), (1, 5), (2, 6)])
    with torch.no_grad():
        re_base.lam.copy_(
            0.2 * torch.randn(J_active, generator=gen, dtype=torch.float64)
        )
        re_base.category_count.copy_(
            torch.tensor([-0.3, -0.4, -0.2], dtype=torch.float64)
        )
        re_base.global_count.fill_(-0.1)
        re_base.link.copy_(torch.tensor([0.35, -0.25, 0.2], dtype=torch.float64))

    M_cat = 0.2 * torch.randn((3, C), generator=gen, dtype=torch.float64)
    M_prod = 0.2 * torch.randn((4, J_active), generator=gen, dtype=torch.float64)
    comp_model = CompositeWholeLawEnergy(cfg, re_base, M_cat, M_prod)
    mission_model = CalibratedMissionLaw(cfg)
    with torch.no_grad():
        mission_model.price.copy_(
            torch.rand((C, P_max), generator=gen, dtype=torch.float64) + 0.5
        )
        mission_model._mask_parameters_()

    ctx_packed = {
        "house": ctx_padded["house"],
        "dlp": mission_model.pack_active(dlp_padded),
        "baseline": mission_model.pack_active(base_padded),
    }

    for name, model in (
        ("structured_energy", re_base),
        ("composite_whole_law", comp_model),
        ("calibrated_mission_law", mission_model),
    ):
        for b_idx in range(B):
            sub_pad = {
                k: v[b_idx : b_idx + 1].expand(n_sets, *([-1] * (v.ndim - 1)))
                for k, v in ctx_padded.items()
            }
            sub_pack = {
                k: v[b_idx : b_idx + 1].expand(n_sets, *([-1] * (v.ndim - 1)))
                for k, v in ctx_packed.items()
            }
            with torch.no_grad():
                lp_pad = model.log_probability(sub_pad, padded_baskets)
                lp_pack = model.log_probability(sub_pack, packed_baskets)
            max_diff = float((lp_pad - lp_pack).abs().max().item())
            assert (
                max_diff < 1e-12
            ), f"{name} padded vs packed log_probability mismatch: {max_diff}"
            total_prob = float(torch.logsumexp(lp_pad, dim=0).exp().item())
            assert (
                abs(total_prob - 1.0) < 1e-10
            ), f"{name} failed ragged exact normalization: sum={total_prob}"

        # Verify invalid basket with selection on padded slot gets -inf log_probability:
        bad_y = torch.zeros((B, C, P_max), dtype=torch.float64)
        bad_y[:, 0, 0] = 1.0
        bad_y[:, 0, 1] = 1.0  # slot 1 in category 0 (P_0=1) is invalid!
        with torch.no_grad():
            lp_bad = model.log_probability(ctx_padded, bad_y)
        assert torch.isneginf(
            lp_bad
        ).all(), f"{name} did not assign -inf to invalid padded basket"

        # Verify input price autograd gradient has zero gradient on padded slots:
        d_req = dlp_padded.clone().requires_grad_(True)
        valid_y = padded_baskets[:B]
        lp_g = model.log_probability({**ctx_padded, "dlp": d_req}, valid_y)
        (-lp_g.mean()).backward()
        assert torch.isfinite(d_req.grad).all()
        assert float(d_req.grad[:, 0, 1:].abs().max().item()) == 0.0
        assert float(d_req.grad[:, 2, 2:].abs().max().item()) == 0.0

        # Verify sampler never activates padded slots and produces non-empty baskets:
        samples = model.sample(ctx_padded, seed=314)
        assert samples.shape == (B, C, P_max)
        assert float(samples[:, 0, 1:].abs().max().item()) == 0.0
        assert float(samples[:, 2, 2:].abs().max().item()) == 0.0
        assert samples.sum(dim=(1, 2)).min().item() >= 1.0

    # 3. Short optimization smoke check; final-weight stationarity is tested in test_ragged_pipeline.py:
    fit_model = CalibratedMissionLaw(cfg)
    fit_model.set_regularization_config(
        {"w_pr_within": 10.0, "w_pr_between": 5.0, "l2_price": 0.2, "w_cat_var": 20.0}
    )
    n_train = 24
    train_ctx = {
        "house": torch.randint(0, 3, (n_train,), generator=gen, dtype=torch.long),
        "dlp": 0.15
        * torch.randn((n_train, C, P_max), generator=gen, dtype=torch.float64),
        "baseline": -1.0 * torch.ones((n_train, C, P_max), dtype=torch.float64),
    }
    train_y = mission_model.sample(train_ctx, seed=555)
    fit_model.initialize((train_ctx, train_y))
    with torch.no_grad():
        nll_before = float(
            (-fit_model.log_probability(train_ctx, train_y).mean()).item()
        )
    fit_lbfgsb(
        fit_model,
        (train_ctx, train_y),
        (train_ctx, train_y),
        ridge=10.0,
        maxiter=15,
    )
    with torch.no_grad():
        nll_after = float(
            (-fit_model.log_probability(train_ctx, train_y).mean()).item()
        )
    assert (
        nll_after < nll_before
    ), f"L-BFGS-B did not reduce ragged NLL: {nll_before} -> {nll_after}"
    assert float(fit_model.price[0, 1:].abs().max().item()) == 0.0
    assert float(fit_model.price[2, 2:].abs().max().item()) == 0.0


if __name__ == "__main__":
    test_race_pl_quadrature_all_regimes()
    test_category_permutation_invariance_of_regularization()
    test_large_category_end_to_end_both_models()
    test_composite_input_price_gradient_and_binomial_sampler()
    test_all_models_exact_normalization_and_sampling()
    test_ragged_category_sizes()
    print(
        "ALL QUADRATURE, PERMUTATION-INVARIANCE, LARGE-P END-TO-END, PRICE-GRADIENT, 13-MODEL NORMALIZATION, AND RAGGED P_c TESTS PASSED!"
    )
