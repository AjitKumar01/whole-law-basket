# Whole-Law Basket Choice & Generation (`whole-law-basket`)

Standalone repository containing the **Proposed Whole-Law Basket Architecture** (`CompositeWholeLawEnergy` and `CalibratedMissionLaw`), the **Vectorized Differentiable Exponential-Race Plackett–Luce (PL) Quadrature** (`race_subset_log_probabilities`, `race_observed_log_probability`), and a **11-Baseline Multi-Paradigm Benchmark Suite** evaluated on synthetic (`missions_a`, `zero_price`) and real retail scanner datasets (**ERIM** 28-product and **Dunnhumby Complete Journey** 64-product catalogues).

## Repository Structure

```text
whole_law_basket_repo/
├── pyproject.toml
├── README.md
├── whole_law_basket/
│   ├── __init__.py
│   ├── race_pl.py          # O(P*M) differentiable exponential-race PL Gauss-Legendre quadrature
│   ├── proposed_model.py   # CompositeWholeLawEnergy, CalibratedMissionLaw, StructuredEnergy
│   ├── baselines.py        # 10 econometric, ESP, latent-mission, AR, SHOPPER-factor, VAE & NADE baselines
│   └── metrics.py          # L-BFGS-B preconditioned optimizer, predictive & generation evaluation suite
├── benchmarks/
│   └── run_all_benchmarks.py  # End-to-end benchmark runner across Synthetic, ERIM, and Dunnhumby
├── tests/
│   └── test_whole_law.py      # Exact 2^J - 1 partition normalization & sampler tests for all 13 models
└── results/
    ├── benchmark_summary.json
    ├── synthetic_comparison.json
    ├── synthetic_missions_a_comparison.csv
    ├── synthetic_zero_price_comparison.csv
    ├── erim_comparison.csv
    ├── erim_comparison.json
    ├── dunnhumby_comparison.csv
    └── dunnhumby_comparison.json
```

## Implemented Models (Proposed + 11 Baselines)

Every model in `whole_law_basket` defines an **exact probability distribution over all $2^J - 1$ nonempty binary baskets** ($\sum_{y \in \{0,1\}^J \setminus \{0\}} P(y \mid x) = 1$) and provides an exact, rejection-free `.sample(ctx, seed)` method:

1. **`proposed_whole_law` (`CompositeWholeLawEnergy` / `CalibratedMissionLaw`)**:
   - **Real Data (`CompositeWholeLawEnergy`)**: Combines (a) the category-frontier dynamic programming (DP) energy (`DP_N`) augmented with a nonparametric global basket-size potential $g(N)$, household-specific count slopes $s_h(N-1)$, and an exact size-1 quick-trip utility stratum $W_1 = \operatorname{logsumexp}(u^{(1)}, -1)$, (b) an exact-partition pre-trip category mission mixture ($K_c$ components over nonempty category subsets), and (c) an exact-partition product bundle mission mixture ($K_p$ SVD-initialized prototypes).
   - **Synthetic Worlds (`CalibratedMissionLaw`)**: Joint pre-trip mission + category count + differentiable exponential-race PL choice model with non-negative validation-calibrated hierarchical cross-price pooling.
2. **`structured_energy` (`StructuredEnergy`)**: Category-frontier DP energy model with quadratic count curvature and screened intra-/cross-category pairwise links (pre-fix whole-law energy baseline).
3. **`neural_nade` (`FastNeuralNADE`)**: Conditional Neural Autoregressive Distribution Estimator (hidden width 16, rank-8 cross-price bridge, exact nonempty conditioning).
4. **`logistic_normal_vae` (`LogisticNormalFactorMixture`)**: Latent Gaussian factor / Multi-VAE basket choice law (Liang et al. 2018 / Blei & Lafferty) integrating over $M$ quasi-Monte Carlo latent trip factor states $z_m \in \mathbb{R}^K$ with exact nonempty normalization per latent state.
5. **`shopper_factor_ar` (`ShopperFactorAR`)**: Low-rank sequential basket factor model inspired by SHOPPER (Ruiz, Athey, Blei 2020), with bilinear user-item preferences, bilinear basket-history interaction prefix $y_{<j} A B^\top$, and basket-size slope.
6. **`pairwise_ar` (`PairwiseBinaryAR`)**: Personalized conditional binary autoregression with lower-triangular pairwise product interaction matrix $\operatorname{tril}(W, -1)$ and rank-4 cross-price matrix.
7. **`mission_esp` (`LatentMissionESP`)**: $K=2$ latent-mission mixture of category-independent Elementary Symmetric Polynomial (ESP) laws.
8. **`independent_categories_esp` (`IndependentCategoriesESP`)**: Hierarchical category participation + within-category ESP count curvature ($\rho_c n_c(n_c-1)/2 - \nu_c(n_c-1)$) and inclusive-value coupling ($\tau_c \text{IV}_c$).
9. **`size_aware_esp` (`SizeAwareESP`)**: Two-stage predicted basket size $P(N \mid h, \bar{p})$ + conditional ESP subset law $\propto \exp(\sum_j u_j y_j) / e_N(e^u)$.
10. **`independent_parent` (`PersonalizedIndependent`)**: Personalized household/store/calendar/price/display/mailer independent binary logit conditioned on nonempty basket (MVP Logit parent).
11. **`no_price` (`NoPriceIndependent`)**: Personalized independent binary logit with price sensitivity frozen at zero (price ablation baseline).
12. **`popularity` (`PooledPopularity`)**: Pooled conditional independent Bernoulli model (item intercepts only).

## Running Tests and Benchmarks

```bash
# Run unit tests (verifies quadrature at P=4, 16, 64 and exact normalization of all 13 models)
python3 tests/test_whole_law.py

# Run full benchmark suite across Synthetic, ERIM, and Dunnhumby
python3 benchmarks/run_all_benchmarks.py
```
