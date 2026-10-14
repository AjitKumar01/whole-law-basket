# Whole-Law Basket

Train conditional basket models, generate synthetic shopping baskets, and compare their responses to price changes. The repository contains the current proposed models and baseline implementations for synthetic worlds, ERIM, and Dunnhumby.

The synthetic and real-data proposed models are different architectures. Real causal price accuracy and superiority over fully converged baselines remain open research questions. Historical scores are preserved separately from the corrected benchmark runner.

## Install and test

Python 3.11 or newer is required. Computation uses CPU PyTorch, NumPy, and SciPy.

```bash
git clone https://github.com/AjitKumar01/whole-law-basket.git
cd whole-law-basket
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -c requirements-tested.txt
python -m pytest -q tests
```

## Train and evaluate

Synthetic data are generated deterministically when prepared splits are absent. No dataset download or pretrained checkpoint is required for synthetic training.

```bash
# One synthetic world, proposed model and its unregularized ablation.
python benchmarks/run_all_benchmarks.py --stage synthetic --world missions_b --models proposed_calibrated_mission_law,uncalibrated_mission_law --mode fresh

# All synthetic worlds and baselines.
python benchmarks/run_all_benchmarks.py --stage synthetic --mode fresh

# Real-data comparisons, after supplying the prepared inputs described below.
python benchmarks/run_all_benchmarks.py --stage erim --mode fresh
python benchmarks/run_all_benchmarks.py --stage dunnhumby --mode fresh

# Reuse matching checkpoints and recompute evaluation from their weights.
python benchmarks/run_all_benchmarks.py --stage synthetic --mode resume

# Complete suite, including quadrature checks and final summary.
python benchmarks/run_all_benchmarks.py --stage all --mode fresh
```

Use `--models NAME` or a comma-separated list for focused runs. `--models all` is the default. Models and groups are listed below and in the runner's `REAL_GROUPS` / `SYNTHETIC_GROUPS` constants. A requested model's fitted dependencies run first.

Use `--results-dir PATH` to isolate experiments and `--threads N` to change CPU threads (default 1). Outputs default to `results/verified_v2/`. The full suite can take substantially longer than a focused run; no training starts during installation or testing.

## Real-data inputs

Retail observations and local checkpoints are excluded from Git. Supply the existing prepared data under `data/`:

```text
data/
  erim/
    config.json
    train.npz
    validation.npz
    test.npz
  erim_parent/
    config.json
    best.pt                  # final.pt is accepted when best.pt is absent
    fit.json                 # optional historical cost record
  erim_ref/
    screen.json
  dunnhumby/
    config.json
    train.npz
    validation.npz
    test.npz
    dlp.npy
    disp.npy
    mail.npy
  dunnhumby_parent/
    config.json
    best.pt
    fit.json
  dunnhumby_ref/
    screen.json
```

The real loader implements the current 28-product ERIM selection and 64-product Dunnhumby projection. `erim/*.npz` contains `house`, `store`, `week`, `member`, `dlp`, `disp`, and `mail`. The packed Dunnhumby splits contain `house`, `store`, `week`, `context`, and `packed_member`; context indices address the shared offer arrays. Configurations define the catalogue, mappings, and calendar support. Parent weights must match `ParentControls` and their accompanying configuration. Graph `screen.json` files provide the selected `edges`.

Real experiments fit correction models conditional on the supplied frozen preference controls and screened graph. Raw transaction preprocessing and retraining those historical controls are outside this runner. Reference `*_init.pt` files are unnecessary and are never loaded.

## Models

| Runner name | Class / purpose |
|---|---|
| `proposed_calibrated_mission_law` | Synthetic proposed model: two pre-trip missions, category participation, counts, and race-PL product choice with jointly fitted hierarchical price penalties |
| `uncalibrated_mission_law` | Synthetic ablation using the same law without hierarchical pooling; ordinary ridge remains |
| `proposed_whole_law` | Real proposed model: `CompositeWholeLawEnergy`, combining a fitted structured-energy base, basket-size adjustments, and category/product mission mixtures |
| `structured_energy` | Real-data category-frontier DP energy baseline |
| `neural_nade` | Conditional neural autoregressive basket distribution |
| `pairwise_ar` / `pairwise_binary_ar` | Real / synthetic names for binary autoregression with pairwise interactions |
| `low_rank_bilinear_ar` | Binary autoregression with low-rank interactions; not a reproduction of SHOPPER |
| `gaussian_factor_mixture` | Fixed-node Gaussian factor mixture; not a variational autoencoder |
| `mission_esp` | Two-component latent-mission category ESP mixture |
| `independent_categories_esp` | Category participation and within-category count/choice model |
| `size_aware_esp` | Predicted basket-size distribution plus conditional ESP product selection |
| `independent_parent` | Nonempty-conditioned independent logit; a residual model on shared controls in real experiments |
| `no_price` | Price ablation |
| `popularity` | Pooled item-intercept baseline |

## Generation and price scenarios

Every model exposes `log_probability(ctx, baskets)` and `sample(ctx, seed)`. Baskets have shape `[batch, categories, products_per_category]` and represent distinct-product presence, conditional on a nonempty basket. The context contains household identity and relative log prices (`dlp`); real models also use the prepared price-free preference `baseline`.

This minimal example trains a model from scratch and generates baskets:

```python
from whole_law_basket import (
    PersonalizedIndependent, generate_synthetic_splits, fit_lbfgsb, subset,
)

config, train, validation, test = generate_synthetic_splits(
    'latent_missions_gumbel', 941001, sizes=(1000, 250, 250)
)
model = PersonalizedIndependent(config)
fit = fit_lbfgsb(model, train[:2], validation[:2], ridge=10.0)
context = subset(test[0], slice(0, 10))
baskets = model.sample(context, seed=42)
```

For a price scenario, copy the context and change `dlp`: a 10% discount adds `log(0.9)` to the affected product's relative log price. Evaluate or sample the baseline and modified contexts, then compare product incidence, category participation, and basket size. These are model-implied responses; they require independent evidence before being interpreted causally on real data.

The synthetic runner compares complete basket distributions to the known generator, evaluates own/same-category/cross-category price responses, and measures promotion margin and regret under explicitly hypothetical costs and funding. Real evaluation measures held-out likelihood, calibration, Recall@5, product frequencies, size distributions, category pairs, and multiple purchases. It uses 48 draws on identical contexts; synthetic generation metrics use exact enumeration of the 12-product basket distribution.

## Checkpoints and convergence

New checkpoints contain `state_dict`, model `config`, selected `regularization_config`, and trainable parameter names. Sidecar records fingerprint prepared splits, code, fixed state, dependencies, versions, fitting policy, and checkpoint bytes. Matching checkpoints are recertified and re-evaluated; saved scores are never substituted. Legacy or mismatched checkpoints are rejected.

Convergence checks cover the actual saved active parameters using both raw and scaled projected-gradient thresholds of `2e-4`. Failed dependencies block subsequent fitting. Regularization candidates are selected using validation likelihood only after final-weight convergence checks. Stationarity does not establish global optimality or causal validity.

`fit_seconds` for the regularized mission model includes initialization, all candidates, count refinement, and certification. The unregularized ablation is already included in that cost. `pipeline_fit_seconds` adds fitted dependencies; incomplete historical preference-control costs are reported separately.

## Code layout and research status

```text
whole_law_basket/             Model laws, baselines, fitting, loaders, provenance
benchmarks/run_all_benchmarks.py
                             Training and evaluation CLI
tests/                      Mathematical and benchmark-protocol regression tests
docs/THEORY.md              Model equations and assumptions
results/historical/         Historical tables; not corrected fresh-suite results
```

Historical neural fits on the synthetic worlds are unconverged. Historical real proposed fits used pretrained/frozen components, and their reported training times were incomplete. The corrected complete benchmark has not been rerun. Full-data size-baseline convergence has been verified on ERIM and Dunnhumby; the test suite covers normalization, sampling, numerical quadrature, price derivatives, permutation-invariant penalties, and checkpoint identity/fresh/resume behavior.

The remaining scientific questions include real price identification, cross-price behavior beyond the synthetic graph, sensitivity to pooling assumptions, and full-catalogue scalability. See [the model theory](docs/THEORY.md) for equations and limits.
