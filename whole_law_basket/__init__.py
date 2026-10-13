"""Whole-Law Basket Choice & Generation Package."""

from .race_pl import (
    subset_log_probabilities,
    race_subset_log_probabilities,
    race_observed_log_probability,
    verify_race_pl_quadrature,
)
from .proposed_model import (
    StructuredEnergy,
    CompositeWholeLawEnergy,
    CalibratedMissionLaw,
    log_esp,
    count_pairs,
)
from .baselines import (
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
from .metrics import (
    fit_lbfgsb,
    draws,
    generation_metrics,
    predictive_metrics,
    cluster_interval,
    subset,
    gradient_norm,
)

__all__ = [
    "subset_log_probabilities",
    "race_subset_log_probabilities",
    "race_observed_log_probability",
    "verify_race_pl_quadrature",
    "StructuredEnergy",
    "CompositeWholeLawEnergy",
    "CalibratedMissionLaw",
    "PooledPopularity",
    "NoPriceIndependent",
    "PersonalizedIndependent",
    "SizeAwareESP",
    "IndependentCategoriesESP",
    "LatentMissionESP",
    "PairwiseBinaryAR",
    "ShopperFactorAR",
    "LogisticNormalFactorMixture",
    "FastNeuralNADE",
    "fit_lbfgsb",
    "draws",
    "generation_metrics",
    "predictive_metrics",
    "cluster_interval",
    "subset",
    "gradient_norm",
]
