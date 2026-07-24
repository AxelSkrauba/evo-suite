"""evo-ens — Evolutionary Ensemble Builder.

A scikit-learn-compatible ensemble constructor built on DEAP. It uses a
Genetic Algorithm to co-optimize predictive score and prediction diversity
(Yule's Q-statistic for classification, absolute Pearson correlation for
regression) between candidate models, pre-computing out-of-fold predictions
once so evolution stays cheap even with large populations.

Examples
--------
>>> from sklearn.datasets import load_breast_cancer
>>> from evo_ens import EvoEnsConfig, EvoEnsembleClassifier
>>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
>>> clf = EvoEnsembleClassifier(
...     config=EvoEnsConfig(population_size=30, n_generations=10, verbose=False)
... )
"""

from __future__ import annotations

from evo_ens.benchmark.runner import EvoEnsBenchmarkRunner
from evo_ens.core.candidates import default_candidates, estimator_display_name
from evo_ens.core.config import (
    EnsembleMember,
    EvoEnsConfig,
    EvoEnsResult,
    EvolutionStats,
    ParetoSolution,
)
from evo_ens.core.estimator import EvoEnsembleClassifier, EvoEnsembleRegressor
from evo_ens.core.evaluator import EnsembleFitnessEvaluator
from evo_ens.utils.diversity import diversity_pearson_mean, diversity_q_mean, q_statistic_pair
from evo_ens.visualization.plots import EvoEnsPlotter

__version__ = "0.1.0"

__all__ = [
    "EnsembleFitnessEvaluator",
    "EnsembleMember",
    "EvoEnsBenchmarkRunner",
    "EvoEnsConfig",
    "EvoEnsPlotter",
    "EvoEnsResult",
    "EvoEnsembleClassifier",
    "EvoEnsembleRegressor",
    "EvolutionStats",
    "ParetoSolution",
    "__version__",
    "default_candidates",
    "diversity_pearson_mean",
    "diversity_q_mean",
    "estimator_display_name",
    "q_statistic_pair",
]
