"""evo-gpfe — Genetic Programming Feature Engineer.

A scikit-learn-compatible symbolic feature constructor for tabular data, built
on DEAP. It evolves expression trees that combine the original features into new
ones, complementing ``evo-gafs`` (which selects among existing features).

Examples
--------
>>> from sklearn.tree import DecisionTreeRegressor
>>> from evo_gpfe import GPFeatureEngineer, GPConfig
>>> engineer = GPFeatureEngineer(
...     estimator=DecisionTreeRegressor(random_state=0),
...     config=GPConfig(population_size=100, n_generations=20, verbose=False),
... )
"""

from __future__ import annotations

from evo_gpfe.benchmark.runner import GPBenchmarkRunner
from evo_gpfe.core.config import GeneratedFeature, GPConfig, GPEngineeringResult
from evo_gpfe.core.engineer import GPFeatureEngineer
from evo_gpfe.core.evaluator import GPFitnessEvaluator
from evo_gpfe.visualization.plots import GPPlotter

__version__ = "0.1.0"

__all__ = [
    "GPBenchmarkRunner",
    "GPConfig",
    "GPEngineeringResult",
    "GPFeatureEngineer",
    "GPFitnessEvaluator",
    "GPPlotter",
    "GeneratedFeature",
    "__version__",
]
