"""gafs — Genetic Algorithm Feature Selector.

A scikit-learn-compatible wrapper feature selector for tabular data, built on
DEAP. Supports single-objective (weighted) and multi-objective (NSGA-II,
Pareto front) selection for both classification and regression.

Examples
--------
>>> from sklearn.tree import DecisionTreeClassifier
>>> from gafs import GAFeatureSelector, GAConfig
>>> selector = GAFeatureSelector(
...     estimator=DecisionTreeClassifier(random_state=0),
...     config=GAConfig(population_size=20, n_generations=10, verbose=False),
... )
"""

from gafs.benchmark.runner import BenchmarkRunner
from gafs.core.config import EvolutionStats, GAConfig, SelectionResult
from gafs.core.evaluator import FitnessEvaluator
from gafs.core.selector import GAFeatureSelector
from gafs.visualization.plots import GAPlotter

__version__ = "0.1.0"

__all__ = [
    "BenchmarkRunner",
    "EvolutionStats",
    "FitnessEvaluator",
    "GAConfig",
    "GAFeatureSelector",
    "GAPlotter",
    "SelectionResult",
    "__version__",
]
