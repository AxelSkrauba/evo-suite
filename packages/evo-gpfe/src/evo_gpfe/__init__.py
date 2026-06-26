"""evo-gpfe — Genetic Programming Feature Engineer.

A scikit-learn-compatible symbolic feature constructor for tabular data, built
on DEAP. It evolves expression trees that combine the original features into new
ones, complementing ``evo-gafs`` (which selects among existing features).
"""

from __future__ import annotations

from evo_gpfe.core.config import GeneratedFeature, GPConfig, GPEngineeringResult
from evo_gpfe.core.engineer import GPFeatureEngineer
from evo_gpfe.core.evaluator import GPFitnessEvaluator

__version__ = "0.1.0"

__all__ = [
    "GPConfig",
    "GPEngineeringResult",
    "GPFeatureEngineer",
    "GPFitnessEvaluator",
    "GeneratedFeature",
    "__version__",
]
