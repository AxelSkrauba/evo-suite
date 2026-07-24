"""Configuration and result dataclasses for the evolutionary ensemble builder.

These containers depend only on numpy, keeping them cheap to import and easy
to serialize. DEAP/scikit-learn live in the heavier modules.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

VALID_MODES = ("single", "multiobjective")
VALID_WEIGHT_METHODS = ("softmax", "abs_norm", "uniform")


@dataclass
class EvoEnsConfig:
    """Configuration of the evolutionary ensemble builder.

    Genetic Algorithm parameters
    -----------------------------
    population_size : int, default=80
        Number of individuals per generation. Cost per individual is
        ``O(n_samples)`` thanks to out-of-fold pre-computation, so larger
        populations remain cheap relative to the pre-computation step.
    n_generations : int, default=100
        Maximum number of generations evolved (subject to early stopping).
    crossover_prob : float, default=0.75
        Probability of applying uniform crossover to a pair of offspring.
    mutation_prob : float, default=0.20
        Probability of applying Gaussian mutation to an individual.
    mutation_sigma : float, default=0.25
        Standard deviation of the Gaussian mutation perturbation.
    mutation_indpb : float or None, default=None
        Per-gene mutation probability. Defaults to ``1 / (2 * n_candidates)``
        when ``None`` (resolved at fit time, since it depends on the pool
        size).
    tournament_size : int, default=4
        Tournament size used by single-objective selection.
    elite_size : int, default=2
        Number of elites preserved verbatim each generation (single mode).

    Mode
    ----
    mode : {'single', 'multiobjective'}, default='single'
        ``'single'`` optimizes a scalar fitness with a configurable
        diversity penalty; ``'multiobjective'`` runs NSGA-II over
        ``(score, compression)`` and returns the full Pareto front.

    Fitness parameters
    -------------------
    scoring : str or None, default=None
        Evaluation metric. Classification: ``'accuracy'``, ``'f1_macro'``,
        ``'roc_auc'``. Regression: ``'r2'``, ``'neg_rmse'``. Auto-detected
        by task type when ``None``.
    diversity_beta : float, default=0.10
        Weight of the diversity penalty (single mode only)::

            fitness = score - diversity_beta * mean_diversity

        ``0.0`` ignores diversity entirely; higher values force more
        decorrelated ensembles, potentially at the cost of raw score.
    weight_method : {'softmax', 'abs_norm', 'uniform'}, default='softmax'
        How raw weight genes are normalized into ensemble weights.

    Ensemble parameters
    ---------------------
    min_models : int, default=2
        Minimum number of active models; individuals below this are
        penalized to a zero fitness.
    max_models : int or None, default=None
        Maximum number of active models (``None`` = unlimited). Useful for
        edge-deployment constraints; excess models are dropped by lowest
        weight.
    cv_folds : int, default=5
        Number of folds for the out-of-fold pre-computation.

    Control parameters
    --------------------
    random_seed : int or None, default=42
    verbose : bool, default=True
    early_stopping_rounds : int or None, default=25
        Stop evolving if the best fitness has not improved by more than
        ``early_stopping_tol`` over this many generations (single mode).
    early_stopping_tol : float, default=1e-5
    """

    # Genetic Algorithm
    population_size: int = 80
    n_generations: int = 100
    crossover_prob: float = 0.75
    mutation_prob: float = 0.20
    mutation_sigma: float = 0.25
    mutation_indpb: float | None = None
    tournament_size: int = 4
    elite_size: int = 2

    # Mode
    mode: str = "single"

    # Fitness
    scoring: str | None = None
    diversity_beta: float = 0.10
    weight_method: str = "softmax"

    # Ensemble
    min_models: int = 2
    max_models: int | None = None
    cv_folds: int = 5

    # Control
    random_seed: int | None = 42
    verbose: bool = True
    early_stopping_rounds: int | None = 25
    early_stopping_tol: float = 1e-5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the configuration, raising :class:`ValueError` on error."""
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.weight_method not in VALID_WEIGHT_METHODS:
            raise ValueError(
                f"weight_method must be one of {VALID_WEIGHT_METHODS}, got {self.weight_method!r}"
            )
        if self.diversity_beta < 0.0:
            raise ValueError(f"diversity_beta must be >= 0, got {self.diversity_beta}")
        if self.min_models < 1:
            raise ValueError(f"min_models must be >= 1, got {self.min_models}")
        if self.max_models is not None and self.max_models < self.min_models:
            raise ValueError("max_models must be >= min_models when set")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {self.n_generations}")
        if self.tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {self.tournament_size}")
        if self.elite_size < 0:
            raise ValueError(f"elite_size must be >= 0, got {self.elite_size}")
        if not 0.0 < self.crossover_prob <= 1.0:
            raise ValueError(f"crossover_prob must be in (0, 1], got {self.crossover_prob}")
        if not 0.0 <= self.mutation_prob <= 1.0:
            raise ValueError(f"mutation_prob must be in [0, 1], got {self.mutation_prob}")
        if self.mutation_sigma < 0.0:
            raise ValueError(f"mutation_sigma must be >= 0, got {self.mutation_sigma}")
        if self.mutation_indpb is not None and not 0.0 <= self.mutation_indpb <= 1.0:
            raise ValueError(f"mutation_indpb must be in [0, 1], got {self.mutation_indpb}")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be >= 1 when set")

    def resolved_mutation_indpb(self, n_candidates: int) -> float:
        """Return the per-gene mutation probability for a pool of this size."""
        return self.mutation_indpb if self.mutation_indpb is not None else 1.0 / (2 * n_candidates)

    def to_dict(self) -> dict:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


@dataclass
class EnsembleMember:
    """A single fitted (or fittable) model within the final ensemble.

    Attributes
    ----------
    name : str
        Human-readable label (``estimator_display_name``).
    estimator : sklearn estimator
        The (unfitted, cloned) candidate; the estimator actually used for
        prediction lives in the owning estimator's ``estimators_``.
    weight : float
        Normalized ensemble weight.
    candidate_index : int
        Index into the original candidate pool.
    oof_score : float
        Standalone out-of-fold score of this candidate (for reference).
    """

    name: str
    estimator: Any
    weight: float
    candidate_index: int
    oof_score: float


@dataclass
class ParetoSolution:
    """A single solution on the NSGA-II Pareto front (multiobjective mode)."""

    score: float
    compression: float
    n_models: int
    members: list[EnsembleMember]
    weights: np.ndarray


@dataclass
class EvolutionStats:
    """Per-generation statistics recorded during evolution."""

    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_n_models: int
    mean_n_models: float
    elapsed_time: float

    def __repr__(self) -> str:
        return (
            f"Gen {self.generation:4d} | "
            f"BestFit={self.best_fitness:.4f} | "
            f"MeanFit={self.mean_fitness:.4f}+/-{self.std_fitness:.4f} | "
            f"BestModels={self.best_n_models} | "
            f"MeanModels={self.mean_n_models:.1f} | "
            f"t={self.elapsed_time:.2f}s"
        )


@dataclass
class EvoEnsResult:
    """Outcome of an evolutionary ensemble-building run."""

    members: list[EnsembleMember]
    n_models: int
    weights: np.ndarray

    oof_score_ensemble: float
    oof_score_best_single: float
    oof_score_improvement: float
    diversity_score: float
    compression_ratio: float

    scoring: str
    task_type: str
    n_candidates: int

    history: list[EvolutionStats]
    total_time: float
    precompute_time: float
    n_evaluations: int

    config: EvoEnsConfig | None = None
    pareto_front: list[ParetoSolution] | None = None

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        sep = "=" * 65
        lines = [
            sep,
            "  EvoEnsemble Builder - Result",
            sep,
            f"  Task / scoring       : {self.task_type} / {self.scoring}",
            f"  Candidates evaluated : {self.n_candidates}",
            f"  Ensemble size        : {self.n_models} ({self.compression_ratio:.0%} compression)",
            f"  Best single OOF      : {self.oof_score_best_single:.4f}",
            f"  Ensemble OOF         : {self.oof_score_ensemble:.4f} "
            f"({self.oof_score_improvement:+.4f})",
            f"  Diversity score      : {self.diversity_score:.4f}",
            f"  Evaluations / time   : {self.n_evaluations} / {self.total_time:.2f}s "
            f"(precompute: {self.precompute_time:.2f}s)",
            "-" * 65,
            "  Ensemble composition:",
        ]
        for m in sorted(self.members, key=lambda x: -x.weight):
            lines.append(
                f"    [{m.candidate_index:2d}] w={m.weight:.3f} | OOF={m.oof_score:.4f} | {m.name}"
            )
        if self.pareto_front:
            lines.append(f"\n  Pareto front: {len(self.pareto_front)} solutions")
        lines.append(sep)
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (estimator objects excluded)."""
        return {
            "task_type": self.task_type,
            "scoring": self.scoring,
            "n_candidates": int(self.n_candidates),
            "n_models": int(self.n_models),
            "compression_ratio": float(self.compression_ratio),
            "oof_score_ensemble": float(self.oof_score_ensemble),
            "oof_score_best_single": float(self.oof_score_best_single),
            "oof_score_improvement": float(self.oof_score_improvement),
            "diversity_score": float(self.diversity_score),
            "n_evaluations": int(self.n_evaluations),
            "total_time": float(self.total_time),
            "precompute_time": float(self.precompute_time),
            "members": [
                {
                    "name": m.name,
                    "weight": float(m.weight),
                    "candidate_index": int(m.candidate_index),
                    "oof_score": float(m.oof_score),
                }
                for m in self.members
            ],
            "config": self.config.to_dict() if self.config is not None else None,
        }

    def save_json(self, path: str | Path) -> None:
        """Write the JSON-serializable summary to ``path``."""
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def save(self, path: str | Path) -> None:
        """Pickle the full result object to ``path``."""
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> EvoEnsResult:
        """Load a pickled :class:`EvoEnsResult` from ``path``."""
        return pickle.loads(Path(path).read_bytes())
