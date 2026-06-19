"""Configuration and result dataclasses for the genetic feature selector.

This module holds the plain data containers used across ``gafs``:

* :class:`GAConfig` — the full configuration of the genetic algorithm.
* :class:`EvolutionStats` — per-generation statistics collected during a run.
* :class:`SelectionResult` — the final outcome of a feature-selection run.

None of these classes depend on DEAP or scikit-learn, which keeps them cheap to
import and trivial to serialize.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VALID_MODES = ("single", "multiobjective")


@dataclass
class GAConfig:
    """Full configuration of the genetic algorithm.

    Parameters
    ----------
    population_size : int, default=50
        Number of individuals in the population. Larger populations explore the
        search space better at a higher computational cost. Typical: 30-100.
    n_generations : int, default=100
        Number of generations (iterations of the GA). Typical: 50-200.
    crossover_prob : float, default=0.8
        Probability of applying crossover between two individuals. Recommended
        range: ``[0.6, 0.9]``.
    mutation_prob : float, default=0.15
        Probability of applying the mutation operator to an individual.
        Recommended range: ``[0.05, 0.3]``.
    mutation_indpb : float or None, default=None
        Independent probability of flipping each bit when an individual is
        mutated. If ``None``, it is set to ``1 / n_features`` at fit time.
    tournament_size : int, default=3
        Tournament size for tournament selection (``mode='single'``). Larger
        values increase selective pressure. Typical: 2-7.
    mode : {'single', 'multiobjective'}, default='single'
        ``'single'`` uses a weighted scalar fitness; ``'multiobjective'`` uses
        NSGA-II and returns a Pareto front.
    alpha : float, default=0.8
        Weight of the performance metric in ``mode='single'``::

            fitness = alpha * cv_score + (1 - alpha) * compression_ratio

        ``alpha=1.0`` is a pure wrapper; lower values favour compression
        (useful for edge deployment).
    cv_folds : int, default=5
        Number of cross-validation folds used to evaluate fitness.
    min_features : int, default=1
        Minimum number of selected features. Individuals below this threshold
        are repaired/penalised.
    elite_size : int, default=2
        Number of best individuals carried over unchanged each generation
        (elitism, ``mode='single'`` only).
    random_seed : int or None, default=42
        Seed for reproducibility.
    n_jobs : int, default=1
        Parallelism passed to scikit-learn's cross-validation.
    verbose : bool, default=True
        If ``True``, print a log line for some generations and a final summary.
    early_stopping_rounds : int or None, default=None
        If the best fitness does not improve for this many generations, stop.
        ``None`` disables early stopping (``mode='single'`` only).
    early_stopping_tol : float, default=1e-4
        Minimum improvement considered significant for early stopping.
    """

    population_size: int = 50
    n_generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.15
    mutation_indpb: float | None = None
    tournament_size: int = 3
    mode: str = "single"
    alpha: float = 0.8
    cv_folds: int = 5
    min_features: int = 1
    elite_size: int = 2
    random_seed: int | None = 42
    n_jobs: int = 1
    verbose: bool = True
    early_stopping_rounds: int | None = None
    early_stopping_tol: float = 1e-4

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the configuration, raising :class:`ValueError` on error.

        Unlike ``assert`` statements, these checks are always enforced, even
        when Python runs with optimisations (``-O``).
        """
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 < self.crossover_prob <= 1.0:
            raise ValueError(f"crossover_prob must be in (0, 1], got {self.crossover_prob}")
        if not 0.0 < self.mutation_prob <= 1.0:
            raise ValueError(f"mutation_prob must be in (0, 1], got {self.mutation_prob}")
        if self.mutation_indpb is not None and not 0.0 < self.mutation_indpb <= 1.0:
            raise ValueError(f"mutation_indpb must be in (0, 1], got {self.mutation_indpb}")
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {self.n_generations}")
        if self.tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {self.tournament_size}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        if self.min_features < 1:
            raise ValueError(f"min_features must be >= 1, got {self.min_features}")
        if self.elite_size < 0:
            raise ValueError(f"elite_size must be >= 0, got {self.elite_size}")

    def to_dict(self) -> dict:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


@dataclass
class EvolutionStats:
    """Statistics for a single generation during evolution."""

    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_n_features: int
    mean_n_features: float
    elapsed_time: float

    def __repr__(self) -> str:
        return (
            f"Gen {self.generation:4d} | "
            f"BestFit={self.best_fitness:.4f} | "
            f"MeanFit={self.mean_fitness:.4f}±{self.std_fitness:.4f} | "
            f"BestFeats={self.best_n_features:3d} | "
            f"MeanFeats={self.mean_n_features:.1f} | "
            f"Time={self.elapsed_time:.2f}s"
        )


@dataclass
class SelectionResult:
    """Final outcome of a GA feature-selection run.

    Attributes
    ----------
    selected_mask : numpy.ndarray
        Boolean vector of length ``n_features``. ``True`` marks a selected
        feature.
    selected_indices : numpy.ndarray
        Indices of the selected features.
    selected_feature_names : list of str
        Names of the selected features.
    best_fitness : float
        Best fitness achieved.
    best_cv_score : float
        Best cross-validation score (raw metric, unweighted).
    n_selected : int
        Number of selected features.
    compression_ratio : float
        Fraction of features removed (``1 - n_selected / n_total``).
    history : list of EvolutionStats
        Per-generation statistics.
    pareto_front : list of dict or None
        Only in ``mode='multiobjective'``. Each entry holds ``mask``,
        ``cv_score``, ``compression`` and ``n_features``.
    config : GAConfig or None
        Configuration used for the run.
    total_time : float
        Total wall-clock time in seconds.
    n_evaluations : int
        Total number of fitness evaluations performed.
    """

    selected_mask: np.ndarray
    selected_indices: np.ndarray
    selected_feature_names: list[str]
    best_fitness: float
    best_cv_score: float
    n_selected: int
    compression_ratio: float
    history: list[EvolutionStats] = field(default_factory=list)
    pareto_front: list[dict] | None = None
    config: GAConfig | None = None
    total_time: float = 0.0
    n_evaluations: int = 0

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the result."""
        lines = [
            "=" * 60,
            "  RESULT - GA Feature Selection",
            "=" * 60,
            f"  Original features   : {len(self.selected_mask)}",
            f"  Selected features   : {self.n_selected}",
            f"  Compression ratio   : {self.compression_ratio:.1%}",
            f"  Best CV score       : {self.best_cv_score:.4f}",
            f"  Best fitness        : {self.best_fitness:.4f}",
            f"  Total evaluations   : {self.n_evaluations}",
            f"  Total time          : {self.total_time:.2f}s",
            "-" * 60,
            "  Selected features:",
        ]
        for i, name in zip(self.selected_indices, self.selected_feature_names):
            lines.append(f"    [{int(i):3d}] {name}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary (e.g. for logging/MLflow)."""
        return {
            "selected_indices": [int(i) for i in self.selected_indices],
            "selected_feature_names": list(self.selected_feature_names),
            "best_fitness": float(self.best_fitness),
            "best_cv_score": float(self.best_cv_score),
            "n_selected": int(self.n_selected),
            "compression_ratio": float(self.compression_ratio),
            "total_time": float(self.total_time),
            "n_evaluations": int(self.n_evaluations),
            "config": self.config.to_dict() if self.config is not None else None,
        }

    def save_json(self, path: str | Path) -> None:
        """Write the JSON-serialisable summary to ``path``."""
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def save(self, path: str | Path) -> None:
        """Pickle the full result object to ``path``."""
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> SelectionResult:
        """Load a pickled :class:`SelectionResult` from ``path``."""
        return pickle.loads(Path(path).read_bytes())
