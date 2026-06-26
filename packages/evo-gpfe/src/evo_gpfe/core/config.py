"""Configuration and result dataclasses for the GP feature engineer.

These containers depend only on numpy, keeping them cheap to import and easy to
serialize. DEAP/scikit-learn live in the heavier modules.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evo_gpfe.primitives.functions import VALID_FUNCTION_SETS

VALID_METRICS = ("mutual_info", "correlation", "spearman", "model_score")
VALID_TASK_TYPES = ("auto", "classification", "regression")


@dataclass
class GPConfig:
    """Configuration of the genetic-programming feature engineer.

    Parameters
    ----------
    population_size : int, default=300
        Number of trees per generation. GP needs larger populations than a GA
        because the search space (programs) is much larger.
    n_generations : int, default=40
        Generations evolved per generated feature.
    crossover_prob : float, default=0.75
        Probability of subtree crossover (the dominant GP operator).
    mutation_prob : float, default=0.20
        Probability of applying a mutation to an individual.
    p_subtree_mutation, p_hoist_mutation, p_point_mutation : float
        Relative weights of the three mutation kinds; they are normalised to
        sum to 1 (see :meth:`mutation_split`).
    tournament_size : int, default=5
        Tournament size for selection.
    elite_size : int, default=3
        Best individuals carried over unchanged each generation.
    init_depth_min, init_depth_max : int, default=2, 4
        Depth range for ramped half-and-half initialisation.
    max_tree_height : int, default=6
        Static height limit (anti-bloat). Must be >= ``init_depth_max``.
    use_constants : bool, default=True
        Include random ephemeral constants as terminals.
    constant_range : tuple of float, default=(-2.0, 2.0)
        Range of the ephemeral constants.
    function_set : {'basic', 'extended', 'full', 'nonlinear'}, default='extended'
        Which primitive set to use.
    fitness_metric : {'mutual_info', 'correlation', 'spearman', 'model_score'}
        Relevance metric between a candidate feature and the target.
    redundancy_beta : float, default=0.30
        Penalty weight for redundancy with already-generated features.
    parsimony_coeff : float, default=0.002
        Penalty per tree node (Occam's razor).
    redundancy_threshold : float, default=0.95
        A new feature whose correlation with an existing one exceeds this is
        discarded (an alternative is sought).
    n_features_to_generate : int, default=5
        Number of new features to construct.
    augment_original : bool, default=True
        If True, generated features are concatenated to the originals.
    normalize_output : bool, default=True
        Standardise each generated feature before returning it.
    task_type : {'auto', 'classification', 'regression'}, default='auto'
        Problem type (auto-detected from ``y`` when 'auto').
    early_stopping_rounds : int, default=20
        Stop evolving a feature after this many generations without improvement.
    random_seed : int or None, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Print progress.
    warm_start_population : bool, default=True
        Reuse the previous feature's final population to seed the next one.
    """

    population_size: int = 300
    n_generations: int = 40
    crossover_prob: float = 0.75
    mutation_prob: float = 0.20
    p_subtree_mutation: float = 0.60
    p_hoist_mutation: float = 0.15
    p_point_mutation: float = 0.25
    tournament_size: int = 5
    elite_size: int = 3

    init_depth_min: int = 2
    init_depth_max: int = 4
    max_tree_height: int = 6
    use_constants: bool = True
    constant_range: tuple[float, float] = (-2.0, 2.0)
    function_set: str = "extended"

    fitness_metric: str = "mutual_info"
    redundancy_beta: float = 0.30
    parsimony_coeff: float = 0.002
    redundancy_threshold: float = 0.95

    n_features_to_generate: int = 5
    augment_original: bool = True
    normalize_output: bool = True
    task_type: str = "auto"

    early_stopping_rounds: int = 20
    random_seed: int | None = 42
    verbose: bool = True
    warm_start_population: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the configuration, raising :class:`ValueError` on error."""
        if self.function_set not in VALID_FUNCTION_SETS:
            raise ValueError(
                f"function_set must be one of {VALID_FUNCTION_SETS}, got {self.function_set!r}"
            )
        if self.fitness_metric not in VALID_METRICS:
            raise ValueError(
                f"fitness_metric must be one of {VALID_METRICS}, got {self.fitness_metric!r}"
            )
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got {self.task_type!r}")
        if not 0.0 <= self.redundancy_beta <= 1.0:
            raise ValueError(f"redundancy_beta must be in [0, 1], got {self.redundancy_beta}")
        if not 0.0 <= self.redundancy_threshold <= 1.0:
            raise ValueError(
                f"redundancy_threshold must be in [0, 1], got {self.redundancy_threshold}"
            )
        if self.parsimony_coeff < 0.0:
            raise ValueError(f"parsimony_coeff must be >= 0, got {self.parsimony_coeff}")
        if self.n_features_to_generate < 1:
            raise ValueError(
                f"n_features_to_generate must be >= 1, got {self.n_features_to_generate}"
            )
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
        if self.init_depth_min < 0 or self.init_depth_max < self.init_depth_min:
            raise ValueError("require 0 <= init_depth_min <= init_depth_max")
        if self.max_tree_height < self.init_depth_max:
            raise ValueError("max_tree_height must be >= init_depth_max")
        mut_total = self.p_subtree_mutation + self.p_hoist_mutation + self.p_point_mutation
        if min(self.p_subtree_mutation, self.p_hoist_mutation, self.p_point_mutation) < 0:
            raise ValueError("mutation probabilities must be non-negative")
        if mut_total <= 0.0:
            raise ValueError("mutation probabilities must sum to a positive value")

    def mutation_split(self) -> tuple[float, float, float]:
        """Return the mutation weights normalised to sum to 1.0 (no mutation)."""
        total = self.p_subtree_mutation + self.p_hoist_mutation + self.p_point_mutation
        return (
            self.p_subtree_mutation / total,
            self.p_hoist_mutation / total,
            self.p_point_mutation / total,
        )

    def to_dict(self) -> dict:
        """Return the configuration as a plain dictionary."""
        return asdict(self)


@dataclass
class GeneratedFeature:
    """A single feature produced by genetic programming.

    Attributes
    ----------
    expression : str
        Human-readable symbolic expression of the tree.
    fitness : float
        Fitness of the tree when selected (relevance minus penalties).
    mi_score : float
        Raw relevance to the target (unpenalised).
    redundancy_score : float
        Maximum correlation with previously generated features (0 for the first).
    n_nodes : int
        Number of nodes (complexity).
    height : int
        Tree height.
    feature_index : int
        Zero-based position in the generated block.
    generation_found : int
        Generation at which the best tree was found.
    values : numpy.ndarray
        Feature values on the training data.
    """

    expression: str
    fitness: float
    mi_score: float
    redundancy_score: float
    n_nodes: int
    height: int
    feature_index: int
    generation_found: int
    values: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self) -> str:
        return (
            f"GP-Feature[{self.feature_index}]: {self.expression} "
            f"(MI={self.mi_score:.4f}, redund={self.redundancy_score:.3f}, "
            f"nodes={self.n_nodes}, height={self.height})"
        )


@dataclass
class GPEngineeringResult:
    """Outcome of a GP feature-engineering run."""

    generated_features: list[GeneratedFeature]
    X_original_shape: tuple[int, int]
    X_transformed_shape: tuple[int, int]
    baseline_cv_score: float
    augmented_cv_score: float
    score_improvement: float
    total_time: float
    n_evaluations: int
    task_type: str
    scoring: str
    feature_names_original: list[str]
    feature_names_generated: list[str]
    config: GPConfig | None = None

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        baseline = max(abs(self.baseline_cv_score), 1e-9)
        lines = [
            "=" * 65,
            "  GP Feature Engineering - Result",
            "=" * 65,
            f"  Original dataset     : {self.X_original_shape}",
            f"  Transformed dataset  : {self.X_transformed_shape}",
            f"  Generated features   : {len(self.generated_features)}",
            f"  Scoring              : {self.scoring}",
            f"  Baseline CV score    : {self.baseline_cv_score:.4f}",
            f"  Augmented CV score   : {self.augmented_cv_score:.4f}",
            f"  Improvement          : {self.score_improvement:+.4f} "
            f"({self.score_improvement / baseline * 100:+.1f}%)",
            f"  Total evaluations    : {self.n_evaluations}",
            f"  Total time           : {self.total_time:.2f}s",
            "-" * 65,
            "  Generated symbolic expressions:",
        ]
        for gf in self.generated_features:
            lines.append(
                f"    [{gf.feature_index}] MI={gf.mi_score:.4f} "
                f"| nodes={gf.n_nodes:2d} | {gf.expression}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (feature value arrays excluded)."""
        return {
            "task_type": self.task_type,
            "scoring": self.scoring,
            "baseline_cv_score": float(self.baseline_cv_score),
            "augmented_cv_score": float(self.augmented_cv_score),
            "score_improvement": float(self.score_improvement),
            "n_evaluations": int(self.n_evaluations),
            "total_time": float(self.total_time),
            "feature_names_generated": list(self.feature_names_generated),
            "generated_features": [
                {
                    "expression": gf.expression,
                    "mi_score": float(gf.mi_score),
                    "redundancy_score": float(gf.redundancy_score),
                    "n_nodes": int(gf.n_nodes),
                    "height": int(gf.height),
                }
                for gf in self.generated_features
            ],
            "config": self.config.to_dict() if self.config is not None else None,
        }

    def save_json(self, path: str | Path) -> None:
        """Write the JSON-serialisable summary to ``path``."""
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def save(self, path: str | Path) -> None:
        """Pickle the full result object to ``path``."""
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> GPEngineeringResult:
        """Load a pickled :class:`GPEngineeringResult` from ``path``."""
        return pickle.loads(Path(path).read_bytes())
