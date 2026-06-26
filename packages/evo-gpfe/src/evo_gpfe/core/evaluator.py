"""Fitness evaluation for GP trees: relevance, redundancy and parsimony."""

from __future__ import annotations

import warnings

import numpy as np
from deap import gp
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from evo_gpfe.core.config import GPConfig

_EPS = 1e-10
_CLIP = 1e8


class GPFitnessEvaluator:
    """Evaluate a GP tree as a candidate engineered feature.

    The fitness rewards relevance to the target and penalises redundancy with
    already-accepted features (a "hall of fame") and tree complexity::

        fitness = relevance(tree, y)
                  - redundancy_beta * max_corr(tree, hof)
                  - parsimony_coeff * n_nodes

    Results are cached by the tree's symbolic string to avoid recomputing
    identical expressions across generations. The cache is cleared whenever the
    hall of fame changes (redundancy of every individual then changes).

    Parameters
    ----------
    X, y : numpy.ndarray
        Training data and (numeric) target.
    pset : deap.gp.PrimitiveSet
        Primitive set used to compile trees.
    config : GPConfig
        Configuration (provides metric, penalties, seed).
    task_type : {'classification', 'regression'}
        Used by mutual-information and model-score metrics.
    estimator : sklearn estimator, optional
        Only used when ``fitness_metric='model_score'``.
    scoring : str, default='r2'
        Scoring string for the model-score metric.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pset: gp.PrimitiveSet,
        config: GPConfig,
        task_type: str,
        estimator: BaseEstimator | None = None,
        scoring: str = "r2",
    ) -> None:
        self.X = X
        self.y = y
        self.pset = pset
        self.config = config
        self.task_type = task_type
        self.estimator = estimator
        self.scoring = scoring
        self._seed = config.random_seed if config.random_seed is not None else 0

        self._hof_values: list[np.ndarray] = []
        # symbolic expression -> (fitness, relevance, redundancy)
        self._cache: dict[str, tuple[float, float, float]] = {}
        self._eval_count = 0

    # ── relevance metrics ─────────────────────────────────────────────────────

    def _relevance(self, vals: np.ndarray) -> float:
        metric = self.config.fitness_metric
        if metric == "mutual_info":
            fn = (
                mutual_info_classif
                if self.task_type == "classification"
                else mutual_info_regression
            )
            return float(fn(vals.reshape(-1, 1), self.y, random_state=self._seed)[0])
        if metric == "correlation":
            if np.std(vals) < _EPS:
                return 0.0
            return abs(float(pearsonr(vals, self.y)[0]))
        if metric == "spearman":
            if np.std(vals) < _EPS:
                return 0.0
            return abs(float(spearmanr(vals, self.y)[0]))
        return self._model_score(vals)  # model_score

    def _model_score(self, vals: np.ndarray) -> float:
        if self.estimator is None:
            return 0.0
        cv = (
            StratifiedKFold(3, shuffle=True, random_state=self._seed)
            if self.task_type == "classification"
            else KFold(3, shuffle=True, random_state=self._seed)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                clone(self.estimator),
                vals.reshape(-1, 1),
                self.y,
                scoring=self.scoring,
                cv=cv,
                error_score=0.0,
            )
        return float(np.mean(scores))

    # ── tree evaluation ───────────────────────────────────────────────────────

    def _compute_vals(self, individual) -> np.ndarray | None:
        """Compile and evaluate the tree over ``X``; return a clean vector."""
        func = gp.compile(individual, self.pset)
        try:
            vals = np.array([func(*row) for row in self.X], dtype=float)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return None
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(vals, -_CLIP, _CLIP)

    def _redundancy(self, vals: np.ndarray) -> float:
        """Maximum absolute Pearson correlation with the current hall of fame."""
        if not self._hof_values:
            return 0.0
        if np.std(vals) < _EPS:
            return 1.0
        best = 0.0
        for hv in self._hof_values:
            if np.std(hv) < _EPS:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = pearsonr(vals, hv)[0]
            if np.isfinite(c):
                best = max(best, abs(float(c)))
        return best

    def __call__(self, individual) -> tuple[float]:
        """Return the (single-objective) fitness tuple for ``individual``."""
        key = str(individual)
        if key in self._cache:
            return (self._cache[key][0],)

        self._eval_count += 1
        vals = self._compute_vals(individual)
        if vals is None or np.std(vals) < _EPS:
            self._cache[key] = (0.0, 0.0, 0.0)
            return (0.0,)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                relevance = self._relevance(vals)
                if not np.isfinite(relevance):
                    relevance = 0.0
            except (ValueError, FloatingPointError):
                relevance = 0.0

        redundancy = self._redundancy(vals)
        penalty = self.config.parsimony_coeff * len(individual)
        fitness = max(0.0, relevance - self.config.redundancy_beta * redundancy - penalty)

        self._cache[key] = (fitness, relevance, redundancy)
        return (fitness,)

    # ── hall of fame ──────────────────────────────────────────────────────────

    def relevance_and_redundancy(self, individual) -> tuple[float, float]:
        """Return ``(relevance, redundancy)`` for a tree (using cache if present)."""
        key = str(individual)
        if key in self._cache:
            _, rel, red = self._cache[key]
            return rel, red
        vals = self._compute_vals(individual)
        if vals is None:
            return 0.0, 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                rel = self._relevance(vals)
            except (ValueError, FloatingPointError):
                rel = 0.0
        return rel, self._redundancy(vals)

    def add_to_hof(self, individual) -> None:
        """Record a tree's values for future redundancy penalties; clear cache."""
        vals = self._compute_vals(individual)
        if vals is not None and np.std(vals) > _EPS:
            self._hof_values.append(vals.copy())
            self._cache.clear()

    def compute_feature_values(self, individual, X_new: np.ndarray) -> np.ndarray:
        """Evaluate a tree over new data (used by ``transform``)."""
        func = gp.compile(individual, self.pset)
        try:
            vals = np.array([func(*row) for row in X_new], dtype=float)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return np.zeros(len(X_new))
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(vals, -_CLIP, _CLIP)

    @property
    def eval_count(self) -> int:
        """Number of (non-cached) evaluations performed."""
        return self._eval_count
