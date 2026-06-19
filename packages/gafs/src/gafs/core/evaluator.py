"""Fitness evaluation via cross-validation (the wrapper criterion)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from gafs.core.config import GAConfig

logger = logging.getLogger("gafs")


class FitnessEvaluator:
    """Evaluate an individual (binary feature mask) with cross-validation.

    The evaluator is instantiated once per fit and registered with DEAP as the
    ``evaluate`` operator. It caches results per individual to avoid re-running
    cross-validation for genomes that have already been seen.

    Penalisation strategy
    ----------------------
    * Fewer than ``min_features`` active features -> fitness of zero.

    Fitness by mode
    ---------------
    * ``'single'``::

          fitness = alpha * cv_score + (1 - alpha) * compression

      where ``compression = 1 - n_selected / n_total``.
    * ``'multiobjective'``: returns ``(cv_score, compression)``, both maximised
      (DEAP/NSGA-II handles the Pareto front).

    Parameters
    ----------
    estimator : sklearn estimator
        Model used as the wrapper criterion. It is cloned for each evaluation.
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Target vector.
    scoring : str
        scikit-learn scoring string.
    cv : cross-validation splitter
        Splitter used for the score.
    config : GAConfig
        Configuration (provides ``mode``, ``alpha``, ``min_features``,
        ``n_jobs``).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str,
        cv: object,
        config: GAConfig,
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.config = config
        self.n_features = X.shape[1]
        self._eval_count = 0
        self._cache: dict[tuple[int, ...], tuple[float, ...]] = {}

    def __call__(self, individual: list[int]) -> tuple[float, ...]:
        """Evaluate one individual and return its fitness tuple.

        Returns
        -------
        tuple of float
            ``(fitness,)`` in single-objective mode, ``(cv_score, compression)``
            in multi-objective mode.
        """
        key = tuple(individual)
        if key in self._cache:
            return self._cache[key]

        self._eval_count += 1
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        n_selected = len(selected)

        if n_selected < self.config.min_features:
            penalty: tuple[float, ...] = (
                (0.0, 0.0) if self.config.mode == "multiobjective" else (0.0,)
            )
            self._cache[key] = penalty
            return penalty

        cv_score = self._cross_val_score(selected)
        compression = 1.0 - (n_selected / self.n_features)

        if self.config.mode == "multiobjective":
            result: tuple[float, ...] = (cv_score, compression)
        else:
            alpha = self.config.alpha
            result = (alpha * cv_score + (1.0 - alpha) * compression,)

        self._cache[key] = result
        return result

    def cv_score(self, selected: list[int]) -> float:
        """Public helper: raw (unweighted) CV score for a feature subset."""
        return self._cross_val_score(selected)

    def _cross_val_score(self, selected: list[int]) -> float:
        X_sub = self.X[:, selected]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(
                    clone(self.estimator),
                    X_sub,
                    self.y,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.config.n_jobs,
                    error_score=0.0,
                )
                return float(np.mean(scores))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Cross-validation failed for a candidate: %s", exc)
                return 0.0

    @property
    def eval_count(self) -> int:
        """Number of (non-cached) fitness evaluations performed."""
        return self._eval_count
