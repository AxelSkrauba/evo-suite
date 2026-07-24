"""Fitness evaluation for ensemble individuals, using pre-computed OOF predictions.

The bottleneck of any wrapper-style evolutionary search is fitness
evaluation. Naively re-training and cross-validating every candidate
ensemble would cost ``O(n_gen * n_pop * n_candidates * cv_cost)``. Instead,
out-of-fold (OOF) predictions for every candidate are pre-computed *once*
(``O(n_candidates * cv_cost)``), and each individual's fitness becomes a
cheap ``O(n_samples)`` weighted combination of already-computed arrays.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score

from evo_ens.core.config import EvoEnsConfig
from evo_ens.utils.diversity import diversity_pearson_mean, diversity_q_mean


def compute_score(combined: np.ndarray, y: np.ndarray, scoring: str, task_type: str) -> float:
    """Score a combined (weighted-average) prediction array against ``y``."""
    if task_type == "classification":
        y_pred = combined.argmax(axis=1)
        if scoring == "accuracy":
            return float(accuracy_score(y, y_pred))
        if scoring == "f1_macro":
            return float(f1_score(y, y_pred, average="macro", zero_division=0))
        if scoring == "roc_auc":
            try:
                if combined.shape[1] == 2:
                    return float(roc_auc_score(y, combined[:, 1]))
                return float(roc_auc_score(y, combined, multi_class="ovr"))
            except ValueError:
                return float(accuracy_score(y, y_pred))
        return float(accuracy_score(y, y_pred))

    if scoring == "r2":
        return float(r2_score(y, combined))
    if scoring == "neg_rmse":
        return -float(np.sqrt(mean_squared_error(y, combined)))
    return float(r2_score(y, combined))


class EnsembleFitnessEvaluator:
    """Evaluate the fitness of an ensemble individual.

    An individual is a flat vector of ``2 * n_candidates`` continuous genes:
    ``[b_0, ..., b_{N-1}, w_0, ..., w_{N-1}]`` where ``b_i > 0.5`` activates
    candidate ``i`` and ``w_i`` is its raw (pre-normalization) weight. The
    object is callable so it can be registered directly on a DEAP toolbox.

    Parameters
    ----------
    oof_predictions : list of numpy.ndarray
        Out-of-fold predictions per candidate: shape ``(n_samples,
        n_classes)`` for classification (probabilities) or ``(n_samples,)``
        for regression.
    y : numpy.ndarray
        Target vector.
    config : EvoEnsConfig
    task_type : {'classification', 'regression'}
    scoring : str
    n_candidates : int
    """

    def __init__(
        self,
        oof_predictions: list[np.ndarray],
        y: np.ndarray,
        config: EvoEnsConfig,
        task_type: str,
        scoring: str,
        n_candidates: int,
    ) -> None:
        self.oof = oof_predictions
        self.y = y
        self.cfg = config
        self.task_type = task_type
        self.scoring = scoring
        self.n_candidates = n_candidates
        self._eval_count = 0
        self._cache: dict[tuple, tuple[float, ...]] = {}

    def normalize_weights(self, raw: np.ndarray) -> np.ndarray:
        """Normalize raw weight genes according to ``config.weight_method``."""
        method = self.cfg.weight_method
        if method == "softmax":
            e = np.exp(raw - raw.max())
            return e / e.sum()
        if method == "abs_norm":
            a = np.abs(raw)
            s = a.sum()
            return a / s if s > 1e-10 else np.ones(len(raw)) / len(raw)
        return np.ones(len(raw)) / len(raw)  # uniform

    def combine(self, active_idx: list[int], weights: np.ndarray) -> np.ndarray:
        """Weighted-combine the OOF predictions of the active candidates."""
        combined = weights[0] * self.oof[active_idx[0]]
        for j in range(1, len(active_idx)):
            combined = combined + weights[j] * self.oof[active_idx[j]]
        return combined

    def score(self, combined: np.ndarray) -> float:
        """Score a combined prediction array against ``self.y``."""
        return compute_score(combined, self.y, self.scoring, self.task_type)

    def diversity(self, active_idx: list[int]) -> float:
        """Diversity metric among the active candidates (higher = less diverse)."""
        if len(active_idx) < 2:
            return 0.0
        if self.task_type == "classification":
            preds = [self.oof[i].argmax(1) for i in active_idx]
            return diversity_q_mean(self.y, preds)
        preds = [self.oof[i] for i in active_idx]
        return diversity_pearson_mean(preds)

    def __call__(self, individual: list[float]) -> tuple[float, ...]:
        """Evaluate ``individual``, returning a fitness tuple.

        Single mode: ``(score - beta * diversity,)``.
        Multiobjective mode: ``(score, compression)`` where
        ``compression = 1 - n_selected / n_candidates``.
        """
        n = self.n_candidates
        bits = tuple(1 if b > 0.5 else 0 for b in individual[:n])
        weight_key = tuple(round(w, 4) for w in individual[n:])
        cache_key = (bits, weight_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._eval_count += 1
        active_idx = [i for i, b in enumerate(bits) if b == 1]
        n_sel = len(active_idx)

        if n_sel < self.cfg.min_models:
            result = (0.0, 0.0) if self.cfg.mode == "multiobjective" else (0.0,)
            self._cache[cache_key] = result
            return result

        if self.cfg.max_models is not None and n_sel > self.cfg.max_models:
            raw = np.array([individual[n + i] for i in active_idx])
            top_k = int(self.cfg.max_models)
            keep = np.argsort(raw)[-top_k:]
            active_idx = [active_idx[k] for k in sorted(keep)]
            n_sel = len(active_idx)

        raw_weights = np.array([individual[n + i] for i in active_idx])
        weights = self.normalize_weights(raw_weights)
        combined = self.combine(active_idx, weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = self.score(combined)
        if not np.isfinite(score):
            score = 0.0

        if self.cfg.mode == "multiobjective":
            compression = 1.0 - n_sel / n
            result = (float(score), float(compression))
        else:
            diversity = self.diversity(active_idx)
            fitness_val = float(score) - self.cfg.diversity_beta * float(diversity)
            result = (max(0.0, fitness_val),)

        self._cache[cache_key] = result
        return result

    @property
    def eval_count(self) -> int:
        """Number of (non-cached) fitness evaluations performed."""
        return self._eval_count

    def decode_individual(
        self, individual: list[float], oof_scores: list[float]
    ) -> tuple[list[int], np.ndarray]:
        """Return ``(active_indices, normalized_weights)`` decoded from an individual.

        Falls back to the single best standalone candidate if no gene is
        active (can happen after mutation/crossover on the boundary).
        """
        n = self.n_candidates
        active_idx = [i for i, b in enumerate(individual[:n]) if b > 0.5]
        if not active_idx:
            active_idx = [int(np.argmax(oof_scores))]
        raw_weights = np.array([individual[n + i] for i in active_idx])
        weights = self.normalize_weights(raw_weights)
        return active_idx, weights
