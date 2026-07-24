"""Input validation and lightweight data-preparation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def infer_task_type(y: np.ndarray, declared: str = "auto") -> str:
    """Infer ``'classification'`` or ``'regression'`` from the target.

    Parameters
    ----------
    y : numpy.ndarray
        Target vector.
    declared : str, default='auto'
        If not ``'auto'`` it is returned unchanged (after validation).

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.
    """
    if declared != "auto":
        if declared not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'auto', 'classification' or 'regression', got {declared!r}"
            )
        return declared

    unique = np.unique(y)
    integral = np.all(unique == unique.astype(int))
    if len(unique) <= 20 and integral and len(unique) <= len(y) * 0.5:
        return "classification"
    return "regression"


def resolve_scoring(scoring: str | None, task_type: str) -> str:
    """Return an explicit scoring string, defaulting by task type."""
    if scoring is not None:
        return scoring
    return "accuracy" if task_type == "classification" else "r2"


def build_cv_splitter(y: np.ndarray, task_type: str, cv_folds: int, random_seed: int | None):
    """Build a stratified/plain K-fold splitter, defensively clamped.

    Clamps ``n_splits`` to the smallest class count (classification) or to
    ``n_samples`` (regression) so tiny inputs (e.g. from
    ``check_estimator``) never raise.
    """
    n_samples = len(y)
    if task_type == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError(
                f"EvoEnsemble requires at least 2 classes for cross-validation, "
                f"got 1 class with n_samples={n_samples}."
            )
        if counts.min() < 2:
            raise ValueError(
                "EvoEnsemble requires at least 2 samples per class for "
                f"out-of-fold cross-validation; the smallest class has {int(counts.min())} sample."
            )
        n_splits = min(cv_folds, int(counts.min()), n_samples)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    if n_samples < 2:
        raise ValueError(
            f"EvoEnsemble requires at least 2 samples for cross-validation, got n_samples={n_samples}."
        )
    n_splits = max(2, min(cv_folds, n_samples))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
