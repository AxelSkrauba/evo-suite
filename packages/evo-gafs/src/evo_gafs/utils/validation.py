"""Input validation and lightweight data-preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_target(y: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce the target to a numpy array, label-encoding non-numeric labels."""
    y_array = np.asarray(y)
    if y_array.dtype == object or np.issubdtype(y_array.dtype, np.str_):
        y_array = LabelEncoder().fit_transform(y_array)
    return y_array


def infer_task_type(y: np.ndarray, declared: str = "auto") -> str:
    """Infer ``'classification'`` or ``'regression'`` from the target.

    Parameters
    ----------
    y : numpy.ndarray
        Target vector.
    declared : str, default='auto'
        If not ``'auto'`` it is returned unchanged.

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
    is_integer = y.dtype.kind in "iu" or y.dtype == bool
    if len(unique) <= 20 and is_integer:
        return "classification"
    return "regression"


def resolve_scoring(scoring: str | None, task_type: str) -> str:
    """Return an explicit scoring string, defaulting by task type."""
    if scoring is not None:
        return scoring
    return "accuracy" if task_type == "classification" else "r2"
