"""Diversity metrics between ensemble members.

An ensemble outperforms its best individual member when the members' errors
are decorrelated. For classification, the Yule Q-statistic measures the
error correlation between two classifiers; for regression, the (absolute)
Pearson correlation between predictions is used as an analogous proxy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def q_statistic_pair(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> float:
    """Yule's Q-statistic between two classifiers' class predictions.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        Ground-truth (integer-encoded) labels.
    pred1, pred2 : numpy.ndarray of shape (n_samples,)
        Predicted class labels for each classifier.

    Returns
    -------
    float
        ``Q in [-1, 1]``. ``Q == 0`` means independent errors; ``Q < 0``
        means anti-correlated errors (ideal for an ensemble); returns
        ``0.0`` when both denominators terms vanish (degenerate agreement).
    """
    c1 = pred1.astype(int)
    c2 = pred2.astype(int)
    yt = y_true.astype(int)
    n11 = np.sum((c1 == yt) & (c2 == yt))
    n00 = np.sum((c1 != yt) & (c2 != yt))
    n10 = np.sum((c1 == yt) & (c2 != yt))
    n01 = np.sum((c1 != yt) & (c2 == yt))
    denom = n11 * n00 + n10 * n01
    return float((n11 * n00 - n10 * n01) / denom) if denom > 0 else 0.0


def diversity_q_mean(y_true: np.ndarray, preds_list: list[np.ndarray]) -> float:
    """Average pairwise Q-statistic across every pair of an ensemble.

    Returns ``0.0`` for ensembles with fewer than two members.
    """
    n = len(preds_list)
    if n < 2:
        return 0.0
    qs = [
        q_statistic_pair(y_true, preds_list[i], preds_list[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(qs))


def diversity_pearson_mean(preds_list: list[np.ndarray]) -> float:
    """Average pairwise absolute Pearson correlation (regression proxy for Q).

    Members with (near-)constant predictions are skipped for a given pair
    (undefined correlation). Returns ``0.0`` when fewer than two members are
    given or no pair yields a finite correlation.
    """
    n = len(preds_list)
    if n < 2:
        return 0.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(preds_list[i]) > 1e-10 and np.std(preds_list[j]) > 1e-10:
                c, _ = pearsonr(preds_list[i], preds_list[j])
                corrs.append(abs(c) if np.isfinite(c) else 0.0)
    return float(np.mean(corrs)) if corrs else 0.0
