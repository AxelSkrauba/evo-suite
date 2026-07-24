"""Smoke test: scikit-learn estimator compliance via ``check_estimator``.

Each fit pre-computes out-of-fold predictions for the whole candidate pool
and runs a full GA loop, so the full ``check_estimator`` battery runs with a
deliberately tiny configuration and pool. Marked ``slow`` so it can be
skipped during quick iterations with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import warnings

import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from evo_ens import EvoEnsConfig, EvoEnsembleClassifier, EvoEnsembleRegressor


def _tiny_config() -> EvoEnsConfig:
    return EvoEnsConfig(
        population_size=8,
        n_generations=3,
        min_models=1,
        cv_folds=2,
        verbose=False,
    )


@pytest.mark.slow
def test_check_estimator_classifier():
    estimator = EvoEnsembleClassifier(
        candidate_estimators=[
            LogisticRegression(max_iter=200),
            DecisionTreeClassifier(max_depth=3, random_state=0),
            DecisionTreeClassifier(max_depth=6, random_state=1),
        ],
        config=_tiny_config(),
    )
    # Degenerate candidate models on the tiny synthetic data used by the
    # checks (e.g. a single class in a CV fold) emit benign warnings; the
    # evaluator handles them gracefully (fitness 0.0 / dummy OOF arrays).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_estimator(estimator)


@pytest.mark.slow
def test_check_estimator_regressor():
    estimator = EvoEnsembleRegressor(
        candidate_estimators=[
            Ridge(alpha=1.0),
            DecisionTreeRegressor(max_depth=3, random_state=0),
            DecisionTreeRegressor(max_depth=6, random_state=1),
        ],
        config=_tiny_config(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_estimator(estimator)
