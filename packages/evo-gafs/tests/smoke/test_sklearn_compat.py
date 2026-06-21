"""Smoke test: scikit-learn estimator compliance via ``check_estimator``.

The GA selector is a *wrapper* method: every candidate subset is scored with
cross-validation, so the full ``check_estimator`` battery is comparatively
expensive. We therefore run it with a deliberately tiny configuration. The
check is marked ``slow`` so it can be skipped during quick iterations with
``pytest -m "not slow"``.
"""

from __future__ import annotations

import warnings

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

from evo_gafs import GAConfig, GAFeatureSelector


@pytest.mark.slow
def test_check_estimator_passes():
    estimator = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=0),
        config=GAConfig(population_size=6, n_generations=2, cv_folds=2, verbose=False),
    )
    # Candidate subsets that are infeasible on the tiny synthetic data used by
    # the checks emit benign warnings; they are handled gracefully (score 0.0).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_estimator(estimator)
