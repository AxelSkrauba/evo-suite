"""Smoke test: scikit-learn estimator compliance via ``check_estimator``.

Genetic programming is comparatively expensive (every candidate tree is
evaluated over the training data), so the full ``check_estimator`` battery
runs with a deliberately tiny configuration. Marked ``slow`` so it can be
skipped during quick iterations with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import warnings

import pytest
from sklearn.utils.estimator_checks import check_estimator

from evo_gpfe import GPConfig, GPFeatureEngineer


@pytest.mark.slow
def test_check_estimator_passes():
    estimator = GPFeatureEngineer(
        config=GPConfig(
            population_size=10,
            n_generations=3,
            n_features_to_generate=2,
            verbose=False,
        )
    )
    # Infeasible candidate trees on the tiny synthetic data used by the checks
    # emit benign warnings; they are handled gracefully (fitness 0.0).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_estimator(estimator)
