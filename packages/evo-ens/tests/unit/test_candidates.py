"""Unit tests for evo_ens.core.candidates."""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator, clone

from evo_ens.core.candidates import default_candidates, estimator_display_name


class TestDefaultCandidates:
    def test_classification_pool(self):
        pool = default_candidates("classification", random_seed=0)
        assert len(pool) >= 10
        assert all(isinstance(est, BaseEstimator) for est in pool)

    def test_regression_pool(self):
        pool = default_candidates("regression", random_seed=0)
        assert len(pool) >= 10
        assert all(isinstance(est, BaseEstimator) for est in pool)

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError):
            default_candidates("bogus")

    def test_pools_are_clonable_and_independent(self):
        pool = default_candidates("classification", random_seed=1)
        cloned = [clone(est) for est in pool]
        assert len(cloned) == len(pool)
        assert cloned[0] is not pool[0]

    def test_seed_affects_paired_random_states(self):
        pool_a = default_candidates("classification", random_seed=1)
        pool_b = default_candidates("classification", random_seed=2)
        rf_a = pool_a[0].get_params()["random_state"]
        rf_b = pool_b[0].get_params()["random_state"]
        assert rf_a != rf_b


class TestEstimatorDisplayName:
    def test_includes_index_and_class_name(self):
        pool = default_candidates("classification", random_seed=0)
        name = estimator_display_name(pool[0], 0)
        assert name.startswith("00:")
        assert "RandomForestClassifier" in name

    def test_distinctive_param_included(self):
        pool = default_candidates("regression", random_seed=0)
        names = [estimator_display_name(est, i) for i, est in enumerate(pool)]
        assert any("alpha=" in n for n in names)
