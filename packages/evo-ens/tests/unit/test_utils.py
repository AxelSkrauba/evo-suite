"""Unit tests for evo_ens.utils (diversity metrics, DEAP types, validation)."""

from __future__ import annotations

import numpy as np
import pytest
from deap import creator

from evo_ens.utils.deap_utils import create_types, new_uid
from evo_ens.utils.diversity import (
    diversity_pearson_mean,
    diversity_q_mean,
    q_statistic_pair,
)
from evo_ens.utils.validation import build_cv_splitter, infer_task_type, resolve_scoring


class TestQStatistic:
    def test_perfect_agreement_on_errors(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        p1 = np.array([1, 1, 1, 1, 1, 1])  # errs on indices 1, 3, 5
        p2 = np.array([1, 1, 1, 1, 1, 1])  # identical errors
        q = q_statistic_pair(y, p1, p2)
        assert q == pytest.approx(1.0)

    def test_zero_denominator_returns_zero(self):
        y = np.array([1, 1, 1])
        p1 = np.array([1, 1, 1])
        p2 = np.array([0, 0, 0])
        # n11=3 (p1 correct all), n00=0, n10=0, n01=0 -> denom 0
        assert q_statistic_pair(y, p1, p2) == 0.0

    def test_range(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p1 = rng.integers(0, 2, 200)
        p2 = rng.integers(0, 2, 200)
        q = q_statistic_pair(y, p1, p2)
        assert -1.0 <= q <= 1.0


class TestDiversityMean:
    def test_fewer_than_two_members_is_zero(self):
        y = np.array([0, 1, 0])
        assert diversity_q_mean(y, []) == 0.0
        assert diversity_q_mean(y, [np.array([0, 1, 0])]) == 0.0
        assert diversity_pearson_mean([]) == 0.0
        assert diversity_pearson_mean([np.array([1.0, 2.0])]) == 0.0

    def test_averages_all_pairs(self):
        y = np.array([1, 0, 1, 0])
        preds = [np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])]
        q = diversity_q_mean(y, preds)
        assert isinstance(q, float)

    def test_pearson_skips_constant_members(self):
        preds = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])]
        assert diversity_pearson_mean(preds) == 0.0

    def test_pearson_positive_correlation(self):
        preds = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])]
        assert diversity_pearson_mean(preds) == pytest.approx(1.0)


class TestDeapUtils:
    def test_new_uid_is_unique(self):
        assert new_uid() != new_uid()

    def test_create_types_single(self):
        uid = new_uid()
        types = create_types(uid, mode="single")
        try:
            assert hasattr(creator, types.fitness_name)
            assert hasattr(creator, types.individual_name)
            ind = types.individual_cls([0.1, 0.2])
            ind.fitness.values = (1.0,)
            assert ind.fitness.values == (1.0,)
        finally:
            types.cleanup()
        assert not hasattr(creator, types.fitness_name)
        assert not hasattr(creator, types.individual_name)

    def test_create_types_multiobjective(self):
        uid = new_uid()
        types = create_types(uid, mode="multiobjective")
        try:
            ind = types.individual_cls([0.1, 0.2])
            ind.fitness.values = (1.0, 0.5)
            assert ind.fitness.values == (1.0, 0.5)
        finally:
            types.cleanup()

    def test_cleanup_is_idempotent(self):
        uid = new_uid()
        types = create_types(uid, mode="single")
        types.cleanup()
        types.cleanup()  # should not raise


class TestValidation:
    def test_infer_task_type_classification(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        assert infer_task_type(y) == "classification"

    def test_infer_task_type_regression(self):
        y = np.linspace(0.0, 10.0, 50)
        assert infer_task_type(y) == "regression"

    def test_infer_task_type_declared(self):
        y = np.linspace(0.0, 10.0, 50)
        assert infer_task_type(y, declared="classification") == "classification"

    def test_infer_task_type_invalid_declared(self):
        with pytest.raises(ValueError):
            infer_task_type(np.array([0, 1]), declared="bogus")

    def test_resolve_scoring_defaults(self):
        assert resolve_scoring(None, "classification") == "accuracy"
        assert resolve_scoring(None, "regression") == "r2"

    def test_resolve_scoring_explicit(self):
        assert resolve_scoring("f1_macro", "classification") == "f1_macro"

    def test_build_cv_splitter_clamps_to_smallest_class(self):
        y = np.array([0, 0, 0, 0, 1, 1])  # smallest class has 2 members
        cv = build_cv_splitter(y, "classification", cv_folds=5, random_seed=0)
        assert cv.get_n_splits() == 2

    def test_build_cv_splitter_regression_clamps_to_n_samples(self):
        y = np.linspace(0, 1, 3)
        cv = build_cv_splitter(y, "regression", cv_folds=5, random_seed=0)
        assert cv.get_n_splits() == 3
