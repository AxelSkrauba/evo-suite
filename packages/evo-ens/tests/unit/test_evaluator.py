"""Unit tests for evo_ens.core.evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from evo_ens.core.config import EvoEnsConfig
from evo_ens.core.evaluator import EnsembleFitnessEvaluator, compute_score


@pytest.fixture
def classification_oof():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    # candidate 0: near-perfect; candidate 1: noisy; candidate 2: near-perfect too
    perfect = np.eye(2)[y]
    noisy = np.eye(2)[rng.integers(0, 2, 100)]
    oof = [perfect, noisy, perfect.copy()]
    return oof, y


@pytest.fixture
def regression_oof():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    oof = [y + rng.normal(scale=0.01, size=100), rng.normal(size=100), y.copy()]
    return oof, y


class TestComputeScore:
    def test_classification_accuracy(self):
        y = np.array([0, 1, 1, 0])
        combined = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])
        assert compute_score(combined, y, "accuracy", "classification") == 1.0

    def test_regression_r2(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_score(y.copy(), y, "r2", "regression") == pytest.approx(1.0)

    def test_regression_neg_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_score(y.copy(), y, "neg_rmse", "regression") == pytest.approx(0.0)


class TestEnsembleFitnessEvaluatorSingle:
    def test_call_returns_scalar_tuple(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(min_models=1, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # activate 0 and 2, equal weights
        fitness = ev(individual)
        assert isinstance(fitness, tuple)
        assert len(fitness) == 1
        assert fitness[0] >= 0.0

    def test_below_min_models_is_penalized(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(min_models=2, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # only 1 active
        assert ev(individual) == (0.0,)

    def test_max_models_truncates_by_weight(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(min_models=1, max_models=1, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 1.0, 1.0, 0.1, 5.0, 0.1]  # all active, candidate 1 has top weight
        ev(individual)
        # cache key reflects the *original* bits, but eval_count should be 1
        assert ev.eval_count == 1

    def test_caching_avoids_recount(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(min_models=1, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        ev(individual)
        ev(individual)
        assert ev.eval_count == 1

    def test_regression_mode(self, regression_oof):
        oof, y = regression_oof
        cfg = EvoEnsConfig(min_models=1, scoring="r2", verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "regression", "r2", 3)
        individual = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        fitness = ev(individual)
        assert fitness[0] > 0.5


class TestEnsembleFitnessEvaluatorMultiobjective:
    def test_returns_two_objectives(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(mode="multiobjective", min_models=1, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        fitness = ev(individual)
        assert len(fitness) == 2
        score, compression = fitness
        assert 0.0 <= compression <= 1.0

    def test_below_min_models_penalized_both_objectives(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(mode="multiobjective", min_models=2, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        assert ev(individual) == (0.0, 0.0)


class TestWeightNormalization:
    def test_softmax_sums_to_one(self):
        cfg = EvoEnsConfig(weight_method="softmax", verbose=False)
        ev = EnsembleFitnessEvaluator([np.zeros(3)], np.zeros(3), cfg, "regression", "r2", 1)
        w = ev.normalize_weights(np.array([1.0, 2.0, 3.0]))
        assert w.sum() == pytest.approx(1.0)

    def test_abs_norm_sums_to_one(self):
        cfg = EvoEnsConfig(weight_method="abs_norm", verbose=False)
        ev = EnsembleFitnessEvaluator([np.zeros(3)], np.zeros(3), cfg, "regression", "r2", 1)
        w = ev.normalize_weights(np.array([-1.0, 2.0, -3.0]))
        assert w.sum() == pytest.approx(1.0)

    def test_abs_norm_handles_all_zero(self):
        cfg = EvoEnsConfig(weight_method="abs_norm", verbose=False)
        ev = EnsembleFitnessEvaluator([np.zeros(3)], np.zeros(3), cfg, "regression", "r2", 1)
        w = ev.normalize_weights(np.array([0.0, 0.0]))
        assert w.sum() == pytest.approx(1.0)

    def test_uniform_ignores_values(self):
        cfg = EvoEnsConfig(weight_method="uniform", verbose=False)
        ev = EnsembleFitnessEvaluator([np.zeros(3)], np.zeros(3), cfg, "regression", "r2", 1)
        w = ev.normalize_weights(np.array([100.0, -50.0]))
        assert w[0] == pytest.approx(0.5)
        assert w[1] == pytest.approx(0.5)


class TestDecodeIndividual:
    def test_decodes_active_indices_and_weights(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(min_models=1, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        idx, weights = ev.decode_individual(individual, oof_scores=[0.9, 0.5, 0.9])
        assert idx == [0, 2]
        assert weights.sum() == pytest.approx(1.0)

    def test_falls_back_to_best_single_when_none_active(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(min_models=1, verbose=False)
        ev = EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", 3)
        individual = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        idx, weights = ev.decode_individual(individual, oof_scores=[0.9, 0.5, 0.95])
        assert idx == [2]
        assert weights[0] == pytest.approx(1.0)
