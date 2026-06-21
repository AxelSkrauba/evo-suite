"""Unit tests for the FitnessEvaluator and DEAP type management."""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from evo_gafs import GAConfig
from evo_gafs.core.evaluator import FitnessEvaluator
from evo_gafs.utils.deap_utils import create_types


def _make_evaluator(mode="single"):
    X, y = load_iris(return_X_y=True)
    cfg = GAConfig(mode=mode, min_features=1, verbose=False, alpha=0.5)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    return FitnessEvaluator(DecisionTreeClassifier(random_state=0), X, y, "accuracy", cv, cfg)


def test_single_objective_returns_scalar_tuple():
    ev = _make_evaluator("single")
    result = ev([1, 1, 1, 1])
    assert len(result) == 1
    assert 0.0 <= result[0] <= 1.0


def test_multiobjective_returns_two_values():
    ev = _make_evaluator("multiobjective")
    result = ev([1, 0, 1, 0])
    assert len(result) == 2


def test_empty_individual_is_penalised():
    ev = _make_evaluator("single")
    assert ev([0, 0, 0, 0]) == (0.0,)


def test_cache_avoids_recomputation():
    ev = _make_evaluator("single")
    ev([1, 0, 1, 0])
    first = ev.eval_count
    ev([1, 0, 1, 0])  # same genome
    assert ev.eval_count == first


def test_cv_score_helper_matches_range():
    ev = _make_evaluator("single")
    assert 0.0 <= ev.cv_score([0, 1, 2, 3]) <= 1.0


def test_create_and_cleanup_types():
    from deap import creator

    types = create_types("multiobjective")
    assert hasattr(creator, types.individual_name)
    ind = types.individual_cls([1, 0])
    assert ind.fitness.weights == (1.0, 1.0)
    types.cleanup()
    assert not hasattr(creator, types.individual_name)
    types.cleanup()  # idempotent
