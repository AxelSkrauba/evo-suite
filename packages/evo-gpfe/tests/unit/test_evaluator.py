"""Unit tests for the GP fitness evaluator."""

from __future__ import annotations

import numpy as np
import pytest
from deap import gp

from evo_gpfe.core.config import GPConfig
from evo_gpfe.core.evaluator import GPFitnessEvaluator
from evo_gpfe.primitives import build_pset
from evo_gpfe.utils import new_uid


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + X[:, 1]  # strongly depends on x0 + x1
    pset = build_pset(3, "extended", uid=new_uid(), use_constants=False)
    return X, y, pset


def _tree(expr: str, pset) -> gp.PrimitiveTree:
    return gp.PrimitiveTree.from_string(expr, pset)


def _evaluator(X, y, pset, **cfg):
    config = GPConfig(verbose=False, **cfg)
    return GPFitnessEvaluator(X, y, pset, config, task_type="regression")


def test_fitness_is_nonnegative_tuple(setup):
    ev = _evaluator(*setup)
    fit = ev(_tree("add(x0, x1)", setup[2]))
    assert isinstance(fit, tuple) and len(fit) == 1
    assert fit[0] >= 0.0


def test_relevant_feature_beats_noise(setup):
    X, y, pset = setup
    ev = _evaluator(X, y, pset, fitness_metric="correlation")
    good = ev(_tree("add(x0, x1)", pset))[0]
    weak = ev(_tree("x2", pset))[0]
    assert good > weak


def test_cache_avoids_recomputation(setup):
    ev = _evaluator(*setup)
    tree = _tree("mul(x0, x1)", setup[2])
    ev(tree)
    n = ev.eval_count
    ev(_tree("mul(x0, x1)", setup[2]))  # same expression
    assert ev.eval_count == n


def test_constant_feature_is_penalised(setup):
    X, y, pset = setup
    ev = _evaluator(X, y, pset, fitness_metric="correlation")
    # sub(x0, x0) == 0 everywhere -> zero variance -> fitness 0
    assert ev(_tree("sub(x0, x0)", pset))[0] == 0.0


def test_add_to_hof_increases_redundancy(setup):
    X, y, pset = setup
    ev = _evaluator(X, y, pset, fitness_metric="correlation")
    tree = _tree("add(x0, x1)", pset)
    _, red_before = ev.relevance_and_redundancy(tree)
    assert red_before == 0.0
    ev.add_to_hof(tree)
    _, red_after = ev.relevance_and_redundancy(tree)
    assert red_after == pytest.approx(1.0, abs=1e-6)


def test_compute_feature_values_shape(setup):
    X, y, pset = setup
    ev = _evaluator(X, y, pset)
    vals = ev.compute_feature_values(_tree("add(x0, x1)", pset), X)
    assert vals.shape == (X.shape[0],)
    assert np.all(np.isfinite(vals))
