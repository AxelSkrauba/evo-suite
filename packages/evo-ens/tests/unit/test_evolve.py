"""Unit tests for evo_ens.algorithms.evolve."""

from __future__ import annotations

import numpy as np
import pytest

from evo_ens.algorithms.evolve import build_toolbox, run_nsga2, run_single
from evo_ens.core.config import EvoEnsConfig
from evo_ens.core.evaluator import EnsembleFitnessEvaluator
from evo_ens.utils.deap_utils import create_types, new_uid


@pytest.fixture
def classification_oof():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    oof = [np.eye(2)[y], np.eye(2)[rng.integers(0, 2, 60)], np.eye(2)[y].copy()]
    return oof, y


def _make_evaluator(oof, y, cfg):
    return EnsembleFitnessEvaluator(oof, y, cfg, "classification", "accuracy", n_candidates=3)


class TestRunSingle:
    def test_evolves_a_valid_population(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(
            population_size=12,
            n_generations=5,
            min_models=1,
            random_seed=0,
            verbose=False,
            early_stopping_rounds=None,
        )
        uid = new_uid()
        types = create_types(uid, mode="single")
        try:
            evaluator = _make_evaluator(oof, y, cfg)
            toolbox = build_toolbox(types, n_candidates=3, evaluator=evaluator, config=cfg)
            population, history = run_single(toolbox, 3, cfg)
            assert len(population) == cfg.population_size
            assert len(history) == cfg.n_generations
            assert all(ind.fitness.valid for ind in population)
        finally:
            types.cleanup()

    def test_early_stopping_can_shorten_run(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(
            population_size=10,
            n_generations=50,
            min_models=1,
            random_seed=0,
            verbose=False,
            crossover_prob=0.01,
            mutation_prob=0.0,
            early_stopping_rounds=3,
            early_stopping_tol=1e-3,
        )
        uid = new_uid()
        types = create_types(uid, mode="single")
        try:
            evaluator = _make_evaluator(oof, y, cfg)
            toolbox = build_toolbox(types, n_candidates=3, evaluator=evaluator, config=cfg)
            _, history = run_single(toolbox, 3, cfg)
            assert len(history) < cfg.n_generations
        finally:
            types.cleanup()


class TestRunNsga2:
    def test_evolves_pareto_population(self, classification_oof):
        oof, y = classification_oof
        cfg = EvoEnsConfig(
            mode="multiobjective",
            population_size=12,
            n_generations=5,
            min_models=1,
            random_seed=0,
            verbose=False,
        )
        uid = new_uid()
        types = create_types(uid, mode="multiobjective")
        try:
            evaluator = _make_evaluator(oof, y, cfg)
            toolbox = build_toolbox(types, n_candidates=3, evaluator=evaluator, config=cfg)
            population, history = run_nsga2(toolbox, 3, cfg)
            assert len(population) == cfg.population_size
            assert len(history) == cfg.n_generations
            assert all(len(ind.fitness.values) == 2 for ind in population)
        finally:
            types.cleanup()
