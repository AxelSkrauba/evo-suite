"""Unit tests for the genetic operators."""

from __future__ import annotations

import random

from gafs.operators import (
    cx_uniform_with_repair,
    init_individual,
    mut_flip_with_repair,
    repair_individual,
)


def test_repair_activates_minimum_features():
    genome = [0, 0, 0, 0, 0]
    repair_individual(genome, min_features=2)
    assert sum(genome) >= 2


def test_repair_leaves_feasible_individual_untouched():
    genome = [1, 0, 1, 0]
    repair_individual(genome, min_features=1)
    assert sum(genome) == 2


def test_init_individual_respects_min_features():
    random.seed(0)

    class _Ind(list):
        pass

    for _ in range(20):
        ind = init_individual(_Ind, n_features=6, min_features=2)
        assert isinstance(ind, _Ind)
        assert sum(ind) >= 2
        assert len(ind) == 6


def test_crossover_keeps_min_features():
    random.seed(0)
    a = [1, 0, 0, 0]
    b = [0, 0, 0, 1]
    c1, c2 = cx_uniform_with_repair(a, b, indpb=0.9, min_features=1)
    assert sum(c1) >= 1
    assert sum(c2) >= 1


def test_mutation_keeps_min_features_and_returns_tuple():
    random.seed(0)
    ind = [1, 1, 1, 1]
    out = mut_flip_with_repair(ind, indpb=1.0, min_features=2)
    assert isinstance(out, tuple)
    assert sum(out[0]) >= 2
