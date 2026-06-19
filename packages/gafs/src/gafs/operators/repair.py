"""Constraint-handling (repair) and initialisation operators.

In feature selection the most common constraint is a *minimum number of active
features*: an individual that selects zero (or too few) features is infeasible
because it cannot be evaluated with cross-validation. The repair operator
guarantees feasibility after the variation operators run.
"""

from __future__ import annotations

import random
from typing import Any


def repair_individual(individual: list[int], min_features: int) -> list[int]:
    """Repair an individual so that it has at least ``min_features`` active bits.

    The repair is applied in place after crossover and mutation. If the
    individual selects fewer than ``min_features`` features, randomly chosen
    inactive bits are turned on until the constraint is met.

    Parameters
    ----------
    individual : list of int
        Binary genome (modified in place).
    min_features : int
        Minimum number of active bits required.

    Returns
    -------
    list of int
        The same (repaired) individual.
    """
    active = sum(individual)
    if active < min_features:
        inactive = [i for i, bit in enumerate(individual) if bit == 0]
        needed = min_features - active
        for i in random.sample(inactive, min(needed, len(inactive))):
            individual[i] = 1
    return individual


def init_individual(ind_class: type, n_features: int, min_features: int) -> Any:
    """Create a feasible individual with an adaptive activation probability.

    A uniform 0.5 activation probability is valid but tends to produce either
    too few or too many active features on small problems. We use a slightly
    higher probability for small feature spaces to keep the initial population
    diverse and feasible, then repair to satisfy ``min_features``.

    Parameters
    ----------
    ind_class : type
        The DEAP individual class to instantiate.
    n_features : int
        Total number of features (genome length).
    min_features : int
        Minimum number of active bits.

    Returns
    -------
    ind_class
        A newly created, feasible individual.
    """
    p_on = 0.7 if n_features <= 10 else 0.5
    genome = [1 if random.random() < p_on else 0 for _ in range(n_features)]
    repair_individual(genome, min_features)
    return ind_class(genome)
