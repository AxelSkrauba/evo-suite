"""Crossover operators with built-in repair."""

from __future__ import annotations

from deap import tools

from evo_gafs.operators.repair import repair_individual


def cx_uniform_with_repair(
    ind1: list[int],
    ind2: list[int],
    indpb: float = 0.5,
    min_features: int = 1,
) -> tuple[list[int], list[int]]:
    """Uniform crossover followed by repair on both children.

    Each bit is swapped between the two parents with probability ``indpb``.
    After the swap, both offspring are repaired so they keep at least
    ``min_features`` active features.

    Parameters
    ----------
    ind1, ind2 : list of int
        Parent genomes (modified in place to become the offspring).
    indpb : float, default=0.5
        Independent probability of swapping each bit.
    min_features : int, default=1
        Minimum number of active features each child must keep.

    Returns
    -------
    tuple of (list of int, list of int)
        The two repaired offspring.
    """
    tools.cxUniform(ind1, ind2, indpb=indpb)
    repair_individual(ind1, min_features)
    repair_individual(ind2, min_features)
    return ind1, ind2
