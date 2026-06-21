"""Mutation operators with built-in repair."""

from __future__ import annotations

from deap import tools

from evo_gafs.operators.repair import repair_individual


def mut_flip_with_repair(
    individual: list[int],
    indpb: float,
    min_features: int = 1,
) -> tuple[list[int]]:
    """Bit-flip mutation followed by repair.

    Parameters
    ----------
    individual : list of int
        Genome to mutate (modified in place).
    indpb : float
        Independent probability of flipping each bit.
    min_features : int, default=1
        Minimum number of active features the individual must keep.

    Returns
    -------
    tuple of (list of int,)
        A one-tuple with the repaired individual, as required by DEAP.
    """
    tools.mutFlipBit(individual, indpb=indpb)
    repair_individual(individual, min_features)
    return (individual,)
