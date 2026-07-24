"""Safe management of DEAP's global ``creator`` registrations.

DEAP registers fitness/individual types on the global :mod:`deap.creator`
module. Re-fitting an estimator (common in notebooks) or running
single/multiobjective mode side by side would collide with stale
registrations, so each run uses a unique UUID suffix and cleans up after
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from deap import base, creator


@dataclass
class DeapTypes:
    """Handle to the per-run DEAP types created on :mod:`deap.creator`."""

    individual_cls: type
    fitness_name: str
    individual_name: str
    uid: str

    def cleanup(self) -> None:
        """Remove the created types. Idempotent."""
        for name in (self.individual_name, self.fitness_name):
            if hasattr(creator, name):
                delattr(creator, name)


def new_uid() -> str:
    """Return a short unique identifier for per-instance DEAP registrations."""
    return uuid4().hex[:8]


def create_types(uid: str, mode: str) -> DeapTypes:
    """Create uniquely named fitness and individual (list) types.

    Parameters
    ----------
    uid : str
        Unique suffix (typically from :func:`new_uid`).
    mode : {'single', 'multiobjective'}
        ``'single'`` registers a single-objective (maximizing) fitness;
        ``'multiobjective'`` registers a two-objective (both maximizing)
        fitness for NSGA-II.

    Returns
    -------
    DeapTypes
        Handle holding the individual class and the registered names.
    """
    fitness_name = f"_EvoEnsFitness_{uid}"
    individual_name = f"_EvoEnsIndividual_{uid}"
    weights = (1.0, 1.0) if mode == "multiobjective" else (1.0,)

    creator.create(fitness_name, base.Fitness, weights=weights)
    creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

    return DeapTypes(
        individual_cls=getattr(creator, individual_name),
        fitness_name=fitness_name,
        individual_name=individual_name,
        uid=uid,
    )
