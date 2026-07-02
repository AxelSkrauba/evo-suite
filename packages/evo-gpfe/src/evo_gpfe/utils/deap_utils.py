"""Safe management of DEAP's global ``creator``/``gp`` registrations.

DEAP registers fitness/individual types on the global :mod:`deap.creator`
module, and ephemeral constants on the :mod:`deap.gp` module. Re-fitting the
engineer (common in notebooks) would collide with stale registrations, so each
instance uses a unique UUID suffix and cleans up after itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from deap import base, creator, gp


@dataclass
class DeapTypes:
    """Handle to the per-run DEAP types created on :mod:`deap.creator`."""

    individual_cls: type
    fitness_name: str
    individual_name: str
    uid: str

    def cleanup(self) -> None:
        """Remove the created types and ephemeral constant. Idempotent."""
        for name in (self.individual_name, self.fitness_name):
            if hasattr(creator, name):
                delattr(creator, name)
        remove_ephemeral_constant(self.uid)


def new_uid() -> str:
    """Return a short unique identifier for per-instance DEAP registrations."""
    return uuid4().hex[:8]


def create_types(uid: str) -> DeapTypes:
    """Create uniquely named fitness and ``PrimitiveTree`` individual types.

    Parameters
    ----------
    uid : str
        Unique suffix (typically from :func:`new_uid`).

    Returns
    -------
    DeapTypes
        Handle holding the individual class and the registered names.
    """
    fitness_name = f"_GPFEFitness_{uid}"
    individual_name = f"_GPFEIndividual_{uid}"

    creator.create(fitness_name, base.Fitness, weights=(1.0,))
    creator.create(individual_name, gp.PrimitiveTree, fitness=getattr(creator, fitness_name))

    return DeapTypes(
        individual_cls=getattr(creator, individual_name),
        fitness_name=fitness_name,
        individual_name=individual_name,
        uid=uid,
    )


def remove_ephemeral_constant(uid: str) -> None:
    """Remove the ephemeral-constant class registered for ``uid`` (if any).

    ``deap.gp.PrimitiveSet.addEphemeralConstant`` stores a generated class on the
    ``deap.gp`` module so it stays picklable; this deletes it to keep the global
    namespace clean across runs.
    """
    name = f"rc_{uid}"
    if hasattr(gp, name):
        delattr(gp, name)
