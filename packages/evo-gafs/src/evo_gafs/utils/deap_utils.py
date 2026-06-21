"""Safe management of DEAP's global ``creator`` types.

DEAP registers fitness and individual types as attributes on the global
:mod:`deap.creator` module. Re-running the selector (very common in notebooks)
with the previous fixed-name approach silently reused stale types, and switching
between single- and multi-objective modes caused weight collisions.

We avoid this by giving every selector instance its own uniquely named types
(suffixed with a short UUID) and cleaning them up once the run is done.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from deap import base, creator


@dataclass
class DeapTypes:
    """Handle to the per-run DEAP types created on :mod:`deap.creator`.

    Attributes
    ----------
    individual_cls : type
        The dynamically created individual class.
    fitness_name : str
        Attribute name of the fitness type on ``creator``.
    individual_name : str
        Attribute name of the individual type on ``creator``.
    """

    individual_cls: type
    fitness_name: str
    individual_name: str

    def cleanup(self) -> None:
        """Remove the created types from :mod:`deap.creator`.

        Safe to call multiple times.
        """
        for name in (self.individual_name, self.fitness_name):
            if hasattr(creator, name):
                delattr(creator, name)


def create_types(mode: str) -> DeapTypes:
    """Create uniquely named DEAP fitness/individual types for one run.

    Parameters
    ----------
    mode : {'single', 'multiobjective'}
        ``'single'`` builds a single-objective fitness (weights ``(1.0,)``);
        ``'multiobjective'`` builds a two-objective fitness, both maximised
        (weights ``(1.0, 1.0)`` for ``cv_score`` and ``compression``).

    Returns
    -------
    DeapTypes
        Handle holding the individual class and the registered type names.
    """
    uid = uuid4().hex[:8]
    fitness_name = f"_evo_gafsFitness_{uid}"
    individual_name = f"_evo_gafsIndividual_{uid}"

    weights = (1.0, 1.0) if mode == "multiobjective" else (1.0,)
    creator.create(fitness_name, base.Fitness, weights=weights)
    creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

    return DeapTypes(
        individual_cls=getattr(creator, individual_name),
        fitness_name=fitness_name,
        individual_name=individual_name,
    )
