"""Evolutionary loops: single-objective GA and multi-objective NSGA-II."""

from evo_gafs.algorithms.nsga2 import run_nsga2
from evo_gafs.algorithms.single import run_single_objective

__all__ = ["run_nsga2", "run_single_objective"]
