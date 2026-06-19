"""Evolutionary loops: single-objective GA and multi-objective NSGA-II."""

from gafs.algorithms.nsga2 import run_nsga2
from gafs.algorithms.single import run_single_objective

__all__ = ["run_nsga2", "run_single_objective"]
