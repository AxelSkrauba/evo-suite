"""evo-gpfe — Genetic Programming Feature Engineer.

A scikit-learn-compatible symbolic feature constructor for tabular data, built
on DEAP. It evolves expression trees that combine the original features into new
ones, complementing ``evo-gafs`` (which selects among existing features).

The public API is assembled incrementally as the package is built out.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
