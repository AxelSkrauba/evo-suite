"""Builder for the DEAP :class:`~deap.gp.PrimitiveSet` used by the engineer."""

from __future__ import annotations

import functools
import random

from deap import gp

from evo_gpfe.primitives.functions import PRIMITIVE_SETS, VALID_FUNCTION_SETS


def sanitize_name(name: str, index: int) -> str:
    """Return a Python-identifier-safe argument name for DEAP.

    Feature names may contain spaces or punctuation; DEAP requires valid
    identifiers. Invalid names fall back to ``x{index}``.
    """
    safe = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    if not safe or safe[0].isdigit():
        safe = f"x{index}"
    return safe


def build_pset(
    n_features: int,
    function_set: str,
    uid: str,
    feature_names: list[str] | None = None,
    use_constants: bool = True,
    constant_range: tuple[float, float] = (-2.0, 2.0),
) -> gp.PrimitiveSet:
    """Build a primitive set for expression trees over ``n_features`` inputs.

    Parameters
    ----------
    n_features : int
        Number of input features (tree terminals/arguments).
    function_set : {'basic', 'extended', 'full', 'nonlinear'}
        Which named primitive set to use.
    uid : str
        Unique suffix to keep DEAP's global names collision-free across
        instances (the primitive set name and the ephemeral constant).
    feature_names : list of str, optional
        Names used to label the tree arguments (sanitised). Defaults to
        ``x0, x1, ...``.
    use_constants : bool, default=True
        Whether to add a random ephemeral constant terminal.
    constant_range : tuple of float, default=(-2.0, 2.0)
        ``(low, high)`` range for the ephemeral constant.

    Returns
    -------
    deap.gp.PrimitiveSet
        The configured primitive set.
    """
    if function_set not in PRIMITIVE_SETS:
        raise ValueError(f"function_set must be one of {VALID_FUNCTION_SETS}, got {function_set!r}")

    pset = gp.PrimitiveSet(f"MAIN_{uid}", n_features)

    for i in range(n_features):
        raw = feature_names[i] if feature_names is not None and i < len(feature_names) else f"x{i}"
        pset.renameArguments(**{f"ARG{i}": sanitize_name(raw, i)})

    for prim in PRIMITIVE_SETS[function_set]:
        pset.addPrimitive(prim.func, prim.arity, name=prim.name)

    if use_constants:
        low, high = constant_range
        pset.addEphemeralConstant(f"rc_{uid}", functools.partial(random.uniform, low, high))

    return pset
