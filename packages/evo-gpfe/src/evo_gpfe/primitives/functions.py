"""Protected mathematical primitives and named primitive sets.

Genetic-programming trees combine these scalar operators. The "protected"
variants keep evaluation total (no exceptions, no NaN/inf) on arbitrary inputs,
which is essential when expressions are assembled randomly during evolution.
"""

from __future__ import annotations

import math
import operator
from typing import Callable, NamedTuple

_EPS = 1e-10


def protected_div(a: float, b: float) -> float:
    """Division that returns ``1.0`` when the denominator is ~0."""
    return a / b if abs(b) > _EPS else 1.0


def protected_log(a: float) -> float:
    """Natural log of ``|a|``; returns ``0.0`` when ``a`` is ~0."""
    return math.log(abs(a)) if abs(a) > _EPS else 0.0


def protected_sqrt(a: float) -> float:
    """Square root of ``|a|`` (always defined)."""
    return math.sqrt(abs(a))


def square(a: float) -> float:
    """``a ** 2``."""
    return a * a


def cube(a: float) -> float:
    """``a ** 3``."""
    return a * a * a


def sigmoid(a: float) -> float:
    """Numerically stable logistic sigmoid."""
    if a >= 0:
        z = math.exp(-a)
        return 1.0 / (1.0 + z)
    z = math.exp(a)
    return z / (1.0 + z)


def relu(a: float) -> float:
    """Rectified linear unit: ``max(0, a)``."""
    return a if a > 0.0 else 0.0


def abs_val(a: float) -> float:
    """Absolute value."""
    return abs(a)


class Primitive(NamedTuple):
    """A GP primitive: a callable, its arity and the symbol shown in expressions."""

    func: Callable[..., float]
    arity: int
    name: str


# Named primitive sets, from conservative to expressive.
PRIMITIVE_SETS: dict[str, list[Primitive]] = {
    "basic": [
        Primitive(operator.add, 2, "add"),
        Primitive(operator.sub, 2, "sub"),
        Primitive(operator.mul, 2, "mul"),
        Primitive(protected_div, 2, "div"),
        Primitive(operator.neg, 1, "neg"),
        Primitive(abs_val, 1, "abs"),
    ],
    "extended": [
        Primitive(operator.add, 2, "add"),
        Primitive(operator.sub, 2, "sub"),
        Primitive(operator.mul, 2, "mul"),
        Primitive(protected_div, 2, "div"),
        Primitive(protected_log, 1, "log"),
        Primitive(protected_sqrt, 1, "sqrt"),
        Primitive(square, 1, "sq"),
        Primitive(operator.neg, 1, "neg"),
        Primitive(abs_val, 1, "abs"),
    ],
    "full": [
        Primitive(operator.add, 2, "add"),
        Primitive(operator.sub, 2, "sub"),
        Primitive(operator.mul, 2, "mul"),
        Primitive(protected_div, 2, "div"),
        Primitive(protected_log, 1, "log"),
        Primitive(protected_sqrt, 1, "sqrt"),
        Primitive(square, 1, "sq"),
        Primitive(cube, 1, "cube"),
        Primitive(operator.neg, 1, "neg"),
        Primitive(abs_val, 1, "abs"),
        Primitive(sigmoid, 1, "sigmoid"),
        Primitive(relu, 1, "relu"),
    ],
    "nonlinear": [
        Primitive(operator.mul, 2, "mul"),
        Primitive(protected_div, 2, "div"),
        Primitive(protected_log, 1, "log"),
        Primitive(protected_sqrt, 1, "sqrt"),
        Primitive(square, 1, "sq"),
        Primitive(cube, 1, "cube"),
        Primitive(sigmoid, 1, "sigmoid"),
        Primitive(relu, 1, "relu"),
        Primitive(operator.neg, 1, "neg"),
    ],
}

VALID_FUNCTION_SETS = tuple(PRIMITIVE_SETS)
