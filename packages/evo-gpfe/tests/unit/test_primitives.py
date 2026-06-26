"""Unit tests for protected primitives and primitive-set construction."""

from __future__ import annotations

import math
from uuid import uuid4

import pytest

from evo_gpfe.primitives import (
    PRIMITIVE_SETS,
    build_pset,
    cube,
    protected_div,
    protected_log,
    protected_sqrt,
    relu,
    sanitize_name,
    sigmoid,
    square,
)


class TestProtectedFunctions:
    def test_protected_div(self):
        assert protected_div(6.0, 3.0) == 2.0
        assert protected_div(1.0, 0.0) == 1.0  # protected
        assert protected_div(1.0, 1e-15) == 1.0

    def test_protected_log(self):
        assert protected_log(0.0) == 0.0
        assert protected_log(math.e) == pytest.approx(1.0)
        assert protected_log(-math.e) == pytest.approx(1.0)  # |a|

    def test_protected_sqrt(self):
        assert protected_sqrt(9.0) == 3.0
        assert protected_sqrt(-9.0) == 3.0  # |a|

    def test_square_cube(self):
        assert square(3.0) == 9.0
        assert cube(2.0) == 8.0

    def test_sigmoid_range_and_stability(self):
        assert sigmoid(0.0) == pytest.approx(0.5)
        assert 0.0 < sigmoid(1000.0) <= 1.0
        assert 0.0 <= sigmoid(-1000.0) < 1.0  # no overflow

    def test_relu(self):
        assert relu(-2.0) == 0.0
        assert relu(2.0) == 2.0


class TestPrimitiveSets:
    @pytest.mark.parametrize("name", ["basic", "extended", "full", "nonlinear"])
    def test_sets_are_well_formed(self, name):
        prims = PRIMITIVE_SETS[name]
        assert len(prims) > 0
        for prim in prims:
            assert prim.arity in (1, 2)
            assert callable(prim.func)
            assert isinstance(prim.name, str)


class TestSanitizeName:
    @pytest.mark.parametrize(
        ("raw", "idx", "expected"),
        [
            ("mean radius", 0, "mean_radius"),
            ("a.b-c", 1, "a_b_c"),
            ("1bad", 2, "x2"),
            ("", 3, "x3"),
        ],
    )
    def test_sanitize(self, raw, idx, expected):
        assert sanitize_name(raw, idx) == expected


class TestBuildPset:
    def test_basic_construction(self):
        pset = build_pset(4, "extended", uid=uuid4().hex[:8])
        # 4 terminals (arguments) registered
        assert len(pset.arguments) == 4

    def test_feature_names_are_used(self):
        pset = build_pset(2, "basic", uid=uuid4().hex[:8], feature_names=["mean radius", "x-1"])
        assert len(pset.arguments) == 2

    def test_invalid_function_set_raises(self):
        with pytest.raises(ValueError):
            build_pset(3, "does-not-exist", uid=uuid4().hex[:8])

    def test_without_constants(self):
        pset = build_pset(3, "basic", uid=uuid4().hex[:8], use_constants=False)
        assert len(pset.arguments) == 3
