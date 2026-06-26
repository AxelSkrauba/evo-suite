"""Unit tests for DEAP type management and validation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from deap import gp

from evo_gpfe.primitives import build_pset
from evo_gpfe.utils import (
    create_types,
    infer_task_type,
    new_uid,
    prepare_target,
    remove_ephemeral_constant,
    resolve_scoring,
)


class TestDeapTypes:
    def test_create_and_cleanup(self):
        from deap import creator

        types = create_types(new_uid())
        assert hasattr(creator, types.individual_name)
        assert issubclass(types.individual_cls, gp.PrimitiveTree)
        assert types.individual_cls.fitness.weights == (1.0,)
        types.cleanup()
        assert not hasattr(creator, types.individual_name)
        types.cleanup()  # idempotent

    def test_new_uid_unique(self):
        assert new_uid() != new_uid()
        assert len(new_uid()) == 8

    def test_ephemeral_constant_cleanup_is_safe(self):
        # The ephemeral constant lives in pset.mapping; on DEAP versions that
        # also register it on the gp module, cleanup removes it. Either way the
        # call must be safe and idempotent, and distinct uids must not collide.
        uid_a, uid_b = new_uid(), new_uid()
        ps_a = build_pset(3, "basic", uid=uid_a, use_constants=True)
        build_pset(3, "basic", uid=uid_b, use_constants=True)
        assert f"rc_{uid_a}" in ps_a.mapping
        remove_ephemeral_constant(uid_a)
        remove_ephemeral_constant(uid_a)  # idempotent
        assert not hasattr(gp, f"rc_{uid_a}")


class TestValidation:
    def test_prepare_target_numeric(self):
        out = prepare_target([0, 1, 1, 0])
        assert out.dtype == float
        assert list(out) == [0.0, 1.0, 1.0, 0.0]

    def test_prepare_target_strings(self):
        out = prepare_target(np.array(["a", "b", "a"]))
        assert out.dtype == float
        assert len(np.unique(out)) == 2

    def test_infer_classification(self):
        assert infer_task_type(np.array([0, 1, 2, 1, 0])) == "classification"

    def test_infer_regression(self):
        assert infer_task_type(np.linspace(0, 1, 50)) == "regression"

    def test_infer_declared_passthrough(self):
        assert infer_task_type(np.array([0.5, 1.5]), "classification") == "classification"

    def test_infer_invalid_declared_raises(self):
        with pytest.raises(ValueError):
            infer_task_type(np.array([1.0]), "nope")

    def test_resolve_scoring(self):
        assert resolve_scoring(None, "classification") == "accuracy"
        assert resolve_scoring(None, "regression") == "r2"
        assert resolve_scoring("f1_macro", "classification") == "f1_macro"
