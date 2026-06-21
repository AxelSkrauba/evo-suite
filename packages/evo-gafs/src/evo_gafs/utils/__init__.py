"""Utilities: safe DEAP type management and input validation."""

from evo_gafs.utils.deap_utils import DeapTypes, create_types
from evo_gafs.utils.validation import infer_task_type, prepare_target, resolve_scoring

__all__ = [
    "DeapTypes",
    "create_types",
    "infer_task_type",
    "prepare_target",
    "resolve_scoring",
]
