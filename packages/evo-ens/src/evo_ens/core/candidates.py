"""Default candidate pools used when the user does not supply their own."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

VALID_TASK_TYPES = ("classification", "regression")


def default_candidates(task_type: str, random_seed: int = 42) -> list[BaseEstimator]:
    """Return a structurally diverse pool of default candidate models.

    Diversity in the *candidate pool* (tree-based, linear, kernel, instance-
    based) favors low Q-statistics / correlations naturally, giving the
    Genetic Algorithm more room to find a decorrelated ensemble.

    Parameters
    ----------
    task_type : {'classification', 'regression'}
    random_seed : int, default=42
        Base seed; candidates that accept ``random_state`` use ``rs`` or
        ``rs + 1`` so paired configurations are not perfectly correlated.

    Returns
    -------
    list of sklearn estimators
    """
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got {task_type!r}")

    rs = random_seed
    if task_type == "classification":
        return [
            RandomForestClassifier(n_estimators=100, random_state=rs),
            RandomForestClassifier(
                n_estimators=100, max_features="sqrt", min_samples_leaf=4, random_state=rs + 1
            ),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=rs),
            GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=4, random_state=rs + 1
            ),
            ExtraTreesClassifier(n_estimators=100, random_state=rs),
            LogisticRegression(max_iter=500, C=1.0, random_state=rs),
            LogisticRegression(max_iter=500, C=0.1, random_state=rs + 1),
            SVC(kernel="rbf", probability=True, C=1.0, random_state=rs),
            SVC(kernel="linear", probability=True, C=1.0, random_state=rs),
            DecisionTreeClassifier(max_depth=5, random_state=rs),
            DecisionTreeClassifier(max_depth=10, random_state=rs + 1),
            KNeighborsClassifier(n_neighbors=5),
            KNeighborsClassifier(n_neighbors=15),
        ]
    return [
        RandomForestRegressor(n_estimators=100, random_state=rs),
        RandomForestRegressor(
            n_estimators=100, max_features=0.5, min_samples_leaf=4, random_state=rs + 1
        ),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=rs),
        GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=rs + 1
        ),
        ExtraTreesRegressor(n_estimators=100, random_state=rs),
        Ridge(alpha=1.0),
        Ridge(alpha=10.0),
        Lasso(alpha=0.1),
        ElasticNet(alpha=0.1, l1_ratio=0.5),
        SVR(kernel="rbf", C=1.0),
        DecisionTreeRegressor(max_depth=5, random_state=rs),
        KNeighborsRegressor(n_neighbors=5),
        KNeighborsRegressor(n_neighbors=15),
    ]


def estimator_display_name(est: BaseEstimator, idx: int) -> str:
    """Human-readable ``NN:ClassName(distinctive_param=value)`` label."""
    name = type(est).__name__
    params = est.get_params()
    if "n_estimators" in params:
        name += f"(n={params['n_estimators']})"
    elif "C" in params:
        name += f"(C={params['C']})"
    elif "alpha" in params:
        name += f"(alpha={params['alpha']})"
    elif "n_neighbors" in params:
        name += f"(k={params['n_neighbors']})"
    elif params.get("max_depth"):
        name += f"(d={params['max_depth']})"
    return f"{idx:02d}:{name}"
