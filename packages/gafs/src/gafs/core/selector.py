"""The :class:`GAFeatureSelector` estimator."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import replace
from typing import Callable, Union

import numpy as np
import pandas as pd
from deap import base, tools
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted, validate_data

from gafs.algorithms.nsga2 import run_nsga2
from gafs.algorithms.single import run_single_objective
from gafs.core.config import EvolutionStats, GAConfig, SelectionResult
from gafs.core.evaluator import FitnessEvaluator
from gafs.operators.crossover import cx_uniform_with_repair
from gafs.operators.mutation import mut_flip_with_repair
from gafs.operators.repair import init_individual
from gafs.utils.deap_utils import create_types
from gafs.utils.validation import infer_task_type, prepare_target, resolve_scoring

logger = logging.getLogger("gafs")

ArrayLike = Union[np.ndarray, pd.DataFrame]
Callback = Callable[[int, EvolutionStats, list], bool]


class GAFeatureSelector(SelectorMixin, BaseEstimator):
    """Genetic-algorithm wrapper feature selector, compatible with scikit-learn.

    The selector searches for the subset of features that maximises a
    cross-validated score of ``estimator`` (the wrapper criterion), optionally
    trading raw performance for a smaller feature set.

    Parameters
    ----------
    estimator : sklearn estimator
        Model used to score candidate feature subsets. Must implement ``fit``
        and ``predict``. Fast estimators (decision trees, linear models) keep
        the search affordable. It is cloned for every evaluation, never fitted
        in place.
    config : GAConfig, optional
        Genetic-algorithm configuration. If ``None``, defaults are used.
    scoring : str, optional
        scikit-learn scoring string (e.g. ``'accuracy'``, ``'f1_macro'``,
        ``'r2'``, ``'neg_mean_squared_error'``). If ``None`` it is chosen from
        ``task_type``.
    task_type : {'auto', 'classification', 'regression'}, default='auto'
        Problem type. ``'auto'`` infers it from ``y``.
    feature_names : list of str, optional
        Names of the input features. Inferred from a DataFrame's columns or
        generated as ``f0, f1, ...`` when not given.

    Attributes
    ----------
    result_ : SelectionResult
        Full result of the run (set after :meth:`fit`).
    support_ : numpy.ndarray of bool
        Boolean mask of selected features.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Names of features seen during :meth:`fit` (only when ``X`` is a
        DataFrame).

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gafs import GAFeatureSelector, GAConfig
    >>> config = GAConfig(population_size=20, n_generations=10, verbose=False)
    >>> selector = GAFeatureSelector(
    ...     estimator=DecisionTreeClassifier(random_state=42),
    ...     config=config,
    ... )
    >>> selector.fit(X_train, y_train)             # doctest: +SKIP
    >>> X_selected = selector.transform(X_test)    # doctest: +SKIP
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        config: GAConfig | None = None,
        scoring: str | None = None,
        task_type: str = "auto",
        feature_names: list[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.config = config
        self.scoring = scoring
        self.task_type = task_type
        self.feature_names = feature_names

    # ── scikit-learn API ──────────────────────────────────────────────────────

    def fit(
        self, X: ArrayLike, y: ArrayLike, callbacks: list[Callback] | None = None
    ) -> GAFeatureSelector:
        """Run the genetic algorithm to find the best feature subset.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training data. Sparse matrices are not supported.
        y : array-like of shape (n_samples,)
            Target values. Integers/strings for classification, floats for
            regression.
        callbacks : list of callable, optional
            Functions ``f(gen, stats, population) -> bool``. Returning ``True``
            stops evolution early.

        Returns
        -------
        self : GAFeatureSelector
            The fitted selector.
        """
        config = self.config if self.config is not None else GAConfig()
        config.validate()

        # validate_data rejects complex/NaN input and records n_features_in_
        # and (for DataFrames) feature_names_in_.
        X_array = validate_data(self, X, reset=True, dtype="numeric")
        y_array = prepare_target(y)
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError(f"X has {X_array.shape[0]} samples but y has {y_array.shape[0]}.")

        n_features = X_array.shape[1]
        feature_names = self._resolve_names(n_features)
        task_type = infer_task_type(y_array, self.task_type)
        scoring = resolve_scoring(self.scoring, task_type)
        cv = self._build_cv(y_array, task_type, config)

        # Resolve mutation_indpb without mutating the user's config object.
        effective_config = replace(
            config,
            mutation_indpb=(
                config.mutation_indpb if config.mutation_indpb is not None else 1.0 / n_features
            ),
        )

        logger.info(
            "GA feature selection | n_features=%d | mode=%s | scoring=%s | task=%s",
            n_features,
            effective_config.mode,
            scoring,
            task_type,
        )

        if effective_config.random_seed is not None:
            random.seed(effective_config.random_seed)
            np.random.seed(effective_config.random_seed)

        evaluator = FitnessEvaluator(
            estimator=self.estimator,
            X=X_array,
            y=y_array,
            scoring=scoring,
            cv=cv,
            config=effective_config,
        )

        deap_types = create_types(effective_config.mode)
        try:
            toolbox = self._build_toolbox(
                n_features, effective_config, deap_types.individual_cls, evaluator
            )
            start = time.time()
            if effective_config.mode == "multiobjective":
                population, _, history = run_nsga2(toolbox, effective_config, callbacks)
            else:
                population, _, history = run_single_objective(toolbox, effective_config, callbacks)
            total_time = time.time() - start

            result = self._extract_result(
                population=population,
                history=history,
                feature_names=feature_names,
                n_features=n_features,
                evaluator=evaluator,
                config=effective_config,
                total_time=total_time,
            )
        finally:
            deap_types.cleanup()

        self.result_ = result
        self.support_ = result.selected_mask

        if effective_config.verbose:
            print(result.summary())
        return self

    def _get_support_mask(self) -> np.ndarray:
        """Return the boolean mask of selected features (sklearn contract)."""
        check_is_fitted(self, "support_")
        return self.support_

    # ── Convenience ─────────────────────────────────────────────────────────--

    def summary(self) -> str:
        """Return a human-readable summary of the fitted result."""
        check_is_fitted(self, "result_")
        return self.result_.summary()

    # ── Internals ───────────────────────────────────────────────────────────--

    def _resolve_names(self, n_features: int) -> list[str]:
        """Resolve feature names for reporting (DataFrame > explicit > generated)."""
        if getattr(self, "feature_names_in_", None) is not None:
            return [str(name) for name in self.feature_names_in_]
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise ValueError(
                    f"feature_names has length {len(self.feature_names)} but X has "
                    f"{n_features} columns."
                )
            return list(self.feature_names)
        return [f"f{i}" for i in range(n_features)]

    def _build_cv(self, y: np.ndarray, task_type: str, config: GAConfig) -> object:
        """Build a CV splitter, clamping folds to the smallest class if needed."""
        if task_type == "classification":
            _, counts = np.unique(y, return_counts=True)
            n_splits = max(2, min(config.cv_folds, int(counts.min())))
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
        n_splits = max(2, min(config.cv_folds, y.shape[0]))
        return KFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)

    def _build_toolbox(
        self,
        n_features: int,
        config: GAConfig,
        individual_cls: type,
        evaluator: FitnessEvaluator,
    ) -> base.Toolbox:
        """Configure the DEAP toolbox with all genetic operators."""
        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            init_individual,
            ind_class=individual_cls,
            n_features=n_features,
            min_features=config.min_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluator)
        toolbox.register(
            "mate", cx_uniform_with_repair, indpb=0.5, min_features=config.min_features
        )
        toolbox.register(
            "mutate",
            mut_flip_with_repair,
            indpb=config.mutation_indpb,
            min_features=config.min_features,
        )
        if config.mode == "multiobjective":
            toolbox.register("select", tools.selNSGA2)
        else:
            toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
        return toolbox

    def _extract_result(
        self,
        population: list,
        history: list[EvolutionStats],
        feature_names: list[str],
        n_features: int,
        evaluator: FitnessEvaluator,
        config: GAConfig,
        total_time: float,
    ) -> SelectionResult:
        """Extract the best individual and assemble a :class:`SelectionResult`."""
        pareto_data: list[dict] | None = None

        if config.mode == "multiobjective":
            front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            best_ind = max(front, key=lambda ind: ind.fitness.values[0])
            best_cv_score = float(best_ind.fitness.values[0])
            best_fitness = best_cv_score
            pareto_data = [
                {
                    "mask": list(ind),
                    "cv_score": float(ind.fitness.values[0]),
                    "compression": float(ind.fitness.values[1]),
                    "n_features": int(sum(ind)),
                }
                for ind in front
            ]
        else:
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = float(best_ind.fitness.values[0])
            selected = [i for i, bit in enumerate(best_ind) if bit == 1]
            best_cv_score = evaluator.cv_score(selected)

        mask = np.array([bool(bit) for bit in best_ind])
        indices = np.where(mask)[0]
        n_selected = int(mask.sum())

        return SelectionResult(
            selected_mask=mask,
            selected_indices=indices,
            selected_feature_names=[feature_names[i] for i in indices],
            best_fitness=best_fitness,
            best_cv_score=best_cv_score,
            n_selected=n_selected,
            compression_ratio=1.0 - (n_selected / n_features),
            history=history,
            pareto_front=pareto_data,
            config=config,
            total_time=total_time,
            n_evaluations=evaluator.eval_count,
        )

    # ── Estimator tags (scikit-learn >= 1.6) ──────────────────────────────────

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags
