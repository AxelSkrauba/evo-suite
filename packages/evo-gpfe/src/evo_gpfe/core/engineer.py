"""The :class:`GPFeatureEngineer` estimator."""

from __future__ import annotations

import copy
import logging
import operator
import random
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd
from deap import base, gp, tools
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from evo_gpfe.algorithms.evolve import evolve_one_feature
from evo_gpfe.core.config import GeneratedFeature, GPConfig, GPEngineeringResult
from evo_gpfe.core.evaluator import GPFitnessEvaluator
from evo_gpfe.primitives.pset import build_pset
from evo_gpfe.utils.deap_utils import create_types, new_uid
from evo_gpfe.utils.validation import infer_task_type, prepare_target, resolve_scoring

logger = logging.getLogger("evo_gpfe")

ArrayLike = Union[np.ndarray, pd.DataFrame]


class GPFeatureEngineer(TransformerMixin, BaseEstimator):
    """Construct new features from existing ones via genetic programming.

    Evolves symbolic expression trees that map the original features to new,
    informative ones. A sequential hall-of-fame strategy rewards relevance to
    the target while penalising redundancy with already-generated features and
    tree complexity, yielding diverse, interpretable expressions such as
    ``log(x2) * (x5 + x1^2)``.

    Parameters
    ----------
    config : GPConfig, optional
        Genetic-programming configuration. If ``None``, defaults are used.
    scoring : str, optional
        scikit-learn scoring string for the baseline/augmented evaluation and
        the ``model_score`` fitness metric. Defaults by task type.
    estimator : sklearn estimator, optional
        Downstream model used to estimate the improvement and (when
        ``fitness_metric='model_score'``) to score candidates. Defaults to a
        linear model for the task.

    Attributes
    ----------
    result_ : GPEngineeringResult
        Full result of the run (set after :meth:`fit`).
    trees_ : list
        The selected expression trees.
    n_features_in_ : int
        Number of input features seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Input feature names (only when ``X`` is a DataFrame).

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from evo_gpfe import GPFeatureEngineer, GPConfig
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> eng = GPFeatureEngineer(
    ...     config=GPConfig(population_size=60, n_generations=10,
    ...                     n_features_to_generate=3, verbose=False)
    ... )
    >>> X_aug = eng.fit_transform(X, y)            # doctest: +SKIP
    """

    def __init__(
        self,
        config: GPConfig | None = None,
        scoring: str | None = None,
        estimator: BaseEstimator | None = None,
    ) -> None:
        self.config = config
        self.scoring = scoring
        self.estimator = estimator

    # ── scikit-learn API ──────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, y: ArrayLike) -> GPFeatureEngineer:
        """Evolve the engineered features.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (required: GP optimises relevance to ``y``).

        Returns
        -------
        self : GPFeatureEngineer
        """
        config = self.config if self.config is not None else GPConfig()
        config.validate()

        X_arr = validate_data(self, X, reset=True, dtype="numeric")
        y_arr = prepare_target(y)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")

        n_features = X_arr.shape[1]
        feature_names = self._resolve_names(n_features)
        task_type = infer_task_type(y_arr, config.task_type)
        scoring = resolve_scoring(self.scoring, task_type)

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        estimator = (
            self.estimator if self.estimator is not None else self._default_estimator(task_type)
        )
        cv = self._build_cv(y_arr, task_type, config)
        baseline_cv = self._cv_score(estimator, X_arr, y_arr, scoring, cv)

        uid = new_uid()
        pset = build_pset(
            n_features=n_features,
            function_set=config.function_set,
            uid=uid,
            feature_names=feature_names,
            use_constants=config.use_constants,
            constant_range=config.constant_range,
        )
        deap_types = create_types(uid)
        start = time.time()
        try:
            evaluator = GPFitnessEvaluator(
                X=X_arr,
                y=y_arr,
                pset=pset,
                config=config,
                task_type=task_type,
                estimator=clone(estimator) if config.fitness_metric == "model_score" else None,
                scoring=scoring,
            )
            toolbox = self._build_toolbox(deap_types.individual_cls, pset, config, evaluator)
            generated, trees, scalers = self._evolve_features(toolbox, evaluator, config, X_arr)
        finally:
            deap_types.cleanup()

        total_time = time.time() - start

        X_aug = self._augment(X_arr, generated, config)
        augmented_cv = self._cv_score(estimator, X_aug, y_arr, scoring, cv)

        self.trees_ = trees
        self.scalers_ = scalers
        self.feature_names_generated_ = [f"gp_{i}" for i in range(len(trees))]
        self._evaluator = evaluator
        self._uid = uid

        self.result_ = GPEngineeringResult(
            generated_features=generated,
            X_original_shape=(X_arr.shape[0], X_arr.shape[1]),
            X_transformed_shape=(X_aug.shape[0], X_aug.shape[1]),
            baseline_cv_score=baseline_cv,
            augmented_cv_score=augmented_cv,
            score_improvement=augmented_cv - baseline_cv,
            total_time=total_time,
            n_evaluations=evaluator.eval_count,
            task_type=task_type,
            scoring=scoring,
            feature_names_original=feature_names,
            feature_names_generated=list(self.feature_names_generated_),
            config=config,
        )

        if config.verbose:
            print(self.result_.summary())
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply the learned trees, returning engineered (and optionally original) features."""
        check_is_fitted(self, "trees_")
        X_arr = validate_data(self, X, reset=False, dtype="numeric")

        columns = []
        for tree, scaler in zip(self.trees_, self.scalers_):
            vals = self._evaluator.compute_feature_values(tree, X_arr)
            if scaler is not None:
                vals = scaler.transform(vals.reshape(-1, 1)).ravel()
            columns.append(vals)

        X_gp = np.column_stack(columns) if columns else np.empty((X_arr.shape[0], 0))
        if self._augment_original():
            return np.hstack([X_arr, X_gp])
        return X_gp

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names (sklearn contract)."""
        check_is_fitted(self, "trees_")
        generated = list(self.feature_names_generated_)
        if not self._augment_original():
            return np.asarray(generated, dtype=object)

        if input_features is not None:
            original = list(input_features)
        elif hasattr(self, "feature_names_in_"):
            original = list(self.feature_names_in_)
        else:
            original = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(original + generated, dtype=object)

    def summary(self) -> str:
        """Return a human-readable summary of the fitted result."""
        check_is_fitted(self, "result_")
        return self.result_.summary()

    # ── internals ─────────────────────────────────────────────────────────────

    def _augment_original(self) -> bool:
        config = self.config if self.config is not None else GPConfig()
        return config.augment_original

    def _evolve_features(self, toolbox, evaluator, config, X_arr):
        generated: list[GeneratedFeature] = []
        trees: list = []
        scalers: list = []
        population = None

        for feat_idx in range(config.n_features_to_generate):
            best_tree, gen_found, population = evolve_one_feature(
                toolbox, config, population if config.warm_start_population else None
            )
            relevance, redundancy = evaluator.relevance_and_redundancy(best_tree)

            # Reject near-duplicate features; try the next best candidates.
            if redundancy >= config.redundancy_threshold and feat_idx > 0:
                for candidate in tools.selBest(population, min(5, len(population)))[1:]:
                    rel_c, red_c = evaluator.relevance_and_redundancy(candidate)
                    if red_c < config.redundancy_threshold:
                        best_tree, relevance, redundancy = candidate, rel_c, red_c
                        break

            evaluator.add_to_hof(best_tree)
            trees.append(best_tree)

            vals = evaluator.compute_feature_values(best_tree, X_arr)
            scaler = None
            if config.normalize_output:
                scaler = StandardScaler()
                vals = scaler.fit_transform(vals.reshape(-1, 1)).ravel()
            scalers.append(scaler)

            generated.append(
                GeneratedFeature(
                    expression=str(best_tree),
                    fitness=float(best_tree.fitness.values[0]),
                    mi_score=relevance,
                    redundancy_score=redundancy,
                    n_nodes=len(best_tree),
                    height=best_tree.height,
                    feature_index=feat_idx,
                    generation_found=gen_found,
                    values=vals.copy(),
                )
            )
        return generated, trees, scalers

    def _build_toolbox(self, individual_cls, pset, config, evaluator) -> base.Toolbox:
        toolbox = base.Toolbox()
        toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=pset,
            min_=config.init_depth_min,
            max_=config.init_depth_max,
        )
        toolbox.register("individual", tools.initIterate, individual_cls, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("clone", copy.deepcopy)
        toolbox.register("evaluate", evaluator)
        toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate_subtree", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.register("mutate_hoist", gp.mutShrink)
        toolbox.register("mutate_point", gp.mutNodeReplacement, pset=pset)

        height_limit = gp.staticLimit(operator.attrgetter("height"), config.max_tree_height)
        toolbox.decorate("mate", height_limit)
        toolbox.decorate("mutate_subtree", height_limit)
        return toolbox

    def _augment(self, X_arr, generated, config) -> np.ndarray:
        if not generated:
            return X_arr
        block = np.column_stack([gf.values for gf in generated])
        if config.augment_original:
            return np.hstack([X_arr, block])
        return block

    def _resolve_names(self, n_features: int) -> list[str]:
        if getattr(self, "feature_names_in_", None) is not None:
            return [str(name) for name in self.feature_names_in_]
        return [f"x{i}" for i in range(n_features)]

    def _build_cv(self, y, task_type, config):
        if task_type == "classification":
            _, counts = np.unique(y, return_counts=True)
            n_splits = max(2, min(5, int(counts.min())))
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
        n_splits = max(2, min(5, y.shape[0]))
        return KFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)

    @staticmethod
    def _cv_score(estimator, X, y, scoring, cv) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                clone(estimator), X, y, scoring=scoring, cv=cv, error_score=0.0
            )
        return float(np.mean(scores))

    @staticmethod
    def _default_estimator(task_type: str):
        if task_type == "classification":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=300)
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)

    # ── estimator tags (scikit-learn >= 1.6) ──────────────────────────────────

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags
