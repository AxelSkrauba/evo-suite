"""The :class:`EvoEnsembleClassifier` and :class:`EvoEnsembleRegressor` estimators.

Both share a common core (:class:`_BaseEvoEnsemble`) that pre-computes
out-of-fold predictions, runs the Genetic Algorithm and trains the selected
members on the full dataset. They are split into task-specific classes (the
same pattern as scikit-learn's own ``VotingClassifier``/``VotingRegressor``
and ``StackingClassifier``/``StackingRegressor``) so that ``predict``,
scikit-learn tags and ``check_estimator`` behave consistently with a fixed
estimator type, instead of dispatching on a runtime ``task_type='auto'``.
"""

from __future__ import annotations

import random
import time
import warnings

import numpy as np
import pandas as pd
from deap import tools
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, column_or_1d, validate_data

from evo_ens.algorithms.evolve import build_toolbox, run_nsga2, run_single
from evo_ens.core.candidates import default_candidates, estimator_display_name
from evo_ens.core.config import EnsembleMember, EvoEnsConfig, EvoEnsResult, ParetoSolution
from evo_ens.core.evaluator import EnsembleFitnessEvaluator, compute_score
from evo_ens.utils.deap_utils import create_types, new_uid
from evo_ens.utils.diversity import diversity_pearson_mean, diversity_q_mean
from evo_ens.utils.validation import build_cv_splitter, resolve_scoring


class _BaseEvoEnsemble(MetaEstimatorMixin, BaseEstimator):
    """Shared fit/evolve/extract logic for the classifier and regressor."""

    _task_type: str  # set by subclasses

    def __init__(
        self,
        candidate_estimators: list[BaseEstimator] | None = None,
        config: EvoEnsConfig | None = None,
        scoring: str | None = None,
    ) -> None:
        self.candidate_estimators = candidate_estimators
        self.config = config
        self.scoring = scoring

    # ── shared fit machinery ────────────────────────────────────────────────

    def _fit(self, X, y) -> _BaseEvoEnsemble:
        config = self.config if self.config is not None else EvoEnsConfig()
        config.validate()

        X_arr = validate_data(self, X, reset=True, dtype="numeric")
        y_arr = self._prepare_target(y)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}.")

        task_type = self._task_type
        scoring = resolve_scoring(config.scoring or self.scoring, task_type)

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        candidates = self.candidate_estimators or default_candidates(
            task_type, config.random_seed or 42
        )
        candidates = [clone(c) for c in candidates]
        n_candidates = len(candidates)
        candidate_names = [estimator_display_name(c, i) for i, c in enumerate(candidates)]

        if config.verbose:
            print(
                f"EvoEnsemble | {n_candidates} candidates | task={task_type} | "
                f"scoring={scoring} | mode={config.mode}"
            )

        t_pre = time.time()
        oof_preds, oof_scores = self._precompute_oof(
            candidates, X_arr, y_arr, task_type, scoring, config
        )
        precompute_time = time.time() - t_pre
        best_single_score = max(oof_scores)

        uid = new_uid()
        deap_types = create_types(uid, mode=config.mode)
        t_evo = time.time()
        try:
            evaluator = EnsembleFitnessEvaluator(
                oof_predictions=oof_preds,
                y=y_arr,
                config=config,
                task_type=task_type,
                scoring=scoring,
                n_candidates=n_candidates,
            )
            toolbox = build_toolbox(deap_types, n_candidates, evaluator, config)
            if config.mode == "multiobjective":
                population, history = run_nsga2(toolbox, n_candidates, config)
            else:
                population, history = run_single(toolbox, n_candidates, config)
        finally:
            deap_types.cleanup()
        evolution_time = time.time() - t_evo

        result = self._extract_result(
            population=population,
            history=history,
            evaluator=evaluator,
            candidate_names=candidate_names,
            oof_preds=oof_preds,
            oof_scores=oof_scores,
            y_arr=y_arr,
            task_type=task_type,
            scoring=scoring,
            best_single_score=best_single_score,
            n_candidates=n_candidates,
            config=config,
            total_time=precompute_time + evolution_time,
            precompute_time=precompute_time,
        )
        self.result_ = result

        if config.verbose:
            print("Fitting final ensemble members on the full dataset...")
        self.estimators_ = []
        for member in result.members:
            est = clone(candidates[member.candidate_index])
            est.fit(X_arr, y_arr)
            self.estimators_.append(est)

        if config.verbose:
            print(result.summary())

        self._is_fitted = True
        return self

    def _prepare_target(self, y) -> np.ndarray:
        return np.asarray(y)

    def _precompute_oof(self, candidates, X, y, task_type, scoring, config):
        cv = build_cv_splitter(y, task_type, config.cv_folds, config.random_seed)
        predict_method = "predict_proba" if task_type == "classification" else "predict"

        oof_preds: list[np.ndarray] = []
        oof_scores: list[float] = []
        for i, est in enumerate(candidates):
            name = estimator_display_name(est, i)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    oof = cross_val_predict(est, X, y, cv=cv, method=predict_method)
                score = compute_score(oof, y, scoring, task_type)
            except (ValueError, ArithmeticError) as exc:  # pragma: no cover - defensive
                warnings.warn(f"OOF prediction failed for {name}: {exc}", stacklevel=2)
                if task_type == "classification":
                    n_classes = len(np.unique(y))
                    oof = np.ones((len(y), n_classes)) / n_classes
                else:
                    oof = np.zeros(len(y))
                score = 0.0
            oof_preds.append(oof)
            oof_scores.append(float(score))
            if config.verbose:
                print(f"  [{i + 1:2d}/{len(candidates)}] {name:40s} -> {scoring}={score:.4f}")

        return oof_preds, oof_scores

    def _extract_result(
        self,
        population,
        history,
        evaluator: EnsembleFitnessEvaluator,
        candidate_names,
        oof_preds,
        oof_scores,
        y_arr,
        task_type,
        scoring,
        best_single_score,
        n_candidates,
        config,
        total_time,
        precompute_time,
    ) -> EvoEnsResult:
        pareto_solutions: list[ParetoSolution] | None = None
        if config.mode == "multiobjective":
            pareto_inds = tools.sortNondominated(
                population, len(population), first_front_only=True
            )[0]
            best_ind = max(pareto_inds, key=lambda ind: ind.fitness.values[0])
            pareto_solutions = []
            for ind in sorted(pareto_inds, key=lambda ind: -ind.fitness.values[0]):
                members_p, weights_p = self._members_from_individual(
                    ind, evaluator, candidate_names, oof_scores
                )
                pareto_solutions.append(
                    ParetoSolution(
                        score=ind.fitness.values[0],
                        compression=ind.fitness.values[1],
                        n_models=sum(1 for b in ind[:n_candidates] if b > 0.5),
                        members=members_p,
                        weights=weights_p,
                    )
                )
        else:
            best_ind = tools.selBest(population, 1)[0]

        members, weights = self._members_from_individual(
            best_ind, evaluator, candidate_names, oof_scores
        )
        active_idx = [m.candidate_index for m in members]
        combined = evaluator.combine(active_idx, weights)
        oof_score_ens = evaluator.score(combined)

        if task_type == "classification":
            preds_div = [oof_preds[i].argmax(1) for i in active_idx]
            diversity = diversity_q_mean(y_arr, preds_div)
        else:
            preds_div = [oof_preds[i] for i in active_idx]
            diversity = diversity_pearson_mean(preds_div)

        return EvoEnsResult(
            members=members,
            n_models=len(members),
            weights=weights,
            oof_score_ensemble=oof_score_ens,
            oof_score_best_single=best_single_score,
            oof_score_improvement=oof_score_ens - best_single_score,
            diversity_score=diversity,
            compression_ratio=1.0 - len(members) / n_candidates,
            scoring=scoring,
            task_type=task_type,
            n_candidates=n_candidates,
            history=history,
            total_time=total_time,
            precompute_time=precompute_time,
            n_evaluations=evaluator.eval_count,
            config=config,
            pareto_front=pareto_solutions,
        )

    @staticmethod
    def _members_from_individual(individual, evaluator, candidate_names, oof_scores):
        active_idx, weights = evaluator.decode_individual(individual, oof_scores)
        members = [
            EnsembleMember(
                name=candidate_names[i],
                estimator=None,
                weight=float(weights[j]),
                candidate_index=i,
                oof_score=oof_scores[i],
            )
            for j, i in enumerate(active_idx)
        ]
        return members, weights

    def get_ensemble_info(self) -> pd.DataFrame:
        """Return a DataFrame describing each selected ensemble member."""
        check_is_fitted(self, "_is_fitted")
        rows = [
            {
                "index": m.candidate_index,
                "name": m.name,
                "weight": round(m.weight, 4),
                "oof_score": round(m.oof_score, 4),
            }
            for m in self.result_.members
        ]
        return pd.DataFrame(rows).sort_values("weight", ascending=False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags


class EvoEnsembleClassifier(ClassifierMixin, _BaseEvoEnsemble):
    """Evolutionary ensemble builder for classification.

    Optimizes, via a Genetic Algorithm, which candidate models to include in
    a weighted-voting ensemble and how much weight to give each, trading off
    predictive score against prediction diversity (Yule's Q-statistic)
    between members.

    Parameters
    ----------
    candidate_estimators : list of sklearn classifiers, optional
        Candidate pool. If ``None``, :func:`evo_ens.core.candidates.default_candidates`
        is used.
    config : EvoEnsConfig, optional
        Genetic Algorithm configuration. If ``None``, defaults are used.
    scoring : str, optional
        Evaluation metric (``'accuracy'``, ``'f1_macro'``, ``'roc_auc'``).
        Defaults to ``'accuracy'``.

    Attributes
    ----------
    result_ : EvoEnsResult
        Full result of the evolutionary run (set after :meth:`fit`).
    estimators_ : list of fitted sklearn classifiers
        The selected ensemble members, fitted on the full training data.
    classes_ : numpy.ndarray
        Class labels seen during :meth:`fit`.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from evo_ens import EvoEnsConfig, EvoEnsembleClassifier
    >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    >>> clf = EvoEnsembleClassifier(
    ...     config=EvoEnsConfig(population_size=30, n_generations=10, verbose=False)
    ... )
    >>> clf.fit(X, y).predict(X)  # doctest: +SKIP
    """

    _task_type = "classification"

    def fit(self, X, y) -> EvoEnsembleClassifier:
        """Run the Genetic Algorithm and fit the selected ensemble members."""
        self._fit(X, y)
        return self

    def _prepare_target(self, y) -> np.ndarray:
        y_arr = column_or_1d(y, warn=True)
        self.classes_ = unique_labels(y_arr)
        if y_arr.dtype == object or np.issubdtype(y_arr.dtype, np.str_):
            self._label_encoder = LabelEncoder()
            return self._label_encoder.fit_transform(y_arr).astype(float)
        self._label_encoder = None
        return y_arr.astype(float)

    def predict_proba(self, X) -> np.ndarray:
        """Return weighted-averaged class probabilities."""
        check_is_fitted(self, "_is_fitted")
        X_arr = validate_data(self, X, reset=False, dtype="numeric")
        weights = self.result_.weights
        combined = weights[0] * self.estimators_[0].predict_proba(X_arr)
        for j in range(1, len(self.estimators_)):
            combined = combined + weights[j] * self.estimators_[j].predict_proba(X_arr)
        return combined

    def predict(self, X) -> np.ndarray:
        """Return the ensemble's predicted class labels."""
        proba = self.predict_proba(X)
        indices = proba.argmax(axis=1)
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(indices)
        return self.classes_[indices]


class EvoEnsembleRegressor(RegressorMixin, _BaseEvoEnsemble):
    """Evolutionary ensemble builder for regression.

    Optimizes, via a Genetic Algorithm, which candidate models to include in
    a weighted-average ensemble and how much weight to give each, trading
    off predictive score against prediction diversity (absolute Pearson
    correlation) between members.

    Parameters
    ----------
    candidate_estimators : list of sklearn regressors, optional
        Candidate pool. If ``None``, :func:`evo_ens.core.candidates.default_candidates`
        is used.
    config : EvoEnsConfig, optional
        Genetic Algorithm configuration. If ``None``, defaults are used.
    scoring : str, optional
        Evaluation metric (``'r2'``, ``'neg_rmse'``). Defaults to ``'r2'``.

    Attributes
    ----------
    result_ : EvoEnsResult
        Full result of the evolutionary run (set after :meth:`fit`).
    estimators_ : list of fitted sklearn regressors
        The selected ensemble members, fitted on the full training data.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from evo_ens import EvoEnsConfig, EvoEnsembleRegressor
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> reg = EvoEnsembleRegressor(
    ...     config=EvoEnsConfig(population_size=30, n_generations=10, verbose=False)
    ... )
    >>> reg.fit(X, y).predict(X)  # doctest: +SKIP
    """

    _task_type = "regression"

    def fit(self, X, y) -> EvoEnsembleRegressor:
        """Run the Genetic Algorithm and fit the selected ensemble members."""
        self._fit(X, y)
        return self

    def _prepare_target(self, y) -> np.ndarray:
        return column_or_1d(y, warn=True).astype(float)

    def predict(self, X) -> np.ndarray:
        """Return the ensemble's weighted-average prediction."""
        check_is_fitted(self, "_is_fitted")
        X_arr = validate_data(self, X, reset=False, dtype="numeric")
        weights = self.result_.weights
        combined = weights[0] * self.estimators_[0].predict(X_arr)
        for j in range(1, len(self.estimators_)):
            combined = combined + weights[j] * self.estimators_[j].predict(X_arr)
        return combined
