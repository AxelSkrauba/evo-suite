"""Multi-dataset benchmarking utility for :class:`GAFeatureSelector`."""

from __future__ import annotations

import copy
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from evo_gafs.core.config import GAConfig
from evo_gafs.core.selector import GAFeatureSelector

ArrayLike = Union[np.ndarray, pd.DataFrame]


class BenchmarkRunner:
    """Run and compare GA feature selection across several datasets.

    For each registered dataset the runner records:

    * the model's cross-validated score using **all** features (baseline),
    * the score using the features **selected** by the GA,
    * the compression ratio, and
    * the wall-clock time.

    Examples
    --------
    >>> runner = BenchmarkRunner()                          # doctest: +SKIP
    >>> runner.add_dataset("Iris", X, y, task_type="classification")  # doctest: +SKIP
    >>> runner.run(DecisionTreeClassifier())                # doctest: +SKIP
    >>> runner.report()                                     # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._datasets: list[dict] = []
        self._results: list[dict] = []

    def add_dataset(
        self,
        name: str,
        X: ArrayLike,
        y: ArrayLike,
        task_type: str = "auto",
        description: str = "",
    ) -> BenchmarkRunner:
        """Register a dataset for the benchmark. Returns ``self`` for chaining."""
        self._datasets.append(
            {"name": name, "X": X, "y": y, "task_type": task_type, "description": description}
        )
        return self

    def run(
        self,
        estimator: BaseEstimator,
        config: GAConfig | None = None,
        scoring: str | None = None,
        verbose: bool = True,
        estimator_regression: BaseEstimator | None = None,
    ) -> list[dict]:
        """Run the benchmark over all registered datasets.

        Parameters
        ----------
        estimator : sklearn estimator
            Model for classification datasets (and for all datasets when
            ``estimator_regression`` is not given).
        config : GAConfig, optional
            Configuration applied to every run.
        scoring : str, optional
            Scoring string; auto-selected per task when ``None``.
        verbose : bool, default=True
            Print a per-dataset progress report.
        estimator_regression : sklearn estimator, optional
            Alternative model for regression datasets.

        Returns
        -------
        list of dict
            One result entry per dataset.
        """
        config = config or GAConfig(verbose=False)
        self._results = []

        for dataset in self._datasets:
            self._results.append(
                self._run_one(dataset, estimator, config, scoring, verbose, estimator_regression)
            )
        return self._results

    def _run_one(
        self,
        dataset: dict,
        estimator: BaseEstimator,
        config: GAConfig,
        scoring: str | None,
        verbose: bool,
        estimator_regression: BaseEstimator | None,
    ) -> dict:
        name, X, y = dataset["name"], dataset["X"], dataset["y"]
        X_array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_array = np.asarray(y)
        if y_array.dtype == object:
            y_array = LabelEncoder().fit_transform(y_array)

        task = dataset["task_type"]
        if task == "auto":
            unique = np.unique(y_array)
            task = (
                "classification"
                if len(unique) <= 20 and y_array.dtype.kind in "iub"
                else "regression"
            )

        est = clone(
            estimator_regression
            if task == "regression" and estimator_regression is not None
            else estimator
        )
        scoring_resolved = scoring or ("accuracy" if task == "classification" else "r2")

        if verbose:
            print(
                f"\n{'=' * 60}\n  Dataset: {name}\n  Shape: {X_array.shape}\n"
                f"  Task: {task}\n{'=' * 60}"
            )

        cv = self._build_cv(task, config, y_array)
        cv_score_full = self._score(est, X_array, y_array, scoring_resolved, cv, config)
        if verbose:
            print(f"  Baseline CV score (all features): {cv_score_full:.4f}")

        start = time.time()
        selector = GAFeatureSelector(
            estimator=clone(est),
            config=copy.deepcopy(config),
            scoring=scoring_resolved,
            task_type=task,
            feature_names=list(X.columns) if isinstance(X, pd.DataFrame) else None,
        )
        selector.fit(X, y)
        elapsed = time.time() - start
        result = selector.result_

        X_selected = X_array[:, result.selected_indices]
        cv_score_selected = self._score(est, X_selected, y_array, scoring_resolved, cv, config)

        if verbose:
            print(f"  GA CV score (selected features):  {cv_score_selected:.4f}")
            print(
                f"  Features: {X_array.shape[1]} -> {result.n_selected} "
                f"({result.compression_ratio:.1%} compression)"
            )
            print(f"  Time: {elapsed:.2f}s | Evaluations: {result.n_evaluations}")

        return {
            "name": name,
            "n_samples": X_array.shape[0],
            "n_features_original": X_array.shape[1],
            "n_features_selected": result.n_selected,
            "compression_ratio": result.compression_ratio,
            "cv_score_full": cv_score_full,
            "cv_score_selected": cv_score_selected,
            "score_delta": cv_score_selected - cv_score_full,
            "time_seconds": elapsed,
            "n_evaluations": result.n_evaluations,
            "scoring": scoring_resolved,
            "task_type": task,
            "selected_features": result.selected_feature_names,
            "result": result,
            "selector": selector,
        }

    @staticmethod
    def _build_cv(task: str, config: GAConfig, y: np.ndarray) -> object:
        if task == "classification":
            return StratifiedKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed
            )
        return KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    @staticmethod
    def _score(estimator, X, y, scoring, cv, config: GAConfig) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                clone(estimator), X, y, scoring=scoring, cv=cv, n_jobs=config.n_jobs
            )
        return float(np.mean(scores))

    def report(self) -> pd.DataFrame:
        """Return (and print) a summary :class:`pandas.DataFrame` of the runs."""
        if not self._results:
            print("No results. Call run() first.")
            return pd.DataFrame()

        rows = [
            {
                "Dataset": r["name"],
                "Samples": r["n_samples"],
                "Feats. original": r["n_features_original"],
                "Feats. selected": r["n_features_selected"],
                "Compression": f"{r['compression_ratio']:.1%}",
                "CV (all)": f"{r['cv_score_full']:.4f}",
                "CV (selected)": f"{r['cv_score_selected']:.4f}",
                "Delta": f"{r['score_delta']:+.4f}",
                "Time (s)": f"{r['time_seconds']:.1f}",
                "Evals": r["n_evaluations"],
                "Task": r["task_type"],
                "Scoring": r["scoring"],
            }
            for r in self._results
        ]
        df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("  BENCHMARK REPORT - GA Feature Selection")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        return df

    @property
    def results(self) -> list[dict]:
        """The list of result entries from the last :meth:`run`."""
        return self._results
