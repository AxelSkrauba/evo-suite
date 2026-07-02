"""Multi-dataset benchmarking utility for :class:`GPFeatureEngineer`."""

from __future__ import annotations

import copy
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from evo_gpfe.core.config import GPConfig
from evo_gpfe.core.engineer import GPFeatureEngineer
from evo_gpfe.utils.validation import infer_task_type, prepare_target, resolve_scoring

ArrayLike = Union[np.ndarray, pd.DataFrame]


class GPBenchmarkRunner:
    """Run and compare GP feature engineering across several datasets.

    For each registered dataset the runner records:

    * the CV score with the **original** features (baseline),
    * the CV score with original **+ generated** features (augmented),
    * the CV score with **only** the generated features,
    * the discovered symbolic expressions, and
    * the maximum mutual information among original vs. generated features.

    Examples
    --------
    >>> runner = GPBenchmarkRunner()                           # doctest: +SKIP
    >>> runner.add_dataset("diabetes", X, y, task_type="regression")  # doctest: +SKIP
    >>> runner.run()                                            # doctest: +SKIP
    >>> runner.report()                                         # doctest: +SKIP
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
    ) -> GPBenchmarkRunner:
        """Register a dataset for the benchmark. Returns ``self`` for chaining."""
        self._datasets.append(
            {"name": name, "X": X, "y": y, "task_type": task_type, "description": description}
        )
        return self

    def run(
        self,
        config: GPConfig | None = None,
        estimator: BaseEstimator | None = None,
        scoring: str | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Run the benchmark over all registered datasets.

        Parameters
        ----------
        config : GPConfig, optional
            Configuration applied to every run (task type is overridden per
            dataset). Defaults to :class:`GPConfig` with ``verbose=False``.
        estimator : sklearn estimator, optional
            Downstream model; defaults to a linear model per task type.
        scoring : str, optional
            Scoring string; auto-selected per task when ``None``.
        verbose : bool, default=True
            Print a per-dataset progress report.

        Returns
        -------
        list of dict
            One result entry per dataset.
        """
        config = config or GPConfig(verbose=False)
        self._results = []

        for dataset in self._datasets:
            self._results.append(self._run_one(dataset, config, estimator, scoring, verbose))
        return self._results

    def _run_one(
        self,
        dataset: dict,
        config: GPConfig,
        estimator: BaseEstimator | None,
        scoring: str | None,
        verbose: bool,
    ) -> dict:
        name, X, y = dataset["name"], dataset["X"], dataset["y"]
        X_array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_array = prepare_target(y)
        task_type = infer_task_type(y_array, dataset["task_type"])
        scoring_resolved = resolve_scoring(scoring, task_type)

        if verbose:
            print(
                f"\n{'=' * 60}\n  Dataset: {name} | Shape: {X_array.shape} | "
                f"Task: {task_type}\n{'=' * 60}"
            )

        cfg = copy.deepcopy(config)
        cfg.verbose = verbose
        cfg.task_type = task_type

        est = (
            clone(estimator)
            if estimator is not None
            else GPFeatureEngineer._default_estimator(task_type)
        )
        cv = self._build_cv(y_array, task_type, cfg.random_seed)

        start = time.time()
        engineer = GPFeatureEngineer(config=cfg, scoring=scoring_resolved, estimator=clone(est))
        engineer.fit(X, y)
        elapsed = time.time() - start
        result = engineer.result_

        X_full = engineer.transform(X)
        n_orig = X_array.shape[1]
        X_gp_only = X_full[:, n_orig:] if cfg.augment_original else X_full
        gp_only_cv = self._score(est, X_gp_only, y_array, scoring_resolved, cv)

        mi_fn = mutual_info_classif if task_type == "classification" else mutual_info_regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_original = mi_fn(X_array, y_array, random_state=cfg.random_seed)

        if verbose:
            print(f"\n  Summary for {name}:")
            print(f"    Baseline CV:  {result.baseline_cv_score:.4f}")
            print(
                f"    Augmented CV: {result.augmented_cv_score:.4f} "
                f"({result.score_improvement:+.4f})"
            )
            print(f"    GP-only CV:   {gp_only_cv:.4f}")
            print(f"    Time:         {elapsed:.1f}s")

        return {
            "name": name,
            "n_samples": X_array.shape[0],
            "n_features_original": n_orig,
            "n_features_generated": len(result.generated_features),
            "baseline_cv": result.baseline_cv_score,
            "augmented_cv": result.augmented_cv_score,
            "gp_only_cv": gp_only_cv,
            "score_delta": result.score_improvement,
            "time_seconds": elapsed,
            "n_evaluations": result.n_evaluations,
            "scoring": scoring_resolved,
            "task_type": task_type,
            "max_mi_original": float(np.max(mi_original)),
            "max_mi_gp": max(gf.mi_score for gf in result.generated_features),
            "expressions": [gf.expression for gf in result.generated_features],
            "result": result,
            "engineer": engineer,
        }

    @staticmethod
    def _build_cv(y: np.ndarray, task_type: str, seed) -> object:
        if task_type == "classification":
            _, counts = np.unique(y, return_counts=True)
            n_splits = max(2, min(5, int(counts.min())))
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        n_splits = max(2, min(5, y.shape[0]))
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    @staticmethod
    def _score(estimator, X, y, scoring, cv) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                clone(estimator), X, y, scoring=scoring, cv=cv, error_score=0.0
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
                "Feats. GP": r["n_features_generated"],
                "Baseline CV": f"{r['baseline_cv']:.4f}",
                "Augmented CV": f"{r['augmented_cv']:.4f}",
                "GP-only CV": f"{r['gp_only_cv']:.4f}",
                "Delta": f"{r['score_delta']:+.4f}",
                "MI orig (max)": f"{r['max_mi_original']:.4f}",
                "MI GP (max)": f"{r['max_mi_gp']:.4f}",
                "Time (s)": f"{r['time_seconds']:.1f}",
                "Task": r["task_type"],
            }
            for r in self._results
        ]
        df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("  BENCHMARK REPORT - GP Feature Engineering")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        return df

    @property
    def results(self) -> list[dict]:
        """The list of result entries from the last :meth:`run`."""
        return self._results
