"""Multi-dataset benchmark runner for :class:`EvoEnsembleClassifier`/`Regressor`."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from evo_ens.core.config import EvoEnsConfig
from evo_ens.core.estimator import EvoEnsembleClassifier, EvoEnsembleRegressor


class EvoEnsBenchmarkRunner:
    """Run :class:`EvoEnsembleClassifier`/`Regressor` over multiple datasets.

    Collects the ensemble's out-of-fold improvement over the best single
    candidate, diversity score, compression ratio and wall-clock time, for
    easy side-by-side comparison.
    """

    def __init__(self) -> None:
        self._datasets: list[dict[str, Any]] = []
        self._results: list[dict[str, Any]] = []

    def add_dataset(
        self,
        name: str,
        X,
        y,
        task_type: str = "classification",
        candidate_estimators: list[BaseEstimator] | None = None,
        description: str = "",
    ) -> EvoEnsBenchmarkRunner:
        """Register a dataset to be run by :meth:`run`.

        Parameters
        ----------
        task_type : {'classification', 'regression'}
            Selects whether :class:`EvoEnsembleClassifier` or
            :class:`EvoEnsembleRegressor` is used for this dataset.
        """
        if task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'")
        self._datasets.append(
            {
                "name": name,
                "X": X,
                "y": y,
                "task_type": task_type,
                "candidates": candidate_estimators,
                "description": description,
            }
        )
        return self

    def run(self, config: EvoEnsConfig | None = None, verbose: bool = True) -> list[dict[str, Any]]:
        """Fit an ensemble on every registered dataset and collect the results."""
        base_config = config or EvoEnsConfig(verbose=False)
        self._results = []

        for ds in self._datasets:
            name = ds["name"]
            if verbose:
                print(f"\n{'=' * 60}\n  Dataset: {name}\n{'=' * 60}")

            cfg_copy = copy.deepcopy(base_config)
            cfg_copy.verbose = verbose

            estimator_cls = (
                EvoEnsembleClassifier
                if ds["task_type"] == "classification"
                else EvoEnsembleRegressor
            )
            builder = estimator_cls(candidate_estimators=ds["candidates"], config=cfg_copy)

            t0 = time.time()
            builder.fit(ds["X"], ds["y"])
            elapsed = time.time() - t0
            result = builder.result_

            entry = {
                "name": name,
                "n_samples": np.asarray(ds["X"]).shape[0],
                "n_features": np.asarray(ds["X"]).shape[1],
                "n_candidates": result.n_candidates,
                "n_models": result.n_models,
                "compression_ratio": result.compression_ratio,
                "best_single_oof": result.oof_score_best_single,
                "ensemble_oof": result.oof_score_ensemble,
                "score_improvement": result.oof_score_improvement,
                "diversity_score": result.diversity_score,
                "time_seconds": elapsed,
                "n_evaluations": result.n_evaluations,
                "scoring": result.scoring,
                "task_type": result.task_type,
                "result": result,
                "estimator": builder,
            }
            self._results.append(entry)

            if verbose:
                print(
                    f"\n  Summary {name}: best_single={result.oof_score_best_single:.4f} | "
                    f"ensemble={result.oof_score_ensemble:.4f} "
                    f"({result.oof_score_improvement:+.4f}) | "
                    f"{result.n_candidates}->{result.n_models} models | "
                    f"{elapsed:.1f}s"
                )

        return self._results

    def report(self) -> pd.DataFrame:
        """Return a tabular summary of the last :meth:`run` call."""
        if not self._results:
            return pd.DataFrame()
        rows = [
            {
                "Dataset": r["name"],
                "Samples": r["n_samples"],
                "Candidates": r["n_candidates"],
                "Selected": r["n_models"],
                "Compression": f"{r['compression_ratio']:.0%}",
                "Best single": f"{r['best_single_oof']:.4f}",
                "Ensemble": f"{r['ensemble_oof']:.4f}",
                "Delta": f"{r['score_improvement']:+.4f}",
                "Diversity": f"{r['diversity_score']:.4f}",
                "Time (s)": f"{r['time_seconds']:.1f}",
            }
            for r in self._results
        ]
        return pd.DataFrame(rows)
