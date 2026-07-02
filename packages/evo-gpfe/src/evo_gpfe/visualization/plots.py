"""Matplotlib visualisations for analysing a GP feature-engineering run.

``matplotlib`` is an optional dependency (install with ``pip install evo-gpfe[viz]``)
and is imported lazily so the core package stays lightweight.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from evo_gpfe.core.config import GPEngineeringResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install evo-gpfe[viz]`."
        ) from exc
    return plt


class GPPlotter:
    """Static helpers to visualise generated-feature quality and benchmarks."""

    @staticmethod
    def plot_feature_quality(result: GPEngineeringResult, figsize: tuple[float, float] = (13, 5)):
        """Plot MI score, redundancy and complexity of each generated feature."""
        plt = _require_matplotlib()

        features = result.generated_features
        idx = range(len(features))
        labels = [f"gp_{i}" for i in idx]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax = axes[0]
        bars = ax.bar(
            idx, [gf.mi_score for gf in features], color="#2ecc71", edgecolor="black", alpha=0.85
        )
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Mutual information with target")
        ax.set_title("Relevance per generated feature")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, gf in zip(bars, features):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{gf.mi_score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax = axes[1]
        threshold = result.config.redundancy_threshold if result.config is not None else 0.95
        colors = [
            "#2ecc71"
            if gf.redundancy_score < 0.3
            else "#f39c12"
            if gf.redundancy_score < 0.6
            else "#e74c3c"
            for gf in features
        ]
        ax.bar(
            idx,
            [gf.redundancy_score for gf in features],
            color=colors,
            edgecolor="black",
            alpha=0.85,
        )
        ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Redundancy (corr. with prior hall-of-fame)")
        ax.set_title("Redundancy between generated features")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[2]
        ax.bar(idx, [gf.n_nodes for gf in features], color="#9b59b6", edgecolor="black", alpha=0.85)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Number of nodes")
        ax.set_title("Tree complexity")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Quality of GP-generated features", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_benchmark_comparison(
        benchmark_results: list[dict],
        figsize: tuple[float, float] = (13, 5),
        title: str = "GP feature-engineering benchmark",
    ):
        """Plot baseline vs. augmented score, improvement and time across datasets."""
        plt = _require_matplotlib()

        names = [r["name"] for r in benchmark_results]
        baseline = [r["baseline_cv"] for r in benchmark_results]
        augmented = [r["augmented_cv"] for r in benchmark_results]
        improvement = [a - b for a, b in zip(augmented, baseline)]
        times = [r["time_seconds"] for r in benchmark_results]

        x = np.arange(len(names))
        width = 0.35

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax = axes[0]
        ax.bar(
            x - width / 2,
            baseline,
            width,
            label="Baseline (original)",
            color="#3498db",
            alpha=0.85,
            edgecolor="black",
        )
        ax.bar(
            x + width / 2,
            augmented,
            width,
            label="GP augmented",
            color="#2ecc71",
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("CV score")
        ax.set_title("Baseline vs GP-augmented")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in improvement]
        bars = ax.bar(x, improvement, color=colors, edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Delta CV score")
        ax.set_title("Improvement from GP feature engineering")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, improvement):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.001 if v >= 0 else -0.003),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )

        ax = axes[2]
        ax.bar(x, times, color="#e67e22", edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Time (s)")
        ax.set_title("Execution time")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_mi_comparison(
        result: GPEngineeringResult,
        X_original: np.ndarray,
        y: np.ndarray,
        figsize: tuple[float, float] = (12, 5),
    ):
        """Compare mutual information of original vs. GP-generated features."""
        plt = _require_matplotlib()

        mi_fn = (
            mutual_info_classif if result.task_type == "classification" else mutual_info_regression
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_original = mi_fn(X_original, y)
        mi_gp = [gf.mi_score for gf in result.generated_features]
        best_orig = float(np.max(mi_original))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        names = result.feature_names_original
        order = np.argsort(mi_original)[::-1]
        ax.barh(
            range(len(mi_original)),
            mi_original[order],
            color="#3498db",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_yticks(range(len(mi_original)))
        ax.set_yticklabels([names[i] for i in order], fontsize=8)
        ax.set_xlabel("Mutual information with target")
        ax.set_title("Original features\n(sorted by MI)")
        ax.axvline(
            best_orig,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Best original: {best_orig:.4f}",
        )
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        ax = axes[1]
        ax.barh(range(len(mi_gp)), mi_gp, color="#2ecc71", edgecolor="black", alpha=0.85)
        ax.set_yticks(range(len(mi_gp)))
        ax.set_yticklabels([f"gp_{i}" for i in range(len(mi_gp))], fontsize=8)
        ax.set_xlabel("Mutual information with target")
        ax.set_title("GP-generated features\n(generation order)")
        ax.axvline(
            best_orig,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Best original: {best_orig:.4f}",
        )
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle(
            f"MI: original vs GP-generated features\n"
            f"(CV improvement: {result.baseline_cv_score:.4f} -> {result.augmented_cv_score:.4f})",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig
