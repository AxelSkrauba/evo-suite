"""Matplotlib visualizations for analyzing an evolutionary ensemble run.

``matplotlib`` is an optional dependency (install with ``pip install evo-ens[viz]``)
and is imported lazily so the core package stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evo_ens.core.config import EvoEnsResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install evo-ens[viz]`."
        ) from exc
    return plt


class EvoEnsPlotter:
    """Static helpers to visualize an ensemble run and benchmark results."""

    @staticmethod
    def plot_evolution(
        result: EvoEnsResult, title_prefix: str = "", figsize: tuple[float, float] = (10, 5)
    ):
        """Plot best/mean fitness and ensemble size across generations."""
        plt = _require_matplotlib()

        gens = [h.generation for h in result.history]
        best = [h.best_fitness for h in result.history]
        mean = [h.mean_fitness for h in result.history]
        std = [h.std_fitness for h in result.history]
        n_models = [h.mean_n_models for h in result.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(gens, best, label="Best fitness", color="#2ecc71", lw=2)
        ax1.plot(gens, mean, label="Mean fitness", color="#3498db", lw=1.5)
        mean_arr, std_arr = np.array(mean), np.array(std)
        ax1.fill_between(gens, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15, color="#3498db")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title("Fitness evolution")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(gens, n_models, color="#9b59b6", lw=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Mean ensemble size")
        ax2.set_title("Ensemble size evolution")
        ax2.grid(True, alpha=0.3)

        title = f"{title_prefix} | " if title_prefix else ""
        plt.suptitle(f"{title}EvoEnsemble evolution", fontsize=11, fontweight="bold")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_ensemble_composition(result: EvoEnsResult, figsize: tuple[float, float] = (11, 5)):
        """Plot member weights and standalone OOF scores side by side."""
        plt = _require_matplotlib()

        members = sorted(result.members, key=lambda m: -m.weight)
        names = [m.name for m in members]
        weights = [m.weight for m in members]
        oof_scores = [m.oof_score for m in members]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.barh(range(len(members)), weights, color="#3498db", edgecolor="black", alpha=0.85)
        ax1.set_yticks(range(len(members)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel("Weight")
        ax1.set_title("Ensemble composition")
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis="x")

        colors = ["#3498db" if s >= result.oof_score_ensemble else "#e74c3c" for s in oof_scores]
        ax2.barh(range(len(members)), oof_scores, color=colors, edgecolor="black", alpha=0.85)
        ax2.axvline(
            result.oof_score_ensemble,
            color="green",
            lw=2,
            linestyle="--",
            label=f"Ensemble ({result.oof_score_ensemble:.4f})",
        )
        ax2.axvline(
            result.oof_score_best_single,
            color="red",
            lw=1.5,
            linestyle=":",
            label=f"Best single ({result.oof_score_best_single:.4f})",
        )
        ax2.set_yticks(range(len(members)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel(result.scoring)
        ax2.set_title("Standalone OOF score per member")
        ax2.invert_yaxis()
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="x")

        plt.suptitle(
            f"Ensemble composition | diversity={result.diversity_score:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_pareto_front(result: EvoEnsResult, figsize: tuple[float, float] = (8, 6)):
        """Plot the NSGA-II Pareto front (score vs. ensemble size)."""
        plt = _require_matplotlib()

        if not result.pareto_front:
            raise ValueError(
                "No Pareto front available (result was not run in 'multiobjective' mode)."
            )

        pf = result.pareto_front
        scores = [s.score for s in pf]
        compressions = [s.compression for s in pf]
        n_models = [s.n_models for s in pf]

        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(
            n_models,
            scores,
            c=compressions,
            cmap="RdYlGn",
            s=120,
            edgecolors="black",
            linewidths=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Compression")
        best_idx = int(np.argmax(scores))
        ax.scatter(
            n_models[best_idx],
            scores[best_idx],
            s=300,
            marker="*",
            color="gold",
            edgecolors="black",
            linewidths=1.5,
            label=f"Best score ({scores[best_idx]:.4f})",
        )
        ax.set_xlabel("Number of models in ensemble")
        ax.set_ylabel(result.scoring)
        ax.set_title("Pareto front - score vs. ensemble size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_benchmark(
        benchmark_results: list[dict[str, Any]], figsize: tuple[float, float] = (14, 5)
    ):
        """Plot a multi-dataset benchmark comparison (see :class:`EvoEnsBenchmarkRunner`)."""
        plt = _require_matplotlib()

        names = [r["name"] for r in benchmark_results]
        baseline = [r["best_single_oof"] for r in benchmark_results]
        ensemble = [r["ensemble_oof"] for r in benchmark_results]
        times = [r["time_seconds"] for r in benchmark_results]

        x = np.arange(len(names))
        width = 0.35
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax = axes[0]
        ax.bar(x - width / 2, baseline, width, label="Best single", color="#3498db", alpha=0.85)
        ax.bar(x + width / 2, ensemble, width, label="EvoEnsemble", color="#2ecc71", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("OOF score")
        ax.set_title("Best single vs. EvoEnsemble")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        deltas = [e - b for e, b in zip(ensemble, baseline)]
        colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
        ax.bar(x, deltas, color=colors, edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Delta OOF score")
        ax.set_title("Ensemble improvement")
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[2]
        ax.bar(x, times, color="#9b59b6", edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Time (s)")
        ax.set_title("Total time (precompute + evolution)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("EvoEnsemble benchmark", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig
