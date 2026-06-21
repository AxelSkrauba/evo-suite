"""Matplotlib visualisations for analysing a GA run.

``matplotlib`` is an optional dependency (install with ``pip install evo_gafs[viz]``)
and is imported lazily so the core package stays lightweight.
"""

from __future__ import annotations

import numpy as np

from evo_gafs.core.config import SelectionResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install evo_gafs[viz]`."
        ) from exc
    return plt


class GAPlotter:
    """Static helpers to visualise evolution curves and Pareto fronts."""

    @staticmethod
    def plot_evolution(
        result: SelectionResult,
        figsize: tuple[float, float] = (14, 5),
        title_prefix: str = "",
    ):
        """Plot fitness and feature-count evolution over generations."""
        plt = _require_matplotlib()

        history = result.history
        gens = [s.generation for s in history]
        best_fit = [s.best_fitness for s in history]
        mean_fit = [s.mean_fitness for s in history]
        std_fit = [s.std_fitness for s in history]
        best_feats = [s.best_n_features for s in history]
        mean_feats = [s.mean_n_features for s in history]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        prefix = f"{title_prefix} - " if title_prefix else ""

        ax1 = axes[0]
        ax1.plot(gens, best_fit, "b-", linewidth=2, label="Best fitness")
        ax1.plot(gens, mean_fit, "g--", linewidth=1.5, label="Mean fitness")
        ax1.fill_between(
            gens,
            [m - s for m, s in zip(mean_fit, std_fit)],
            [m + s for m, s in zip(mean_fit, std_fit)],
            alpha=0.2,
            color="green",
            label="±1 std",
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(f"{prefix}Fitness evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(gens, best_feats, "r-", linewidth=2, label="Features (best ind.)")
        ax2.plot(gens, mean_feats, "m--", linewidth=1.5, label="Features (mean)")
        ax2.axhline(
            y=result.n_selected,
            color="darkred",
            linestyle=":",
            label=f"Final selected: {result.n_selected}",
        )
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Number of features")
        ax2.set_title(f"{prefix}Selected-feature evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_pareto_front(
        result: SelectionResult,
        figsize: tuple[float, float] = (8, 6),
        title: str = "Pareto front",
    ):
        """Plot the Pareto front (multi-objective mode only)."""
        plt = _require_matplotlib()

        if result.pareto_front is None:
            raise ValueError("No Pareto front available (single-objective mode).")

        front = result.pareto_front
        cv_scores = [p["cv_score"] for p in front]
        n_feats = [p["n_features"] for p in front]
        compressions = [p["compression"] for p in front]

        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            n_feats,
            cv_scores,
            c=compressions,
            cmap="RdYlGn",
            s=100,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        fig.colorbar(scatter, ax=ax, label="Compression (1 - feature ratio)")

        best_idx = int(np.argmax(cv_scores))
        ax.scatter(
            n_feats[best_idx],
            cv_scores[best_idx],
            s=250,
            marker="*",
            color="gold",
            edgecolors="black",
            linewidths=1.5,
            label=f"Best CV score\n({cv_scores[best_idx]:.3f}, {n_feats[best_idx]} feats)",
            zorder=10,
        )
        ax.set_xlabel("Number of selected features")
        ax.set_ylabel("CV score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_selected_features(
        result: SelectionResult,
        feature_names: list[str] | None = None,
        figsize: tuple[float, float] = (10, 6),
        title: str = "Selected vs removed features",
    ):
        """Show which features were selected versus removed."""
        plt = _require_matplotlib()
        from matplotlib.patches import Patch

        names = (
            feature_names
            if feature_names is not None
            else [f"f{i}" for i in range(len(result.selected_mask))]
        )
        n_total = len(names)
        colors = ["#2ecc71" if m else "#e74c3c" for m in result.selected_mask]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(n_total), [1] * n_total, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_total))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks([])
        ax.set_title(title)
        ax.legend(
            handles=[
                Patch(facecolor="#2ecc71", label=f"Selected ({result.n_selected})"),
                Patch(facecolor="#e74c3c", label=f"Removed ({n_total - result.n_selected})"),
            ],
            loc="lower right",
        )
        ax.invert_yaxis()
        fig.tight_layout()
        return fig
