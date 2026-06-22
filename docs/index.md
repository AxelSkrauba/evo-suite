# evo-suite

**Evolutionary computation for tabular data engineering.**

`evo-suite` is a family of independent, [scikit-learn](https://scikit-learn.org)-compatible
packages that apply evolutionary computation to the data-preprocessing stage of a
machine-learning pipeline. They share one repository, CI and documentation, but
are published to PyPI independently.

| Distribution | Import | Technique | Role | Status |
|--------------|--------|-----------|------|--------|
| `evo-gafs` | `evo_gafs` | Genetic Algorithm | Feature **selection** | Available |
| `evo-gpfe` | `evo_gpfe` | Genetic Programming | Feature **engineering** | Planned |

This documentation currently covers **`evo-gafs`**, a genetic-algorithm *wrapper*
feature selector for tabular data.

## Highlights

- **Explicit accuracy ↔ compression trade-off** via a single `alpha` parameter —
  ideal for edge/embedded deployment.
- **Multi-objective NSGA-II** mode exposing the full Pareto front.
- **Native scikit-learn estimator**: `fit` / `transform` / `get_support`, usable
  in a `Pipeline` and tunable with `GridSearchCV`.
- **Repair operator** guaranteeing feasible subsets, evaluation **cache**, and a
  built-in multi-dataset **benchmark runner**.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

guide/concepts
guide/configuration
guide/multiobjective
guide/pipeline
guide/benchmark
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

api
changelog
```
