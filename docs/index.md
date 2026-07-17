# evo-suite

**Evolutionary computation for tabular data engineering.**

`evo-suite` is a family of independent, [scikit-learn](https://scikit-learn.org)-compatible
packages that apply evolutionary computation to the data-preprocessing stage of a
machine-learning pipeline. They share one repository, CI and documentation, but
are published to PyPI independently.

| Distribution | Import | Technique | Role | Status |
|--------------|--------|-----------|------|--------|
| [`evo-gafs`](evo-gafs/quickstart.md) | `evo_gafs` | Genetic Algorithm | Feature **selection** | Available |
| [`evo-gpfe`](evo-gpfe/quickstart.md) | `evo_gpfe` | Genetic Programming | Feature **engineering** | Available |

The two packages are complementary — `evo-gpfe` *constructs* new features
through evolved symbolic expressions, and `evo-gafs` *selects* the best subset
from the (possibly augmented) feature set. Chaining them (GP, then GA) is a
common pipeline; see the `06_gp_then_ga_pipeline.py` example.

## evo-gafs — Genetic Algorithm Feature Selector

A genetic-algorithm *wrapper* feature selector.

- **Explicit accuracy ↔ compression trade-off** via a single `alpha` parameter —
  ideal for edge/embedded deployment.
- **Multi-objective NSGA-II** mode exposing the full Pareto front.
- **Native scikit-learn estimator**: `fit` / `transform` / `get_support`, usable
  in a `Pipeline` and tunable with `GridSearchCV`.
- **Repair operator** guaranteeing feasible subsets, evaluation **cache**, and a
  built-in multi-dataset **benchmark runner**.

→ [Quickstart](evo-gafs/quickstart.md) · [User guide](evo-gafs/guide/concepts.md) · [API reference](evo-gafs/api.md)

## evo-gpfe — Genetic Programming Feature Engineer

A genetic-programming *symbolic feature constructor*.

- **Sequential hall-of-fame** strategy: relevance to the target, penalised by
  redundancy with already-generated features and tree complexity.
- **Protected primitives** (safe division, log, sqrt, ...) so randomly
  assembled expressions always evaluate.
- **Native scikit-learn transformer**: `fit` / `transform` /
  `get_feature_names_out`, usable in a `Pipeline` and tunable with
  `GridSearchCV`.
- **Anti-bloat controls** (static height limit, parsimony penalty) and a
  built-in multi-dataset **benchmark runner**.

→ [Quickstart](evo-gpfe/quickstart.md) · [User guide](evo-gpfe/guide/concepts.md) · [API reference](evo-gpfe/api.md)

```{toctree}
:maxdepth: 1
:hidden:

installation
```

```{toctree}
:maxdepth: 2
:caption: evo-gafs
:hidden:

evo-gafs/quickstart
evo-gafs/guide/concepts
evo-gafs/guide/configuration
evo-gafs/guide/multiobjective
evo-gafs/guide/pipeline
evo-gafs/guide/benchmark
evo-gafs/api
evo-gafs/changelog
```

```{toctree}
:maxdepth: 2
:caption: evo-gpfe
:hidden:

evo-gpfe/quickstart
evo-gpfe/guide/concepts
evo-gpfe/guide/configuration
evo-gpfe/guide/benchmark
evo-gpfe/api
evo-gpfe/changelog
```
