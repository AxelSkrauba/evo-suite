# Configuration

All genetic-programming behaviour is controlled by {class}`~evo_gpfe.GPConfig`,
a dataclass passed to the engineer. It validates its values on construction
(raising {class}`ValueError` on invalid input).

```python
from evo_gpfe import GPConfig

config = GPConfig(
    population_size=200,
    n_generations=40,
    n_features_to_generate=5,
    fitness_metric="mutual_info",
    redundancy_beta=0.3,
    parsimony_coeff=0.002,
    random_seed=42,
)
```

## Key parameters

```{list-table}
:header-rows: 1
:widths: 24 12 64

* - Parameter
  - Default
  - Description
* - `population_size`
  - `300`
  - Trees per generation. GP needs larger populations than a GA because the
    search space (programs) is much larger.
* - `n_generations`
  - `40`
  - Generations evolved **per generated feature**.
* - `n_features_to_generate`
  - `5`
  - Number of new features to construct (see
    [Concepts](concepts.md#sequential-hall-of-fame-strategy)).
* - `function_set`
  - `'extended'`
  - Primitive set: `'basic'`, `'extended'`, `'full'` or `'nonlinear'`.
* - `fitness_metric`
  - `'mutual_info'`
  - Relevance metric: `'mutual_info'`, `'correlation'`, `'spearman'` or
    `'model_score'`.
* - `redundancy_beta`
  - `0.30`
  - Penalty for correlation with the hall of fame. `0` risks near-duplicate
    features; higher values favour diversity.
* - `parsimony_coeff`
  - `0.002`
  - Penalty per tree node (Occam's razor).
* - `redundancy_threshold`
  - `0.95`
  - A candidate whose correlation with the hall of fame exceeds this is
    rejected in favour of an alternative.
* - `max_tree_height`
  - `6`
  - Static height limit (anti-bloat). Must be `>= init_depth_max`.
* - `augment_original`
  - `True`
  - Concatenate generated features to the originals (vs. returning only the
    generated ones).
* - `normalize_output`
  - `True`
  - Standardise each generated feature before returning it.
* - `early_stopping_rounds`
  - `20`
  - Stop evolving a feature after this many generations without improvement.
* - `warm_start_population`
  - `True`
  - Seed each new feature's population with the previous one.
* - `random_seed`
  - `42`
  - Seed for reproducible runs.
```

See the full list and defaults in the {class}`~evo_gpfe.GPConfig` API reference.

## Choosing a fitness metric

| Goal | Suggested `fitness_metric` |
|------|----------------------------|
| General-purpose default | `mutual_info` |
| Fast iteration on large data | `correlation` or `spearman` |
| Directly optimise the downstream model (higher cost) | `model_score` |

See `02_fitness_metrics.py` in the examples for a head-to-head comparison.
