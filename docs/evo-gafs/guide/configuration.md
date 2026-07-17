# Configuration

All genetic-algorithm behaviour is controlled by {class}`~evo_gafs.GAConfig`, a
dataclass passed to the selector. It validates its values on construction
(raising {class}`ValueError` on invalid input).

```python
from evo_gafs import GAConfig

config = GAConfig(
    population_size=50,
    n_generations=100,
    alpha=0.8,
    cv_folds=5,
    random_seed=42,
)
```

## Key parameters

```{list-table}
:header-rows: 1
:widths: 22 12 66

* - Parameter
  - Default
  - Description
* - `population_size`
  - `50`
  - Number of candidate masks per generation. Larger explores more, costs more.
* - `n_generations`
  - `100`
  - Number of evolutionary iterations.
* - `mode`
  - `'single'`
  - `'single'` (weighted fitness) or `'multiobjective'` (NSGA-II).
* - `alpha`
  - `0.8`
  - Performance vs compression weight in single-objective mode (see
    [Concepts](concepts.md)).
* - `cv_folds`
  - `5`
  - Cross-validation folds used to score each subset.
* - `min_features`
  - `1`
  - Minimum number of selected features (enforced by the repair operator).
* - `crossover_prob` / `mutation_prob`
  - `0.8` / `0.15`
  - Probabilities of crossover and mutation.
* - `tournament_size`
  - `3`
  - Tournament size for selection (single-objective).
* - `elite_size`
  - `2`
  - Best individuals carried over unchanged each generation.
* - `early_stopping_rounds`
  - `None`
  - Stop if the best fitness stagnates for this many generations.
* - `n_jobs`
  - `1`
  - Parallelism passed to scikit-learn's cross-validation.
* - `random_seed`
  - `42`
  - Seed for reproducible runs.
```

See the full list and defaults in the {class}`~evo_gafs.GAConfig` API reference.

## Choosing `alpha`

| Goal | Suggested `alpha` |
|------|-------------------|
| Maximum accuracy, size irrelevant | `1.0` |
| Balanced (general use) | `0.8` |
| Edge/embedded, small models | `0.5`–`0.7` |

If you are unsure which trade-off you want, use
[multi-objective mode](multiobjective.md) and inspect the Pareto front instead.
