# Configuration

All algorithm knobs live on {class}`~evo_ens.EvoEnsConfig`, a validated
dataclass passed to `EvoEnsembleClassifier`/`EvoEnsembleRegressor` via the
`config` parameter.

```python
from evo_ens import EvoEnsConfig

config = EvoEnsConfig(
    population_size=80,
    n_generations=100,
    mode="single",
    diversity_beta=0.10,
    weight_method="softmax",
    min_models=2,
    max_models=None,
    cv_folds=5,
    random_seed=42,
)
```

## Genetic Algorithm

| Parameter | Default | Notes |
|---|---|---|
| `population_size` | 80 | Cost per individual is `O(n_samples)` thanks to OOF pre-computation, so larger populations stay cheap relative to pre-computation. |
| `n_generations` | 100 | Maximum generations; subject to early stopping (single mode). |
| `crossover_prob` | 0.75 | Uniform crossover probability. |
| `mutation_prob` | 0.20 | Probability of mutating an individual. |
| `mutation_sigma` | 0.25 | Gaussian mutation standard deviation. |
| `mutation_indpb` | `None` | Per-gene mutation probability; defaults to `1 / (2 * n_candidates)`. |
| `tournament_size` | 4 | Single mode only. |
| `elite_size` | 2 | Elites preserved verbatim each generation (single mode only). |

## Mode and fitness

| Parameter | Default | Notes |
|---|---|---|
| `mode` | `'single'` | `'single'` or `'multiobjective'` (NSGA-II) — see [Concepts](concepts.md). |
| `scoring` | `None` | `'accuracy'`/`'f1_macro'`/`'roc_auc'` (classification) or `'r2'`/`'neg_rmse'` (regression); auto-selected by task when `None`. |
| `diversity_beta` | 0.10 | Diversity penalty weight (single mode). `0.0` disables it; higher values favor decorrelated ensembles over raw score. |
| `weight_method` | `'softmax'` | `'softmax'` (always positive, never exactly zero), `'abs_norm'` (simple, can zero out a member), or `'uniform'` (ignores the weight genes). |

## Ensemble constraints

| Parameter | Default | Notes |
|---|---|---|
| `min_models` | 2 | Individuals selecting fewer models are penalized to zero fitness. |
| `max_models` | `None` | Caps ensemble size (excess candidates dropped by lowest weight) — useful for edge deployment. |
| `cv_folds` | 5 | Folds for the out-of-fold pre-computation; automatically clamped to the smallest class count (classification) or `n_samples` (regression) for tiny inputs. |

## Control

| Parameter | Default | Notes |
|---|---|---|
| `random_seed` | 42 | Seeds both `random` and `numpy.random`. |
| `verbose` | `True` | Print progress during `fit`. |
| `early_stopping_rounds` | 25 | Stop if the best fitness has not improved by more than `early_stopping_tol` over this many generations (single mode only). |
| `early_stopping_tol` | 1e-5 | See above. |

## Candidate pool

By default, `EvoEnsembleClassifier`/`Regressor` use
{func}`evo_ens.default_candidates`: a structurally diverse pool (tree-based,
linear, kernel, instance-based models) chosen to favor naturally low
Q-statistics between members. Pass your own list of (unfitted) scikit-learn
estimators via `candidate_estimators` to override it — see
`04_custom_candidates_and_pipeline.py`.

```{note}
`EvoEnsConfig.validate()` raises `ValueError` (not a bare `assert`) on
invalid combinations, e.g. `max_models < min_models`, and is called
automatically on construction and again defensively at the start of `fit`.
```
