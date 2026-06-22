# Concepts

## Wrapper vs filter selection

Feature-selection methods fall into two broad families:

- **Filter** methods rank features by a statistic (mutual information,
  correlation, ANOVA F-value) computed independently of any model. They are
  fast but ignore feature *interactions* and are not tailored to the model you
  will actually use.
- **Wrapper** methods evaluate candidate subsets by training and
  cross-validating the *target model* on them. They are more expensive but
  account for interactions and optimise the subset for the specific estimator.

`evo-gafs` is a **wrapper**: the fitness of a feature subset is a cross-validated
score of your `estimator`.

## Genetic representation

Each individual in the population is a **binary mask** of length `n_features`: a
`1` keeps the feature, a `0` drops it. The genetic algorithm evolves this
population with selection, uniform crossover and bit-flip mutation. A **repair
operator** guarantees every individual keeps at least `min_features` active
features, so infeasible (empty) subsets never reach evaluation.

## The `alpha` trade-off (single-objective)

In the default single-objective mode, fitness combines performance and
compression into one scalar:

```
fitness     = alpha * cv_score + (1 - alpha) * compression
compression = 1 - n_selected / n_total
```

- `alpha = 1.0` → a pure wrapper: maximise performance, ignore size.
- `alpha ≈ 0.7` → a balanced default, good for edge/embedded deployment.
- lower `alpha` → aggressively smaller subsets, trading some accuracy.

This makes the performance/size decision — which every engineer faces — an
explicit, tunable parameter.

## Multi-objective (NSGA-II)

When you do not want to commit to a single `alpha`, multi-objective mode uses
**NSGA-II** to optimise performance and compression *simultaneously*, returning
the entire **Pareto front** of non-dominated trade-offs. You then pick a point
on the front after inspecting it. See [Multi-objective](multiobjective.md).

## Reproducibility & efficiency

- `random_seed` makes runs deterministic.
- An internal **cache** memoises fitness per genome, so repeated subsets are not
  re-evaluated.
- Cross-validation folds are clamped to the smallest class count, keeping the
  selector robust on small datasets.
