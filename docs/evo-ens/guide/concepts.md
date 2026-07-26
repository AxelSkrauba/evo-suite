# Concepts

## Why diversity matters

An ensemble outperforms its best individual member when the members' errors
are decorrelated. For regression, the Krogh & Vedelsby (1995) theorem
formalizes this:

```
Error_ensemble = Mean_Error - Mean_Diversity
```

For classification, Yule's Q-statistic measures the error correlation
between two classifiers:

```
Q(i, j) = (N11*N00 - N10*N01) / (N11*N00 + N10*N01)
```

where `N11`/`N00` count samples where both classifiers are simultaneously
correct/wrong, and `N10`/`N01` count disagreements. `Q == 0` means
independent errors; `Q < 0` means anti-correlated errors — the ideal case for
an ensemble. `evo-ens` penalizes high `Q` (equivalently, rewards diversity)
via `diversity_beta`. For regression, the absolute Pearson correlation
between members' predictions is used as an analogous proxy.

## Genetic representation

Each individual is a flat vector of `2 * n_candidates` continuous genes:

```
individual = [b_0, ..., b_{N-1},  w_0, ..., w_{N-1}]
              activation thresholds       raw weights
```

`b_i > 0.5` activates candidate `i`; `w_i` is normalized (softmax, absolute
value, or uniform — see `weight_method`) into the final ensemble weight. A
single continuous representation lets standard DEAP operators
(`cxUniform`, `mutGaussian`) evolve *which* models to use and *how much* to
weight them jointly, so the two decisions can co-adapt.

## Out-of-fold pre-computation

Naively evaluating a candidate ensemble would require re-training and
cross-validating every active candidate for every individual in every
generation — `O(n_gen * n_pop * n_candidates * cv_cost)`, infeasible for
non-trivial populations. Instead, `evo-ens` computes out-of-fold (OOF)
predictions for **every candidate once**, before evolution starts. Each
individual's fitness is then a cheap weighted combination of already-computed
arrays:

```
Without OOF:  O(n_gen * n_pop * n_candidates * cv_cost)   <- infeasible
With OOF:     O(n_candidates * cv_cost) + O(n_gen * n_pop * n_samples)
```

The pre-computation step (dominant cost) runs once regardless of how large a
population or how many generations are configured.

## Optimization modes

- **`single`** (default): scalar fitness, `score - diversity_beta * mean_diversity`,
  evolved with tournament selection and elitism. Fast, returns one ensemble.
- **`multiobjective`**: NSGA-II over `(score, compression)`, where
  `compression = 1 - n_selected / n_candidates`. Returns the full Pareto
  front (`result_.pareto_front`) so a smaller ensemble can be picked after
  the fact (e.g. for edge deployment) without re-running the search — see
  `03_multiobjective_pareto.py`.

## Classifier / Regressor split

`EvoEnsembleClassifier` and `EvoEnsembleRegressor` share the same evolutionary
core but are separate estimator classes — the same pattern scikit-learn uses
for `VotingClassifier`/`VotingRegressor` and
`StackingClassifier`/`StackingRegressor`. This keeps scikit-learn's estimator
tags (target type, `predict_proba` availability) fixed per class instead of
resolved at `fit` time, which is required for `check_estimator` compliance
and predictable behavior inside `Pipeline`/`GridSearchCV`.

## Reproducibility

`random_seed` makes runs deterministic (both Python's `random` module, used
by DEAP, and NumPy's RNG are seeded). Per-instance DEAP types are registered
with a UUID suffix and cleaned up after `fit`, so re-instantiating the
estimator (e.g. in a notebook, or across `single`/`multiobjective` runs) never
collides with a previous run.
