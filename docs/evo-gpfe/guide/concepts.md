# Concepts

## Selection vs. construction

`evo-gafs` **selects** among existing features (a binary mask over the input
columns). `evo-gpfe` **constructs** new features by combining the existing ones
through evolved mathematical expressions. They are complementary: a GP-then-GA
pipeline can construct candidate features and then select the best subset from
the union of original and generated ones (see the `06_gp_then_ga_pipeline.py`
example).

## Genetic representation

Each individual is an **expression tree** (a `deap.gp.PrimitiveTree`) whose
internal nodes are mathematical primitives (`add`, `mul`, `protected_div`,
`sqrt`, `log`, ...) and whose leaves are input features or random constants.
Evaluating the tree over a row of `X` produces one scalar — the candidate
engineered feature. The GP loop evolves this population with subtree
crossover, mixed mutation (subtree replacement, hoist, node replacement) and a
static height limit that prevents uncontrolled tree growth ("bloat").

## Protected primitives

Arithmetic operators are total (never raise, never produce NaN/inf) so that
randomly assembled expressions always evaluate: division by ~0 returns `1.0`,
`log`/`sqrt` operate on `|x|`, and outputs are clipped to a safe range. Four
named primitive sets trade off expressiveness: `basic`, `extended` (default),
`full` and `nonlinear` — see {func}`evo_gpfe.primitives.build_pset`.

## Sequential hall-of-fame strategy

Generating several diverse features one at a time, instead of independently, is
key to avoiding near-duplicate expressions (a common failure mode with a plain
relevance-only fitness). For each feature to generate:

1. Evolve a population maximising relevance to `y`.
2. Take the best tree and check its correlation with every feature already in
   the **hall of fame** (HoF); if too redundant, try the next-best candidates
   from the final population instead.
3. Add the accepted tree to the HoF — future fitness evaluations now penalise
   correlation with it.
4. Optionally warm-start the next feature's population with the current one
   (`warm_start_population=True`, the default) for faster convergence.

## The fitness formula

```
fitness = relevance(tree, y)
          - redundancy_beta * max_corr(tree, hall_of_fame)
          - parsimony_coeff * n_nodes
```

- **Relevance** (`fitness_metric`): `mutual_info` (default, captures
  non-linear dependence), `correlation` / `spearman` (cheap association), or
  `model_score` (directly optimises the downstream estimator's CV score — the
  most informative but most expensive option).
- **Redundancy** (`redundancy_beta`): penalises correlation with the hall of
  fame; higher values push toward more diverse features.
- **Parsimony** (`parsimony_coeff`): penalises tree size (Occam's razor),
  favouring simpler, more interpretable expressions.

See [Configuration](configuration.md) for the full parameter reference and
`03_redundancy_and_parsimony.py` in the examples for these knobs in action.

## Reproducibility

`random_seed` makes runs deterministic. Per-instance DEAP types are registered
with a UUID suffix and cleaned up after `fit`, so re-instantiating the engineer
(e.g. in a notebook) never collides with a previous run.
