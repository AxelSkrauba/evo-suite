# Changelog

All notable changes to `evo-ens` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-25

### Added

- Initial release of `evo-ens`: `EvoEnsembleClassifier` / `EvoEnsembleRegressor`,
  a scikit-learn-compatible evolutionary ensemble builder that co-optimizes
  predictive score and prediction diversity (Q-statistic / Pearson) with a
  Genetic Algorithm, using an out-of-fold pre-computation strategy for fast
  fitness evaluation.
- `single` (score − β·diversity) and `multiobjective` (NSGA-II: score vs.
  compression) optimization modes.
- `EvoEnsConfig`, `EvoEnsResult`, `EnsembleMember`, `ParetoSolution`,
  `EvolutionStats` dataclasses.
- `EvoEnsBenchmarkRunner` and `EvoEnsPlotter` for multi-dataset benchmarking
  and visualization.
