# Changelog

All notable changes to `gafs` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial modular package extracted from the educational monolith.
- `GAFeatureSelector` now inherits from `BaseEstimator` and `SelectorMixin`,
  exposing `fit`/`transform`/`get_support` and working inside scikit-learn
  pipelines.
- Per-instance DEAP type registration (UUID-suffixed) to avoid global type
  collisions when re-instantiating the selector (e.g. in notebooks).
- Robust input validation and configuration validation (`ValueError` instead of
  `assert`).
- `SelectionResult` serialization helpers: `to_json`, `save_json`, `save`,
  `load`.
- Callback-driven early stopping in both single- and multi-objective loops.

### Changed
- `fit` no longer mutates the user-provided `GAConfig` (the effective
  `mutation_indpb` is resolved on a copy).
- Cross-validation folds are clamped to the smallest class count to stay robust
  on small datasets.
