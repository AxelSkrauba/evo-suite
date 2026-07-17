# Changelog

All notable changes to `evo-gpfe` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-17

### Added
- Initial release: a scikit-learn-compatible (`BaseEstimator` + `TransformerMixin`)
  genetic-programming feature constructor.
- Protected mathematical primitives (`add`, `sub`, `mul`, protected `div`/`log`/
  `sqrt`, `sq`, `cube`, `neg`, `abs`, `sigmoid`, `relu`) and four named primitive
  sets (`basic`, `extended`, `full`, `nonlinear`).
- Sequential hall-of-fame strategy: relevance minus a redundancy penalty
  (correlation with already-generated features) minus a parsimony penalty
  (tree size), with automatic rejection/replacement of overly redundant
  candidates.
- Four relevance metrics: `mutual_info` (default), `correlation`, `spearman`,
  and `model_score` (directly optimises the downstream estimator's CV score).
- Anti-bloat controls: static tree-height limit and mixed mutation (subtree,
  hoist, node replacement).
- Per-instance UUID-scoped DEAP types (fitness/`PrimitiveTree`) with cleanup
  after each `fit`, avoiding collisions when re-instantiating the engineer
  (e.g. in notebooks).
- `GPFeatureEngineer.get_feature_names_out`, pipeline/`GridSearchCV`
  compatibility, and `check_estimator` compliance.
- `GPBenchmarkRunner` for multi-dataset comparisons and `GPPlotter` for
  feature-quality, benchmark and mutual-information visualisations.
- `GPEngineeringResult` / `GeneratedFeature` with JSON and pickle
  serialization.
