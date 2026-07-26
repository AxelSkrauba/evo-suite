# API reference

The public API of `evo_ens`.

```{eval-rst}
.. currentmodule:: evo_ens

.. autosummary::
   :nosignatures:

   EvoEnsembleClassifier
   EvoEnsembleRegressor
   EvoEnsConfig
   EvoEnsResult
   EnsembleMember
   ParetoSolution
   EvolutionStats
   EnsembleFitnessEvaluator
   EvoEnsBenchmarkRunner
   EvoEnsPlotter
```

## Estimators

```{eval-rst}
.. autoclass:: evo_ens.EvoEnsembleClassifier
   :members: fit, predict, predict_proba, get_ensemble_info

   In addition to the methods above, ``EvoEnsembleClassifier`` inherits the
   standard scikit-learn classifier API from
   :class:`sklearn.base.ClassifierMixin` and :class:`sklearn.base.BaseEstimator`
   — notably ``score``, ``get_params`` and ``set_params``.

.. autoclass:: evo_ens.EvoEnsembleRegressor
   :members: fit, predict, get_ensemble_info

   Inherits the standard scikit-learn regressor API from
   :class:`sklearn.base.RegressorMixin`.
```

## Configuration & results

```{eval-rst}
.. autoclass:: evo_ens.EvoEnsConfig
   :members:

.. autoclass:: evo_ens.EvoEnsResult
   :members:

.. autoclass:: evo_ens.EnsembleMember
   :members:

.. autoclass:: evo_ens.ParetoSolution
   :members:

.. autoclass:: evo_ens.EvolutionStats
   :members:
```

## Evaluation & diversity metrics

```{eval-rst}
.. autoclass:: evo_ens.EnsembleFitnessEvaluator
   :members:

.. autofunction:: evo_ens.q_statistic_pair

.. autofunction:: evo_ens.diversity_q_mean

.. autofunction:: evo_ens.diversity_pearson_mean
```

## Candidate pools

```{eval-rst}
.. autofunction:: evo_ens.default_candidates

.. autofunction:: evo_ens.estimator_display_name
```

## Benchmarking & visualization

```{eval-rst}
.. autoclass:: evo_ens.EvoEnsBenchmarkRunner
   :members:

.. autoclass:: evo_ens.EvoEnsPlotter
   :members:
```
