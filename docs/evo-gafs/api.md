# API reference

The public API of `evo_gafs`.

```{eval-rst}
.. currentmodule:: evo_gafs

.. autosummary::
   :nosignatures:

   GAFeatureSelector
   GAConfig
   SelectionResult
   EvolutionStats
   FitnessEvaluator
   BenchmarkRunner
   GAPlotter
```

## Selector

```{eval-rst}
.. autoclass:: evo_gafs.GAFeatureSelector
   :members: fit, summary

   In addition to the methods above, ``GAFeatureSelector`` inherits the standard
   scikit-learn transformer API from :class:`sklearn.feature_selection.SelectorMixin`
   and :class:`sklearn.base.BaseEstimator` — notably ``transform``, ``fit_transform``,
   ``get_support``, ``get_params`` and ``set_params``.
```

## Configuration & results

```{eval-rst}
.. autoclass:: evo_gafs.GAConfig
   :members:

.. autoclass:: evo_gafs.SelectionResult
   :members:

.. autoclass:: evo_gafs.EvolutionStats
   :members:
```

## Evaluation

```{eval-rst}
.. autoclass:: evo_gafs.FitnessEvaluator
   :members:
```

## Benchmarking & visualization

```{eval-rst}
.. autoclass:: evo_gafs.BenchmarkRunner
   :members:

.. autoclass:: evo_gafs.GAPlotter
   :members:
```
