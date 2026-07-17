# API reference

The public API of `evo_gpfe`.

```{eval-rst}
.. currentmodule:: evo_gpfe

.. autosummary::
   :nosignatures:

   GPFeatureEngineer
   GPConfig
   GPEngineeringResult
   GeneratedFeature
   GPFitnessEvaluator
   GPBenchmarkRunner
   GPPlotter
```

## Engineer

```{eval-rst}
.. autoclass:: evo_gpfe.GPFeatureEngineer
   :members: fit, transform, get_feature_names_out, summary

   In addition to the methods above, ``GPFeatureEngineer`` inherits the standard
   scikit-learn transformer API from :class:`sklearn.base.TransformerMixin` and
   :class:`sklearn.base.BaseEstimator` — notably ``fit_transform``,
   ``get_params`` and ``set_params``.
```

## Configuration & results

```{eval-rst}
.. autoclass:: evo_gpfe.GPConfig
   :members:

.. autoclass:: evo_gpfe.GPEngineeringResult
   :members:

.. autoclass:: evo_gpfe.GeneratedFeature
   :members:
```

## Evaluation

```{eval-rst}
.. autoclass:: evo_gpfe.GPFitnessEvaluator
   :members:
```

## Primitives

`evo_gpfe.primitives.PRIMITIVE_SETS` maps each named function-set string
(`'basic'`, `'extended'`, `'full'`, `'nonlinear'`) to its list of
{class}`~evo_gpfe.primitives.Primitive` entries (see [Concepts](guide/concepts.md)).

```{eval-rst}
.. autofunction:: evo_gpfe.primitives.build_pset

.. autoclass:: evo_gpfe.primitives.Primitive
   :members:
```

## Benchmarking & visualization

```{eval-rst}
.. autoclass:: evo_gpfe.GPBenchmarkRunner
   :members:

.. autoclass:: evo_gpfe.GPPlotter
   :members:
```
