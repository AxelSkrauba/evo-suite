# Pipelines & model selection

`GAFeatureSelector` is a fully compliant scikit-learn estimator (it passes
`check_estimator`), so it composes with the rest of the ecosystem.

## Inside a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from evo_gafs import GAConfig, GAFeatureSelector

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=20, n_generations=15, verbose=False),
    )),
    ("clf", SVC()),
])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

The selector exposes `get_support()` so downstream tooling that inspects
selected features keeps working.

## Tuning with GridSearchCV

Because `get_params`/`set_params` and `clone` are implemented correctly, you can
search over configurations. Pass alternative {class}`~evo_gafs.GAConfig`
objects as the grid values:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "selector__config": [
        GAConfig(alpha=0.9, population_size=12, n_generations=6, verbose=False),
        GAConfig(alpha=0.6, population_size=12, n_generations=6, verbose=False),
    ]
}
search = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy")
search.fit(X, y)
print(search.best_params_["selector__config"].alpha)
```

```{note}
Wrapper selection is computationally heavy: a grid search multiplies the cost by
the number of candidates and outer folds. Keep `population_size`, `n_generations`
and `cv_folds` modest while tuning, and increase them for the final fit.
```
