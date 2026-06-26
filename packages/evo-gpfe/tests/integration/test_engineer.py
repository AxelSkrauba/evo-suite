"""Integration tests for GPFeatureEngineer."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from evo_gpfe import GPConfig, GPFeatureEngineer


def test_fit_transform_augments(classification_data, tiny_config):
    X, y = classification_data
    eng = GPFeatureEngineer(config=tiny_config).fit(X, y)
    Xt = eng.transform(X)
    assert len(eng.trees_) == tiny_config.n_features_to_generate
    assert Xt.shape == (X.shape[0], X.shape[1] + tiny_config.n_features_to_generate)
    assert np.all(np.isfinite(Xt))


def test_no_augment_returns_only_generated(classification_data):
    X, y = classification_data
    config = GPConfig(
        population_size=30,
        n_generations=5,
        n_features_to_generate=3,
        augment_original=False,
        random_seed=0,
        verbose=False,
    )
    Xt = GPFeatureEngineer(config=config).fit_transform(X, y)
    assert Xt.shape == (X.shape[0], 3)


def test_get_feature_names_out_matches_transform(classification_data, tiny_config):
    X, y = classification_data
    eng = GPFeatureEngineer(config=tiny_config).fit(X, y)
    names = eng.get_feature_names_out()
    assert len(names) == eng.transform(X).shape[1]
    assert list(names[: X.shape[1]]) == list(X.columns)


def test_feature_names_in_for_dataframe(classification_data, tiny_config):
    X, y = classification_data
    eng = GPFeatureEngineer(config=tiny_config).fit(X, y)
    assert list(eng.feature_names_in_) == list(X.columns)


def test_numpy_input_has_no_feature_names_in(regression_data, tiny_config):
    X, y = regression_data
    eng = GPFeatureEngineer(config=tiny_config).fit(X, y)
    assert not hasattr(eng, "feature_names_in_")
    names = eng.get_feature_names_out()
    assert names[0] == "x0"


def test_fit_does_not_mutate_config(classification_data):
    X, y = classification_data
    config = GPConfig(population_size=20, n_generations=4, n_features_to_generate=2, verbose=False)
    snapshot = config.to_dict()
    GPFeatureEngineer(config=config).fit(X, y)
    assert config.to_dict() == snapshot


def test_works_inside_pipeline(classification_data, tiny_config):
    X, y = classification_data
    pipe = Pipeline(
        [
            ("gp", GPFeatureEngineer(config=tiny_config)),
            ("clf", DecisionTreeClassifier(random_state=0)),
        ]
    )
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.5


def test_regression(regression_data, tiny_config):
    X, y = regression_data
    eng = GPFeatureEngineer(config=tiny_config).fit(X, y)
    assert eng.result_.task_type == "regression"
    assert eng.transform(X).shape[1] == X.shape[1] + tiny_config.n_features_to_generate


def test_transform_before_fit_raises(classification_data, tiny_config):
    X, _ = classification_data
    with pytest.raises(NotFittedError):
        GPFeatureEngineer(config=tiny_config).transform(X)


def test_reproducible_with_seed(classification_data):
    X, y = classification_data
    config = GPConfig(
        population_size=25,
        n_generations=5,
        n_features_to_generate=2,
        random_seed=7,
        verbose=False,
    )
    a = GPFeatureEngineer(config=config).fit(X, y)
    b = GPFeatureEngineer(config=config).fit(X, y)
    assert [gf.expression for gf in a.result_.generated_features] == [
        gf.expression for gf in b.result_.generated_features
    ]


def test_clone(tiny_config):
    eng = GPFeatureEngineer(config=tiny_config, scoring="accuracy")
    cloned = clone(eng)
    assert isinstance(cloned, GPFeatureEngineer)
    assert cloned.scoring == "accuracy"
