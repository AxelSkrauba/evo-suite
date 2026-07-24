"""Integration tests for EvoEnsembleClassifier / EvoEnsembleRegressor."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from evo_ens import EvoEnsConfig, EvoEnsembleClassifier, EvoEnsembleRegressor


class TestClassification:
    def test_fit_predict_binary(self, binary_classification_data, tiny_config):
        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape
        assert set(np.unique(preds)) <= set(np.unique(y))
        assert clf.score(X, y) >= 0.5

    def test_fit_predict_multiclass(self, classification_data, tiny_config):
        X, y = classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_string_labels_are_handled(self, binary_classification_data, tiny_config):
        X, y = binary_classification_data
        y_str = np.array(["neg", "pos"])[y]
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y_str)
        preds = clf.predict(X)
        assert set(preds) <= {"neg", "pos"}

    def test_result_and_ensemble_info(self, binary_classification_data, tiny_config):
        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        assert clf.result_.n_models >= tiny_config.min_models
        info = clf.get_ensemble_info()
        assert set(info.columns) == {"index", "name", "weight", "oof_score"}
        assert len(info) == clf.result_.n_models

    def test_custom_candidate_pool(self, binary_classification_data, tiny_config):
        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(
            candidate_estimators=[
                LogisticRegression(max_iter=200),
                DecisionTreeClassifier(max_depth=3, random_state=0),
                DecisionTreeClassifier(max_depth=6, random_state=1),
            ],
            config=tiny_config,
        )
        clf.fit(X, y)
        assert clf.result_.n_candidates == 3


class TestRegression:
    def test_fit_predict(self, regression_data, tiny_config):
        X, y = regression_data
        reg = EvoEnsembleRegressor(config=tiny_config)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape
        assert reg.score(X, y) > -1.0

    def test_custom_candidate_pool(self, regression_data, tiny_config):
        X, y = regression_data
        reg = EvoEnsembleRegressor(
            candidate_estimators=[
                Ridge(alpha=1.0),
                DecisionTreeRegressor(max_depth=3, random_state=0),
                DecisionTreeRegressor(max_depth=6, random_state=1),
            ],
            config=tiny_config,
        )
        reg.fit(X, y)
        assert reg.result_.n_candidates == 3


class TestMultiobjectiveMode:
    def test_classification_pareto_front(self, binary_classification_data, tiny_nsga2_config):
        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_nsga2_config)
        clf.fit(X, y)
        assert clf.result_.pareto_front is not None
        assert len(clf.result_.pareto_front) >= 1
        clf.predict(X)  # the selected solution must still be usable for prediction

    def test_regression_pareto_front(self, regression_data, tiny_nsga2_config):
        X, y = regression_data
        reg = EvoEnsembleRegressor(config=tiny_nsga2_config)
        reg.fit(X, y)
        assert reg.result_.pareto_front is not None
        reg.predict(X)


class TestSklearnIntegration:
    def test_pipeline(self, binary_classification_data, tiny_config):
        X, y = binary_classification_data
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("ensemble", EvoEnsembleClassifier(config=tiny_config))]
        )
        pipe.fit(X, y)
        assert pipe.predict(X).shape == y.shape

    def test_grid_search_cv(self, binary_classification_data):
        X, y = binary_classification_data
        param_grid = {
            "config": [
                EvoEnsConfig(
                    population_size=8,
                    n_generations=3,
                    min_models=1,
                    cv_folds=3,
                    diversity_beta=beta,
                    verbose=False,
                    random_seed=0,
                )
                for beta in (0.0, 0.2)
            ]
        }
        search = GridSearchCV(
            EvoEnsembleClassifier(), param_grid, cv=2, scoring="accuracy", error_score="raise"
        )
        search.fit(X, y)
        assert search.best_estimator_ is not None

    def test_clone_does_not_share_state(self, binary_classification_data, tiny_config):
        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        cloned = clone(clf)
        assert not hasattr(cloned, "result_")
        # clone() deep-copies non-estimator __init__ params (sklearn contract);
        # it must not carry over any fitted state.
        assert cloned.config.to_dict() == clf.config.to_dict()

    def test_config_is_not_mutated_by_fit(self, binary_classification_data, tiny_config):
        before = copy.deepcopy(tiny_config.to_dict())
        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        assert tiny_config.to_dict() == before

    def test_reproducibility_by_seed(self, binary_classification_data):
        X, y = binary_classification_data
        cfg = EvoEnsConfig(
            population_size=10,
            n_generations=4,
            min_models=1,
            cv_folds=3,
            random_seed=123,
            verbose=False,
        )
        clf1 = EvoEnsembleClassifier(config=copy.deepcopy(cfg))
        clf2 = EvoEnsembleClassifier(config=copy.deepcopy(cfg))
        clf1.fit(X, y)
        clf2.fit(X, y)
        np.testing.assert_allclose(clf1.result_.weights, clf2.result_.weights)
        assert clf1.result_.n_models == clf2.result_.n_models


class TestValidationErrors:
    def test_invalid_config_raises_on_fit(self, binary_classification_data):
        X, y = binary_classification_data
        # Bypass EvoEnsConfig's own validation to simulate external mutation.
        cfg = EvoEnsConfig(verbose=False)
        cfg.min_models = -1
        clf = EvoEnsembleClassifier(config=cfg)
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_predict_before_fit_raises(self, binary_classification_data):
        X, _ = binary_classification_data
        clf = EvoEnsembleClassifier()
        with pytest.raises(NotFittedError):
            clf.predict(X)
