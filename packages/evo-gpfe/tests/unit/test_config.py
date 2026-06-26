"""Unit tests for GP configuration and result dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from evo_gpfe.core.config import GeneratedFeature, GPConfig, GPEngineeringResult


class TestGPConfig:
    def test_defaults_valid(self):
        config = GPConfig()
        assert config.function_set == "extended"
        assert config.to_dict()["population_size"] == 300

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"function_set": "nope"},
            {"fitness_metric": "nope"},
            {"task_type": "nope"},
            {"redundancy_beta": 1.5},
            {"redundancy_threshold": -0.1},
            {"parsimony_coeff": -1.0},
            {"n_features_to_generate": 0},
            {"population_size": 1},
            {"n_generations": 0},
            {"tournament_size": 1},
            {"elite_size": -1},
            {"crossover_prob": 0.0},
            {"mutation_prob": 1.5},
            {"init_depth_min": 5, "init_depth_max": 3},
            {"init_depth_max": 10, "max_tree_height": 4},
        ],
    )
    def test_invalid_raises(self, kwargs):
        with pytest.raises(ValueError):
            GPConfig(**kwargs)

    def test_mutation_split_normalises(self):
        config = GPConfig(p_subtree_mutation=2.0, p_hoist_mutation=1.0, p_point_mutation=1.0)
        split = config.mutation_split()
        assert sum(split) == pytest.approx(1.0)
        assert split[0] == pytest.approx(0.5)

    def test_mutation_all_zero_raises(self):
        with pytest.raises(ValueError):
            GPConfig(p_subtree_mutation=0.0, p_hoist_mutation=0.0, p_point_mutation=0.0)


def _feature(idx: int = 0) -> GeneratedFeature:
    return GeneratedFeature(
        expression="mul(x0, x1)",
        fitness=0.5,
        mi_score=0.42,
        redundancy_score=0.1,
        n_nodes=3,
        height=1,
        feature_index=idx,
        generation_found=4,
        values=np.array([1.0, 2.0, 3.0]),
    )


def _result() -> GPEngineeringResult:
    return GPEngineeringResult(
        generated_features=[_feature(0), _feature(1)],
        X_original_shape=(100, 5),
        X_transformed_shape=(100, 7),
        baseline_cv_score=0.80,
        augmented_cv_score=0.86,
        score_improvement=0.06,
        total_time=3.2,
        n_evaluations=1234,
        task_type="classification",
        scoring="accuracy",
        feature_names_original=["a", "b"],
        feature_names_generated=["gp_0", "gp_1"],
        config=GPConfig(verbose=False),
    )


class TestGPEngineeringResult:
    def test_summary_mentions_expression(self):
        text = _result().summary()
        assert "mul(x0, x1)" in text
        assert "Generated features   : 2" in text

    def test_to_json_serialisable(self):
        import json

        payload = _result().to_json()
        assert payload["generated_features"][0]["expression"] == "mul(x0, x1)"
        json.dumps(payload)  # must not raise

    def test_save_load_roundtrip(self, tmp_path):
        result = _result()
        path = tmp_path / "gp.pkl"
        result.save(path)
        loaded = GPEngineeringResult.load(path)
        assert len(loaded.generated_features) == 2
        assert loaded.augmented_cv_score == result.augmented_cv_score

    def test_save_json(self, tmp_path):
        path = tmp_path / "gp.json"
        _result().save_json(path)
        assert path.exists()
