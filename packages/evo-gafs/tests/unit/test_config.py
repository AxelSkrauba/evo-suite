"""Unit tests for configuration and result dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from evo_gafs import GAConfig, SelectionResult


class TestGAConfig:
    def test_defaults_are_valid(self):
        config = GAConfig()
        assert config.mode == "single"
        assert config.to_dict()["population_size"] == 50

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "invalid"},
            {"alpha": 1.5},
            {"alpha": -0.1},
            {"crossover_prob": 0.0},
            {"crossover_prob": 1.5},
            {"mutation_prob": 0.0},
            {"mutation_indpb": 0.0},
            {"population_size": 1},
            {"n_generations": 0},
            {"tournament_size": 1},
            {"cv_folds": 1},
            {"min_features": 0},
            {"elite_size": -1},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValueError):
            GAConfig(**kwargs)

    def test_validate_is_idempotent(self):
        config = GAConfig()
        config.validate()  # should not raise


class TestSelectionResult:
    def _make(self):
        return SelectionResult(
            selected_mask=np.array([True, False, True]),
            selected_indices=np.array([0, 2]),
            selected_feature_names=["a", "c"],
            best_fitness=0.9,
            best_cv_score=0.85,
            n_selected=2,
            compression_ratio=1 / 3,
            config=GAConfig(verbose=False),
            total_time=1.23,
            n_evaluations=42,
        )

    def test_summary_contains_key_numbers(self):
        text = self._make().summary()
        assert "Selected features   : 2" in text
        assert "0.8500" in text

    def test_to_json_is_serialisable(self):
        import json

        payload = self._make().to_json()
        assert payload["n_selected"] == 2
        assert payload["selected_feature_names"] == ["a", "c"]
        json.dumps(payload)  # must not raise

    def test_save_and_load_roundtrip(self, tmp_path):
        result = self._make()
        path = tmp_path / "result.pkl"
        result.save(path)
        loaded = SelectionResult.load(path)
        assert loaded.n_selected == result.n_selected
        assert np.array_equal(loaded.selected_mask, result.selected_mask)

    def test_save_json(self, tmp_path):
        path = tmp_path / "result.json"
        self._make().save_json(path)
        assert path.exists()
