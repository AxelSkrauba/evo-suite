"""Unit tests for evo_ens.core.config dataclasses."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evo_ens.core.config import (
    EnsembleMember,
    EvoEnsConfig,
    EvoEnsResult,
    EvolutionStats,
    ParetoSolution,
)


class TestEvoEnsConfigValidation:
    def test_defaults_are_valid(self):
        cfg = EvoEnsConfig()
        cfg.validate()  # should not raise

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "bogus"},
            {"weight_method": "bogus"},
            {"diversity_beta": -0.1},
            {"min_models": 0},
            {"max_models": 1, "min_models": 2},
            {"cv_folds": 1},
            {"population_size": 1},
            {"n_generations": 0},
            {"tournament_size": 1},
            {"elite_size": -1},
            {"crossover_prob": 0.0},
            {"crossover_prob": 1.5},
            {"mutation_prob": -0.1},
            {"mutation_sigma": -1.0},
            {"mutation_indpb": 1.5},
            {"early_stopping_rounds": 0},
        ],
    )
    def test_invalid_configs_raise(self, kwargs):
        with pytest.raises(ValueError):
            EvoEnsConfig(**kwargs)

    def test_resolved_mutation_indpb_default(self):
        cfg = EvoEnsConfig(mutation_indpb=None)
        assert cfg.resolved_mutation_indpb(10) == pytest.approx(1.0 / 20)

    def test_resolved_mutation_indpb_explicit(self):
        cfg = EvoEnsConfig(mutation_indpb=0.3)
        assert cfg.resolved_mutation_indpb(10) == 0.3

    def test_to_dict_roundtrip(self):
        cfg = EvoEnsConfig(population_size=42)
        d = cfg.to_dict()
        assert d["population_size"] == 42
        assert EvoEnsConfig(**d).population_size == 42

    def test_config_not_mutated_by_validate(self):
        cfg = EvoEnsConfig()
        before = cfg.to_dict()
        cfg.validate()
        assert cfg.to_dict() == before


class TestResultDataclasses:
    def _make_result(self, tmp_path):
        member = EnsembleMember(
            name="00:RandomForestClassifier(n=100)",
            estimator=None,
            weight=0.6,
            candidate_index=0,
            oof_score=0.95,
        )
        cfg = EvoEnsConfig(population_size=10, n_generations=2, verbose=False)
        history = [
            EvolutionStats(
                generation=0,
                best_fitness=0.9,
                mean_fitness=0.8,
                std_fitness=0.05,
                best_n_models=2,
                mean_n_models=1.5,
                elapsed_time=0.01,
            )
        ]
        return EvoEnsResult(
            members=[member],
            n_models=1,
            weights=np.array([1.0]),
            oof_score_ensemble=0.96,
            oof_score_best_single=0.95,
            oof_score_improvement=0.01,
            diversity_score=0.2,
            compression_ratio=0.9,
            scoring="accuracy",
            task_type="classification",
            n_candidates=10,
            history=history,
            total_time=1.23,
            precompute_time=1.0,
            n_evaluations=50,
            config=cfg,
        )

    def test_summary_contains_key_metrics(self, tmp_path):
        result = self._make_result(tmp_path)
        summary = result.summary()
        assert "0.9600" in summary
        assert "RandomForestClassifier" in summary

    def test_summary_mentions_pareto_front_when_present(self, tmp_path):
        result = self._make_result(tmp_path)
        result.pareto_front = [
            ParetoSolution(score=0.9, compression=0.5, n_models=2, members=[], weights=np.array([]))
        ]
        assert "Pareto front" in result.summary()

    def test_to_json_is_serializable(self, tmp_path):
        result = self._make_result(tmp_path)
        payload = result.to_json()
        json.dumps(payload)  # should not raise
        assert payload["n_models"] == 1
        assert payload["config"]["population_size"] == 10

    def test_save_json_writes_file(self, tmp_path):
        result = self._make_result(tmp_path)
        path = tmp_path / "result.json"
        result.save_json(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["task_type"] == "classification"

    def test_save_and_load_pickle_roundtrip(self, tmp_path):
        result = self._make_result(tmp_path)
        path = tmp_path / "result.pkl"
        result.save(path)
        loaded = EvoEnsResult.load(path)
        assert loaded.oof_score_ensemble == result.oof_score_ensemble
        assert loaded.members[0].name == result.members[0].name
