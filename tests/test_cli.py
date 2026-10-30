"""Tests for argument parsing, the sweep grid and config.

The training and indexing entry points cannot run offline, but the parts of
them that decide *what* will run -- flag wiring and the sweep definition -- are
pure and worth pinning. A typo in a sweep arm otherwise shows up six GPU-hours
later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medextract.config import ALT_BASE_MODEL, DEFAULT_BASE_MODEL, Settings
from medextract.report import _fmt
from medextract.sweep import default_grid
from medextract.train import RunConfig, build_parser


class TestTrainParser:
    def test_defaults_are_sane(self):
        ns = build_parser().parse_args([])
        assert ns.base_model == DEFAULT_BASE_MODEL
        assert ns.learning_rate == pytest.approx(3e-5)
        assert ns.batch_size == 16
        assert ns.label_all_subwords is False

    def test_flags_reach_the_run_config(self):
        ns = build_parser().parse_args(
            [
                "--lr",
                "5e-5",
                "--batch-size",
                "32",
                "--epochs",
                "2",
                "--seed",
                "7",
                "--label-all-subwords",
            ]
        )
        config = RunConfig(
            learning_rate=ns.learning_rate,
            batch_size=ns.batch_size,
            epochs=ns.epochs,
            seed=ns.seed,
            label_all_subwords=ns.label_all_subwords,
        )
        assert (config.learning_rate, config.batch_size, config.epochs, config.seed) == (
            5e-5,
            32,
            2.0,
            7,
        )
        assert config.label_all_subwords is True

    def test_output_dir_is_a_path(self):
        ns = build_parser().parse_args(["--output-dir", "/tmp/somewhere"])
        assert isinstance(ns.output_dir, Path)


class TestSweepGrid:
    @pytest.fixture
    def arms(self):
        return default_grid(RunConfig())

    def test_meets_the_six_configuration_minimum(self, arms):
        assert len(arms) >= 6

    def test_every_arm_is_named_and_names_are_unique(self, arms):
        names = [a.run_name for a in arms]
        assert all(names)
        assert len(names) == len(set(names))

    def test_learning_rate_is_varied(self, arms):
        assert len({a.learning_rate for a in arms}) >= 3

    def test_batch_size_and_epochs_are_varied(self, arms):
        assert len({a.batch_size for a in arms}) >= 2
        assert len({a.epochs for a in arms}) >= 2

    def test_both_subword_conventions_are_covered(self, arms):
        assert {a.label_all_subwords for a in arms} == {True, False}

    def test_both_base_checkpoints_are_covered(self, arms):
        assert {DEFAULT_BASE_MODEL, ALT_BASE_MODEL} <= {a.base_model for a in arms}

    def test_seed_repeats_exist_so_variance_is_visible(self, arms):
        """At least one configuration must appear under more than one seed."""
        without_seed: dict[tuple, set[int]] = {}
        for arm in arms:
            key = (
                arm.base_model,
                arm.learning_rate,
                arm.batch_size,
                arm.epochs,
                arm.label_all_subwords,
            )
            without_seed.setdefault(key, set()).add(arm.seed)
        assert any(len(seeds) >= 3 for seeds in without_seed.values())


class TestReportFormatting:
    def test_probabilities_keep_four_decimals(self):
        assert _fmt(0.87654321) == "0.8765"

    def test_nan_renders_as_a_dash(self):
        assert _fmt(float("nan")) == "-"

    def test_none_renders_as_a_dash(self):
        assert _fmt(None) == "-"

    def test_large_numbers_are_not_padded(self):
        assert _fmt(16.0) == "16"

    def test_model_paths_are_shortened_to_the_checkpoint_name(self):
        assert _fmt("dmis-lab/biobert-base-cased-v1.2") == "biobert-base-cased-v1.2"


class TestSettings:
    def test_environment_overrides_defaults(self, monkeypatch):
        monkeypatch.setenv("MEDEXTRACT_BACKEND", "torch")
        monkeypatch.setenv("MEDEXTRACT_MAX_INPUT_CHARS", "500")
        settings = Settings()
        assert settings.backend == "torch"
        assert settings.max_input_chars == 500

    def test_unknown_environment_variables_are_ignored(self, monkeypatch):
        monkeypatch.setenv("MEDEXTRACT_NOT_A_REAL_SETTING", "x")
        assert Settings().backend in {"onnx", "torch"}

    def test_paths_are_coerced(self):
        assert isinstance(Settings(model_dir="/tmp/m").model_dir, Path)


class TestIndexParser:
    def test_fallback_is_off_by_default(self):
        from medextract.retrieval.build_index import build_parser as index_parser

        ns = index_parser().parse_args([])
        assert ns.allow_hashing_fallback is False
        assert ns.recreate is False

    def test_fallback_can_be_opted_into(self):
        from medextract.retrieval.build_index import build_parser as index_parser

        assert index_parser().parse_args(["--allow-hashing-fallback"]).allow_hashing_fallback


class TestExportParser:
    def test_torch_is_the_default_exporter(self):
        from medextract.export_onnx import build_parser as export_parser

        ns = export_parser().parse_args([])
        assert ns.exporter == "torch"
        assert ns.skip_parity is False

    def test_unknown_exporter_is_rejected(self):
        from medextract.export_onnx import build_parser as export_parser

        with pytest.raises(SystemExit):
            export_parser().parse_args(["--exporter", "tensorrt"])

    def test_export_rejects_an_unknown_exporter_at_call_time(self, tiny_model_dir, tmp_path):
        from medextract.export_onnx import export

        with pytest.raises(ValueError, match="Unknown exporter"):
            export(tiny_model_dir, tmp_path, exporter="nope")


class TestPubMedParser:
    def test_default_query_is_restricted_to_records_with_abstracts(self):
        from medextract.retrieval.pubmed import DEFAULT_QUERY

        assert "hasabstract" in DEFAULT_QUERY

    def test_max_and_out_flags(self):
        from medextract.retrieval.pubmed import build_parser as pubmed_parser

        ns = pubmed_parser().parse_args(["--max", "50", "--out", "/tmp/a.jsonl"])
        assert ns.max_results == 50
        assert ns.out == Path("/tmp/a.jsonl")


class TestPerTypeReport:
    def test_single_type_experiments_get_no_extra_columns(self):
        from medextract.report import per_type_columns

        assert per_type_columns(["metrics.eval_f1", "metrics.eval_Disease_f1"]) == []

    def test_the_macro_aggregate_is_not_mistaken_for_a_type(self):
        from medextract.report import per_type_columns

        logged = ["metrics.eval_f1", "metrics.eval_Disease_f1", "metrics.eval_macro_f1"]
        assert per_type_columns(logged) == []

    def test_two_types_get_validation_and_test_columns(self):
        from medextract.report import per_type_columns

        columns = per_type_columns(
            [
                "metrics.eval_f1",
                "metrics.eval_Disease_f1",
                "metrics.eval_Chemical_f1",
                "metrics.test_Chemical_f1",
                "metrics.test_Disease_f1",
            ]
        )
        assert [header for _, header in columns] == [
            "val F1 Chemical",
            "val F1 Disease",
            "test F1 Chemical",
            "test F1 Disease",
        ]

    def test_rendered_table_adds_type_and_restriction_columns(self):
        pandas = pytest.importorskip("pandas")

        from medextract.report import render_table

        runs = pandas.DataFrame(
            [
                {
                    "tags.mlflow.runName": "combined",
                    "params.base_model": "x/biobert",
                    "params.learning_rate": "3e-05",
                    "params.batch_size": "16",
                    "params.epochs": "3",
                    "params.seed": "42",
                    "params.label_all_subwords": "False",
                    "params.keep_types": "all",
                    "metrics.eval_precision": 0.9,
                    "metrics.eval_recall": 0.8,
                    "metrics.eval_f1": 0.85,
                    "metrics.test_f1": 0.84,
                    "metrics.eval_Disease_f1": 0.82,
                    "metrics.eval_Chemical_f1": 0.91,
                    "metrics.test_Disease_f1": 0.81,
                    "metrics.test_Chemical_f1": 0.9,
                },
                {
                    "tags.mlflow.runName": "combined_disease",
                    "params.base_model": "x/biobert",
                    "params.learning_rate": "3e-05",
                    "params.batch_size": "16",
                    "params.epochs": "3",
                    "params.seed": "42",
                    "params.label_all_subwords": "False",
                    "params.keep_types": "Disease",
                    "metrics.eval_precision": 0.9,
                    "metrics.eval_recall": 0.8,
                    "metrics.eval_f1": 0.83,
                    "metrics.test_f1": 0.82,
                    "metrics.eval_Disease_f1": 0.83,
                    "metrics.eval_Chemical_f1": float("nan"),
                    "metrics.test_Disease_f1": 0.82,
                    "metrics.test_Chemical_f1": float("nan"),
                },
            ]
        )
        table = render_table(runs)
        header = table.splitlines()[0]
        assert "| types |" in header
        assert "| val F1 Chemical |" in header
        assert "| test F1 Disease |" in header
        assert "| combined_disease | biobert |" in table
        assert "| Disease |" in table
        assert " - |" in table  # the NaN chemical score of the Disease-only arm


class TestKeepTypes:
    def test_train_flag_reaches_the_run_config(self):
        from medextract.train import build_parser

        ns = build_parser().parse_args(["--keep-types", "Disease", "Chemical"])
        assert ns.keep_types == ["Disease", "Chemical"]

    def test_sweep_flag_is_optional(self):
        from medextract.sweep import build_parser

        assert build_parser().parse_args([]).keep_types is None
        assert build_parser().parse_args(["--keep-types", "Disease"]).keep_types == ["Disease"]

    def test_restricted_arms_are_renamed_after_their_types(self):
        from medextract.sweep import default_grid, suffix_arms
        from medextract.train import RunConfig

        arms = suffix_arms(default_grid(RunConfig()), ("Disease",))
        assert all(arm.run_name.endswith("_disease") for arm in arms)
        assert len({arm.run_name for arm in arms}) == len(arms)
