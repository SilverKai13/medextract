"""Tests for the pipeline orchestrator's decision points.

The pipeline's job is sequencing already-tested phase functions, so what's
worth pinning here is the logic specific to `pipeline.py`: which arm gets
picked, whether a stage is skipped or re-run, and Docker detection -- not the
phases themselves, which have their own tests.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from medextract.pipeline import (
    docker_available,
    maybe_fetch_pubmed,
    maybe_run_sweep,
    select_best_arm,
    update_env_var,
)


class TestSelectBestArm:
    def test_picks_the_highest_validation_f1(self):
        results = [
            {"run_name": "a", "validation": {"eval_f1": 0.80}, "test": {"test_f1": 0.79}},
            {"run_name": "b", "validation": {"eval_f1": 0.91}, "test": {"test_f1": 0.88}},
            {"run_name": "c", "validation": {"eval_f1": 0.85}, "test": {"test_f1": 0.90}},
        ]
        assert select_best_arm(results)["run_name"] == "b"

    def test_ties_are_broken_by_test_f1(self):
        results = [
            {"run_name": "a", "validation": {"eval_f1": 0.90}, "test": {"test_f1": 0.80}},
            {"run_name": "b", "validation": {"eval_f1": 0.90}, "test": {"test_f1": 0.95}},
        ]
        assert select_best_arm(results)["run_name"] == "b"

    def test_empty_results_raise_with_a_clear_message(self):
        with pytest.raises(SystemExit, match="Every sweep arm failed"):
            select_best_arm([])


class TestMaybeRunSweep:
    def _arms(self):
        from medextract.train import RunConfig

        return [RunConfig(run_name="arm1"), RunConfig(run_name="arm2")]

    def test_skips_when_every_arm_is_already_in_the_results_file(self, tmp_path, monkeypatch):
        output_root = tmp_path / "sweep"
        output_root.mkdir()
        existing = [
            {"run_name": "arm1", "validation": {"eval_f1": 0.9}, "test": {"test_f1": 0.9}},
            {"run_name": "arm2", "validation": {"eval_f1": 0.8}, "test": {"test_f1": 0.8}},
        ]
        (output_root / "sweep_results.json").write_text(json.dumps(existing), encoding="utf-8")

        called = {"n": 0}

        def fake_run_sweep(*args, **kwargs):
            called["n"] += 1
            return []

        monkeypatch.setattr("medextract.sweep.run_sweep", fake_run_sweep)

        results = maybe_run_sweep(self._arms(), output_root, "exp", force_retrain=False)
        assert called["n"] == 0
        assert results == existing

    def test_runs_when_the_results_file_is_missing_arms(self, tmp_path, monkeypatch):
        output_root = tmp_path / "sweep"
        output_root.mkdir()
        (output_root / "sweep_results.json").write_text(
            json.dumps([{"run_name": "arm1", "validation": {}, "test": {}}]),
            encoding="utf-8",
        )

        called = {"n": 0}

        def fake_run_sweep(arms, root, experiment):
            called["n"] += 1
            return [{"run_name": a.run_name} for a in arms]

        monkeypatch.setattr("medextract.sweep.run_sweep", fake_run_sweep)

        results = maybe_run_sweep(self._arms(), output_root, "exp", force_retrain=False)
        assert called["n"] == 1
        assert {r["run_name"] for r in results} == {"arm1", "arm2"}

    def test_force_retrain_reruns_even_if_the_file_is_complete(self, tmp_path, monkeypatch):
        output_root = tmp_path / "sweep"
        output_root.mkdir()
        (output_root / "sweep_results.json").write_text(
            json.dumps(
                [
                    {"run_name": "arm1", "validation": {}, "test": {}},
                    {"run_name": "arm2", "validation": {}, "test": {}},
                ]
            ),
            encoding="utf-8",
        )

        called = {"n": 0}

        def fake_run_sweep(arms, root, experiment):
            called["n"] += 1
            return [{"run_name": a.run_name} for a in arms]

        monkeypatch.setattr("medextract.sweep.run_sweep", fake_run_sweep)

        maybe_run_sweep(self._arms(), output_root, "exp", force_retrain=True)
        assert called["n"] == 1


class TestDockerAvailable:
    def test_false_when_docker_is_not_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert docker_available() is False

    def test_false_when_the_daemon_does_not_respond(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "docker info")

        monkeypatch.setattr("subprocess.run", fake_run)
        assert docker_available() is False

    def test_true_when_docker_info_succeeds(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
        assert docker_available() is True


class TestUpdateEnvVar:
    def test_appends_a_new_key(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("MEDEXTRACT_MODEL_DIR=artifacts/onnx\n", encoding="utf-8")
        update_env_var(path, "MEDEXTRACT_QDRANT_URL", "http://localhost:6333")
        text = path.read_text(encoding="utf-8")
        assert "MEDEXTRACT_MODEL_DIR=artifacts/onnx" in text
        assert "MEDEXTRACT_QDRANT_URL=http://localhost:6333" in text

    def test_overwrites_an_existing_key_in_place(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("MEDEXTRACT_QDRANT_URL=old\nMEDEXTRACT_LOG_LEVEL=INFO\n", encoding="utf-8")
        update_env_var(path, "MEDEXTRACT_QDRANT_URL", "new")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert "MEDEXTRACT_QDRANT_URL=new" in lines
        assert "MEDEXTRACT_LOG_LEVEL=INFO" in lines
        assert len(lines) == 2

    def test_creates_the_file_if_absent_and_no_example_to_copy(self, tmp_path, monkeypatch):
        monkeypatch.setattr("medextract.pipeline.REPO_ROOT", tmp_path)
        path = tmp_path / ".env"
        update_env_var(path, "MEDEXTRACT_QDRANT_URL", "http://localhost:6333")
        assert "MEDEXTRACT_QDRANT_URL=http://localhost:6333" in path.read_text(encoding="utf-8")


class TestMaybeFetchPubmed:
    def test_skips_when_the_dump_already_exists(self, tmp_path, monkeypatch):
        from medextract.retrieval.pubmed import Abstract, write_jsonl

        out_path = tmp_path / "pubmed.jsonl"
        write_jsonl([Abstract("1", "t", "a")], out_path)

        called = {"n": 0}
        monkeypatch.setattr(
            "medextract.pipeline.fetch_pubmed_abstracts",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1) or 0,
        )

        count = maybe_fetch_pubmed("query", 10, out_path, force_fetch=False)
        assert called["n"] == 0
        assert count == 1

    def test_force_fetch_refetches_even_if_present(self, tmp_path, monkeypatch):
        from medextract.retrieval.pubmed import Abstract, write_jsonl

        out_path = tmp_path / "pubmed.jsonl"
        write_jsonl([Abstract("1", "t", "a")], out_path)

        monkeypatch.setattr("medextract.pipeline.fetch_pubmed_abstracts", lambda *a, **k: 5)

        assert maybe_fetch_pubmed("query", 10, out_path, force_fetch=True) == 5
