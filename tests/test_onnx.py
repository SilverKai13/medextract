"""ONNX export and parity.

The point of these is that the exported graph is not silently a different
model. They run against the tiny random-weight fixture, so they verify the
export and comparison machinery rather than any quality claim -- the real
numbers come from running export_onnx.py on a fine-tuned checkpoint.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from medextract.export_onnx import PARITY_TEXTS, check_parity
from medextract.inference import NerEngine
from medextract.integrity import PARITY_FILE, IntegrityError, file_sha256

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_export_produces_a_graph(tiny_onnx_dir):
    assert (tiny_onnx_dir / "model.onnx").exists()
    assert (tiny_onnx_dir / "config.json").exists()


def test_onnx_engine_loads_and_predicts(tiny_onnx_dir):
    engine = NerEngine(tiny_onnx_dir, backend="onnx")
    text = "the patient has acute hepatic failure"
    for entity in engine.predict_one(text):
        assert text[entity.start : entity.end] == entity.text


def test_logits_match_within_tolerance(tiny_model_dir, tiny_onnx_dir):
    torch_engine = NerEngine(tiny_model_dir, backend="torch")
    onnx_engine = NerEngine(tiny_onnx_dir, backend="onnx")

    encoded = torch_engine.encode(PARITY_TEXTS)
    inputs = {
        k: np.asarray(v)
        for k, v in encoded.items()
        if k in {"input_ids", "attention_mask", "token_type_ids"}
    }

    torch_logits = torch_engine.logits(inputs)
    onnx_logits = onnx_engine.logits(inputs)

    assert torch_logits.shape == onnx_logits.shape
    np.testing.assert_allclose(onnx_logits, torch_logits, atol=1e-4, rtol=1e-3)


def test_check_parity_reports_and_passes(tiny_model_dir, tiny_onnx_dir):
    result = check_parity(tiny_model_dir, tiny_onnx_dir)
    assert result["entity_spans_match"] is True
    assert result["argmax_agreement"] == pytest.approx(1.0)
    assert result["max_abs_logit_diff"] < 1e-4


def test_check_parity_fails_when_tolerance_is_impossible(tiny_model_dir, tiny_onnx_dir):
    """The check must be capable of failing, not just of passing."""
    with pytest.raises(AssertionError):
        check_parity(tiny_model_dir, tiny_onnx_dir, atol=0.0, rtol=0.0)


def test_both_backends_decode_identical_entities(tiny_model_dir, tiny_onnx_dir):
    torch_engine = NerEngine(tiny_model_dir, backend="torch")
    onnx_engine = NerEngine(tiny_onnx_dir, backend="onnx")

    for text in PARITY_TEXTS:
        a = [(e.start, e.end, e.label) for e in torch_engine.predict_one(text)]
        b = [(e.start, e.end, e.label) for e in onnx_engine.predict_one(text)]
        assert a == b


def test_benchmark_runs_and_reports_percentiles(tiny_model_dir, tiny_onnx_dir):
    from medextract.benchmark import run_benchmark, to_markdown

    results = run_benchmark(
        tiny_model_dir,
        tiny_onnx_dir,
        batch_sizes=(1, 4),
        iterations=3,
        warmup=1,
    )

    for backend in ("pytorch_eager", "onnxruntime"):
        for batch in ("1", "4"):
            stats = results["backends"][backend][batch]
            assert stats["p50_ms"] > 0
            assert stats["p99_ms"] >= stats["p50_ms"]
            assert stats["max_ms"] >= stats["min_ms"]

    assert set(results["speedup_p50"]) == {"1", "4"}
    assert "throughput" in to_markdown(results)


class TestArtefactIntegrity:
    """The graph that is served is the graph the parity check passed."""

    def test_the_parity_record_carries_the_graph_hash(self, tiny_model_dir, tiny_onnx_dir):
        result = check_parity(tiny_model_dir, tiny_onnx_dir)
        record = json.loads((tiny_onnx_dir / PARITY_FILE).read_text(encoding="utf-8"))
        graph = tiny_onnx_dir / "model.onnx"
        assert record["model_sha256"] == result["model_sha256"] == file_sha256(graph)
        assert record["model_bytes"] == graph.stat().st_size
        assert "torch" in record["exported_with"]

    def test_a_failed_parity_check_writes_no_record(self, tiny_model_dir, tiny_onnx_dir, tmp_path):
        fresh = tmp_path / "onnx"
        shutil.copytree(tiny_onnx_dir, fresh)
        (fresh / PARITY_FILE).unlink(missing_ok=True)
        with pytest.raises(AssertionError):
            check_parity(tiny_model_dir, fresh, atol=0.0, rtol=0.0)
        assert not (fresh / PARITY_FILE).exists()

    def test_the_engine_verifies_the_graph_against_the_record(self, tiny_model_dir, tiny_onnx_dir):
        check_parity(tiny_model_dir, tiny_onnx_dir)
        engine = NerEngine(tiny_onnx_dir, backend="onnx")
        assert engine.integrity_verified is True
        assert engine.model_sha256 == file_sha256(tiny_onnx_dir / "model.onnx")

    def test_a_tampered_graph_is_refused_at_load(self, tiny_model_dir, tiny_onnx_dir, tmp_path):
        check_parity(tiny_model_dir, tiny_onnx_dir)
        tampered = tmp_path / "onnx"
        shutil.copytree(tiny_onnx_dir, tampered)
        with (tampered / "model.onnx").open("ab") as handle:
            handle.write(b"\0")
        with pytest.raises(IntegrityError, match="does not match"):
            NerEngine(tampered, backend="onnx")

    def test_a_graph_without_a_record_loads_but_unverified(self, tiny_onnx_dir, tmp_path):
        bare = tmp_path / "onnx"
        shutil.copytree(tiny_onnx_dir, bare)
        (bare / PARITY_FILE).unlink(missing_ok=True)
        engine = NerEngine(bare, backend="onnx")
        assert engine.integrity_verified is False
        assert len(engine.model_sha256) == 64

    def test_the_torch_backend_has_nothing_to_verify(self, tiny_model_dir):
        engine = NerEngine(tiny_model_dir, backend="torch")
        assert engine.model_sha256 is None
        assert engine.integrity_verified is False
