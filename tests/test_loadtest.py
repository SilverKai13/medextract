"""The concurrency harness, driven in-process through an ASGI transport.

No server is started and no socket is opened: the real application runs under
httpx's ASGI transport, so the harness is exercised end to end against the real
routes with the tiny model. Numbers it produces here are meaningless; the shape
of what it produces, and how it behaves when the service misbehaves, are not.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from medextract.api.main import create_app
from medextract.loadtest import build_parser, run_loadtest, sample_corpus, to_markdown

TEXTS = [
    "Familial adenomatous polyposis with APC mutations.",
    "Acute hepatic failure after cisplatin.",
    "No abnormality.",
]


async def _drive(app, texts, **kwargs):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_loadtest(client, texts, **kwargs)


class TestRamp:
    def test_every_level_is_run_and_reports_percentiles(self, settings_for):
        app = create_app(settings_for())
        results = asyncio.run(_drive(app, TEXTS, levels=(1, 2), requests_per_level=6, warmup=1))

        assert set(results["levels"]) == {"1", "2"}
        for stats in results["levels"].values():
            assert stats["completed"] == 6
            assert stats["errors"] == 0
            assert stats["p50_ms"] > 0
            assert stats["p99_ms"] >= stats["p50_ms"]
            assert stats["throughput_req_per_s"] > 0
        assert results["payload"]["texts"] == len(TEXTS)
        assert results["backend"] == "torch"

    def test_concurrency_level_is_recorded_as_requested(self, settings_for):
        app = create_app(settings_for())
        results = asyncio.run(_drive(app, TEXTS, levels=(3,), requests_per_level=4, warmup=0))
        assert results["levels"]["3"]["concurrency"] == 3
        assert results["levels"]["3"]["requests"] == 4


class TestFailureModes:
    def test_a_degraded_service_is_refused_rather_than_timed(self, settings_for, tmp_path):
        app = create_app(settings_for(model_dir=tmp_path / "absent"))
        with pytest.raises(RuntimeError, match="degraded"):
            asyncio.run(_drive(app, TEXTS, levels=(1,), requests_per_level=2, warmup=0))

    def test_server_errors_are_counted_not_raised(self):
        """A 500 mid-run is a data point about the server, not a crash of the harness."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"model_loaded": True, "backend": "onnx"})
            return httpx.Response(500, json={"detail": "boom"})

        async def run():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await run_loadtest(
                    client, TEXTS, levels=(2,), requests_per_level=5, warmup=0
                )

        results = asyncio.run(run())
        level = results["levels"]["2"]
        assert level["errors"] == 5
        assert level["completed"] == 0
        assert level["error_kinds"] == ["HTTP 500"]
        assert "p50_ms" not in level


class TestReporting:
    def test_markdown_carries_levels_and_payload_description(self, settings_for):
        app = create_app(settings_for())
        results = asyncio.run(_drive(app, TEXTS, levels=(1, 2), requests_per_level=3, warmup=0))
        markdown = to_markdown(results)
        assert "| 1 | 3 |" in markdown
        assert "| 2 | 3 |" in markdown
        assert "Closed loop" in markdown
        assert f"{len(TEXTS)} distinct texts" in markdown

    def test_results_are_json_serialisable(self, settings_for):
        app = create_app(settings_for())
        results = asyncio.run(_drive(app, TEXTS, levels=(1,), requests_per_level=2, warmup=0))
        json.dumps(results)


class TestPayloadSampling:
    def test_sampling_is_seeded_and_bounded(self, monkeypatch, tmp_path):
        class Stub:
            def __init__(self, text):
                self.abstract = text

        corpus = [Stub(f"abstract {i}") for i in range(20)] + [Stub("")]
        monkeypatch.setattr("medextract.retrieval.pubmed.read_jsonl", lambda path: corpus)

        first = sample_corpus(tmp_path / "x.jsonl", 5, seed=1)
        second = sample_corpus(tmp_path / "x.jsonl", 5, seed=1)
        assert first == second
        assert len(first) == 5
        assert all(text for text in first)
        assert len(sample_corpus(tmp_path / "x.jsonl", 500, seed=1)) == 20

    def test_an_empty_corpus_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr("medextract.retrieval.pubmed.read_jsonl", lambda path: [])
        with pytest.raises(ValueError, match="no abstracts"):
            sample_corpus(tmp_path / "x.jsonl", 5)


def test_parser_defaults_ramp_to_sixteen():
    ns = build_parser().parse_args([])
    assert ns.levels == [1, 2, 4, 8, 16]
    assert ns.requests == 200
    assert ns.corpus is None
