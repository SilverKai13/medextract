"""API tests: happy path, empty input, oversized input, malformed JSON."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medextract.api.main import REQUEST_ID_HEADER, create_app


class TestHealth:
    def test_reports_ok_with_a_model_loaded(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "B-Disease" in body["labels"]
        assert body["uptime_seconds"] >= 0

    def test_reports_degraded_without_a_model(self, settings_for, tmp_path):
        """A missing model must not stop the container answering /health."""
        with TestClient(create_app(settings_for(model_dir=tmp_path / "absent"))) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["model_loaded"] is False


class TestExtract:
    def test_happy_path(self, client):
        text = "The patient has familial adenomatous polyposis."
        response = client.post("/extract", json={"text": text})
        assert response.status_code == 200

        body = response.json()
        assert body["text"] == text
        assert body["model_version"] == "test"
        assert body["took_ms"] >= 0
        for entity in body["entities"]:
            assert text[entity["start"] : entity["end"]] == entity["text"]
            assert 0.0 <= entity["score"] <= 1.0

    def test_empty_text_is_rejected(self, client):
        response = client.post("/extract", json={"text": ""})
        assert response.status_code == 422

    def test_missing_field_is_rejected(self, client):
        assert client.post("/extract", json={}).status_code == 422

    def test_malformed_json_is_rejected(self, client):
        response = client.post(
            "/extract",
            content=b"{not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_oversized_input_is_rejected_with_413(self, settings_for):
        with TestClient(create_app(settings_for(max_input_chars=100))) as client:
            response = client.post("/extract", json={"text": "word " * 200})
        assert response.status_code == 413
        assert "limit is 100" in response.json()["detail"]

    def test_min_score_filters_results(self, client):
        text = "acute hepatic failure with cisplatin"
        unfiltered = client.post("/extract", json={"text": text}).json()["entities"]
        filtered = client.post("/extract", json={"text": text, "min_score": 1.0}).json()["entities"]
        assert len(filtered) <= len(unfiltered)

    def test_min_score_out_of_range_is_rejected(self, client):
        assert client.post("/extract", json={"text": "x", "min_score": 1.5}).status_code == 422

    def test_returns_503_when_the_model_is_missing(self, settings_for, tmp_path):
        with TestClient(create_app(settings_for(model_dir=tmp_path / "absent"))) as client:
            response = client.post("/extract", json={"text": "fever"})
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_unicode_offsets_are_character_offsets(self, client):
        """Offsets must index Python characters, not UTF-8 bytes."""
        # Greek and accented characters are the point of the test, so the
        # ambiguous-character lint is suppressed rather than the text changed.
        text = "Patient α-thalassaemia — no fever. Café."  # noqa: RUF001
        body = client.post("/extract", json={"text": text}).json()
        for entity in body["entities"]:
            assert text[entity["start"] : entity["end"]] == entity["text"]


class TestBatchExtract:
    def test_returns_one_result_per_text(self, client):
        texts = ["acute renal failure", "no abnormality", "cisplatin therapy"]
        body = client.post("/extract/batch", json={"texts": texts}).json()
        assert len(body["results"]) == len(texts)

    def test_empty_list_is_rejected(self, client):
        assert client.post("/extract/batch", json={"texts": []}).status_code == 422

    def test_batch_matches_single_calls(self, client):
        texts = ["the patient has carcinoma", "acute hepatic failure"]
        batched = client.post("/extract/batch", json={"texts": texts}).json()["results"]
        singles = [client.post("/extract", json={"text": t}).json()["entities"] for t in texts]
        assert batched == singles


class TestSearch:
    def test_returns_503_when_retrieval_is_off(self, client):
        response = client.post("/search", json={"query": "lung cancer"})
        assert response.status_code == 503
        assert "Retrieval unavailable" in response.json()["detail"]

    def test_top_k_bounds_are_enforced(self, client):
        assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
        assert client.post("/search", json={"query": "x", "top_k": 999}).status_code == 422


class TestRequestContext:
    def test_request_id_is_returned(self, client):
        assert client.get("/health").headers[REQUEST_ID_HEADER]

    def test_caller_supplied_request_id_is_echoed(self, client):
        response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"

    def test_request_ids_differ_between_calls(self, client):
        first = client.get("/health").headers[REQUEST_ID_HEADER]
        second = client.get("/health").headers[REQUEST_ID_HEADER]
        assert first != second


class TestOpenAPI:
    def test_schema_documents_the_endpoints(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/extract", "/extract/batch", "/search", "/health"} <= set(paths)


@pytest.mark.parametrize("path", ["/extract", "/extract/batch", "/search"])
def test_get_is_not_allowed_on_post_routes(client, path):
    assert client.get(path).status_code == 405
