"""Retrieval tests.

E-utilities is mocked with recorded-shape XML and Qdrant runs in its embedded
in-memory mode, so the whole retrieval path is exercised without touching the
network or a container.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from medextract.retrieval.embed import HashingEmbedder, l2_normalise
from medextract.retrieval.index import (
    build_client,
    ensure_collection,
    index_abstracts,
    pmid_to_point_id,
)
from medextract.retrieval.pubmed import (
    Abstract,
    PubMedClient,
    parse_pubmed_xml,
    read_jsonl,
    write_jsonl,
)
from medextract.retrieval.search import Searcher

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>EGFR mutations in lung adenocarcinoma</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">EGFR mutations drive tumour growth.</AbstractText>
          <AbstractText Label="RESULTS">Gefitinib improved survival.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Journal of Oncology</Title>
          <JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <ArticleTitle>Cisplatin nephrotoxicity</ArticleTitle>
        <Abstract><AbstractText>Renal injury followed cisplatin therapy.</AbstractText></Abstract>
        <Journal><Title>Nephrology</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33333333</PMID>
      <Article><ArticleTitle>Editorial with no abstract</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestParsing:
    def test_records_without_abstracts_are_dropped(self):
        parsed = parse_pubmed_xml(SAMPLE_XML)
        assert [a.pmid for a in parsed] == ["12345678", "22222222"]

    def test_structured_abstract_sections_are_joined_with_labels(self):
        first = parse_pubmed_xml(SAMPLE_XML)[0]
        assert "BACKGROUND: EGFR mutations drive tumour growth." in first.abstract
        assert "RESULTS: Gefitinib improved survival." in first.abstract

    def test_metadata_is_captured(self):
        first = parse_pubmed_xml(SAMPLE_XML)[0]
        assert first.title == "EGFR mutations in lung adenocarcinoma"
        assert first.journal == "Journal of Oncology"
        assert first.year == 2019

    def test_missing_year_is_none_not_a_crash(self):
        assert parse_pubmed_xml(SAMPLE_XML)[1].year is None

    def test_text_property_joins_title_and_abstract(self):
        first = parse_pubmed_xml(SAMPLE_XML)[0]
        assert first.text.startswith("EGFR mutations in lung adenocarcinoma")

    def test_empty_article_set_parses_to_nothing(self):
        assert parse_pubmed_xml("<PubmedArticleSet></PubmedArticleSet>") == []


class TestClient:
    def _client(self, handler):
        transport = httpx.MockTransport(handler)
        return PubMedClient(email="test@example.com", client=httpx.Client(transport=transport))

    def test_search_paginates_until_max(self):
        seen_starts = []

        def handler(request: httpx.Request) -> httpx.Response:
            retstart = int(request.url.params.get("retstart", 0))
            seen_starts.append(retstart)
            remaining = max(0, 250 - retstart)
            ids = [str(retstart + i) for i in range(min(400, remaining))]
            return httpx.Response(200, json={"esearchresult": {"idlist": ids}})

        pmids = self._client(handler).search("cancer", max_results=250)
        assert len(pmids) == 250
        assert seen_starts[0] == 0

    def test_search_stops_on_empty_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})

        assert self._client(handler).search("nothing", max_results=100) == []

    def test_fetch_parses_the_xml(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_XML)

        abstracts = self._client(handler).fetch(["12345678", "22222222"])
        assert len(abstracts) == 2

    def test_contact_details_are_sent(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.url.params)
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})

        self._client(handler).search("x", max_results=10)
        assert captured["email"] == "test@example.com"
        assert captured["tool"] == "medextract"

    def test_transient_failure_is_retried(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})

        assert self._client(handler).search("x", max_results=1) == ["1"]
        assert attempts["n"] == 2

    def test_gives_up_after_retries(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        with pytest.raises(RuntimeError, match="failed after"):
            self._client(handler).search("x", max_results=1)

    def test_rate_limiter_spaces_requests(self):
        import time

        from medextract.retrieval.pubmed import RateLimiter

        limiter = RateLimiter(per_second=20.0)
        started = time.monotonic()
        for _ in range(3):
            limiter.wait()
        assert time.monotonic() - started >= 0.05


class TestJsonl:
    def test_round_trip(self, tmp_path):
        abstracts = parse_pubmed_xml(SAMPLE_XML)
        path = tmp_path / "abstracts.jsonl"
        write_jsonl(abstracts, path)
        assert read_jsonl(path) == abstracts


class TestEmbedder:
    def test_output_shape_and_normalisation(self):
        embedder = HashingEmbedder(dimension=64)
        vectors = embedder.encode(["lung cancer", "renal failure"])
        assert vectors.shape == (2, 64)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_deterministic(self):
        embedder = HashingEmbedder(dimension=64)
        assert np.array_equal(embedder.encode(["EGFR"]), embedder.encode(["EGFR"]))

    def test_empty_string_does_not_produce_nan(self):
        vectors = HashingEmbedder(dimension=32).encode(["", "  "])
        assert np.isfinite(vectors).all()

    def test_l2_normalise_handles_zero_vector(self):
        assert np.isfinite(l2_normalise(np.zeros((1, 4), dtype=np.float32))).all()


class TestIndexAndSearch:
    @pytest.fixture
    def indexed(self):
        abstracts = [
            Abstract(
                "1",
                "EGFR mutations in lung adenocarcinoma",
                "Gefitinib improves survival in EGFR mutant lung cancer.",
            ),
            Abstract(
                "2",
                "Cisplatin nephrotoxicity",
                "Renal injury after cisplatin chemotherapy in adults.",
            ),
            Abstract(
                "3",
                "Familial adenomatous polyposis",
                "APC germline mutations cause colorectal polyps.",
            ),
        ]
        embedder = HashingEmbedder(dimension=256)
        client = build_client(":memory:")
        index_abstracts(client, "test_collection", abstracts, embedder)
        return client, embedder

    def test_indexing_reports_the_count(self):
        client = build_client(":memory:")
        embedder = HashingEmbedder(dimension=64)
        total = index_abstracts(
            client, "c", [Abstract("1", "t", "a"), Abstract("2", "t", "a")], embedder
        )
        assert total == 2

    def test_reindexing_is_idempotent(self):
        client = build_client(":memory:")
        embedder = HashingEmbedder(dimension=64)
        abstracts = [Abstract("1", "t", "a"), Abstract("2", "t", "a")]
        index_abstracts(client, "c", abstracts, embedder)
        index_abstracts(client, "c", abstracts, embedder)
        assert client.count("c").count == 2

    def test_point_ids_are_stable_for_a_pmid(self):
        assert pmid_to_point_id("12345") == pmid_to_point_id("12345")
        assert pmid_to_point_id("12345") != pmid_to_point_id("12346")

    def test_ensure_collection_is_safe_to_call_twice(self):
        client = build_client(":memory:")
        ensure_collection(client, "c", 8)
        ensure_collection(client, "c", 8)
        assert client.collection_exists("c")

    def test_build_client_accepts_a_local_path_for_docker_free_retrieval(self, tmp_path):
        """A filesystem path gets Qdrant's on-disk embedded mode -- persistent,
        no server process, so retrieval works without Docker."""
        qdrant_dir = tmp_path / "qdrant_local"
        client = build_client(str(qdrant_dir))
        embedder = HashingEmbedder(dimension=32)
        index_abstracts(client, "c", [Abstract("1", "t", "a")], embedder)
        assert client.count("c").count == 1
        assert qdrant_dir.exists()

    def test_search_returns_requested_number_of_hits(self, indexed):
        client, embedder = indexed
        hits = Searcher(client, "test_collection", embedder).search(
            "cisplatin kidney damage", top_k=2
        )
        assert len(hits) == 2

    def test_hits_carry_payload_and_scores(self, indexed):
        client, embedder = indexed
        hits = Searcher(client, "test_collection", embedder).search("lung cancer", top_k=3)
        for hit in hits:
            assert hit["pmid"] and hit["title"] and hit["abstract"]
            assert -1.0 <= hit["score"] <= 1.0
        assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)

    def test_lexically_close_query_ranks_the_right_abstract_first(self, indexed):
        """Hashing embeddings only capture surface overlap, which is enough
        for a near-verbatim query and is all this test claims."""
        client, embedder = indexed
        hits = Searcher(client, "test_collection", embedder).search(
            "Renal injury after cisplatin chemotherapy", top_k=3
        )
        assert hits[0]["pmid"] == "2"

    def test_top_k_larger_than_the_corpus_is_not_an_error(self, indexed):
        client, embedder = indexed
        assert len(Searcher(client, "test_collection", embedder).search("x", top_k=50)) == 3


class TestSearcherConstruction:
    def test_missing_collection_fails_loudly(self, tmp_path):
        from medextract.config import Settings

        settings = Settings(model_dir=tmp_path, qdrant_url=":memory:", qdrant_collection="absent")
        with pytest.raises(RuntimeError, match="does not exist"):
            Searcher.from_settings(settings)


class TestSearchEndpoint:
    def test_search_returns_hits_with_entities(self, settings_for, tiny_model_dir):
        """The full /search path with a stub index behind it."""
        from fastapi.testclient import TestClient

        from medextract.api.main import create_app

        embedder = HashingEmbedder(dimension=128)
        client = build_client(":memory:")
        index_abstracts(
            client,
            "test_collection",
            [
                Abstract("1", "Acute hepatic failure", "The patient has acute hepatic failure."),
                Abstract("2", "Cisplatin therapy", "Cisplatin therapy caused renal failure."),
            ],
            embedder,
        )

        app = create_app(settings_for())
        with TestClient(app) as test_client:
            # Inject the stub searcher in place of a live Qdrant connection.
            test_client.app.state.ctx.searcher = Searcher(client, "test_collection", embedder)
            response = test_client.post(
                "/search", json={"query": "acute hepatic failure", "top_k": 2}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "acute hepatic failure"
        assert len(body["hits"]) == 2
        assert body["hits"][0]["pmid"] == "1"
        for hit in body["hits"]:
            for entity in hit["entities"]:
                assert hit["abstract"][entity["start"] : entity["end"]] == entity["text"]

    def test_extract_entities_false_skips_tagging(self, settings_for):
        from fastapi.testclient import TestClient

        from medextract.api.main import create_app

        embedder = HashingEmbedder(dimension=128)
        client = build_client(":memory:")
        index_abstracts(client, "c", [Abstract("1", "Fever", "The patient has a fever.")], embedder)

        with TestClient(create_app(settings_for())) as test_client:
            test_client.app.state.ctx.searcher = Searcher(client, "c", embedder)
            body = test_client.post(
                "/search", json={"query": "fever", "extract_entities": False}
            ).json()

        assert body["hits"][0]["entities"] == []
