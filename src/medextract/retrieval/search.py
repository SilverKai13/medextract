"""Query-time retrieval against the Qdrant index."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from medextract.retrieval.index import build_client

if TYPE_CHECKING:  # pragma: no cover
    from medextract.config import Settings
    from medextract.retrieval.embed import Embedder

log = logging.getLogger(__name__)


class Searcher:
    def __init__(self, client: Any, collection: str, embedder: Embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    @classmethod
    def from_settings(cls, settings: Settings) -> Searcher:
        """Build from configuration, failing loudly if the collection is missing.

        A searcher pointed at a collection that was never built would return
        empty results forever and look like a ranking problem; better to refuse
        to construct and let /health report it.
        """
        from medextract.retrieval.embed import build_embedder

        client = build_client(settings.qdrant_url)
        if not client.collection_exists(settings.qdrant_collection):
            raise RuntimeError(
                f"Qdrant collection {settings.qdrant_collection!r} does not exist at "
                f"{settings.qdrant_url}. Build it with medextract-index."
            )
        embedder = build_embedder(settings.embedding_model, onnx_dir=settings.embedding_onnx_dir)
        return cls(client, settings.qdrant_collection, embedder)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector = self.embedder.encode([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                {
                    "pmid": payload.get("pmid", ""),
                    "title": payload.get("title", ""),
                    "abstract": payload.get("abstract", ""),
                    "journal": payload.get("journal"),
                    "year": payload.get("year"),
                    "score": float(point.score),
                    "entities": [],
                }
            )
        return hits
