"""Qdrant collection management and upserts."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from medextract.retrieval.embed import Embedder
    from medextract.retrieval.pubmed import Abstract

log = logging.getLogger(__name__)

UPSERT_BATCH = 128
# Qdrant point ids must be uint64 or UUID; PMIDs are neither reliably, so they
# are hashed into a stable UUID and kept in the payload for lookups.
POINT_NAMESPACE = uuid.UUID("6f1c1f5e-6a3d-4c2a-9a56-0b6b1a1f9d21")


def pmid_to_point_id(pmid: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, pmid))


def build_client(url: str):
    """Connect to Qdrant.

    Three forms: `:memory:` for an embedded instance (tests), an `http(s)://`
    URL for a running server (Docker), or a filesystem path for Qdrant's
    on-disk embedded mode -- persistent, no server process, so retrieval works
    without Docker.
    """
    from qdrant_client import QdrantClient

    if url in {":memory:", "memory"}:
        return QdrantClient(":memory:")
    if url.startswith("http://") or url.startswith("https://"):
        return QdrantClient(url=url)

    from pathlib import Path

    path = Path(url)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def ensure_collection(client: Any, name: str, dimension: int, *, recreate: bool = False) -> None:
    from qdrant_client.models import Distance, VectorParams

    exists = client.collection_exists(name)
    if exists and recreate:
        log.info("dropping existing collection %s", name)
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            # Vectors are normalised at encode time, so cosine scores come back
            # directly comparable across queries.
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        log.info("created collection %s (dim=%d, cosine)", name, dimension)


def index_abstracts(
    client: Any,
    collection: str,
    abstracts: list[Abstract],
    embedder: Embedder,
    *,
    batch_size: int = UPSERT_BATCH,
    recreate: bool = False,
) -> int:
    """Embed and upsert. Re-running is idempotent: point ids derive from PMIDs."""
    from qdrant_client.models import PointStruct

    ensure_collection(client, collection, embedder.dimension, recreate=recreate)

    total = 0
    for start in range(0, len(abstracts), batch_size):
        chunk = abstracts[start : start + batch_size]
        vectors = np.asarray(embedder.encode([a.text for a in chunk]), dtype=np.float32)

        points = [
            PointStruct(
                id=pmid_to_point_id(abstract.pmid),
                vector=vector.tolist(),
                payload={
                    "pmid": abstract.pmid,
                    "title": abstract.title,
                    "abstract": abstract.abstract,
                    "journal": abstract.journal,
                    "year": abstract.year,
                },
            )
            for abstract, vector in zip(chunk, vectors, strict=True)
        ]
        client.upsert(collection_name=collection, points=points)
        total += len(points)
        log.info("indexed %d/%d", total, len(abstracts))

    return total
