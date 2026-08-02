"""Phase 4: build the Qdrant index from a downloaded abstract dump.

python -m medextract.retrieval.build_index --input data/pubmed.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from medextract.config import REPO_ROOT, get_settings
from medextract.retrieval.embed import build_embedder
from medextract.retrieval.index import build_client, index_abstracts
from medextract.retrieval.pubmed import read_jsonl

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed PubMed abstracts into Qdrant.")
    p.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "pubmed.jsonl")
    p.add_argument("--qdrant-url", default=None, help="overrides MEDEXTRACT_QDRANT_URL")
    p.add_argument("--collection", default=None, help="overrides MEDEXTRACT_QDRANT_COLLECTION")
    p.add_argument("--embedding-model", default=None)
    p.add_argument(
        "--recreate",
        action="store_true",
        help="drop the collection first (needed when the embedding model changes)",
    )
    p.add_argument(
        "--allow-hashing-fallback",
        action="store_true",
        help="index with hash vectors if the real encoder cannot be loaded (for smoke tests only)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)
    settings = get_settings()

    if not ns.input.exists():
        raise SystemExit(
            f"{ns.input} not found. Download abstracts first: "
            "python -m medextract.retrieval.pubmed --max 3000"
        )

    abstracts = read_jsonl(ns.input)
    log.info("read %d abstracts from %s", len(abstracts), ns.input)

    # No hashing fallback here: an index built from hash vectors would look
    # fine and retrieve badly, and rebuilding costs a full pass over the corpus.
    embedder = build_embedder(
        ns.embedding_model or settings.embedding_model,
        onnx_dir=settings.embedding_onnx_dir,
        allow_fallback=ns.allow_hashing_fallback,
    )
    client = build_client(ns.qdrant_url or settings.qdrant_url)
    collection = ns.collection or settings.qdrant_collection

    total = index_abstracts(client, collection, abstracts, embedder, recreate=ns.recreate)
    print(f"indexed {total} abstracts into {collection!r} (dim={embedder.dimension})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
