"""Embedding backends for the retrieval index.

`Embedder` is a protocol rather than a concrete class so the index and search
code can be exercised without downloading a sentence-transformers checkpoint --
the tests inject a deterministic stub through the same interface the real
model uses.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dimension)."""
        ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard the zero vector rather than emitting NaNs into the index.
    norms[norms == 0] = 1.0
    return vectors / norms


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model.

    Vectors are normalised on the way out so the Qdrant collection can use
    cosine distance and the scores it returns are cosine similarities directly.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class OnnxEmbedder:
    """Runs an exported sentence encoder on onnxruntime.

    This is what the container uses. Pooling and normalisation are done in
    numpy so the graph stays a plain transformer forward pass, which keeps the
    export simple and lets the pooling mode be read from config at load time
    rather than baked into the graph.
    """

    def __init__(
        self, model_dir: str | Path, *, max_length: int = 256, batch_size: int = 32
    ) -> None:
        import json

        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No model.onnx in {self.model_dir}. Export one with "
                "python -m medextract.retrieval.export_embedder"
            )

        config_path = self.model_dir / "embedder_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.pooling = config.get("pooling", "mean")
        self.model_name = config.get("model_name", str(self.model_dir))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self.max_length = max_length
        self.batch_size = batch_size
        self.dimension = int(config.get("dimension") or self.session.get_outputs()[0].shape[-1])

    def _pool(self, hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        # Mean over real tokens only -- averaging padding drags every vector
        # toward the pad embedding and shortens short texts' effective norm.
        mask = attention_mask[..., None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                chunk,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            feed = {
                k: np.asarray(v, dtype=np.int64)
                for k, v in encoded.items()
                if k in self._input_names
            }
            hidden = self.session.run(None, feed)[0]
            vectors.append(self._pool(hidden, np.asarray(encoded["attention_mask"])))

        return l2_normalise(np.vstack(vectors).astype(np.float32))


class HashingEmbedder:
    """Deterministic character n-gram hashing, no download required.

    Not competitive with a trained biomedical encoder -- it has no notion of
    synonymy, so "myocardial infarction" and "heart attack" land nowhere near
    each other. It exists so the indexing and search paths can be tested
    offline and so the service can start without network access.
    """

    def __init__(self, dimension: int = 256, ngram: int = 4) -> None:
        self.dimension = dimension
        self.ngram = ngram
        self.model_name = f"hashing-{dimension}d"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        cleaned = " ".join(text.lower().split())
        if not cleaned:
            return vector
        for i in range(max(1, len(cleaned) - self.ngram + 1)):
            gram = cleaned[i : i + self.ngram]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimension
            vector[index] += 1.0
        return vector

    def encode(self, texts: list[str]) -> np.ndarray:
        return l2_normalise(np.vstack([self._vector(t) for t in texts]))


def build_embedder(
    model_name: str,
    device: str = "cpu",
    *,
    onnx_dir: Path | str | None = None,
    allow_fallback: bool = True,
) -> Embedder:
    """Pick an embedding backend.

    Preference order: an exported ONNX encoder (what the container runs), then
    sentence-transformers (what indexing normally uses), then hashing.

    The hashing fallback keeps the service answering when neither is available,
    but it is a materially worse retriever, so it is logged loudly. Pass
    `allow_fallback=False` when building an index -- silently indexing three
    thousand abstracts with hash vectors is worse than failing.
    """
    if onnx_dir and Path(onnx_dir).exists():
        try:
            return OnnxEmbedder(onnx_dir)
        except Exception as exc:
            log.warning("ONNX embedder at %s unusable (%s)", onnx_dir, exc)

    try:
        return SentenceTransformerEmbedder(model_name, device=device)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"Could not load embedding model {model_name!r}: {exc}. Refusing to "
                "build an index with fallback hashing vectors."
            ) from exc
        log.warning(
            "could not load %s (%s); falling back to hashing embeddings. "
            "Retrieval quality will be poor -- fix the model before reporting "
            "any search results.",
            model_name,
            exc,
        )
        return HashingEmbedder()
