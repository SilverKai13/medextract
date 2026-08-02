"""Configuration: dataset specs, model defaults, and runtime settings.

Runtime settings come from the environment (see `.env.example`); training
settings come from CLI flags so that every value that affects a run ends up in
the MLflow record rather than in someone's shell history.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class DatasetSpec:
    """Where a corpus lives on the Hub and what its columns are called.

    Column names differ between corpora (`ner_tags` on ncbi_disease, `tags` on
    the tner mirrors), so they are part of the spec rather than hardcoded in
    the loader.
    """

    hf_id: str
    tokens_column: str = "tokens"
    tags_column: str = "ner_tags"
    config_name: str | None = None
    # Only needed when the column is plain ints rather than a ClassLabel, in
    # which case the loader cannot recover the names from the features.
    label_names: tuple[str, ...] | None = None
    # Some corpora ship no validation split; carve one out of train instead.
    validation_split: str = "validation"
    test_split: str = "test"


DATASETS: dict[str, DatasetSpec] = {
    "ncbi_disease": DatasetSpec(
        hf_id="ncbi_disease",
        tokens_column="tokens",
        tags_column="ner_tags",
    ),
    "bc5cdr": DatasetSpec(
        hf_id="tner/bc5cdr",
        tokens_column="tokens",
        tags_column="tags",
        label_names=("O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"),
    ),
}

DEFAULT_DATASET = "ncbi_disease"
DEFAULT_BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"

# Alternative base checkpoint from the brief. PubMedBERT is uncased and trained
# from scratch on PubMed rather than initialised from general-domain BERT, so
# it is worth including as a sweep arm rather than treating as interchangeable.
ALT_BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Serving-time configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MEDEXTRACT_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # Path to a saved model directory (the output of `train.py`, or of
    # `export_onnx.py` when backend is "onnx").
    model_dir: Path = REPO_ROOT / "artifacts" / "model"
    # "torch" or "onnx".
    backend: str = "onnx"
    model_version: str = "dev"

    max_input_chars: int = 20_000
    # Longer inputs are chunked on sentence-ish boundaries rather than truncated,
    # so entities past the 512th token are still found.
    max_sequence_length: int = 256
    stride: int = 32

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pubmed_abstracts"
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    # An exported query encoder. When present the container encodes queries on
    # onnxruntime and needs no PyTorch at all.
    embedding_onnx_dir: Path = REPO_ROOT / "artifacts" / "embedder"
    # Set false to run the API without a vector store; /search then returns 503.
    retrieval_enabled: bool = True

    log_level: str = "INFO"
    # NCBI asks for an identifying email on E-utilities traffic.
    ncbi_email: str = ""
    ncbi_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
