"""Configuration: dataset specs, model defaults, and runtime settings.

Runtime settings come from the environment (see `.env.example`); training
settings come from CLI flags so that every value that affects a run ends up in
the MLflow record rather than in someone's shell history.
"""

from __future__ import annotations

import os
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
    # Direct-download source, used when the Hub repo turns out to be
    # script-only. Both corpora below are in that state: their repos contain a
    # loader script and no data files, and datasets>=3 refuses to execute
    # loader scripts, so `load_dataset(hf_id)` alone is not a source that can
    # be relied on. These URLs point at the same upstream files the official
    # loader scripts download, so the corpus is identical either way.
    direct_urls: dict[str, str] | None = None
    direct_format: str = "conll"  # "conll" (two-column TSV) or "jsonl"


NCBI_CONLL = "https://github.com/spyysalo/ncbi-disease/raw/master/conll"
BC5CDR_JSON = "https://huggingface.co/datasets/tner/bc5cdr/resolve/main/dataset"

DATASETS: dict[str, DatasetSpec] = {
    "ncbi_disease": DatasetSpec(
        hf_id="ncbi/ncbi_disease",
        tokens_column="tokens",
        tags_column="ner_tags",
        # Order matters: it is the integer encoding of the corpus, and it is
        # what the exported config.json's id2label will carry into serving.
        label_names=("O", "B-Disease", "I-Disease"),
        direct_urls={
            "train": f"{NCBI_CONLL}/train.tsv",
            "validation": f"{NCBI_CONLL}/devel.tsv",
            "test": f"{NCBI_CONLL}/test.tsv",
        },
        direct_format="conll",
    ),
    "bc5cdr": DatasetSpec(
        hf_id="tner/bc5cdr",
        tokens_column="tokens",
        tags_column="tags",
        label_names=("O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"),
        direct_urls={
            "train": f"{BC5CDR_JSON}/train.json",
            "validation": f"{BC5CDR_JSON}/valid.json",
            "test": f"{BC5CDR_JSON}/test.json",
        },
        direct_format="jsonl",
    ),
}

DEFAULT_DATASET = "ncbi_disease"
DEFAULT_BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"

# Alternative base checkpoint from the brief. PubMedBERT is uncased and trained
# from scratch on PubMed rather than initialised from general-domain BERT, so
# it is worth including as a sweep arm rather than treating as interchangeable.
ALT_BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

REPO_ROOT = Path(__file__).resolve().parents[2]


MLFLOW_DB = REPO_ROOT / "mlflow.db"


def mlflow_tracking_uri() -> str:
    """Where sweep runs are recorded. SQLite, not a directory of files.

    Two reasons it is not the `./mlruns` file store the MLflow docs used to
    lead with:

    - MLflow 3 put that backend into maintenance mode and now raises rather
      than opening one, so it is not a working default any more.
    - It writes a tree of thousands of very small files, one per metric and
      per parameter per run. Nine arms of that inside a synced folder is a lot
      of churn for data that a single file holds just as well.

    Building the URI by hand also avoids a Windows trap worth naming, since the
    file store is still reachable by setting MLFLOW_TRACKING_URI: interpolating
    a path into a "file://" prefix yields backwards separators and only two
    slashes, so MLflow reads the drive letter as a hostname and writes runs
    somewhere nobody looks. as_posix() and as_uri() are the forms that survive
    both platforms.

    train.py and report.py both come through here, so they cannot disagree
    about where runs live.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{MLFLOW_DB.as_posix()}"


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
