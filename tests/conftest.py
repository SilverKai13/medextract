"""Test fixtures.

The suite must run with no network access, so it cannot pull BioBERT. Instead
it builds a genuinely tiny BERT (2 layers, hidden size 32) with random weights
and a hand-written WordPiece vocabulary, saved in the normal HuggingFace
layout. Its predictions are meaningless, which is fine -- what is under test is
the plumbing: label alignment, span decoding, ONNX parity, request handling.
Model quality is what the MLflow sweep measures, not what pytest measures.
"""

from __future__ import annotations

import pytest

# The vocabulary is deliberately built so that biomedical terms fragment into
# several subwords, which is the case the alignment code has to get right.
VOCAB_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "the",
    "patient",
    "has",
    "a",
    "and",
    "with",
    "of",
    "in",
    "was",
    "no",
    "gene",
    "cell",
    "acute",
    "chronic",
    "severe",
    "treatment",
    "therapy",
    "aden",
    "##oma",
    "##tous",
    "poly",
    "##posis",
    "familial",
    "carcin",
    "##oma",
    "neuro",
    "##blast",
    "##oma",
    "hepat",
    "##ic",
    "fail",
    "##ure",
    "renal",
    "cis",
    "##plat",
    "##in",
    "acetamin",
    "##ophen",
    "aspirin",
    "brca",
    "##1",
    "apc",
    "egfr",
    ".",
    ",",
    "(",
    ")",
    "-",
    "%",
    "12",
]

LABEL_NAMES = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]


def _write_vocab(path):
    seen: list[str] = []
    for token in VOCAB_TOKENS:
        if token not in seen:
            seen.append(token)
    path.write_text("\n".join(seen) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A saved token-classification model directory, built offline."""
    from transformers import (
        AutoModelForTokenClassification,
        BertConfig,
        BertTokenizerFast,
    )

    model_dir = tmp_path_factory.mktemp("tiny-model")
    _write_vocab(model_dir / "vocab.txt")

    tokenizer = BertTokenizerFast(
        vocab_file=str(model_dir / "vocab.txt"),
        do_lower_case=True,
        model_max_length=256,
    )
    tokenizer.save_pretrained(str(model_dir))

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=len(LABEL_NAMES),
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    model = AutoModelForTokenClassification.from_config(config)
    model.save_pretrained(str(model_dir))
    return model_dir


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tiny_model_dir), use_fast=True)


@pytest.fixture(scope="session")
def tiny_onnx_dir(tiny_model_dir, tmp_path_factory):
    """The same model exported to ONNX through the project's own exporter."""
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

    from medextract.export_onnx import export

    onnx_dir = tmp_path_factory.mktemp("tiny-onnx")
    export(tiny_model_dir, onnx_dir, exporter="torch")
    return onnx_dir


@pytest.fixture
def settings_for(tiny_model_dir):
    """Build Settings pointed at the tiny model, retrieval off by default."""

    def _make(**overrides):
        from medextract.config import Settings

        base = {
            "model_dir": tiny_model_dir,
            "backend": "torch",
            "model_version": "test",
            "retrieval_enabled": False,
            "log_level": "WARNING",
        }
        return Settings(**{**base, **overrides})

    return _make


@pytest.fixture
def client(settings_for):
    from fastapi.testclient import TestClient

    from medextract.api.main import create_app

    with TestClient(create_app(settings_for())) as test_client:
        yield test_client


SAMPLE_TEXTS = [
    "The patient has familial adenomatous polyposis with APC gene mutations.",
    "Acute hepatic failure was treated with acetaminophen and cisplatin.",
    "No abnormality.",
]
