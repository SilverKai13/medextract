"""Tests for the dataset tokenisation path.

These run the real batched map function against hand-built batches, so they
cover the code that touches the corpus without needing the corpus downloaded.
"""

from __future__ import annotations

import pytest

from medextract.config import DATASETS, DatasetSpec
from medextract.data import build_tokenize_fn, resolve_label_names
from medextract.labels import IGNORE_INDEX

LABEL_NAMES = ["O", "B-Disease", "I-Disease"]
SPEC = DatasetSpec(hf_id="test", tokens_column="tokens", tags_column="ner_tags")

BATCH = {
    "tokens": [
        ["the", "patient", "has", "familial", "adenomatous", "polyposis"],
        ["no", "abnormality"],
    ],
    "ner_tags": [
        [0, 0, 0, 1, 2, 2],
        [0, 0],
    ],
}


@pytest.fixture
def tokenize(tiny_tokenizer):
    return build_tokenize_fn(tiny_tokenizer, SPEC, LABEL_NAMES, max_length=64)


def test_produces_the_model_inputs(tokenize):
    encoded = tokenize(BATCH)
    assert set(encoded) >= {"input_ids", "attention_mask", "labels"}


def test_labels_align_with_input_ids(tokenize):
    encoded = tokenize(BATCH)
    for ids, labels in zip(encoded["input_ids"], encoded["labels"], strict=True):
        assert len(ids) == len(labels)


def test_label_ids_are_valid_or_ignored(tokenize):
    encoded = tokenize(BATCH)
    for labels in encoded["labels"]:
        for label in labels:
            assert label == IGNORE_INDEX or 0 <= label < len(LABEL_NAMES)


def test_supervised_labels_recover_the_original_tags(tokenize):
    """One label per word, in order -- the invariant that alignment exists for."""
    encoded = tokenize(BATCH)
    for row, tags in enumerate(BATCH["ner_tags"]):
        supervised = [x for x in encoded["labels"][row] if x != IGNORE_INDEX]
        assert supervised == tags


def test_all_outside_sentence_has_no_entity_labels(tokenize):
    encoded = tokenize({"tokens": [["no", "abnormality"]], "ner_tags": [[0, 0]]})
    assert set(encoded["labels"][0]) <= {IGNORE_INDEX, 0}


def test_string_tags_are_accepted(tokenize):
    """Some Hub mirrors ship BIO strings rather than ClassLabel ints."""
    encoded = tokenize(
        {"tokens": [["familial", "polyposis"]], "ner_tags": [["B-Disease", "I-Disease"]]}
    )
    assert [x for x in encoded["labels"][0] if x != IGNORE_INDEX] == [1, 2]


def test_truncation_does_not_desynchronise_labels(tiny_tokenizer):
    tokenize = build_tokenize_fn(tiny_tokenizer, SPEC, LABEL_NAMES, max_length=8)
    encoded = tokenize(
        {
            "tokens": [["the", "patient", "has", "acute", "hepatic", "failure"] * 5],
            "ner_tags": [[0, 0, 0, 1, 2, 2] * 5],
        }
    )
    assert len(encoded["input_ids"][0]) == len(encoded["labels"][0]) <= 8


def test_slow_tokenizer_is_rejected_with_an_explanation(tiny_model_dir):
    from transformers import BertTokenizer

    slow = BertTokenizer(vocab_file=str(tiny_model_dir / "vocab.txt"))
    with pytest.raises(ValueError, match="fast tokenizer"):
        build_tokenize_fn(slow, SPEC, LABEL_NAMES)


class TestLabelResolution:
    def test_prefers_the_dataset_class_label_names(self):
        class FakeFeature:
            def __init__(self, names):
                self.feature = type("Inner", (), {"names": names})()

        split = type("Split", (), {"features": {"ner_tags": FakeFeature(LABEL_NAMES)}})()
        assert resolve_label_names({"train": split}, SPEC) == LABEL_NAMES

    def test_falls_back_to_the_spec_override(self):
        split = type("Split", (), {"features": {"tags": object()}})()
        spec = DatasetSpec(hf_id="x", tags_column="tags", label_names=("O", "B-Chemical"))
        assert resolve_label_names({"train": split}, spec) == ["O", "B-Chemical"]

    def test_unresolvable_labels_raise_an_actionable_error(self):
        split = type("Split", (), {"features": {"ner_tags": object()}})()
        with pytest.raises(ValueError, match="label_names override"):
            resolve_label_names({"train": split}, SPEC)


def test_known_datasets_are_registered():
    assert {"ncbi_disease", "bc5cdr"} <= set(DATASETS)
    assert DATASETS["bc5cdr"].label_names is not None
