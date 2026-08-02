"""Tests for label alignment -- the step the brief flags as the silent killer.

If alignment is wrong the model still trains and the loss still falls; only the
F1 is quietly wrong. So these assert the exact positions, not just that the
lengths match.
"""

from __future__ import annotations

import pytest

from medextract.labels import (
    IGNORE_INDEX,
    align_labels_with_tokens,
    build_label_maps,
    entity_type,
    labels_to_ids,
    to_inside,
)


def test_entity_type():
    assert entity_type("B-Disease") == "Disease"
    assert entity_type("I-Chemical") == "Chemical"
    assert entity_type("O") is None


def test_to_inside_only_converts_b():
    assert to_inside("B-Disease") == "I-Disease"
    assert to_inside("I-Disease") == "I-Disease"
    assert to_inside("O") == "O"


def test_special_tokens_are_ignored():
    # [CLS] one-word sentence [SEP]
    word_ids = [None, 0, None]
    assert align_labels_with_tokens(word_ids, ["B-Disease"]) == [
        IGNORE_INDEX,
        "B-Disease",
        IGNORE_INDEX,
    ]


def test_only_first_subword_is_labelled_by_default():
    # "familial adenomatous polyposis" where word 1 fragments into 3 subwords
    # and word 2 into 2.
    word_ids = [None, 0, 1, 1, 1, 2, 2, None]
    word_labels = ["B-Disease", "I-Disease", "I-Disease"]

    aligned = align_labels_with_tokens(word_ids, word_labels)

    assert aligned == [
        IGNORE_INDEX,
        "B-Disease",
        "I-Disease",
        IGNORE_INDEX,
        IGNORE_INDEX,
        "I-Disease",
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_label_all_subwords_converts_b_to_i_on_continuations():
    # A word that begins an entity and splits into three pieces must not be
    # read as three separate entity beginnings.
    word_ids = [None, 0, 0, 0, None]
    aligned = align_labels_with_tokens(word_ids, ["B-Disease"], label_all_subwords=True)

    assert aligned == [IGNORE_INDEX, "B-Disease", "I-Disease", "I-Disease", IGNORE_INDEX]


def test_alignment_length_always_matches_input():
    word_ids = [None, 0, 0, 1, 2, 2, 2, None, None]
    labels = ["B-Chemical", "O", "B-Disease"]
    for flag in (True, False):
        assert len(align_labels_with_tokens(word_ids, labels, label_all_subwords=flag)) == len(
            word_ids
        )


def test_repeated_word_id_after_gap_is_treated_as_new_word():
    # Two adjacent words can share a label but must each get a labelled first
    # subword; only equal *consecutive* ids mean "same word".
    word_ids = [0, 1, 0]
    aligned = align_labels_with_tokens(word_ids, ["B-Disease", "O"])
    assert aligned == ["B-Disease", "O", "B-Disease"]


def test_labels_to_ids_passes_ignore_index_through():
    label2id, _ = build_label_maps(["O", "B-Disease", "I-Disease"])
    aligned = [IGNORE_INDEX, "B-Disease", IGNORE_INDEX, "I-Disease", "O"]
    assert labels_to_ids(aligned, label2id) == [IGNORE_INDEX, 1, IGNORE_INDEX, 2, 0]


def test_build_label_maps_round_trips():
    names = ["O", "B-Disease", "I-Disease"]
    label2id, id2label = build_label_maps(names)
    assert [id2label[label2id[n]] for n in names] == names


def test_real_tokenizer_alignment_end_to_end(tiny_tokenizer):
    """The same check against a real fast tokenizer rather than handmade ids."""
    words = ["the", "patient", "has", "familial", "adenomatous", "polyposis"]
    tags = ["O", "O", "O", "B-Disease", "I-Disease", "I-Disease"]

    encoded = tiny_tokenizer([words], is_split_into_words=True, truncation=True)
    word_ids = encoded.word_ids(batch_index=0)
    aligned = align_labels_with_tokens(word_ids, tags)

    assert len(aligned) == len(encoded["input_ids"][0])

    # Every supervised position must correspond to the first subword of its
    # word, and carry that word's label.
    supervised = [(w, lbl) for w, lbl in zip(word_ids, aligned, strict=True) if lbl != IGNORE_INDEX]
    assert [w for w, _ in supervised] == list(range(len(words)))
    assert [lbl for _, lbl in supervised] == tags


@pytest.mark.parametrize("label_all", [False, True])
def test_no_supervised_position_is_a_special_token(tiny_tokenizer, label_all):
    words = ["acute", "hepatic", "failure", "with", "cisplatin"]
    tags = ["B-Disease", "I-Disease", "I-Disease", "O", "B-Chemical"]

    encoded = tiny_tokenizer(
        [words],
        is_split_into_words=True,
        truncation=True,
        return_special_tokens_mask=True,
    )
    word_ids = encoded.word_ids(batch_index=0)
    aligned = align_labels_with_tokens(word_ids, tags, label_all_subwords=label_all)

    for is_special, label in zip(encoded["special_tokens_mask"][0], aligned, strict=True):
        if is_special:
            assert label == IGNORE_INDEX
