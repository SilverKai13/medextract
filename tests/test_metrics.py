"""Tests for entity-level scoring, including a cross-check against seqeval."""

from __future__ import annotations

import random

import numpy as np
import pytest

from medextract.labels import IGNORE_INDEX
from medextract.metrics import decode_predictions, entity_scores, extract_spans


def test_extract_spans_basic():
    tags = ["O", "B-Disease", "I-Disease", "O", "B-Chemical"]
    assert extract_spans(tags) == {("Disease", 1, 3), ("Chemical", 4, 5)}


def test_span_runs_to_end_of_sequence():
    assert extract_spans(["O", "B-Disease", "I-Disease"]) == {("Disease", 1, 3)}


def test_adjacent_spans_of_same_type_stay_separate():
    # B- always opens a new span, so two adjacent diseases are two entities.
    assert extract_spans(["B-Disease", "B-Disease"]) == {("Disease", 0, 1), ("Disease", 1, 2)}


def test_invalid_continuation_is_discarded_not_repaired():
    # Strict IOB2: I-Chemical cannot follow B-Disease, so it is dropped rather
    # than opening a Chemical span.
    assert extract_spans(["B-Disease", "I-Chemical"]) == {("Disease", 0, 1)}


def test_orphan_i_tag_yields_no_span():
    assert extract_spans(["O", "I-Disease", "I-Disease"]) == set()


def test_a_broken_span_stays_broken_until_the_next_b():
    assert extract_spans(["B-Disease", "I-Chemical", "I-Disease"]) == {("Disease", 0, 1)}
    assert extract_spans(["B-Disease", "O", "I-Disease"]) == {("Disease", 0, 1)}


def test_b_after_b_closes_and_reopens():
    assert extract_spans(["B-Disease", "B-Chemical", "I-Chemical"]) == {
        ("Disease", 0, 1),
        ("Chemical", 1, 3),
    }


def test_strict_scoring_and_lenient_serving_differ_on_purpose():
    """The two decoders disagree on orphan I- tags, by design.

    Scoring drops them so numbers stay comparable to published baselines;
    serving recovers them so a caller is not silently denied an entity the
    model did find.
    """
    import numpy as np

    from medextract.inference import decode_entities

    assert extract_spans(["O", "I-Disease"]) == set()

    served = decode_entities(
        "no fever",
        np.array([0, 1]),
        np.array([0.9, 0.9]),
        np.array([(0, 2), (3, 8)]),
        np.array([False, False]),
        {0: "O", 1: "I-Disease"},
    )
    assert [e.text for e in served] == ["fever"]


def test_perfect_prediction_scores_one():
    gold = [["B-Disease", "I-Disease", "O"], ["O", "B-Chemical"]]
    scores = entity_scores(gold, gold)
    assert scores["f1"] == pytest.approx(1.0)
    assert scores["Disease_f1"] == pytest.approx(1.0)
    assert scores["Chemical_f1"] == pytest.approx(1.0)


def test_predicting_nothing_scores_zero_not_high():
    """The reason token accuracy is not used: all-O looks great token-wise."""
    gold = [["B-Disease", "I-Disease", "O", "O", "O", "O", "O", "O", "O", "O"]]
    pred = [["O"] * 10]
    scores = entity_scores(pred, gold)
    assert scores["f1"] == 0.0
    assert scores["recall"] == 0.0


def test_partial_span_match_counts_as_wrong():
    # Strict matching: getting two of three tokens is a miss and a false alarm,
    # not partial credit.
    gold = [["B-Disease", "I-Disease", "I-Disease"]]
    pred = [["B-Disease", "I-Disease", "O"]]
    scores = entity_scores(pred, gold)
    assert scores["f1"] == 0.0
    assert scores["support"] == 1.0


def test_precision_recall_arithmetic():
    # 2 gold spans, 3 predicted, 1 exact hit.
    gold = [["B-Disease", "O", "B-Disease", "O"]]
    pred = [["B-Disease", "B-Disease", "O", "B-Disease"]]
    scores = entity_scores(pred, gold)
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["recall"] == pytest.approx(1 / 2)
    assert scores["f1"] == pytest.approx(2 * (1 / 3) * (1 / 2) / ((1 / 3) + (1 / 2)))


def test_decode_predictions_drops_ignored_positions():
    id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
    predictions = np.array([[0, 1, 2, 0]])
    labels = np.array([[IGNORE_INDEX, 1, 2, IGNORE_INDEX]])

    pred_seqs, gold_seqs = decode_predictions(predictions, labels, id2label)
    assert pred_seqs == [["B-Disease", "I-Disease"]]
    assert gold_seqs == [["B-Disease", "I-Disease"]]


def test_decode_predictions_accepts_raw_logits():
    id2label = {0: "O", 1: "B-Disease"}
    logits = np.array([[[0.1, 0.9], [0.8, 0.2]]])
    labels = np.array([[1, 0]])
    pred_seqs, _ = decode_predictions(logits, labels, id2label)
    assert pred_seqs == [["B-Disease", "O"]]


def _random_tag_sequences(n_sentences: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    vocabulary = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]
    return [[rng.choice(vocabulary) for _ in range(rng.randint(1, 25))] for _ in range(n_sentences)]


def test_matches_seqeval_on_random_sequences():
    """Cross-check the hand-rolled scorer against seqeval where it is available.

    seqeval is not a runtime dependency (its sdist no longer builds against
    current setuptools), so this skips rather than fails when it is absent.
    """
    seqeval_metrics = pytest.importorskip("seqeval.metrics", reason="seqeval not installed")
    from seqeval.scheme import IOB2

    vocabulary = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]

    for seed in range(25):
        gold = _random_tag_sequences(40, seed)
        # Independent predictions of matching length, so the comparison covers
        # plenty of malformed transitions rather than only clean sequences.
        rng = random.Random(seed + 9973)
        pred = [[rng.choice(vocabulary) for _ in tags] for tags in gold]

        ours = entity_scores(pred, gold)
        theirs = seqeval_metrics.classification_report(
            gold, pred, output_dict=True, zero_division=0, mode="strict", scheme=IOB2
        )
        micro = theirs.get("micro avg")
        if micro is None:  # no spans on either side
            continue

        assert ours["precision"] == pytest.approx(micro["precision"], abs=1e-9)
        assert ours["recall"] == pytest.approx(micro["recall"], abs=1e-9)
        assert ours["f1"] == pytest.approx(micro["f1-score"], abs=1e-9)
