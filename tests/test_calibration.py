"""Reliability measurement for the entity score: the arithmetic, offline."""

from __future__ import annotations

import numpy as np

from medextract.calibration import (
    gold_entities,
    precision_at_thresholds,
    reliability,
    to_markdown,
)


class TestGoldOffsets:
    def test_word_indices_become_character_offsets_into_the_joined_sentence(self):
        tokens = ["acute", "renal", "failure", "."]
        tags = ["O", "B-Disease", "I-Disease", "O"]
        assert gold_entities(tokens, tags) == {(6, 19, "Disease")}
        assert " ".join(tokens)[6:19] == "renal failure"

    def test_two_entities_of_different_types(self):
        tokens = ["cisplatin", "caused", "nephrotoxicity"]
        tags = ["B-Chemical", "O", "B-Disease"]
        assert gold_entities(tokens, tags) == {(0, 9, "Chemical"), (17, 31, "Disease")}

    def test_no_entities(self):
        assert gold_entities(["no", "abnormality"], ["O", "O"]) == set()


class TestReliability:
    def test_perfectly_calibrated_bin_has_zero_gap(self):
        scores = np.full(20, 0.95)
        correct = np.array([True] * 19 + [False])
        result = reliability(scores, correct, bins=10)
        assert result["ece"] == 0.0
        assert result["bins"][0]["gap"] == 0.0
        assert result["n_predictions"] == 20

    def test_overconfidence_shows_as_a_negative_gap(self):
        scores = np.full(10, 0.95)
        correct = np.array([True] * 5 + [False] * 5)
        result = reliability(scores, correct, bins=10)
        assert result["bins"][0]["gap"] == -0.45
        assert result["ece"] == 0.45

    def test_ece_is_support_weighted_across_bins(self):
        # 8 predictions at 0.95 all correct (gap +0.05), 2 at 0.55 all wrong (gap -0.55)
        scores = np.array([0.95] * 8 + [0.55] * 2)
        correct = np.array([True] * 8 + [False] * 2)
        result = reliability(scores, correct, bins=10)
        assert result["ece"] == round(0.8 * 0.05 + 0.2 * 0.55, 4)

    def test_a_score_of_exactly_one_lands_in_the_top_bin(self):
        result = reliability(np.array([1.0, 1.0]), np.array([True, True]), bins=10)
        assert [row["bin"] for row in result["bins"]] == ["0.9-1.0"]

    def test_empty_bins_are_omitted(self):
        result = reliability(np.array([0.05, 0.95]), np.array([False, True]), bins=10)
        assert [row["bin"] for row in result["bins"]] == ["0.0-0.1", "0.9-1.0"]

    def test_no_predictions_is_not_a_division_error(self):
        result = reliability(np.array([]), np.array([], dtype=bool))
        assert result["n_predictions"] == 0
        assert result["ece"] == 0.0


class TestThresholds:
    def test_raising_the_threshold_trades_recall_for_precision(self):
        scores = np.array([0.99, 0.9, 0.6, 0.6])
        correct = np.array([True, True, True, False])
        rows = precision_at_thresholds(scores, correct, n_gold=4, thresholds=(0.5, 0.8))
        assert rows[0] == {"min_score": 0.5, "kept": 4, "precision": 0.75, "recall": 0.75}
        assert rows[1] == {"min_score": 0.8, "kept": 2, "precision": 1.0, "recall": 0.5}

    def test_a_threshold_that_keeps_nothing_reports_zero_not_nan(self):
        rows = precision_at_thresholds(np.array([0.5]), np.array([True]), 1, thresholds=(0.9,))
        assert rows[0]["kept"] == 0
        assert rows[0]["precision"] == 0.0


def test_markdown_lists_bins_and_thresholds():
    result = reliability(np.array([0.95, 0.55]), np.array([True, False]), bins=10)
    result.update(
        {
            "dataset": "ncbi_disease",
            "split": "validation",
            "backend": "onnx",
            "n_sentences": 2,
            "n_gold": 2,
            "thresholds": precision_at_thresholds(
                np.array([0.95, 0.55]), np.array([True, False]), 2, (0.5, 0.9)
            ),
        }
    )
    markdown = to_markdown(result)
    assert "| 0.9-1.0 |" in markdown
    assert "| 0.5-0.6 |" in markdown
    assert "Expected calibration error" in markdown
    assert "| 0.9 | 1 |" in markdown
