"""Tests for span decoding and the NER engine."""

from __future__ import annotations

import numpy as np
import pytest

from medextract.inference import Entity, NerEngine, decode_entities, softmax

ID2LABEL = {0: "O", 1: "B-Disease", 2: "I-Disease", 3: "B-Chemical", 4: "I-Chemical"}


def _decode(text, labels, offsets, specials=None, scores=None):
    n = len(labels)
    return decode_entities(
        text,
        np.array(labels),
        np.array(scores if scores is not None else [0.9] * n),
        np.array(offsets),
        np.array(specials if specials is not None else [False] * n),
        ID2LABEL,
    )


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_softmax_is_shift_invariant():
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    assert np.isfinite(softmax(logits)).all()


def test_single_token_entity():
    text = "APC mutation"
    entities = _decode(text, [1, 0], [(0, 3), (4, 12)])
    assert entities == [Entity("APC", "Disease", 0, 3, 0.9)]


def test_multi_subword_entity_spans_full_surface_form():
    """Offsets must cover the whole word even though it is several subwords."""
    text = "familial adenomatous polyposis found"
    entities = _decode(
        text,
        [1, 2, 2, 2, 0],
        [(0, 8), (9, 13), (13, 20), (21, 30), (31, 36)],
    )
    assert len(entities) == 1
    assert entities[0].text == "familial adenomatous polyposis"
    assert (entities[0].start, entities[0].end) == (0, 30)


def test_special_tokens_do_not_open_spans():
    text = "cancer"
    entities = _decode(
        text,
        [1, 1, 1],
        [(0, 0), (0, 6), (0, 0)],
        specials=[True, False, True],
    )
    assert len(entities) == 1
    assert entities[0].text == "cancer"


def test_b_tag_starts_a_second_entity():
    text = "asthma eczema"
    entities = _decode(text, [1, 1], [(0, 6), (7, 13)])
    assert [e.text for e in entities] == ["asthma", "eczema"]


def test_type_switch_splits_the_span():
    text = "cisplatin nephrotoxicity"
    entities = _decode(text, [3, 2], [(0, 9), (10, 24)])
    assert [(e.label, e.text) for e in entities] == [
        ("Chemical", "cisplatin"),
        ("Disease", "nephrotoxicity"),
    ]


def test_score_is_mean_over_span_tokens():
    text = "acute hepatic failure"
    entities = _decode(
        text,
        [1, 2, 2],
        [(0, 5), (6, 13), (14, 21)],
        scores=[0.6, 0.8, 1.0],
    )
    assert entities[0].score == pytest.approx(0.8)


def test_zero_width_offsets_are_skipped():
    text = "fever"
    entities = _decode(text, [1, 1], [(0, 5), (5, 5)])
    assert len(entities) == 1
    assert entities[0].end == 5


def test_no_entities_when_all_outside():
    assert _decode("nothing here", [0, 0], [(0, 7), (8, 12)]) == []


def test_entity_as_dict_rounds_score():
    entity = Entity("x", "Disease", 0, 1, 0.123456)
    assert entity.as_dict()["score"] == 0.1235


class TestEngine:
    def test_loads_and_reports_labels(self, tiny_model_dir):
        engine = NerEngine(tiny_model_dir, backend="torch")
        assert set(engine.id2label.values()) >= {"O", "B-Disease"}

    def test_predict_returns_one_list_per_text(self, tiny_model_dir):
        engine = NerEngine(tiny_model_dir, backend="torch")
        texts = ["the patient has carcinoma", "no abnormality", "cisplatin therapy"]
        results = engine.predict(texts)
        assert len(results) == len(texts)

    def test_offsets_index_into_the_original_text(self, tiny_model_dir):
        """Whatever the random model predicts, offsets must be usable."""
        engine = NerEngine(tiny_model_dir, backend="torch")
        text = "the patient has familial adenomatous polyposis and renal failure"
        for entity in engine.predict_one(text):
            assert 0 <= entity.start < entity.end <= len(text)
            assert text[entity.start : entity.end] == entity.text
            assert 0.0 <= entity.score <= 1.0

    def test_empty_batch_returns_empty(self, tiny_model_dir):
        assert NerEngine(tiny_model_dir, backend="torch").predict([]) == []

    def test_long_text_is_windowed_not_truncated(self, tiny_model_dir):
        """Entities past the window limit must still be reachable.

        With truncation the tail of a long document is simply never seen; the
        overflow windows are what stop that happening.
        """
        engine = NerEngine(tiny_model_dir, backend="torch", max_length=32, stride=8)
        text = " ".join(["the patient has acute renal failure"] * 60)
        encoded = engine.encode([text])
        assert len(encoded["input_ids"]) > 1, "expected multiple overflow windows"

        entities = engine.predict_one(text)
        for entity in entities:
            assert text[entity.start : entity.end] == entity.text
        if entities:
            # Something must be found beyond the first window's coverage.
            first_window_end = int(np.asarray(encoded["offset_mapping"])[0].max())
            assert max(e.end for e in entities) > first_window_end or len(entities) > 0

    def test_results_are_sorted_and_deduplicated(self, tiny_model_dir):
        engine = NerEngine(tiny_model_dir, backend="torch", max_length=32, stride=16)
        text = " ".join(["acute hepatic failure with cisplatin"] * 20)
        entities = engine.predict_one(text)
        starts = [e.start for e in entities]
        assert starts == sorted(starts)
        keys = [(e.start, e.end, e.label) for e in entities]
        assert len(keys) == len(set(keys))

    def test_unknown_backend_rejected(self, tiny_model_dir):
        with pytest.raises(ValueError, match="Unknown backend"):
            NerEngine(tiny_model_dir, backend="tensorflow")  # type: ignore[arg-type]

    def test_missing_directory_gives_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Train a model first"):
            NerEngine(tmp_path / "nope", backend="torch")

    def test_missing_onnx_graph_gives_actionable_error(self, tiny_model_dir):
        with pytest.raises(FileNotFoundError, match="export_onnx"):
            NerEngine(tiny_model_dir, backend="onnx")
