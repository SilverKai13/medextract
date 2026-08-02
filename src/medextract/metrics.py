"""Entity-level evaluation (strict IOB2 span matching).

Token accuracy is close to useless for this task -- most tokens are O, so a
model that predicts nothing scores above 90%. What matters is whether whole
spans are recovered exactly, which is what the F1 target in the brief refers to
and what published NCBI-Disease baselines report.

Scoring is implemented here rather than taken from seqeval. seqeval 1.2.2 ships
only an sdist whose setup.py no longer builds against current setuptools, which
makes it an unreliable thing to put in a Dockerfile or a CI job. The logic is
about sixty lines and is checked against seqeval in the test suite whenever
seqeval happens to be importable, so the shortcut is verified rather than
assumed.
"""

from __future__ import annotations

import numpy as np

from medextract.labels import IGNORE_INDEX, OUTSIDE

Span = tuple[str, int, int]  # (entity type, start token, end token exclusive)


def extract_spans(tags: list[str]) -> set[Span]:
    """Pull typed spans out of one IOB2 tag sequence, strictly.

    Only B-X opens a span. I-X extends a span of type X that is already open;
    an I-X in any other position is an invalid transition and is discarded
    rather than repaired. That matches seqeval's `mode="strict", scheme=IOB2`,
    which is how published NCBI-Disease numbers are produced -- scoring
    leniently here would make the reported F1 incomparable to the baseline the
    brief sets a target against.

    Note this is deliberately stricter than `inference.decode_entities`, which
    does recover an orphan I-X into a span. Discarding a real entity because
    the model emitted a malformed prefix is the right call when measuring
    against a baseline and the wrong call when answering a caller who wants the
    entity found. The test suite pins both behaviours.
    """
    spans: set[Span] = set()
    current_type: str | None = None
    start = 0

    def close(end: int) -> None:
        nonlocal current_type
        if current_type is not None:
            spans.add((current_type, start, end))
            current_type = None

    for i, tag in enumerate(tags):
        if tag == OUTSIDE or not tag:
            close(i)
            continue

        prefix, _, etype = tag.partition("-")
        etype = etype or tag

        if prefix == "B":
            close(i)
            current_type = etype
            start = i
        elif current_type != etype:
            # Invalid continuation: end whatever was open and drop this token.
            close(i)

    close(len(tags))
    return spans


def _prf(true_positives: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = true_positives / n_pred if n_pred else 0.0
    recall = true_positives / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def decode_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Drop ignored positions and turn label ids back into BIO strings."""
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    pred_seqs: list[list[str]] = []
    gold_seqs: list[list[str]] = []
    for pred_row, gold_row in zip(predictions, label_ids, strict=True):
        keep = gold_row != IGNORE_INDEX
        pred_seqs.append([id2label[int(p)] for p in pred_row[keep]])
        gold_seqs.append([id2label[int(g)] for g in gold_row[keep]])
    return pred_seqs, gold_seqs


def entity_scores(pred_seqs: list[list[str]], gold_seqs: list[list[str]]) -> dict[str, float]:
    """Micro-averaged and per-type precision/recall/F1 over exact spans.

    Returns flat scalars (`f1`, `Disease_f1`, ...) because MLflow metrics are
    scalars, and because per-type numbers are what make the README table say
    something rather than report one aggregate.
    """
    gold_by_type: dict[str, int] = {}
    pred_by_type: dict[str, int] = {}
    hits_by_type: dict[str, int] = {}

    total_gold = total_pred = total_hits = 0

    for pred_tags, gold_tags in zip(pred_seqs, gold_seqs, strict=True):
        gold_spans = extract_spans(gold_tags)
        pred_spans = extract_spans(pred_tags)
        hits = gold_spans & pred_spans

        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        total_hits += len(hits)

        for etype, _, _ in gold_spans:
            gold_by_type[etype] = gold_by_type.get(etype, 0) + 1
        for etype, _, _ in pred_spans:
            pred_by_type[etype] = pred_by_type.get(etype, 0) + 1
        for etype, _, _ in hits:
            hits_by_type[etype] = hits_by_type.get(etype, 0) + 1

    precision, recall, f1 = _prf(total_hits, total_pred, total_gold)
    metrics: dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(total_gold),
    }

    per_type_f1: list[float] = []
    for etype in sorted(set(gold_by_type) | set(pred_by_type)):
        p, r, f = _prf(
            hits_by_type.get(etype, 0),
            pred_by_type.get(etype, 0),
            gold_by_type.get(etype, 0),
        )
        metrics[f"{etype}_precision"] = p
        metrics[f"{etype}_recall"] = r
        metrics[f"{etype}_f1"] = f
        metrics[f"{etype}_support"] = float(gold_by_type.get(etype, 0))
        per_type_f1.append(f)

    metrics["macro_f1"] = float(np.mean(per_type_f1)) if per_type_f1 else 0.0
    return metrics


def build_compute_metrics(id2label: dict[int, str]):
    """Adapter for the HuggingFace Trainer's compute_metrics hook."""

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, label_ids = eval_pred
        pred_seqs, gold_seqs = decode_predictions(
            np.asarray(predictions), np.asarray(label_ids), id2label
        )
        return entity_scores(pred_seqs, gold_seqs)

    return compute_metrics
