"""Is the entity `score` a probability, or just an ordering?

`/extract` returns a score per entity: the mean softmax probability over the
span's tokens. The API has always said it "orders predictions usefully", and
`min_score` invites callers to treat it as a probability of correctness. Those
are different claims, and only one of them had been measured.

This runs the served model over a labelled split, marks each predicted entity
as correct (exact span and type in the gold annotation) or not, bins the
predictions by score and compares the mean score in each bin with the fraction
that were actually correct. Expected calibration error is the support-weighted
mean gap. The reliability table is what tells a caller whether `min_score=0.9`
means roughly nine in ten.

    python -m medextract.calibration --model-dir artifacts/onnx --split validation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from medextract.config import DEFAULT_DATASET, REPO_ROOT
from medextract.metrics import extract_spans

log = logging.getLogger(__name__)

DEFAULT_BINS = 10


def gold_entities(tokens: list[str], tags: list[str]) -> set[tuple[int, int, str]]:
    """Gold spans as character offsets into the space-joined sentence.

    The corpora are word-tokenised, so the sentence the model sees is the words
    joined by single spaces; a word's character span follows from its index.
    """
    starts: list[int] = []
    position = 0
    for token in tokens:
        starts.append(position)
        position += len(token) + 1

    spans: set[tuple[int, int, str]] = set()
    for etype, first, end in extract_spans(tags):  # end is exclusive, as in metrics.py
        spans.add((starts[first], starts[end - 1] + len(tokens[end - 1]), etype))
    return spans


def reliability(scores: np.ndarray, correct: np.ndarray, *, bins: int = DEFAULT_BINS) -> dict:
    """Bin by score and compare mean score with observed precision per bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    # np.digitize puts a score of exactly 1.0 past the last edge; clip it back.
    indices = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, bins - 1)

    rows: list[dict[str, Any]] = []
    ece = 0.0
    total = len(scores)
    for b in range(bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_score = float(scores[mask].mean())
        precision = float(correct[mask].mean())
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "count": count,
                "mean_score": round(mean_score, 4),
                "precision": round(precision, 4),
                "gap": round(precision - mean_score, 4),
            }
        )
        ece += (count / total) * abs(precision - mean_score)

    return {
        "n_predictions": total,
        "n_correct": int(correct.sum()),
        "overall_precision": round(float(correct.mean()), 4) if total else 0.0,
        "mean_score": round(float(scores.mean()), 4) if total else 0.0,
        "ece": round(ece, 4),
        "bins": rows,
    }


def precision_at_thresholds(
    scores: np.ndarray, correct: np.ndarray, n_gold: int, thresholds: tuple[float, ...]
) -> list[dict[str, Any]]:
    """What `min_score` buys: precision and recall of what survives each cut-off."""
    rows = []
    for threshold in thresholds:
        kept = scores >= threshold
        n_kept = int(kept.sum())
        hits = int(correct[kept].sum())
        rows.append(
            {
                "min_score": threshold,
                "kept": n_kept,
                "precision": round(hits / n_kept, 4) if n_kept else 0.0,
                "recall": round(hits / n_gold, 4) if n_gold else 0.0,
            }
        )
    return rows


def evaluate(
    model_dir: Path,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = "validation",
    backend: str = "onnx",
    bins: int = DEFAULT_BINS,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
    limit: int | None = None,
) -> dict[str, Any]:
    from medextract.data import load_corpus, resolve_label_names
    from medextract.inference import NerEngine

    raw, spec, source = load_corpus(dataset)
    label_names = resolve_label_names(raw, spec)
    rows = raw[split] if limit is None else raw[split].select(range(min(limit, len(raw[split]))))

    engine = NerEngine(model_dir, backend=backend)  # type: ignore[arg-type]

    scores: list[float] = []
    correct: list[bool] = []
    n_gold = 0
    batch = 64
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        sentences = [" ".join(tokens) for tokens in chunk[spec.tokens_column]]
        golds = [
            gold_entities(tokens, [label_names[t] if isinstance(t, int) else t for t in tags])
            for tokens, tags in zip(chunk[spec.tokens_column], chunk[spec.tags_column], strict=True)
        ]
        for entities, gold in zip(engine.predict(sentences), golds, strict=True):
            n_gold += len(gold)
            for entity in entities:
                scores.append(entity.score)
                correct.append((entity.start, entity.end, entity.label) in gold)

    score_array = np.asarray(scores, dtype=float)
    correct_array = np.asarray(correct, dtype=bool)
    result = reliability(score_array, correct_array, bins=bins)
    result.update(
        {
            "model_dir": str(model_dir),
            "backend": backend,
            "dataset": dataset,
            "split": split,
            "corpus_source": source,
            "n_sentences": len(rows),
            "n_gold": n_gold,
            "thresholds": precision_at_thresholds(score_array, correct_array, n_gold, thresholds),
        }
    )
    return result


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"Reliability of `score` on {result['dataset']} {result['split']} "
        f"({result['n_sentences']} sentences, {result['n_gold']} gold entities, "
        f"{result['n_predictions']} predictions), `{result['backend']}` backend.",
        "",
        "| score bin | predictions | mean score | precision | gap |",
        "|---|---|---|---|---|",
    ]
    for row in result["bins"]:
        lines.append(
            f"| {row['bin']} | {row['count']} | {row['mean_score']:.4f} | "
            f"{row['precision']:.4f} | {row['gap']:+.4f} |"
        )
    lines += [
        "",
        f"Expected calibration error **{result['ece']:.4f}**; overall precision "
        f"{result['overall_precision']:.4f} against a mean score of {result['mean_score']:.4f}.",
        "",
        "| min_score | entities kept | precision | recall |",
        "|---|---|---|---|",
    ]
    for row in result["thresholds"]:
        lines.append(
            f"| {row['min_score']} | {row['kept']} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure whether entity scores are calibrated.")
    p.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "onnx")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="validation")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--limit", type=int, default=None, help="sentences to score (default: all)")
    p.add_argument("--json-out", type=Path, default=REPO_ROOT / "docs" / "calibration.json")
    p.add_argument("--markdown-out", type=Path, default=REPO_ROOT / "docs" / "calibration.md")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)
    result = evaluate(
        ns.model_dir,
        dataset=ns.dataset,
        split=ns.split,
        backend=ns.backend,
        bins=ns.bins,
        limit=ns.limit,
    )
    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    ns.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = to_markdown(result)
    ns.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
