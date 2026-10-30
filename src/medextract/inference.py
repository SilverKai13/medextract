"""Runtime NER: tokenise, run a backend, decode BIO tags into character spans.

One decoding path is shared by the API, the benchmark and the ONNX parity
check, so a change to span aggregation cannot make the served output disagree
with the measured output.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

log = logging.getLogger(__name__)

Backend = Literal["torch", "onnx"]


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 4),
        }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _load_id2label(model_dir: Path) -> dict[int, str]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if not id2label:
        raise ValueError(f"{model_dir}/config.json has no id2label mapping.")
    return {int(k): v for k, v in id2label.items()}


def _collapse_to_words(
    label_ids: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    special_mask: np.ndarray,
    word_ids: list[int | None],
) -> list[tuple[int, float, int, int]]:
    """Reduce subword predictions to one prediction per word.

    A word's label and score come from its *first* subword, and its span runs
    from that subword's start to the last subword's end.

    This is not a tidiness choice, it is forced by how the model was trained.
    Under the default alignment convention only the first subword of each word
    carries a label; every continuation gets IGNORE_INDEX and contributes
    nothing to the loss. So a continuation subword's prediction was never
    supervised and is not evidence of anything.

    Reading them anyway is actively harmful, because biomedical words fragment
    into many pieces and a single stray `O` in the middle splits an entity.
    Observed on a real fine-tuned model, `nephrotoxicity` tokenises to
    `ne ##ph ##rot ##ox ##icit ##y` and came back predicted
    `B-Disease, O, I-Disease, O, O, O` -- decoding that per subword yields the
    two fragments `"ne"` and `"rot"` instead of one entity.

    Evaluation never saw this: `metrics.decode_predictions` drops positions
    whose gold label is IGNORE_INDEX, so it scores first subwords only. That is
    exactly why the bug was invisible in the F1 -- the reported number was
    honest about the convention it measured, while `/extract` returned shredded
    spans for most multi-subword entities.

    Returns (label_id, score, start_char, end_char) per word, in order.
    """
    words: list[tuple[int, float, int, int]] = []
    current_word: int | None = None

    for position, word_id in enumerate(word_ids):
        if word_id is None or special_mask[position]:
            continue
        token_start, token_end = int(offsets[position][0]), int(offsets[position][1])
        if token_end <= token_start:
            # Zero-width piece (some tokenisers emit these for stripped chars).
            continue

        if word_id == current_word and words:
            label, score, start, _ = words[-1]
            words[-1] = (label, score, start, token_end)
        else:
            words.append(
                (int(label_ids[position]), float(probabilities[position]), token_start, token_end)
            )
            current_word = word_id

    return words


def decode_entities(
    text: str,
    label_ids: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    special_mask: np.ndarray,
    id2label: dict[int, str],
    word_ids: list[int | None] | None = None,
) -> list[Entity]:
    """Aggregate per-token BIO predictions into character-offset spans.

    An I- tag with no matching B- before it opens a span rather than being
    dropped: models do emit that, and discarding it silently loses recall.

    With `word_ids` supplied, subwords are first collapsed to whole words -- see
    `_collapse_to_words` for why that is required rather than optional. Without
    it each position is treated as its own unit, which is the right granularity
    when the caller is working at token level already.
    """
    if word_ids is None:
        units = [
            (int(label_ids[position]), float(probabilities[position]), int(start), int(end))
            for position, (start, end) in enumerate(offsets)
            if not special_mask[position] and int(end) > int(start)
        ]
    else:
        units = _collapse_to_words(label_ids, probabilities, offsets, special_mask, word_ids)

    entities: list[Entity] = []
    current_type: str | None = None
    start_char = 0
    end_char = 0
    scores: list[float] = []

    def flush() -> None:
        nonlocal current_type, scores
        if current_type is not None and end_char > start_char:
            entities.append(
                Entity(
                    text=text[start_char:end_char],
                    label=current_type,
                    start=start_char,
                    end=end_char,
                    score=float(np.mean(scores)) if scores else 0.0,
                )
            )
        current_type = None
        scores = []

    for label_id, score, unit_start, unit_end in units:
        label = id2label.get(label_id, "O")

        if label == "O":
            flush()
            continue

        prefix, _, etype = label.partition("-")
        etype = etype or label

        if prefix == "B" or current_type != etype:
            flush()
            current_type = etype
            start_char = unit_start
            end_char = unit_end
            scores = [score]
        else:
            end_char = unit_end
            scores.append(score)

    flush()
    return entities


def _dedupe(entities: list[Entity]) -> list[Entity]:
    """Overlapping windows can emit the same span twice; keep the best score."""
    best: dict[tuple[int, int, str], Entity] = {}
    for entity in entities:
        key = (entity.start, entity.end, entity.label)
        if key not in best or entity.score > best[key].score:
            best[key] = entity
    return sorted(best.values(), key=lambda e: (e.start, e.end))


class NerEngine:
    """Loads a fine-tuned model once and extracts entities from raw text.

    Long inputs are windowed with overlap rather than truncated, so an entity
    at character 8000 is still found. Windows are decoded independently and
    their spans merged.
    """

    def __init__(
        self,
        model_dir: str | Path,
        backend: Backend = "onnx",
        *,
        max_length: int = 256,
        stride: int = 32,
        intra_op_threads: int | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        from medextract.tokenization import require_word_alignment

        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory {self.model_dir} does not exist. Train a model first "
                "(python -m medextract.train) or point MEDEXTRACT_MODEL_DIR at one."
            )
        self.backend: Backend = backend
        self.max_length = max_length
        self.stride = stride

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        require_word_alignment(self.tokenizer, purpose="character spans")
        self.id2label = _load_id2label(self.model_dir)

        # Filled in by the ONNX backend; the torch backend serves weights that
        # no export-time record describes, so it has nothing to verify against.
        self.model_sha256: str | None = None
        self.integrity_verified = False

        self.loaded_at = time.time()
        if backend == "torch":
            self._init_torch()
        elif backend == "onnx":
            self._init_onnx(intra_op_threads)
        else:
            raise ValueError(f"Unknown backend {backend!r}; expected 'torch' or 'onnx'.")

    def _init_torch(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification

        self._torch = torch
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self._input_names = {"input_ids", "attention_mask", "token_type_ids"}

    def _init_onnx(self, intra_op_threads: int | None) -> None:
        import onnxruntime as ort

        from medextract.integrity import verify_graph

        candidates = [self.model_dir / "model.onnx", self.model_dir / "model_optimized.onnx"]
        onnx_path = next((p for p in candidates if p.exists()), None)
        if onnx_path is None:
            raise FileNotFoundError(
                f"No model.onnx in {self.model_dir}. Run python -m medextract.export_onnx first."
            )

        # Before the session opens the file: a graph that is not the one the
        # parity check passed should fail here, with both hashes, not later as
        # inexplicable predictions. See integrity.py.
        self.model_sha256, self.integrity_verified = verify_graph(onnx_path)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads

        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        # Exported graphs vary in whether they take token_type_ids; only feed
        # the inputs this particular graph declares.
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.onnx_path = onnx_path

    def encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

    def logits(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run the backend on an already-tokenised batch."""
        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        if self.backend == "onnx":
            return self.session.run(None, {k: v.astype(np.int64) for k, v in feed.items()})[0]

        torch = self._torch
        tensors = {k: torch.as_tensor(v, dtype=torch.long) for k, v in feed.items()}
        with torch.no_grad():
            return self.model(**tensors).logits.numpy()

    def predict(self, texts: list[str]) -> list[list[Entity]]:
        if not texts:
            return []

        encoded = self.encode(texts)
        # Maps each window back to the text it came from.
        sample_map = np.asarray(encoded.get("overflow_to_sample_mapping", list(range(len(texts)))))
        model_inputs = {
            k: np.asarray(v)
            for k, v in encoded.items()
            if k in {"input_ids", "attention_mask", "token_type_ids"}
        }

        raw_logits = self.logits(model_inputs)
        probabilities = softmax(raw_logits, axis=-1)
        predicted = probabilities.argmax(axis=-1)
        confidence = probabilities.max(axis=-1)

        offsets = np.asarray(encoded["offset_mapping"])
        special = np.asarray(encoded["special_tokens_mask"]).astype(bool)
        attention = np.asarray(encoded["attention_mask"]).astype(bool)
        # Padding carries an offset of (0, 0) but so does [CLS]; mask both.
        ignore = special | ~attention

        per_text: list[list[Entity]] = [[] for _ in texts]
        for window in range(predicted.shape[0]):
            text_index = int(sample_map[window])
            per_text[text_index].extend(
                decode_entities(
                    texts[text_index],
                    predicted[window],
                    confidence[window],
                    offsets[window],
                    ignore[window],
                    self.id2label,
                    # Without these, decoding reads continuation subwords the
                    # model was never trained to label. See _collapse_to_words.
                    word_ids=encoded.word_ids(batch_index=window),
                )
            )

        return [_dedupe(entities) for entities in per_text]

    def predict_one(self, text: str) -> list[Entity]:
        return self.predict([text])[0]
