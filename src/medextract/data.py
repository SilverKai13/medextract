"""Corpus loading and tokenisation for token classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from medextract.config import DATASETS, DatasetSpec
from medextract.labels import align_labels_with_tokens, build_label_maps, labels_to_ids

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from datasets import DatasetDict
    from transformers import PreTrainedTokenizerFast

log = logging.getLogger(__name__)


@dataclass
class PreparedData:
    tokenized: Any  # DatasetDict
    label_names: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]


def resolve_label_names(raw: DatasetDict, spec: DatasetSpec) -> list[str]:
    """Recover the BIO label vocabulary for a corpus.

    Prefers the dataset's own ClassLabel names so the ordering always matches
    the integers in the corpus. Falls back to the spec override for mirrors
    that ship the tags as bare ints.
    """
    split = next(iter(raw.values()))
    feature = split.features[spec.tags_column]
    names = getattr(getattr(feature, "feature", None), "names", None)
    if names:
        return list(names)
    if spec.label_names:
        return list(spec.label_names)
    raise ValueError(
        f"Cannot determine label names for {spec.hf_id}: the tags column is not a "
        "ClassLabel sequence and no label_names override is set in config.DATASETS."
    )


def load_corpus(
    dataset_key: str = "ncbi_disease", cache_dir: str | None = None
) -> tuple[DatasetDict, DatasetSpec]:
    from datasets import load_dataset

    if dataset_key not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_key!r}. Known: {sorted(DATASETS)}")
    spec = DATASETS[dataset_key]
    raw = load_dataset(spec.hf_id, spec.config_name, cache_dir=cache_dir)

    if spec.validation_split not in raw:
        log.warning(
            "%s has no %r split; holding out 10%% of train instead.",
            spec.hf_id,
            spec.validation_split,
        )
        split = raw["train"].train_test_split(test_size=0.1, seed=13)
        raw["train"] = split["train"]
        raw[spec.validation_split] = split["test"]

    return raw, spec


def build_tokenize_fn(
    tokenizer: PreTrainedTokenizerFast,
    spec: DatasetSpec,
    label_names: list[str],
    *,
    max_length: int = 256,
    label_all_subwords: bool = False,
):
    """Return a batched map() function producing input_ids/attention_mask/labels."""
    label2id, _ = build_label_maps(label_names)

    if not tokenizer.is_fast:
        # word_ids() only exists on the fast tokenisers; without it there is no
        # reliable way to map word labels onto subwords.
        raise ValueError(
            "A fast tokenizer is required for label alignment. Load with "
            "AutoTokenizer.from_pretrained(..., use_fast=True)."
        )

    def tokenize_and_align(batch: dict[str, list]) -> dict[str, list]:
        encoded = tokenizer(
            batch[spec.tokens_column],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        all_labels: list[list[int]] = []
        for i, tags in enumerate(batch[spec.tags_column]):
            word_labels = [label_names[t] if isinstance(t, int) else t for t in tags]
            aligned = align_labels_with_tokens(
                encoded.word_ids(batch_index=i),
                word_labels,
                label_all_subwords=label_all_subwords,
            )
            all_labels.append(labels_to_ids(aligned, label2id))

        encoded["labels"] = all_labels
        return encoded

    return tokenize_and_align


def prepare(
    tokenizer: PreTrainedTokenizerFast,
    dataset_key: str = "ncbi_disease",
    *,
    max_length: int = 256,
    label_all_subwords: bool = False,
    cache_dir: str | None = None,
) -> PreparedData:
    raw, spec = load_corpus(dataset_key, cache_dir=cache_dir)
    label_names = resolve_label_names(raw, spec)
    label2id, id2label = build_label_maps(label_names)

    tokenize_fn = build_tokenize_fn(
        tokenizer,
        spec,
        label_names,
        max_length=max_length,
        label_all_subwords=label_all_subwords,
    )
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=next(iter(raw.values())).column_names,
        desc="tokenising",
    )
    return PreparedData(tokenized, label_names, label2id, id2label)
