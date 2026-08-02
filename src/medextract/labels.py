"""BIO tag helpers.

Kept separate from `data.py` so the alignment logic can be unit tested without
importing `datasets` (and therefore without touching the network).
"""

from __future__ import annotations

IGNORE_INDEX = -100
OUTSIDE = "O"


def entity_type(label: str) -> str | None:
    """Strip the BIO prefix: B-Disease becomes Disease. Returns None for O."""
    if label == OUTSIDE:
        return None
    return label.split("-", 1)[1] if "-" in label else label


def to_inside(label: str) -> str:
    """Turn a beginning tag into a continuation: B-Disease becomes I-Disease.

    Leaves I- and O untouched.

    Used when labelling continuation subwords: the second subword of a token
    that starts an entity is inside that entity, not another beginning.
    """
    if label.startswith("B-"):
        return "I-" + label[2:]
    return label


def align_labels_with_tokens(
    word_ids: list[int | None],
    word_labels: list[str],
    *,
    label_all_subwords: bool = False,
) -> list[str | int]:
    """Project word-level BIO labels onto subword tokens.

    Biomedical vocabulary fragments heavily -- "adenomatous" becomes several
    WordPiece tokens -- so labels given per word have to be mapped onto the
    subwords the model actually sees. `word_ids` comes from a fast tokeniser
    and gives, for each subword position, the index of the word it came from
    (None for special tokens).

    Special tokens always get IGNORE_INDEX so they contribute nothing to the
    loss. For continuation subwords there are two defensible conventions:

    - `label_all_subwords=False` (default): only the first subword of each word
      carries the label, the rest get IGNORE_INDEX. This matches the standard
      HuggingFace token-classification recipe and the published baselines, and
      is what the reported F1 should be produced with.
    - `label_all_subwords=True`: every subword is labelled, with B- converted
      to I- after the first. Included so the choice can be swept rather than
      assumed.

    Returns a list the same length as `word_ids`, holding label strings for
    supervised positions and IGNORE_INDEX for ignored ones.
    """
    aligned: list[str | int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            aligned.append(word_labels[word_id])
        elif label_all_subwords:
            aligned.append(to_inside(word_labels[word_id]))
        else:
            aligned.append(IGNORE_INDEX)
        previous_word_id = word_id

    return aligned


def labels_to_ids(aligned: list[str | int], label2id: dict[str, int]) -> list[int]:
    return [x if isinstance(x, int) else label2id[x] for x in aligned]


def build_label_maps(label_names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = dict(enumerate(label_names))
    return label2id, id2label
