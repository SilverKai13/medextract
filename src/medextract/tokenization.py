"""Loading a tokenizer that matches the checkpoint it is paired with.

There is a specific way this goes wrong that produces no error and no warning.
A checkpoint that ships no `tokenizer_config.json` -- BioBERT is the well-known
case -- gives `AutoTokenizer` nothing to read the casing behaviour from, so it
falls back to the class default, which for BERT is `do_lower_case=True`. Applied
to a *cased* vocabulary that is not merely suboptimal, it is destructive:

    APC          ->  a ##p ##c        (correct: AP ##C)
    EGFR         ->  e ##g ##f ##r    (correct: E ##G ##F ##R)
    Adenomatous  ->  ad ##eno ##mat ##ous  (correct: Aden ##oma ##to ##us)

The cased vocabulary has no lowercase entry for a gene symbol, so every one of
them shatters into single characters. Training still converges, the loss curve
still looks healthy, and the F1 is quietly several points lower.

So the casing is decided by looking at the vocabulary rather than trusting a
default or a substring of the model name, and then verified against a probe
before the tokenizer is handed back. Same reasoning as `labels.py`: isolate the
thing that fails silently and assert it.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Terms chosen because each one is destroyed differently by wrong-cased
# tokenisation: an all-caps gene symbol, a mixed-case one, and a long cased
# word that fragments.
CASING_PROBE = "Familial APC BRCA1 EGFR Adenomatous"

# bert-base-uncased carries uppercase only in its five special tokens, about
# 0.02% of a 30k vocabulary. bert-base-cased is over 20%. Anything above this
# threshold is a cased vocabulary; there is no checkpoint in between.
CASED_VOCAB_THRESHOLD = 0.01


def supports_word_alignment(tokenizer) -> bool:
    """Whether the tokenizer can map subwords back to words and to characters.

    Decided by trying it rather than by reading `is_fast`. The flag was a
    reliable proxy while transformers shipped a pure-Python implementation
    beside the Rust one; from 5.x every tokenizer is Rust-backed and reports
    `is_fast=True`, so a guard written against the flag either never fires or
    has to be pinned to a library version. Asking for `word_ids()` and an
    offset mapping on a probe is the actual requirement, and it holds or fails
    the same way on every version.
    """
    try:
        encoded = tokenizer(
            ["probe", "word"],
            is_split_into_words=True,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded.word_ids(batch_index=0)
        encoded["offset_mapping"]
    except (NotImplementedError, TypeError, ValueError, AttributeError, KeyError):
        return False
    return True


def require_word_alignment(tokenizer, *, purpose: str) -> None:
    """Raise with an actionable message when the tokenizer cannot do `purpose`."""
    if not supports_word_alignment(tokenizer):
        raise ValueError(
            f"A fast tokenizer is required for {purpose}: this one cannot report "
            "word_ids() or an offset mapping. Load with "
            "AutoTokenizer.from_pretrained(..., use_fast=True), or convert the "
            "checkpoint's vocabulary to a `tokenizers`-backed tokenizer."
        )


def vocabulary_is_cased(tokenizer) -> bool:
    """True when the vocabulary itself contains uppercase word pieces.

    This is an observation about the artifact rather than an inference from its
    name, which matters because the checkpoints that get this wrong are exactly
    the ones whose metadata is missing.

    Special and placeholder tokens are excluded before the ratio is taken.
    `[CLS]`, `[SEP]` and friends are uppercase in every BERT vocabulary, cased
    or not, so counting them carries no signal about the corpus -- and on a
    small vocabulary they are numerous enough to swamp the ratio on their own.
    """
    vocab = tokenizer.get_vocab()
    if not vocab:
        return False

    special = set(getattr(tokenizer, "all_special_tokens", None) or ())
    content = [
        token
        for token in vocab
        if token not in special and not (token.startswith("[") and token.endswith("]"))
    ]
    if not content:
        return False

    uppercase = sum(1 for token in content if any(character.isupper() for character in token))
    return (uppercase / len(content)) > CASED_VOCAB_THRESHOLD


def _preserves_case(tokenizer) -> bool:
    pieces = tokenizer.convert_ids_to_tokens(
        tokenizer(CASING_PROBE, add_special_tokens=False)["input_ids"]
    )
    return any(character.isupper() for piece in pieces for character in piece)


def load_fast_tokenizer(model_name_or_path: str | Path, **kwargs):
    """Load a fast tokenizer with the casing its vocabulary actually calls for.

    Raises rather than warns when the result is still wrong: a run that trains
    on shredded input is more expensive to discover later than a failure now.
    """
    from transformers import AutoTokenizer

    name = str(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)

    # word_ids() and offset_mapping are both needed downstream; without them
    # neither label alignment nor character spans are possible.
    require_word_alignment(tokenizer, purpose=f"label alignment ({name})")

    cased = vocabulary_is_cased(tokenizer)
    lowercasing = bool(getattr(tokenizer, "do_lower_case", False))

    if cased and lowercasing and "do_lower_case" not in kwargs:
        log.warning(
            "%s has a cased vocabulary but the tokenizer defaulted to lowercasing "
            "(no tokenizer_config.json in the checkpoint). Reloading with "
            "do_lower_case=False.",
            name,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=True, do_lower_case=False, **kwargs
        )

    if cased and not _preserves_case(tokenizer):
        raise ValueError(
            f"{name} has a cased vocabulary but its tokenizer still lowercases input. "
            f"Every capitalised term in {CASING_PROBE!r} would be split into single "
            "characters, which silently costs several F1. Pass do_lower_case=False "
            "explicitly, or fix the checkpoint's tokenizer_config.json."
        )

    log.info(
        "tokenizer %s loaded (cased_vocab=%s, do_lower_case=%s)",
        name,
        cased,
        getattr(tokenizer, "do_lower_case", "n/a"),
    )
    return tokenizer
