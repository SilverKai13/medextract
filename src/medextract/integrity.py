"""Artefact integrity: the graph being served is the graph the parity check passed.

`export_onnx.check_parity` writes the SHA-256 of `model.onnx` into
`parity.json` next to it. `NerEngine` hashes the file it is about to load and
compares. A partial mount, a truncated copy or a swapped graph then fails at
startup with both hashes in the message, rather than surfacing later as odd
predictions that nobody can explain.

Kept free of torch and onnxruntime imports so the serving path can use it
without pulling anything in.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PARITY_FILE = "parity.json"


class IntegrityError(ValueError):
    """The graph on disk is not the one recorded at export time."""


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streamed, so a 400 MB graph is hashed without being read into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_parity_record(onnx_dir: Path) -> dict[str, Any] | None:
    record_path = onnx_dir / PARITY_FILE
    if not record_path.exists():
        return None
    return json.loads(record_path.read_text(encoding="utf-8"))


def verify_graph(onnx_path: Path) -> tuple[str, bool]:
    """Hash the graph and check it against the export-time record, if there is one.

    Returns `(sha256, verified)`. `verified` is False when no record exists --
    an older export, or a graph that arrived some other way -- which is logged
    rather than refused, because a missing record is a weaker claim, not a
    contradiction. A record that disagrees with the file *is* a contradiction
    and raises.
    """
    actual = file_sha256(onnx_path)
    record = read_parity_record(onnx_path.parent)
    expected = (record or {}).get("model_sha256")

    if expected is None:
        log.warning(
            "no integrity record for %s (%s absent or has no model_sha256); serving it unverified",
            onnx_path,
            PARITY_FILE,
        )
        return actual, False

    if actual != expected:
        raise IntegrityError(
            f"{onnx_path} does not match the graph the parity check passed: "
            f"expected sha256 {expected}, found {actual}. The file was changed, "
            "truncated or replaced after export; re-run python -m medextract.export_onnx."
        )

    log.info("integrity ok: %s sha256=%s", onnx_path.name, actual[:16])
    return actual, True
