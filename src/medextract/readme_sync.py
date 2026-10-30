"""Rewrite the result tables in README.md from real pipeline output.

README.md carries four HTML-comment-delimited blocks -- status banner, sweep
table, benchmark table, and the "not done yet" section. This replaces the
content strictly between each `START`/`END` pair, so the pipeline can update
the README the same way a person would, without touching prose around them.

Markers, not surrounding text, are what gets matched: a README that has
drifted (a marker renamed or removed) fails loudly here instead of silently
leaving stale numbers in place.
"""

from __future__ import annotations

import re
from pathlib import Path

MARKERS = {
    "status": "STATUS",
    "sweep_table": "SWEEP_TABLE",
    "benchmark_table": "BENCHMARK_TABLE",
    "not_done": "NOT_DONE",
    "loadtest_table": "LOADTEST_TABLE",
}


def _replace_block(text: str, marker: str, replacement: str) -> str:
    start = f"<!-- MEDEXTRACT:{marker}:START -->"
    end = f"<!-- MEDEXTRACT:{marker}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README.md is missing the {marker} marker pair ({start} / {end})")
    return pattern.sub(f"{start}\n{replacement}\n{end}", text)


def sync_readme(
    readme_path: Path,
    *,
    sweep_table_md: str,
    benchmark_table_md: str,
    status_text: str,
    not_done_text: str,
    loadtest_table_md: str | None = None,
) -> None:
    """Rewrite the marked blocks in place. Idempotent.

    The load-test block is optional because it needs a live server to produce,
    which a pipeline run without Docker does not have; the other four are
    always available by the time this is called.
    """
    text = readme_path.read_text(encoding="utf-8")
    replacements = {
        MARKERS["status"]: status_text,
        MARKERS["sweep_table"]: sweep_table_md,
        MARKERS["benchmark_table"]: benchmark_table_md,
        MARKERS["not_done"]: not_done_text,
    }
    if loadtest_table_md is not None:
        replacements[MARKERS["loadtest_table"]] = loadtest_table_md
    for marker, replacement in replacements.items():
        text = _replace_block(text, marker, replacement.strip())
    readme_path.write_text(text, encoding="utf-8")
