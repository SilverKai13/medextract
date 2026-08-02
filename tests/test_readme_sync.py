"""Tests for marker-based README rewriting."""

from __future__ import annotations

import pytest

from medextract.readme_sync import sync_readme

SAMPLE_README = """# MedExtract

<!-- MEDEXTRACT:STATUS:START -->
old status
<!-- MEDEXTRACT:STATUS:END -->

## Sweep

<!-- MEDEXTRACT:SWEEP_TABLE:START -->
| old | table |
<!-- MEDEXTRACT:SWEEP_TABLE:END -->

## Benchmark

<!-- MEDEXTRACT:BENCHMARK_TABLE:START -->
| old | bench |
<!-- MEDEXTRACT:BENCHMARK_TABLE:END -->

## Not done

<!-- MEDEXTRACT:NOT_DONE:START -->
old not-done text
<!-- MEDEXTRACT:NOT_DONE:END -->

## Licence

MIT.
"""


def _write(tmp_path, content=SAMPLE_README):
    path = tmp_path / "README.md"
    path.write_text(content)
    return path


class TestSyncReadme:
    def test_replaces_all_four_blocks(self, tmp_path):
        path = _write(tmp_path)
        sync_readme(
            path,
            sweep_table_md="| new | table |",
            benchmark_table_md="| new | bench |",
            status_text="new status",
            not_done_text="new not-done text",
        )
        text = path.read_text()
        assert "new status" in text
        assert "| new | table |" in text
        assert "| new | bench |" in text
        assert "new not-done text" in text
        assert "old status" not in text
        assert "old table" not in text
        assert "old bench" not in text
        assert "old not-done" not in text

    def test_content_outside_markers_is_untouched(self, tmp_path):
        path = _write(tmp_path)
        sync_readme(
            path,
            sweep_table_md="x",
            benchmark_table_md="y",
            status_text="z",
            not_done_text="w",
        )
        text = path.read_text()
        assert "# MedExtract" in text
        assert "## Licence" in text
        assert "MIT." in text

    def test_markers_survive_the_rewrite(self, tmp_path):
        path = _write(tmp_path)
        sync_readme(
            path, sweep_table_md="x", benchmark_table_md="y", status_text="z", not_done_text="w"
        )
        text = path.read_text()
        for marker in ("STATUS", "SWEEP_TABLE", "BENCHMARK_TABLE", "NOT_DONE"):
            assert f"<!-- MEDEXTRACT:{marker}:START -->" in text
            assert f"<!-- MEDEXTRACT:{marker}:END -->" in text

    def test_is_idempotent(self, tmp_path):
        path = _write(tmp_path)
        kwargs = {
            "sweep_table_md": "| new | table |",
            "benchmark_table_md": "| new | bench |",
            "status_text": "new status",
            "not_done_text": "new not-done text",
        }
        sync_readme(path, **kwargs)
        once = path.read_text()
        sync_readme(path, **kwargs)
        assert path.read_text() == once

    def test_missing_marker_raises_with_the_marker_name(self, tmp_path):
        broken = SAMPLE_README.replace("<!-- MEDEXTRACT:STATUS:START -->", "")
        path = _write(tmp_path, broken)
        with pytest.raises(ValueError, match="STATUS"):
            sync_readme(
                path, sweep_table_md="x", benchmark_table_md="y", status_text="z", not_done_text="w"
            )

    def test_real_readme_has_all_four_marker_pairs(self):
        """The actual project README, not a fixture -- catches drift."""
        from medextract.config import REPO_ROOT

        text = (REPO_ROOT / "README.md").read_text()
        for marker in ("STATUS", "SWEEP_TABLE", "BENCHMARK_TABLE", "NOT_DONE"):
            assert f"<!-- MEDEXTRACT:{marker}:START -->" in text
            assert f"<!-- MEDEXTRACT:{marker}:END -->" in text
