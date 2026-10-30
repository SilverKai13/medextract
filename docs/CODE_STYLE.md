# Code Style

## Tooling

Ruff does both linting and formatting. There is no separate black, isort or
flake8 — one tool, one config, in `pyproject.toml`.

```powershell
.\make.ps1 lint      # ruff check + ruff format --check
.\make.ps1 format    # ruff check --fix + ruff format
```

| Setting | Value |
|---|---|
| `line-length` | 100 |
| `target-version` | `py310` |
| Rule sets | `E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`, `RUF` |
| Ignored | `B008` — `Depends()` in a default argument is the documented FastAPI idiom |
| Per-file | `tests/*` exempt from `E501` |

CI fails on either check, so formatting is not a review topic.

## Comments

This is the convention that most shapes how the code reads, so it gets the most
space.

**Comments explain why, never what.** If a line needs a comment saying what it
does, rewrite the line.

```python
# Padding carries an offset of (0, 0) but so does [CLS]; mask both.
ignore = special | ~attention
```

That is worth writing. `# combine the masks` is not.

**Prefer the reason that would otherwise be re-litigated.** The most valuable
comments answer "why not the obvious thing?"

```python
# 429 is a rate-limit signal, not a permanent failure.
if response.status_code == 429:
```

```python
# Literal rather than a starlette constant: the name for 413 was
# renamed between versions and this pins the wire behaviour.
status_code=413,
```

**Module docstrings carry the argument.** Where a module embodies a decision,
the docstring makes the case — see `metrics.py` on why scoring is hand-rolled,
`tokenization.py` on why casing is detected rather than assumed, `data.py` on
why loading has three sources. A reader who disagrees can at least see what was
weighed.

**Name the silent failure.** Several things here break without raising. Those
get an explicit comment saying so, because a future reader has no other signal:

```python
# Getting this wrong does not throw. The loss still falls, training still
# looks healthy, and the F1 is quietly wrong.
```

**Do not comment out code.** Delete it; git remembers.

## Naming

- Functions and variables: `snake_case`, descriptive over short.
  `label_all_subwords`, not `las`.
- Private helpers: leading underscore (`_prf`, `_dedupe`, `_load_id2label`).
- Constants: `UPPER_SNAKE` at module scope (`IGNORE_INDEX`, `DEFAULT_OPSET`,
  `PARITY_TEXTS`).
- Tests: a sentence about behaviour.
  `test_invalid_continuation_is_discarded_not_repaired`, not `test_spans_2`.
  The name should tell you what broke without opening the file.

## Types

`from __future__ import annotations` at the top of every module, so modern
syntax works on 3.10 and annotations stay cheap.

Public functions are annotated. Internal one-liners need not be. `TYPE_CHECKING`
guards imports that exist only for annotations:

```python
if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from datasets import DatasetDict
```

No type checker runs in CI. Annotations are documentation that happens to be
machine-readable.

## Imports

**Heavy imports go inside the function that needs them**, never at module
scope:

```python
def train(config: RunConfig, output_dir: Path) -> dict:
    import mlflow
    from transformers import Trainer, TrainingArguments
```

This is not stylistic. The serving container has no torch, no datasets and no
mlflow, and must still import `medextract.api.main`. A module-level
`import torch` anywhere in the import graph breaks the container.

Light imports (`numpy`, `httpx`, stdlib, first-party) go at the top normally.
Ruff's `I` rules handle ordering.

## Errors

**Error messages say what to do next.** An error a reader cannot act on costs
real time.

```python
raise FileNotFoundError(
    f"Model directory {self.model_dir} does not exist. Train a model first "
    "(python -m medextract.train) or point MEDEXTRACT_MODEL_DIR at one."
)
```

Several tests match on message content, so this is contract, not decoration.

**Fail loudly where a silent failure is expensive.** `build_index` refuses to
index with fallback hashing vectors; `export_onnx` aborts on parity mismatch;
`load_fast_tokenizer` raises rather than warns if case did not survive. The
rule of thumb: fail loudly when the wrong result *looks* fine.

**Fail softly where a partial result is still useful.** The API starts without
a model and reports `degraded`. `run_sweep` logs a failed arm and continues.
The rule of thumb: keep going when the failure is visible and localised.

## Dataclasses

`@dataclass(frozen=True)` for value objects (`Entity`, `DatasetSpec`); plain
`@dataclass` for mutable state (`AppState`, `RunConfig`, `PreparedData`).
`RunConfig` is mutable because `dataclasses.replace` builds the sweep grid from
it.

## CLI modules

Every runnable module follows the same shape, so a new one is predictable:

```python
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: list[str] | None = None) -> int: ...

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

`build_parser` is separate so flag wiring can be tested without running
anything — `test_cli.py` does exactly that. `main` takes `argv` for the same
reason.

## Tests

Covered in [TESTING.md](TESTING.md). The style points:

- Group related assertions in a `class Test...`; keep genuinely standalone
  tests as functions.
- One behaviour per test. A test asserting five things reports one failure.
- Docstrings on tests that pin a *decision* rather than a mechanism:
  `"""The official loader script emits one; an empty example is not data."""`

## Numbers

**Never type a measured number into a document.** The four
`<!-- MEDEXTRACT:*:START -->` blocks in `README.md` are rewritten by
`readme_sync.sync_readme` from generated output; `docs/sweep_table.md` comes
from `report.py`; `docs/benchmark.md` from `benchmark.py`.

A hand-transcribed number is a number that will be wrong later, and the whole
project's claim is that its numbers are real.

## British English

Prose and comments use British spelling — "tokenise", "normalised",
"behaviour". Identifiers from third-party APIs keep their own spelling
(`normalize_embeddings`, `l2_normalise` is ours, `initializer_range` is not).
Not important, but consistent.
