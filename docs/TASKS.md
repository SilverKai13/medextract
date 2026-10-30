# Tasks

Execution breakdown of the work that took this project from "written and
tested, never run" to "run end to end with measured results". Remaining work is
in [TODO.md](TODO.md).

Status: **Done** · **Blocked** · **Open**

---

## Phase 0 — Environment

| # | Task | Status | Notes |
|---|---|---|---|
| 0.1 | Audit the codebase and identify blockers | Done | Six by inspection, four during execution, two by reading real output. [BUGS.md](BUGS.md) |
| 0.2 | Verify hardware and network access | Done | RTX 3050 Ti 4 GB, CUDA 12.7, Docker + compose v2.2.1. HTTP 200 from all four Hub repos and E-utilities |
| 0.3 | Create an isolated virtualenv | Done | `--system-site-packages` to reuse the CUDA torch build; venv shadows the incompatible system `transformers` 5.8.1 |
| 0.4 | Install pinned dependencies | Done | `transformers` 4.57.6, `datasets` 4.8.5, `mlflow` 3.15.1, `accelerate` 1.14.0, `onnxruntime` 1.26.0 |
| 0.5 | Point the HuggingFace cache outside the synced folder | Done | `HF_HOME=C:\hf-cache`, set by `make.ps1` |
| 0.6 | Configure `.env` | Done | Contact address set for E-utilities; `.env` is gitignored |
| 0.7 | Establish a green baseline before changing anything | Done | 163 passed, 1 skipped — after fixing B7 |

## Phase 1 — Fix the blockers

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Three-source corpus loading (B1) | Done | Parquet branch → plain Hub → upstream files. Source recorded in MLflow. [ADR-011](DECISIONS.md#adr-011) |
| 1.2 | CoNLL and JSONL parsers | Done | Match the official loader's field handling; drop the empty end-of-file example |
| 1.3 | Tokenizer casing guard (B2) | Done | Detect from the vocabulary, reload, verify with a probe, raise if still wrong. [ADR-012](DECISIONS.md#adr-012) |
| 1.4 | Single MLflow URI helper (B3) | Done | `config.mlflow_tracking_uri()`, used by both `train.py` and `report.py` |
| 1.5 | Effective-batch resolution (B5) | Done | `resolve_batching()`; product always equals `batch_size`. [ADR-013](DECISIONS.md#adr-013) |
| 1.6 | OOM retry in the sweep | Done | Retries once at half the per-device batch; an OOM becomes a slower arm, not a lost one |
| 1.7 | fp16 auto-detection in the CLI sweep (B5) | Done | `--no-fp16` to override |
| 1.8 | Checkpoint pruning and metadata-only MLflow artefacts | Done | ~3.7 GB of sweep output instead of ~8 GB. [ADR-016](DECISIONS.md#adr-016) |
| 1.9 | Windows task runner (B6) | Done | `make.ps1`, every Makefile target |
| 1.10 | pyarrow/torch load-order guard (B8) | Done | Full suite went from exit 139 to exit 0. [ADR-015](DECISIONS.md#adr-015) |
| 1.11 | `USE_TF=0` guard (B9) | Done | Declines the TF import path rather than installing a shim |
| 1.12 | Migrate MLflow to SQLite (B10) | Done | The file store raises in MLflow 3. [ADR-014](DECISIONS.md#adr-014) |
| 1.13 | Word-level span decoding (B11) | Done | Found by reading real model output, not by a test. [ADR-017](DECISIONS.md#adr-017) |
| 1.14 | Explicit encoding on all text I/O (B12) | Done | Enforced by an AST-level test |
| 1.15 | Tests for all of the above | Done | 37 new tests across `test_corpus_sources.py` and `test_inference.py` |
| 1.16 | Lint and format clean | Done | `ruff check` and `ruff format --check` both pass |

## Phase 2 — Run it

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Training smoke test | Done | 0.05 epochs end to end, before committing GPU hours |
| 2.2 | Full nine-arm sweep | Done | Results in [sweep_table.md](sweep_table.md) and [PROGRESS.md](PROGRESS.md) |
| 2.3 | Generate the comparison table | Done | `report.py` from the tracking store, never transcribed |
| 2.4 | ONNX export with parity check | Done | Logits, argmax and decoded spans all verified |
| 2.5 | Export the query encoder | Done | Pooling read from the model's own config |
| 2.6 | Latency benchmark | Done | Percentiles with hardware recorded, [benchmark.md](benchmark.md) |
| 2.7 | Fetch PubMed abstracts | Done | E-utilities, rate-limited, contact address sent |
| 2.8 | Build the Qdrant index | Done | Real encoder, hashing fallback refused |
| 2.9 | Build and size-check the container | Done | Against the 1.5 GB budget |
| 2.10 | Verify the live API | Done | `/health`, `/extract`, `/extract/batch`, `/search` |
| 2.11 | Rewrite the README result blocks | Done | `readme_sync.sync_readme` |

## Phase 3 — Documentation

| # | Document | Status |
|---|---|---|
| 3.2 | [PROJECT.md](PROJECT.md) | Done |
| 3.3 | [PRD.md](PRD.md) | Done |
| 3.4 | [REQUIREMENTS.md](REQUIREMENTS.md) | Done |
| 3.5 | [SPEC.md](SPEC.md) | Done |
| 3.6 | [ARCHITECTURE.md](ARCHITECTURE.md) | Done |
| 3.7 | [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) | Done |
| 3.8 | [DATA_MODEL.md](DATA_MODEL.md) | Done |
| 3.9 | [DATABASE.md](DATABASE.md) | Done |
| 3.10 | [SECURITY.md](SECURITY.md) | Done |
| 3.11 | [TESTING.md](TESTING.md) | Done |
| 3.12 | [DEPLOYMENT.md](DEPLOYMENT.md) | Done |
| 3.13 | [ENVIRONMENT.md](ENVIRONMENT.md) | Done |
| 3.14 | [CODE_STYLE.md](CODE_STYLE.md) | Done |
| 3.15 | [DECISIONS.md](DECISIONS.md) | Done |
| 3.16 | [../CHANGELOG.md](../CHANGELOG.md) | Done |
| 3.17 | [TODO.md](TODO.md) | Done |
| 3.18 | [TASKS.md](TASKS.md) | Done |
| 3.19 | [BUGS.md](BUGS.md) | Done |
| 3.20 | [PROGRESS.md](PROGRESS.md) | Done |
| 3.20a | [RESULTS.md](RESULTS.md) | Done |
| 3.21 | [HANDOVER.md](HANDOVER.md) | Done |
| 3.22 | [README.md](README.md) — index | Done |

## Sequencing notes

Worth recording, because the order mattered:

- **Baseline before changes.** Phase 0.7 established a green suite *before* any
  edit, so every later failure was attributable. It immediately paid for itself
  by isolating B7 as pre-existing rather than something newly introduced.
- **Smoke test before the sweep.** Task 2.1 ran 0.05 epochs end to end. It cost
  a minute and would have caught B9 and B10 before an hour of GPU time, had
  they not already surfaced.
- **Stages run individually, not via `pipeline.py`.** The sweep is resumable
  through `sweep_results.json`, but a failure in a later stage should not
  discard finished ones. `pipeline.py` remains the unattended path and is
  exercised by its own tests.
- **Status documents written last.** [RESULTS.md](RESULTS.md),
  [PROGRESS.md](PROGRESS.md), [BUGS.md](BUGS.md) and
  [../CHANGELOG.md](../CHANGELOG.md) record what happened, so writing them
  before it happened would have made them fiction.
- **Results split from narrative.** [RESULTS.md](RESULTS.md) is the reference —
  every measurement with the command that produced it. [PROGRESS.md](PROGRESS.md)
  is the story of the run. Keeping both in one document meant the numbers were
  buried in prose and the prose was interrupted by tables.
