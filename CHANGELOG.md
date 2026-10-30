# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`medextract.tokenization`** — a shared fast-tokenizer loader that decides
  casing from the vocabulary rather than trusting a checkpoint's (frequently
  absent) metadata, then verifies the decision against a probe string and
  raises if case did not survive. Used by `train.py` and
  `retrieval/export_embedder.py`.
- **Three-source corpus loading** in `data.py`. `load_corpus` tries the Hub's
  parquet conversion, a plain Hub load, then direct download of the upstream
  corpus files, and reports which answered. Adds `parse_conll` and
  `parse_tagged_jsonl`, plus `direct_urls` / `direct_format` on `DatasetSpec`.
- **Corpus provenance in the run record.** `PreparedData` now carries `source`
  and `split_sizes`, both logged to MLflow alongside the per-device batch,
  accumulation steps and whether the tokenizer lowercases.
- **`RunConfig.max_per_device_batch`** and `resolve_batching()`, splitting an
  effective batch size into per-device batch and gradient accumulation whose
  product is always preserved.
- **OOM recovery in the sweep.** An arm that runs out of GPU memory is retried
  once at half the per-device batch instead of being lost.
- **`config.mlflow_tracking_uri()`** — one helper used by both `train.py` and
  `report.py`, so they cannot disagree about where runs live.
- **`RunConfig.keep_checkpoints` and `log_model_artifact`**, both defaulting to
  off, pruning per-epoch checkpoints and logging model metadata rather than
  ~430 MB of weights per arm.
- **`make.ps1`** — a Windows task runner mirroring every Makefile target,
  routed through the project virtualenv, with `HF_HOME` pointed outside the
  repository and `PYTHONUTF8=1` set.
- **Import-time environment guards** in `medextract/__init__.py`: pyarrow is
  preloaded before torch can claim the same native libraries, and `USE_TF=0` is
  set so a TensorFlow install on the machine cannot break `Trainer`.
- **Word-level span aggregation** in `inference.py`. `_collapse_to_words`
  reduces subword predictions to one per word before decoding, taking label and
  score from the first subword — the only one training supervises.
- **`tests/test_corpus_sources.py`** — 31 tests covering the corpus parsers,
  the three-source fallback order, the casing guard, effective-batch
  arithmetic, sweep fp16 defaults, the tracking URI, and an AST-level check
  that every text read and write names its encoding.
- **`tests/test_inference.py::TestSubwordAggregation`** — 8 tests pinning
  word-level decoding, including the exact fragmentation observed in the wild.
- **Documentation set** — 22 documents under `docs/` and this changelog. Index at [docs/README.md](docs/README.md). Measurements live in
  [docs/RESULTS.md](docs/RESULTS.md) as a reference; the narrative of the run
  that produced them is in [docs/PROGRESS.md](docs/PROGRESS.md).

### Changed

- **MLflow now tracks to SQLite** (`sqlite:///mlflow.db`) rather than the
  `./mlruns` file store, which MLflow 3 refuses to open. Also avoids writing
  thousands of tiny files inside a synced folder.
- **`ncbi_disease` now resolves as `ncbi/ncbi_disease`** and carries an explicit
  `label_names` ordering, required by the direct-download path.
- **`sweep.main()` auto-detects CUDA for mixed precision**, matching what
  `pipeline.py` already did. `--no-fp16` overrides. Previously a CLI sweep
  trained in fp32 on a GPU and took roughly twice as long.
- **`per_device_eval_batch_size` now matches the train batch** rather than
  `max(batch_size, 32)`, so evaluation cannot fail on a card that trained fine.
- **`load_corpus` returns `(dataset, spec, source)`** rather than
  `(dataset, spec)`.
- **`train.py` and `retrieval/export_embedder.py` load tokenizers through
  `load_fast_tokenizer`** instead of calling `AutoTokenizer` directly.

### Fixed

Twelve defects, documented with root causes in [docs/BUGS.md](docs/BUGS.md).
Four of them failed silently — including one that left the headline F1 correct
while degrading what the API actually returned.

- **B1** `load_dataset("ncbi_disease")` failed outright — the Hub repository is
  script-only and `datasets` 3.0 removed script execution. Training could not
  start.
- **B2** BioBERT silently lowercased its own cased vocabulary. With no
  `tokenizer_config.json` in the checkpoint, `AutoTokenizer` defaulted to
  `do_lower_case=True`, turning `APC` into `a ##p ##c` and every other gene
  symbol into single characters. Nothing raised; the F1 was quietly lower.
- **B3** The MLflow tracking URI was built by string interpolation and came out
  malformed on Windows, so runs were written where nothing looked for them.
- **B4** The project was not installed, and the system `transformers` 5.8.1
  violates the documented `<5` pin.
- **B5** No gradient accumulation anywhere, and a CLI sweep never enabled fp16.
- **B6** Every documented command was unrunnable on Windows.
- **B7** `test_readme_sync.py` read the README without an explicit encoding,
  failing on Windows' cp1252 default.
- **B8** The full test suite segfaulted (exit 139) from a pyarrow/torch native
  load-order conflict. The training pipeline was exposed to the same fault.
- **B9** A TensorFlow install with Keras 3 made `transformers.Trainer` raise at
  import.
- **B10** MLflow 3 refuses the filesystem tracking backend.
- **B11** `/extract` shredded most multi-subword entities into fragments —
  `Familial adenomatous polyposis` came back as `'F'` plus
  `'amilial adenomatous polyposis'`, and `nephrotoxicity` as `'ne'` and
  `'rot'`. Decoding read continuation subwords that training never supervised.
  **Invisible to the F1**, because evaluation drops exactly those positions, so
  the headline metric stayed correct while the product degraded.
- **B12** `benchmark.py` wrote `docs/benchmark.md` with the platform default
  encoding, producing a file that was not valid UTF-8 on Windows. Every text
  read and write in the package and test suite now names its encoding, enforced
  by a test.

- Also corrected: `pipeline.not_done_text` claimed every remaining gap was a
  scope decision, and the README's image-size estimate was 42% high because it
  added a dependency measurement to a base image that already provides the
  Python runtime underneath it.

### Verified

- Full nine-arm sweep run on an RTX 3050 Ti (4 GB). Results in
  [docs/sweep_table.md](docs/sweep_table.md) and
  [docs/PROGRESS.md](docs/PROGRESS.md).
- ONNX export passing the three-level parity check: logit tolerance, argmax
  agreement, identical decoded spans.
- 3,000 PubMed abstracts fetched and indexed into Qdrant with the real encoder.
- Container built and measured against the 1.5 GB budget.
- 200 tests passing offline in about 35 seconds; `ruff check` and
  `ruff format --check` clean.

## [0.1.0]

Initial implementation: training with label alignment, entity-level IOB2
scoring, a nine-arm MLflow sweep, ONNX export with a parity check, a FastAPI
service with request tracing and windowed inference, an E-utilities client, a
Qdrant index, a multi-stage Dockerfile and a CI workflow.

Written and tested offline against a stub model. Never executed against real
data — the build environment had no GPU and blocked `huggingface.co` and
`eutils.ncbi.nlm.nih.gov`.
