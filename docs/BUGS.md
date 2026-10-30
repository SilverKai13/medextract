# Bugs

Defects found and fixed while taking this project from "written and tested" to
"run end to end on real data". Every one of them blocked or silently degraded a
real run.

Six were found by inspection before the first run, four surfaced during setup
and execution, and two were found only by reading what the finished system
actually produced. **None had a failing test** before the fix — which is the
common thread: each fails silently, at import time, or only on a platform CI
does not cover.

Status key: **Fixed** — corrected and covered by a test. **Fixed (no test)** —
corrected, not unit-testable in this environment.

---

## B1 — `load_dataset("ncbi_disease")` fails outright

**Severity:** blocker · **Status:** Fixed

The `ncbi_disease` Hub repository contains `.gitattributes`, `README.md` and
`ncbi_disease.py` — a loader *script*, and no data files. `datasets` 3.0
removed script execution, so the documented call cannot work on any current
install. Training could not start.

```python
>>> load_dataset("ncbi_disease")
RuntimeError: dataset scripts are no longer supported
```

**Fix.** `data.load_corpus` now tries three sources in order — the Hub's
`refs/convert/parquet` branch, a plain Hub load, then direct download of the
upstream CoNLL files the official script itself fetches — and records which
answered. See [DECISIONS.md](DECISIONS.md#adr-011).

**Tests.** `test_corpus_sources.py::TestCorpusFallback` (3),
`TestConllParser` (5), `TestJsonlParser` (3).

---

## B2 — BioBERT silently lowercases its own cased vocabulary

**Severity:** high, silent · **Status:** Fixed

`dmis-lab/biobert-base-cased-v1.2` ships **no** `tokenizer_config.json` — only
`config.json`, `pytorch_model.bin` and `vocab.txt`. With nothing to read casing
from, `AutoTokenizer` falls back to the BERT class default `do_lower_case=True`
and applies it to a cased vocabulary:

| Term | Observed | Correct |
|---|---|---|
| `APC` | `a ##p ##c` | `AP ##C` |
| `EGFR` | `e ##g ##f ##r` | `E ##G ##F ##R` |
| `Adenomatous` | `ad ##eno ##mat ##ous` | `Aden ##oma ##to ##us` |

The cased vocabulary has no lowercase entry for a gene symbol, so every one
shatters into single characters. Nothing raises. The loss falls, training looks
healthy, and the F1 is quietly several points lower.

**Fix.** `tokenization.load_fast_tokenizer` detects casing from the vocabulary,
reloads with `do_lower_case=False` when needed, then verifies with a probe and
raises if case still did not survive. The corrected setting is written into the
saved `tokenizer_config.json`, so the ONNX export and the container inherit it.
See [DECISIONS.md](DECISIONS.md#adr-012).

**Tests.** `test_corpus_sources.py::TestTokenizerCasing` (4).

---

## B3 — MLflow tracking URI malformed on Windows

**Severity:** high · **Status:** Fixed

`train.py` and `report.py` both built the URI as
`f"file://{REPO_ROOT / 'mlruns'}"`. On Windows that yields
`file://C:\Users\...` — backwards separators and only two slashes, so MLflow
parses the drive letter as a hostname. Runs go somewhere nobody looks and
`report.py` finds no experiment.

**Fix.** A single `config.mlflow_tracking_uri()`, used by both, which now
returns SQLite (see B10) and builds any path form correctly.

**Tests.** `test_corpus_sources.py::TestTrackingUri` (2).

---

## B4 — Wrong `transformers` installed; package not installed at all

**Severity:** blocker · **Status:** Fixed (no test)

The machine had `transformers` 5.8.1 system-wide. `pyproject.toml` pins `<5`
for a documented reason: 5.x makes every tokenizer report `is_fast=True`,
defeating the guard at `data.py:81` that label alignment depends on. Separately,
`medextract` was not installed in any form — all ten test modules failed at
import with `ModuleNotFoundError`.

**Fix.** A project virtualenv created with `--system-site-packages`, so the
existing CUDA PyTorch build is reused (saving a ~2.5 GB download) while venv
packages shadow the system ones. `make.ps1 venv` does this reproducibly.

---

## B5 — No gradient accumulation; CLI sweep ran fp32 on a GPU

**Severity:** medium · **Status:** Fixed

Two problems in one area:

- `train.py` passed `batch_size` straight to `per_device_train_batch_size` with
  no ceiling, and set `per_device_eval_batch_size=max(batch_size, 32)`. On a
  4 GB card the `bs32` arm risks OOM, and evaluation could fail three epochs in
  even for an arm that trained fine.
- `sweep.main()` built `RunConfig()` without `fp16`. Only `pipeline.py`
  detected CUDA, so a sweep launched from the CLI trained in fp32 and took
  roughly twice as long as the identical sweep launched from the pipeline.

**Fix.** `resolve_batching()` splits the effective batch into per-device batch
and accumulation whose product is preserved; eval matches the train batch;
`run_sweep` retries an OOM arm once at half the per-device batch;
`sweep.main()` auto-detects CUDA with `--no-fp16` to override. See
[DECISIONS.md](DECISIONS.md#adr-013).

**Tests.** `test_corpus_sources.py::TestBatchResolution` (4),
`TestSweepDefaults` (3).

---

## B6 — Every documented command unrunnable on Windows

**Severity:** medium · **Status:** Fixed (no test)

The README documents `make all`, `make sweep`, `make test` and so on. There is
no `make` on Windows, and the Makefile's `clean` and `help` targets are written
against GNU coreutils (`rm -rf`, `find`, `awk`). `PY ?= python3` is also not a
command that exists here.

**Fix.** `make.ps1`, mirroring every Makefile target through the project
virtualenv. It also sets `HF_HOME` outside the synced repository and
`PYTHONUTF8=1`. The Makefile remains the Linux and CI entry point.

---

## B7 — Test read the README with the platform default encoding

**Severity:** low · **Status:** Fixed

`test_readme_sync.py::test_real_readme_has_all_four_marker_pairs` called
`(REPO_ROOT / "README.md").read_text()` with no encoding. On Windows that
defaults to cp1252, and the README contains em-dashes:

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 2985
```

The only failure in the otherwise-green baseline run. `readme_sync.py` itself
was already correct — it passes `encoding="utf-8"` on both read and write.

**Fix.** Pass `encoding="utf-8"` in the test.

---

## B8 — Segfault when the full test suite runs

**Severity:** high · **Status:** Fixed (no test)

Individually every test module passed. Run together, the process died:

```
Windows fatal exception: access violation
  File "...\pyarrow\__init__.py", line 71 in <module>
exit code 139
```

Bisected to `test_cli.py::TestExportParser` followed by
`test_corpus_sources.py::TestCorpusFallback`: once transformers and torch have
initialised, importing pyarrow takes the process down. Both ship overlapping
native dependencies and only load order decides whether they coexist. Not a
catchable exception — a segfault.

The real pipeline was exposed too, not just the tests: `pipeline.py` imports
torch in `environment_banner()` before `train()` reaches `import mlflow`, which
pulls in pandas, which pulls in pyarrow.

**Fix.** `medextract/__init__.py` preloads pyarrow, guarded so it is a no-op
where pyarrow is absent — which is the serving container. See
[DECISIONS.md](DECISIONS.md#adr-015).

**Verified by.** The full suite going from exit 139 to exit 0.

---

## B9 — TensorFlow + Keras 3 broke `Trainer` at import

**Severity:** blocker · **Status:** Fixed (no test)

The machine has TensorFlow 2.21 with Keras 3 installed for unrelated work.
`transformers.Trainer` imports its reporting integrations, which reach for
`TFPreTrainedModel`, which imports the TF activation helpers, which raise:

```
ValueError: Your currently installed version of Keras is Keras 3, but this is
not yet supported in Transformers. Please install the backwards-compatible
tf-keras package
```

Nothing in this project touches TensorFlow — the models are PyTorch and the
runtime is ONNX — so a TF install that happens to be on the machine should have
no say in whether training starts.

**Fix.** `medextract/__init__.py` sets `USE_TF=0` via `setdefault`, so
transformers never probes for TF. Preferred over installing `tf-keras`: a
compatibility shim purely to satisfy an import path never taken. See
[DECISIONS.md](DECISIONS.md#adr-015).

---

## B10 — MLflow 3 refuses the filesystem tracking backend

**Severity:** blocker · **Status:** Fixed

With B3 fixed, training reached MLflow and hit:

```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
maintenance mode and will not receive further updates. Please migrate to a
database backend (e.g., 'sqlite:///mlflow.db')
```

**Fix.** `config.mlflow_tracking_uri()` now returns
`sqlite:///<repo>/mlflow.db`. Better here anyway: the file store writes
thousands of tiny files, one per metric and per parameter per run, which is
poor behaviour in a synced folder. See [DECISIONS.md](DECISIONS.md#adr-014).

`MLFLOW_ALLOW_FILE_STORE=true` would have opted out; migrating was the right
call rather than suppressing a deprecation.

**Tests.** `test_corpus_sources.py::TestTrackingUri` (2).

---

## B11 — Served spans shredded at subword boundaries

**Severity:** high, silent, **invisible to the F1** · **Status:** Fixed

Found by inspecting what a real fine-tuned model actually returned, not by any
test. `/extract` was splitting most multi-subword entities into fragments:

| Input | Returned | Should be |
|---|---|---|
| `Familial adenomatous polyposis ...` | `'F'` [0:1], `'amilial adenomatous polyposis'` [1:30] | `'Familial adenomatous polyposis'` [0:30] |
| `... induced nephrotoxicity ...` | `'ne'` [33:35], `'rot'` [37:40] | `'nephrotoxicity'` [33:47] |

The skipped characters between fragments were the giveaway. Per-subword
predictions for the single word `nephrotoxicity` were:

```
ne     word 4   B-Disease     <- first subword: supervised
##ph   word 4   O             <- continuation: never supervised
##rot  word 4   I-Disease     <- continuation: never supervised
##ox   word 4   O
##icit word 4   O
##y    word 4   O
```

**Root cause.** The default alignment convention labels only the first subword
of each word; every continuation gets `IGNORE_INDEX` and contributes nothing to
the loss. Their predictions were therefore never trained and are noise.
`decode_entities` read them anyway, so one stray `O` mid-word split an entity —
and biomedical words fragment into many pieces, so this hit most entities.

**Why no test caught it.** Evaluation and serving used different code paths at
different granularities. `metrics.decode_predictions` drops positions whose
gold label is `IGNORE_INDEX`, so scoring only ever saw first subwords and was
entirely correct. The reported F1 was honest about the convention it measured
while `/extract` returned fragments. Every existing decode test constructed
one token per word, so the case could not arise.

Confirmed not to be a training problem: the `all-subwords` arm, where
continuations *are* supervised, fragmented too — a single mid-word `O` still
breaks the span.

**Fix.** `_collapse_to_words` reduces subwords to one prediction per word
before the BIO state machine runs — label and score from the first subword,
span from first start to last end. Standard `aggregation_strategy="first"`
behaviour, and the only decoding consistent with how the model was trained. See
[DECISIONS.md](DECISIONS.md#adr-017).

**Side effect worth having:** evaluation and serving now agree on granularity,
so the reported F1 finally predicts what a caller receives. Before the fix the
two were measuring different things.

**Tests.** `test_inference.py::TestSubwordAggregation` (7) and
`::test_predicted_spans_never_start_mid_word`.

---

## B12 — Generated markdown written in the platform's default encoding

**Severity:** medium · **Status:** Fixed

`benchmark.py` wrote `docs/benchmark.md` with `Path.write_text()` and no
encoding. On Windows that is cp1252, so the em-dash in the generated summary
line was written as byte `0x97` and the file was **not valid UTF-8**:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x97 in position 438
```

It renders as a replacement character on GitHub and raises for anyone reading
it as UTF-8 — which is everyone not on Windows. The same class as B7, in the
opposite direction: B7 was a bad *read*, this is a bad *write*, and a bad write
outlives the process that made it.

**Fix.** Every `read_text` / `write_text` in the package and the test suite now
names `encoding="utf-8"`. `make.ps1` additionally sets `PYTHONUTF8=1` so stdout
redirection cannot reintroduce it.

**Tests.** `test_corpus_sources.py::TestTextEncoding` walks the AST of every
source and test file and fails if any call omits `encoding`, plus a check that
the generated documents decode as UTF-8. Asserting behaviour could not catch
this: on a UTF-8 machine the default is already correct, so it only breaks for
someone else, later.

---

## B13 — `datasets.map` re-encoded restricted tags into the wrong label space

**Found by:** `test_data.py::TestTypeRestriction`, on the first run.

**Symptom.** `restrict_types` relabels a two-type corpus to one type and
returns string tags in the reduced label space (`O`, `B-Disease`,
`I-Disease`). The dataset that came back held integers — `[0, 0, 2, 3]` —
and the alignment step then indexed a three-element label list with a 3.

**Root cause.** The map function returned a column with the same name as an
existing `ClassLabel` column. With no explicit `features`, `datasets.map`
keeps the existing feature type for that column and encodes the returned
strings through it, so `"B-Disease"` became its id in the *original*
five-label space. Nothing warned; the column simply held plausible-looking
integers.

**Fix.** Declare the column's type as `Sequence(Value("string"))` in the
`features` passed to `map`, so the strings stay strings and the reduced label
list is the only one in play.

**Silent?** Yes, in the sense that matters: with a corpus where the kept type
happened to have low ids the run would have trained on the wrong labels and
reported a number. It was caught only because the test asserted the exact
tags that should come out rather than that a dataset came back.

## Summary

| ID | Area | Severity | Silent? | Status |
|---|---|---|---|---|
| B1 | Corpus loading | Blocker | No | Fixed |
| B2 | Tokenizer casing | High | **Yes** | Fixed |
| B3 | MLflow URI | High | **Yes** | Fixed |
| B4 | Dependencies | Blocker | No | Fixed (no test) |
| B5 | VRAM / fp16 | Medium | **Partly** | Fixed |
| B6 | Windows tooling | Medium | No | Fixed (no test) |
| B7 | Test encoding | Low | No | Fixed |
| B8 | pyarrow/torch segfault | High | No | Fixed (no test) |
| B9 | TensorFlow/Keras 3 | Blocker | No | Fixed (no test) |
| B10 | MLflow file store | Blocker | No | Fixed |
| B11 | Subword span shredding | High | **Yes** | Fixed |
| B12 | cp1252 generated markdown | Medium | Partly | Fixed |
| B13 | `datasets.map` label re-encoding | High | **Yes** | Fixed |

**Five were silent.** B2 and B3 would have produced a completed run with a
worse number and no indication anything was wrong; B13 would have trained on
the wrong label ids and reported a number too; B5's fp16 half would have
doubled sweep time with no error. B11 is the worst of them: it left the
headline metric intact and correct while degrading the actual product, because
evaluation and serving measured different things. Those are the expensive kind,
and they are why the fixes for B2, B5 and B11 assert an observable outcome
rather than merely not raising.

**B11 is also the one no amount of unit testing would have found**, because
every existing test constructed one token per word. It took looking at real
output from a real model. That is an argument for making "run it and read the
output" a step, not an afterthought.

**Six were environment collisions**, not logic errors: B4, B6, B7, B8, B9 and
partly B10 — with B12 a seventh in effect. They are invisible to CI, which runs clean Ubuntu containers on
Python 3.11 and 3.12 with no TensorFlow, no pre-existing packages and no
Windows. Nothing was wrong with the code as written for that environment. This
is an argument for running on the target platform before claiming it works —
see [ENVIRONMENT.md](ENVIRONMENT.md#known-environment-hazards).

## Open

None. Known limitations that are scope decisions rather than defects are in
[PRD.md](PRD.md); remaining work is in [TODO.md](TODO.md).
