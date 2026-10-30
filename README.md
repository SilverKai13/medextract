# MedExtract

Takes free biomedical or clinical prose and returns the diseases and chemicals
in it — each one tagged with its type, its character offsets in the text you
sent, and a confidence score. A second endpoint takes a question and returns
the most relevant PubMed abstracts from a local index, with the same tagging
applied to each hit.

It is a small, honest version of the extraction layer a clinical NLP product
needs: a fine-tuned model, an optimised runtime, an API, a vector index, and
the CI to keep them working.

<!-- MEDEXTRACT:STATUS:START -->
> **Status: pipeline run completed on 2026-08-20.** Real numbers below, from a fine-tuned model and a 3000-abstract PubMed index. See `docs/sweep_table.md` and `docs/benchmark.md` for full detail.
<!-- MEDEXTRACT:STATUS:END -->

---

## Quickstart: one command

On a machine with network access (and ideally a GPU — CPU works too, just
slower):

```bash
pip install -e ".[dev,serve,train]"
cp .env.example .env && echo 'MEDEXTRACT_NCBI_EMAIL=you@example.com' >> .env
make all
```

`make all` runs the full sweep, picks the best arm by validation F1, exports
it to ONNX with a parity check, benchmarks it against PyTorch, fetches and
indexes a PubMed corpus (no Docker required — falls back to an on-disk Qdrant
if Docker isn't running), brings up the live API, and rewrites the tables
below with the real numbers. It is safe to re-run: finished stages are
skipped unless you pass `--force-retrain` / `--force-fetch`
(`python -m medextract.pipeline --help` for the full flag list).

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Running it](#running-it)
- [The API](#the-api)
- [Phase 1 — fine-tuning](#phase-1--fine-tuning-the-ner-model)
- [Phase 2 — ONNX](#phase-2--onnx-export-and-benchmark)
- [Phase 3 — serving](#phase-3--serving)
- [Phase 4 — retrieval](#phase-4--retrieval)
- [Phase 5 — container and CI](#phase-5--container-and-ci)
- [Design decisions](#design-decisions-worth-defending)
- [Limitations](#limitations)
- [What is not done yet](#what-is-not-done-yet)

---

## What it does

```
POST /extract
{"text": "Familial adenomatous polyposis is caused by germline mutations in the APC gene."}

200 OK
{
  "entities": [
    {"text": "Familial adenomatous polyposis", "label": "Disease",
     "start": 0, "end": 30, "score": 0.9994},
    {"text": "APC", "label": "Disease",
     "start": 70, "end": 73, "score": 0.9990}
  ],
  "text": "Familial adenomatous polyposis is caused by germline mutations in the APC gene.",
  "model_version": "pubmedbert-ncbi-disease-v1",
  "took_ms": 67.33
}
```

Offsets index the string you sent, so `text[start:end]` is the entity. That is
asserted in the test suite, including for non-ASCII input — Python character
offsets, not UTF-8 byte offsets.

## Architecture

```
                    ┌──────────────────────────────────────────┐
  training          │  HuggingFace datasets → NCBI-Disease     │
  (offline,         │       ↓ tokenise + align word labels     │
   run on a box     │  BioBERT + token classification head     │
   with a GPU       │       ↓ Trainer, 9-arm sweep             │
   or patience)     │  MLflow  ──→ comparison table            │
                    │       ↓ best checkpoint                  │
                    │  torch.onnx.export → parity check        │
                    └──────────────────┬───────────────────────┘
                                       │  artifacts/onnx
                                       ▼
  serving           ┌──────────────────────────────────────────┐
  (the container)   │  FastAPI                                 │
                    │   /health    model, backend, labels      │
                    │   /extract   text → entities + offsets   │
                    │   /search    query → abstracts + entities│
                    │       ↓                    ↓             │
                    │  onnxruntime          Qdrant (cosine)    │
                    │  NER graph            3k PubMed abstracts│
                    │  + query encoder                         │
                    └──────────────────────────────────────────┘
```

No PyTorch in the serving container. Both models — the NER tagger and the
query encoder — run on onnxruntime, which is most of how the image stays small.

## Running it

### The whole thing, from a built artifact

```bash
docker compose up --build -d
curl localhost:8000/health
```

Compose brings up the API and Qdrant together. The API reads its model from
`./artifacts`, mounted read-only, so the weights are not baked into the image.
With no artifacts present it still starts, and `/health` reports `degraded`
with the reason — that is deliberate, see [Design decisions](#design-decisions-worth-defending).

### From scratch

```bash
pip install -e ".[dev,serve,train]"

make train                  # one fine-tuning run
make sweep                  # all nine arms, tracked in MLflow
make report                 # regenerate the comparison table
make export                 # ONNX export + parity check + query encoder
make benchmark              # PyTorch vs ONNX latency table
make fetch                  # download PubMed abstracts
make index                  # embed and index them into Qdrant
make serve                  # run the API locally
```

`make help` lists everything. Training needs the `train` extra; serving does not.

### Tests

```bash
make test        # or: pytest -q
make lint        # ruff check + format check
```

The suite runs offline. It builds a tiny two-layer BERT with random weights and
a hand-written vocabulary, then exercises the real code paths against it —
alignment, span decoding, ONNX export and parity, request handling, retrieval.
No network, no downloads, a few seconds. What it verifies is that the plumbing
is correct; what it deliberately does not verify is model quality, which is the
sweep's job.

## The API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/extract` | One text in, entities with offsets and scores out |
| `POST` | `/extract/batch` | Up to 64 texts in one call |
| `POST` | `/search` | Query in, ranked abstracts out, entities tagged in each |
| `GET` | `/health` | Model loaded, backend, label set, retrieval status, uptime |
| `GET` | `/docs` | Generated OpenAPI documentation |

Every response carries an `X-Request-ID`, echoed from the request if you send
one. Logs are JSON, one object per line, with that id attached — including log
lines from library code that knows nothing about the web layer, because the id
travels in a `ContextVar` rather than being passed around.

## Phase 1 — fine-tuning the NER model

Fine-tunes a BioBERT variant for token classification on
[NCBI-Disease](https://huggingface.co/datasets/ncbi/ncbi_disease) (~800 PubMed
abstracts, disease mentions annotated). BC5CDR is registered too, for chemicals
as well as diseases.

**The part that matters is label alignment.** Labels come at word level;
the model sees WordPiece subwords, and biomedical vocabulary fragments hard —
"adenomatous" is three pieces. `word_ids()` from the fast tokeniser maps
subwords back to words. Only the first subword of each word carries the label;
the rest get `-100` so they contribute nothing to the loss.

Getting this wrong does not throw. The loss still falls, training still looks
healthy, and the F1 is quietly wrong. So it is isolated in
[`labels.py`](src/medextract/labels.py) with no heavy imports, and
[`test_labels.py`](tests/test_labels.py) asserts the exact label positions
against a real fast tokeniser — not just that the lengths agree.

The alternative convention (label every subword, converting `B-` to `I-` after
the first) is implemented and run as a sweep arm rather than argued about.

### Sweep

Nine arms: three learning rates, two batch sizes, two epoch counts, the subword
convention, PubMedBERT against BioBERT, and two extra seeds on a middle arm so
run-to-run variance is visible and a 0.005 gap can be read for what it is.

```bash
make sweep && make report
```

<!-- Generated by: python -m medextract.report --experiment medextract-ner -->

<!-- MEDEXTRACT:SWEEP_TABLE:START -->
| run | base model | lr | batch | epochs | seed | all subwords | val P | val R | val F1 | test F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| pubmedbert_lr3e-5_bs16_ep3 | BiomedNLP-PubMedBERT-base-uncased-abstract | 3e-05 | 16 | 3 | 42 | False | 0.8730 | 0.8907 | 0.8818 | 0.8718 |
| lr3e-5_bs16_ep3_seed2024 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 2024 | False | 0.8534 | 0.8729 | 0.8631 | 0.8809 |
| lr3e-5_bs16_ep5 | biobert-base-cased-v1.2 | 3e-05 | 16 | 5 | 42 | False | 0.8548 | 0.8679 | 0.8613 | 0.8702 |
| lr3e-5_bs16_ep3_seed1337 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 1337 | False | 0.8516 | 0.8679 | 0.8597 | 0.8838 |
| lr3e-5_bs16_ep3_all-subwords | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | True | 0.8483 | 0.8666 | 0.8573 | 0.8767 |
| lr3e-5_bs16_ep3 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | 0.8500 | 0.8640 | 0.8570 | 0.8761 |
| lr3e-5_bs32_ep5 | biobert-base-cased-v1.2 | 3e-05 | 32 | 5 | 42 | False | 0.8518 | 0.8615 | 0.8566 | 0.8809 |
| lr5e-5_bs16_ep3 | biobert-base-cased-v1.2 | 5e-05 | 16 | 3 | 42 | False | 0.8405 | 0.8640 | 0.8521 | 0.8746 |
| lr2e-5_bs16_ep3 | biobert-base-cased-v1.2 | 2e-05 | 16 | 3 | 42 | False | 0.8309 | 0.8742 | 0.8520 | 0.8804 |
<!-- MEDEXTRACT:SWEEP_TABLE:END -->

**Target: F1 above 0.85** on NCBI-Disease, which is roughly in line with
published baselines — so a number well below it means something is broken
rather than that the task is hard.

**Met, by every arm.** Validation F1 spans 0.8520–0.8818 and test F1
0.8702–0.8838. The best test F1 is **0.8838**.

The seed repeats are the part worth reading. The same configuration under seeds
42, 1337 and 2024 gives validation F1 0.8570 / 0.8597 / 0.8631 — a spread of
**0.0061** from nothing but initialisation and data order. Set the rest of the
table against that:

- All eight BioBERT arms fall in a 0.0111 band, under two seed-widths. **No
  hyperparameter in this grid is distinguishable from another.**
- The subword convention, the thing that got its own arm, moves validation F1
  by **0.0004**. That is a fifteenth of the noise. It does not matter here.
- PubMedBERT is the one real effect: +0.0187 validation F1 over the best
  BioBERT arm, about three seed-widths, driven by recall (0.8907 vs ~0.87).

And the caution that follows: validation and test F1 are **negatively**
correlated across the nine arms (r = −0.40). PubMedBERT wins on validation and
places seventh of nine on test. With ~900 sentences per split and one seed per
arm, that is what differences this size look like — noise, not a ranking.
Selecting the export by validation F1 is a defensible rule, not a claim that
the selected arm generalises best. Believing any of these gaps would need more
seeds per arm, not more arms.

Scoring is strict entity-level IOB2 matching (exact span and type), which is
what those baselines report. Token accuracy is not used: most tokens are `O`,
so a model predicting nothing scores above 90%. There is a test asserting
exactly that failure mode.

## Phase 2 — ONNX export and benchmark

```bash
make export     # export + parity check, fails the export if they disagree
make benchmark
```

The parity check runs before any timing and hard-fails on mismatch, because a
benchmark of a graph that quietly computes something different is a benchmark
of the wrong thing. It checks three things, in increasing order of what a
caller actually cares about:

1. Max absolute logit difference within `atol=1e-4`, `rtol=1e-3`.
2. Argmax agreement — the predicted label per token.
3. Identical decoded entity spans across both backends.

Bit equality is not on offer: ONNX Runtime and PyTorch fuse and schedule
kernels differently. 1e-4 on logits is tight enough that neither argmax nor the
softmax confidences move.

<!-- Generated by: python -m medextract.benchmark -->

<!-- MEDEXTRACT:BENCHMARK_TABLE:START -->
| backend | batch | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | throughput (seq/s) |
|---|---|---|---|---|---|---|
| pytorch_eager | 1 | 50.086 | 54.795 | 55.816 | 56.91 | 19.97 |
| pytorch_eager | 16 | 187.124 | 198.573 | 201.143 | 217.542 | 85.5 |
| onnxruntime | 1 | 18.387 | 19.572 | 19.955 | 20.929 | 54.39 |
| onnxruntime | 16 | 142.19 | 152.621 | 155.797 | 158.776 | 112.53 |

ONNX Runtime speedup over PyTorch eager at p50 — batch 1: 2.72x, batch 16: 1.32x.

Measured on Windows-11-10.0.26200-SP0, 16 vCPU, torch 2.6.0+cu124, onnxruntime 1.26.0. CPU only, no GPU involved.
<!-- MEDEXTRACT:BENCHMARK_TABLE:END -->

Percentiles rather than means: the mean hides the tail and the tail is what a
caller notices. The script records the hardware and thread settings it ran on
into `docs/benchmark.json`, because a speedup number without them is not a
claim anyone can check.

Measured here: **2.72x at batch 1, 1.32x at batch 16** — 50.1 ms to 18.4 ms
p50 for a single sequence. The gain shrinks with batch size because a larger
batch already saturates the CPU, leaving ONNX Runtime's per-call overhead
savings less to recover. Both numbers come from `make benchmark` on the
hardware named above; on different hardware, measure it again.

## Phase 3 — serving

FastAPI, Pydantic request/response models, model loaded once at startup in the
lifespan handler. Loading per request would dominate latency: a BioBERT-sized
graph takes hundreds of milliseconds to initialise against single-digit
milliseconds to run.

Tested with `pytest` and `TestClient`: happy path, empty input, oversized
input, malformed JSON, missing model, out-of-range parameters, unicode offsets,
and that batch results match single calls.

Long inputs are windowed with overlap rather than truncated, so an entity at
character 8000 is still found. Windows are decoded independently and their
spans merged, keeping the highest score for any span found twice.

## Phase 4 — retrieval

```bash
make fetch    # PubMed abstracts via NCBI E-utilities
make index    # embed and upsert into Qdrant
```

Abstracts are fetched through E-utilities (no key needed at this volume; the
client respects the published rate limits and sends a contact address, because
getting an IP blocked from E-utilities is slow to undo), embedded with a
biomedical sentence encoder, and indexed in Qdrant with cosine distance.

`POST /search` embeds the query, retrieves the top-k abstracts, and runs the
NER model over each hit so entities come back tagged with offsets into the
abstract text.

**This is retrieval plus extraction, not RAG.** Nothing generates an answer. It
returns passages and what was found in them. Calling it RAG would be claiming a
generation step that does not exist.

Structured abstracts — the ones split into `BACKGROUND`, `METHODS`, `RESULTS` —
are reassembled with their section labels rather than having only the first
section taken, which is the easy bug here.

## Phase 5 — container and CI

Multi-stage build. The build stage has compilers and pip caches; the runtime
stage has neither. The runtime installs `requirements-serve.txt`, which
deliberately excludes torch, datasets, mlflow and the training stack — the
container runs ONNX graphs, so the PyTorch wheel alone would be larger than the
entire image. Non-root user (uid 10001), and a `HEALTHCHECK` that hits
`/health` rather than just probing the port.

**Image size budget: under 1.5GB.** CI measures the built image and fails the
build if it exceeds it, so a size regression is caught on the PR rather than at
deploy time. The measured size appears in the workflow summary.

| | |
|---|---|
| Runtime dependencies, measured | **390 MB** (clean venv from `requirements-serve.txt`, torch absent) |
| `python:3.11-slim-bookworm` base | ~150 MB |
| Expected image size | ~560 MB, from adding the two above |
| **Image size, actually built** | **395 MB** — `make docker && make docker-size` |

The 390 MB is a real measurement, not an estimate: a fresh virtualenv built
from `requirements-serve.txt`. The serving path was then run inside it with
PyTorch confirmed absent — ONNX inference, `/health` and `/extract` all work —
which is the check that matters, since a runtime dependency set that is 390 MB
but incomplete would be worse than a larger one that works. The same check
passes in the built image: `torch`, `datasets` and `mlflow` are absent,
`onnxruntime`, `transformers`, `fastapi` and `qdrant-client` are present.

**The 560 MB estimate was wrong, and worth saying why.** It double-counted:
that 390 MB venv includes packages sitting on top of a Python installation the
base image already provides, so adding the two figures counts the shared
substrate twice. The real number is 395 MB — 26% of the budget, not 37%. An
estimate that errs toward caution is the harmless direction, but it was an
estimate, which is exactly why the row above it now holds a measurement.

The workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **lint** — `ruff check` and `ruff format --check`
- **test** — pytest on Python 3.11 and 3.12, with coverage
- **build** — build the image, enforce the size budget, then start the
  container with no model mounted and assert it comes up and answers `/health`
  as degraded rather than crash-looping
- **publish** — on a `v*` tag only, push to GitHub Container Registry

## Design decisions worth defending

Things an interviewer is likely to poke at, and the actual reasons.

**Why multi-stage?** Not for tidiness — to keep `build-essential`, pip's caches
and build artefacts out of the shipped image. They are needed to install the
wheels and never needed to run them. The second stage copies the finished
virtualenv and nothing else.

**Why no PyTorch in the container?** The model runs as an ONNX graph, so torch
is only needed to produce that graph, which happens offline. Dropping it is the
single largest saving in the image. The consequence is that the query encoder
had to be exported to ONNX too, which is why
[`export_embedder.py`](src/medextract/retrieval/export_embedder.py) exists —
`sentence-transformers` would have dragged torch back in.

**Why is scoring stricter than serving?** They disagree on purpose, on exactly
one case: an `I-Disease` tag with no `B-Disease` before it.

- *Scoring* discards it (strict IOB2). Published baselines score this way, and
  scoring leniently would inflate the number against the target the brief sets.
- *Serving* recovers it into a span. A caller wants the entity found; refusing
  to return one because the model emitted a malformed prefix helps nobody.

Both behaviours are pinned by tests so neither drifts into the other.

**Why hand-rolled metrics instead of seqeval?** seqeval 1.2.2 ships only an
sdist whose `setup.py` no longer builds against current setuptools — it fails
at metadata generation, which would break both CI and the Docker build. The
scoring logic is about sixty lines. The test suite cross-checks it against
seqeval across 25 randomised trials whenever seqeval happens to be importable,
so it is verified rather than merely asserted. That cross-check is how the
strict/lenient distinction above was found in the first place.

**Why `torch.onnx.export` rather than `optimum`?** `optimum` is implemented and
selectable with `--exporter optimum`, and it is the better choice if you want
ORTOptimizer's graph fusions. It is not the default because its version
compatibility with torch is narrow — the combination in this environment hits a
bug in its external-data handling. The native path pins input names, output
names, dynamic axes and opset explicitly instead of inheriting a converter's
defaults, and works across torch versions.

**Why does the service start without a model?** Because a container that
crash-loops cannot tell you why. It comes up, `/health` returns 200 with
`status: degraded` and the load error, and `/extract` returns 503 with the same
detail. That is diagnosable from `docker ps` and one curl; a crash loop is not.

**Why refuse the hashing-embedder fallback when indexing?** The service falls
back to hash embeddings if the real encoder cannot load, so it keeps answering.
`build_index` does not: an index built from hash vectors looks fine, retrieves
badly, and costs a full pass over the corpus to rebuild. Failing loudly is
cheaper than that. `--allow-hashing-fallback` overrides it for smoke tests.

## Limitations

- **One entity type, as shipped.** The exported model is trained on
  NCBI-Disease, so it emits `Disease` and nothing else. BC5CDR is registered
  and loads, which would add `Chemical`, but no arm has been run against it.
  Genes are covered by neither corpus and would need a third.
- **Gene symbols get tagged as diseases.** In the worked example above, `APC`
  in "mutations in the APC gene" comes back as a `Disease` at 0.999. That is
  partly the corpus — NCBI-Disease does annotate `APC` where it stands for
  adenomatous polyposis coli, the disease — and partly a genuine
  false positive, since here it is plainly the gene. The model has no way to
  tell those uses apart and nothing downstream disambiguates them.
- **Abstracts, not clinical notes.** Both corpora are PubMed abstracts —
  edited academic prose. Real clinical notes are abbreviated, ungrammatical,
  full of local shorthand and negation. Performance on them would be
  substantially worse, and nothing here measures that.
- **No negation or assertion detection.** "No evidence of carcinoma" yields
  `carcinoma` as a disease entity. For any clinical use that distinction is the
  whole ballgame; this does not attempt it.
- **No entity linking.** Spans are typed but not normalised to MeSH, UMLS or
  any ontology, so "MI", "myocardial infarction" and "heart attack" are three
  unrelated strings.
- **No relation extraction.** BC5CDR carries chemical-induced-disease
  relations; only the entity layer is used.
- **Retrieval is lexical-ish at the corpus scale used.** A few thousand
  abstracts is a demonstration, not a literature search. Recall against the
  full ~35M-record PubMed corpus is not a claim being made.
- **CPU-only benchmarks.** All timings are single-machine CPU. GPU numbers
  would look entirely different, and none are reported.
- **Not a medical device.** Nothing here is validated for clinical use, and it
  should not be used to make decisions about anyone's care.

## What is not done yet

<!-- MEDEXTRACT:NOT_DONE:START -->
Every phase has been run end to end on real data: training, ONNX export with a passing parity check, benchmarking, retrieval indexing, and a running container answering at `http://localhost:8000`. That is a local deployment, not a hosted one -- there is no public URL.

Two kinds of gap remain, and they are not the same kind. The ones under [Limitations](#limitations) -- entity types, clinical-note performance, negation, entity linking, relation extraction, corpus scale -- are scope decisions. The ones in [docs/TODO.md](docs/TODO.md) are genuinely unfinished: latency under concurrency is unmeasured, CI has no dependency vulnerability scan, and the base image is pinned by tag rather than digest.
<!-- MEDEXTRACT:NOT_DONE:END -->

## Licence

MIT.
