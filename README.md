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
> **Status: measured, twice.** Training, ONNX export, the benchmark, a 3000-abstract PubMed index and the container were run end to end on 2026-08-20; the BC5CDR comparison, artefact integrity, score calibration, latency under concurrency and a second container build on 2026-10-30. Every number below was written by a script on the machine named in `docs/ENVIRONMENT.md`. `docs/RESULTS.md` has all of them with the command that produced each.
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
if Docker isn't running), brings up the live API, drives it under rising
concurrency, and rewrites the tables below with the real numbers. It is safe to re-run: finished stages are
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
make loadtest               # latency under concurrency, against a running API
make calibrate              # is the score a probability? (no; measured below)
make fetch                  # download PubMed abstracts
make index                  # embed and index them into Qdrant
make serve                  # run the API locally
```

`make help` lists everything. Training needs the `train` extra; serving does not.

### Tests

```bash
make test        # or: pytest -q
make lint        # ruff check + format check
make audit       # pip-audit over what the serving image installs
```

The suite runs offline. It builds a tiny two-layer BERT with random weights and
a hand-written vocabulary, then exercises the real code paths against it —
alignment, span decoding, ONNX export and parity, request handling, retrieval.
No network, no downloads, a few seconds. What it verifies is that the plumbing
is correct; what it deliberately does not verify is model quality, which is the
sweep's job. It passes unchanged on transformers 4.x and 5.x, and CI runs it
on both.

## The API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/extract` | One text in, entities with offsets and scores out |
| `POST` | `/extract/batch` | Up to 64 texts in one call |
| `POST` | `/search` | Query in, ranked abstracts out, entities tagged in each |
| `GET` | `/health` | Model loaded, backend, label set, retrieval status, uptime, and the hash of the graph being served |
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

### Chemicals as well: BC5CDR

The first sweep left two questions open: does one model trained on both
entity types match two trained on one each, and does the extra chemical
supervision cost any disease F1? Both are answerable on BC5CDR, which
annotates diseases and chemicals in the same 1,500 abstracts, by training
single-type arms on exactly the same sentences with the other type's mentions
relabelled `O` (`--keep-types Disease`). Three combined arms and two
single-type arms, all on the recipe the NCBI sweep settled on:

```bash
python -m medextract.sweep --dataset bc5cdr --experiment medextract-ner-bc5cdr \
    --output-root artifacts/sweep_bc5cdr \
    --only lr3e-5_bs16_ep3 lr3e-5_bs16_ep3_seed1337 pubmedbert_lr3e-5_bs16_ep3
python -m medextract.sweep --dataset bc5cdr --experiment medextract-ner-bc5cdr \
    --output-root artifacts/sweep_bc5cdr --only lr3e-5_bs16_ep3 --keep-types Disease
python -m medextract.sweep --dataset bc5cdr --experiment medextract-ner-bc5cdr \
    --output-root artifacts/sweep_bc5cdr --only lr3e-5_bs16_ep3 --keep-types Chemical
python -m medextract.report --experiment medextract-ner-bc5cdr > docs/sweep_table_bc5cdr.md
```

<!-- Generated by: python -m medextract.report --experiment medextract-ner-bc5cdr; copied from docs/sweep_table_bc5cdr.md -->

| run | base model | lr | batch | epochs | seed | all subwords | types | val P | val R | val F1 | test F1 | val F1 Chemical | val F1 Disease | test F1 Chemical | test F1 Disease |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr3e-5_bs16_ep3_chemical | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | Chemical | 0.9455 | 0.9416 | 0.9436 | 0.9284 | 0.9436 | - | 0.9284 | - |
| pubmedbert_lr3e-5_bs16_ep3 | BiomedNLP-PubMedBERT-base-uncased-abstract | 3e-05 | 16 | 3 | 42 | False | - | 0.9041 | 0.9146 | 0.9093 | 0.9014 | 0.9512 | 0.8561 | 0.9397 | 0.8550 |
| lr3e-5_bs16_ep3 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | - | 0.8947 | 0.8983 | 0.8965 | 0.8886 | 0.9400 | 0.8403 | 0.9312 | 0.8361 |
| lr3e-5_bs16_ep3_seed1337 | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 1337 | False | - | 0.8921 | 0.8988 | 0.8954 | 0.8886 | 0.9408 | 0.8378 | 0.9300 | 0.8385 |
| lr3e-5_bs16_ep3_disease | biobert-base-cased-v1.2 | 3e-05 | 16 | 3 | 42 | False | Disease | 0.8357 | 0.8399 | 0.8378 | 0.8395 | - | 0.8378 | - | 0.8395 |

The per-type columns appear on their own once an experiment has a second
type; on NCBI-Disease they would just repeat the micro score. Read them
against the seed repeat, which moves disease F1 by 0.0025 and chemical F1 by
0.0008 on validation.

- **One model matches two.** The combined BioBERT arm scores disease F1
  0.8403 / 0.8361 (validation / test) against the disease-only arm's
  0.8378 / 0.8395, and chemical F1 0.9400 / 0.9312 against the chemical-only
  arm's 0.9436 / 0.9284. Every gap is inside the seed spread and changes sign
  between the two splits. Two specialised models would cost twice the
  serving footprint for nothing measurable.
- **Chemical supervision does not cost disease F1.** Same comparison, same
  answer: within noise, in both directions.
- **PubMedBERT is the one real effect, again.** +0.0128 validation F1 over
  BioBERT, on both types, and this time it holds on the test split too
  (0.9014 against 0.8886). With 9,586 validation entities the seed repeat
  moves the micro score by 0.0011, so the gap is more than ten seed-widths;
  on NCBI-Disease's 787 entities it was about three.
- **Disease F1 is lower here than on NCBI-Disease** (0.84–0.86 against
  0.86–0.88). Different corpus, different annotation guidelines; the two
  numbers are not comparable and the model is not worse.

The PubMedBERT arm is exported with the same parity check as the default
model, to `artifacts/onnx-bc5cdr`. Point `MEDEXTRACT_MODEL_DIR` at it and
`/extract` returns `Chemical` spans as well as `Disease`. The default stays
the NCBI-Disease model, because that is the corpus the target was set on and
the one the rest of this README measures.

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

### The graph that is served is the graph that was checked

The parity check writes the SHA-256 of `model.onnx` into `parity.json`, after
it passes and not before. The API hashes the file it is about to load and
compares. A mismatch is a load failure with both hashes in the message:
`/health` reports `degraded` and `/extract` returns 503 saying why. A graph
that arrives without a record still loads, with `integrity_verified: false`
on `/health`, because an older export is a weaker claim rather than a
contradiction. The test suite appends one byte to a checked graph and asserts
the refusal.

### Latency under concurrency

The benchmark above has one request in flight, which is the right way to
compare backends and the wrong way to answer "what will a caller see under
load?" `make loadtest` drives the running API over HTTP with 1, 2, 4, 8 and
16 concurrent callers, each waiting for its response before sending the
next, 200 requests per level, payload sampled from the indexed abstracts.
Everything a caller pays for is inside the timing: the socket, JSON,
tokenisation, windowing, the model, serialisation.

<!-- Generated by: python -m medextract.loadtest -->

<!-- MEDEXTRACT:LOADTEST_TABLE:START -->
| concurrency | requests | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) | throughput (req/s) | errors |
|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 273.545 | 422.7 | 429.687 | 566.189 | 601.186 | 4.28 | 0 |
| 2 | 200 | 439.572 | 638.924 | 698.743 | 802.204 | 874.323 | 5.1 | 0 |
| 4 | 200 | 758.13 | 1083.12 | 1152.977 | 1298.373 | 1413.386 | 5.76 | 0 |
| 8 | 200 | 1681.483 | 2318.308 | 2458.576 | 2973.302 | 3145.786 | 5.34 | 0 |
| 16 | 200 | 3821.333 | 5310.078 | 5589.4 | 6173.133 | 6891.5 | 4.72 | 0 |

Client-observed, end to end over HTTP against `/extract` on the `onnx` backend, 64 distinct texts cycled, mean 1616.2 characters (max 4476). Closed loop: each concurrent caller waits for its response before sending the next.

Client on Windows-11-10.0.26200-SP0, 16 vCPU. Server and client share the machine, so these numbers are a ceiling on what this box can serve, not a measurement of the server alone.
<!-- MEDEXTRACT:LOADTEST_TABLE:END -->

Measured against the API served by uvicorn from a clean install of
`requirements-serve.txt` — the image's own dependency set, transformers
5.22.1 and onnxruntime 1.31.1 — with the model mounted from `artifacts/onnx`
and the machine otherwise idle; the Docker engine was unavailable that
evening, so the process ran on the host rather than in the container
([docs/PROGRESS.md](docs/PROGRESS.md#what-got-in-the-way)).

**Throughput is flat.** Four to six requests a second at every level, and
p50 latency rises nearly in step with the number of callers: 274 ms alone,
440 ms with two, 758 ms with four, 1.7 s with eight, 3.8 s with sixteen.
Adding callers adds queueing and nothing else. The reason is that one
request already uses the whole machine: an abstract of 1,600 characters is
two or three 256-token windows, ONNX Runtime runs them across all sixteen
threads, and a second request has nowhere to go. On this box the capacity
is about five abstracts a second — roughly 18,000 an hour — and a p99 under
600 ms needs the callers kept to one.

The first run of this test happened with an unrelated process holding
60–90% of the CPU, and it produced the same figures within three percent.
That is the same point made the other way: the server is the bottleneck,
not the client or the machine. The route to more throughput is more cores
or more replicas, not more concurrency; per-session thread limits with
several workers would be the thing to try next, and it is on the list.

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

### Is the score a probability?

No, and now that is a measurement rather than a caveat. `score` is the mean
softmax probability over a span's tokens. `make calibrate` runs the served
model over the NCBI-Disease validation split, marks each predicted entity
right or wrong against the gold spans, and bins by score:

<!-- Generated by: python -m medextract.calibration; copied from docs/calibration.md -->

| score bin | predictions | mean score | precision | gap |
|---|---|---|---|---|
| 0.5-0.6 | 8 | 0.5631 | 0.1250 | -0.4381 |
| 0.6-0.7 | 15 | 0.6589 | 0.4000 | -0.2589 |
| 0.7-0.8 | 20 | 0.7468 | 0.2000 | -0.5468 |
| 0.8-0.9 | 32 | 0.8657 | 0.5625 | -0.3032 |
| 0.9-1.0 | 742 | 0.9928 | 0.9070 | -0.0858 |

Overconfident in every bin: a mean score of 0.9715 against a precision of
0.8592, expected calibration error 0.1123. Ninety percent of predictions sit
in the top bin, where "0.99" means about 0.91. The *ordering* is sound —
raising `min_score` buys precision at the cost of recall, monotonically:

| min_score | entities kept | precision | recall |
|---|---|---|---|
| 0.5 | 817 | 0.8592 | 0.8920 |
| 0.9 | 742 | 0.9070 | 0.8551 |
| 0.99 | 632 | 0.9383 | 0.7535 |

So `min_score` is a ranking cut-off, and the API documentation says so.
Fitting a calibrator was considered and declined
([ADR-023](docs/DECISIONS.md#adr-023--score-stays-uncalibrated-and-the-miscalibration-is-published)):
it would spend the one split the sweep selection has not already touched,
and the published table lets a caller pick the trade-off for their own
threshold.

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
stage has neither, and no pip or setuptools either: nothing needs an installer
once the image is built, and their vendored packages are what image scanners
flag. The runtime installs `requirements-serve.txt`, which
deliberately excludes torch, datasets, mlflow and the training stack — the
container runs ONNX graphs, so the PyTorch wheel alone would be larger than the
entire image. Non-root user (uid 10001), and a `HEALTHCHECK` that hits
`/health` rather than just probing the port.

**Image size budget: under 1.5GB.** CI measures the built image and fails the
build if it exceeds it, so a size regression is caught on the PR rather than at
deploy time. The measured size appears in the workflow summary.

| | |
|---|---|
| Runtime dependencies, measured | **316 MB** (clean venv from `requirements-serve.txt`, transformers 5.x, torch absent) |
| `python:3.11-slim-bookworm` base, pinned by digest | ~150 MB |
| **Image size, actually built** | **408 MB** — `make docker && make docker-size` (395 MB on the first build, with transformers 4.x) |

The 316 MB is a real measurement, not an estimate: a fresh virtualenv built
from `requirements-serve.txt`, with PyTorch confirmed absent. The same check
passes in the built image: `torch`, `datasets` and `mlflow` are absent,
`onnxruntime`, `transformers`, `fastapi` and `qdrant-client` are present, and
`/health`, `/extract` and `/search` all answer. The image is 27% of its
budget. The first build's README carried a 560 MB estimate next to the
measurement, arrived at by adding the venv to the base image — which counts
the Python installation both of them sit on twice. It was wrong in the
cautious direction, and it is gone because an estimate has no business next
to a number that can be measured.

The workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **lint** — `ruff check` and `ruff format --check`
- **test** — pytest on Python 3.11 and 3.12, each against transformers 4.x
  and 5.x, with coverage
- **audit** — `pip-audit --strict` over what the image installs, failing on
  any known vulnerability; an advisory audit of the training extras
- **build** — build the image, enforce the size budget, scan it with Trivy
  (a fixed HIGH or CRITICAL finding fails the build), then start the
  container with no model mounted and assert it comes up and answers `/health`
  as degraded rather than crash-looping
- **publish** — on a `v*` tag only, push to GitHub Container Registry

### The base image and the dependencies

The base is pinned by digest, not by tag, so a base-image change is a visible
commit rather than a silent rebuild. The serving image installs transformers
from the 5.x line: the first dependency audit found nine published advisories
against the last 4.x release, every fix on 5.x, none of them in a code path
the container runs — and the fix was cheaper than the argument. What actually
crosses from one line to the other is the tokenizer, so that was checked
before the pin moved: byte-identical ids and offsets on every sentence of the
test split and every indexed abstract. Training still runs on 4.x, which is
what produced every number here, and the suite passes on both. The reasoning
is in [docs/SECURITY.md](docs/SECURITY.md#dependency-posture).

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

**Why probe the tokenizer instead of pinning the library?** The alignment
code needs `word_ids()` and an offset mapping. `is_fast` used to be a proxy
for both; from transformers 5 every tokenizer reports it, so a guard on the
flag either never fires or has to be pinned to a version — which is what the
old `<5` pin was. Asking for the two things directly gives the same answer on
every version, and the test feeds it a tokenizer whose flag lies.

**Why does the image run a different transformers line from training?**
Because the two have different jobs. The image is exposed and audited, so it
gets the line with no advisories. Training produced the numbers in this
README on 4.57.6, and re-running forty minutes of GPU to reproduce numbers
that would not change proves nothing. The record says which line produced
what; the tokenizer was shown to be identical across them; CI tests both.

**Why hash the graph in the parity check and not in a deploy script?** The
parity check is the one moment the graph is known to be right, so that is
where the claim gets written. A hash computed anywhere else is a hash of
whatever happened to be on disk.

## Limitations

- **One entity type by default.** The default model is trained on
  NCBI-Disease and emits `Disease` only. A Disease + Chemical model trained on
  BC5CDR is exported alongside it (`artifacts/onnx-bc5cdr`); switching is one
  environment variable. Genes are covered by neither corpus and would need a
  third.
- **The score is not a probability.** Measured, not assumed: overconfident in
  every bin, expected calibration error 0.11 on the validation split. It
  orders predictions, and `min_score` is a ranking cut-off.
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
Every phase has been run on real data: training, ONNX export with a passing parity check, benchmarking, retrieval indexing, a container answering `/health`, `/extract` and `/search` at `http://localhost:8000` with `integrity_verified: true`, and latency under concurrency measured against the same API served from the image's dependency set. That is a local deployment, not a hosted one -- there is no public URL, and no authentication or rate limiting in front of it.

Two kinds of gap remain, and they are not the same kind. The ones under [Limitations](#limitations) -- clinical-note performance, negation, entity linking, relation extraction, corpus scale -- are scope decisions. The ones in [docs/TODO.md](docs/TODO.md) are genuinely unfinished: a hosted deployment behind an ingress, integrity for the query-encoder graph, an open-loop load test, and an evaluation on clinical notes.
<!-- MEDEXTRACT:NOT_DONE:END -->

## Licence

MIT.
