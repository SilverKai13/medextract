# Handover

For picking this project up cold. Aim: productive in about an hour.

## The one-paragraph version

MedExtract extracts diseases and chemicals from biomedical text with character
offsets and confidence scores, and searches a local PubMed index with the same
tagging applied to hits. A BioBERT variant is fine-tuned offline on a GPU,
exported to ONNX, and served by FastAPI in a container that contains no PyTorch
at all. Everything that produces a number — the sweep table, the latency table,
the README result blocks — is written by a script, never typed by hand.

## First hour

```powershell
git clone <repo> && cd medextract
.\make.ps1 venv          # ~3 min
.\make.ps1 test          # ~35s, expect 200 passed, 1 skipped
.\make.ps1 lint          # clean
```

If those pass, the environment is sound. Then read, in this order:

1. [../README.md](../README.md) — what it does and the measured results.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — the shape, in two diagrams.
3. [DECISIONS.md](DECISIONS.md) — why it is shaped that way. This is the one
   that saves the most time; most "why on earth would they…" questions are
   answered there.
4. [BUGS.md](BUGS.md) — twelve defects with root causes. Four of them fail
   *silently*, and knowing which ones will stop you reintroducing them.

Then, to see it work end to end:

```powershell
.\make.ps1 all           # ~1 hour unattended: sweep, export, benchmark, fetch, index, serve
```

## Mental model

Two halves that never share a process:

```
OFFLINE (GPU)                          SERVING (container)
corpus -> tokenise+align -> sweep      FastAPI -> onnxruntime
       -> best arm -> ONNX export  ->  artifacts/ (read-only mount)
       -> query encoder -> ONNX        -> Qdrant
```

The artefact directory is the whole interface between them. That is what lets
the serving image stay small — the PyTorch wheel alone is bigger than the
entire image.

## Seven things that will bite you

Each is fixed in the code. These notes exist so a change does not undo one.
Details in [BUGS.md](BUGS.md).

1. **Never `AutoTokenizer.from_pretrained` directly** — use
   `tokenization.load_fast_tokenizer`. BioBERT ships no `tokenizer_config.json`,
   so the default lowercases a *cased* vocabulary and turns `APC` into
   `a ##p ##c`. Silent; costs several F1.
2. **Never `load_dataset(hf_id)` directly** — use `data.load_corpus`. Both
   corpora live in script-only Hub repos and `datasets` 3.0 removed script
   execution.
3. **`import medextract` must happen first.** Its `__init__.py` preloads
   pyarrow before torch claims the same native libraries; wrong order is a
   segfault on Windows, not an exception. It also sets `USE_TF=0`.
4. **`transformers` must stay `<5`.** 5.x makes every tokenizer report
   `is_fast=True`, defeating the guard label alignment relies on.
5. **Heavy imports go inside functions.** A module-level `import torch`
   anywhere in the import graph breaks the serving container, which has no
   torch.
6. **`batch_size` is the effective batch size**, split into per-device batch
   and accumulation by `resolve_batching()`. Do not reinterpret it.
7. **Decode at word granularity, never subword.** Only a word's first subword
   is supervised, so continuations are untrained noise; reading them shreds
   `nephrotoxicity` into `"ne"` and `"rot"`. Evaluation drops those positions,
   so the F1 will *not* tell you if this regresses.

## Where things live

| Looking for | Go to |
|---|---|
| Label alignment | `labels.py` — no heavy imports, so it is testable cheaply |
| Scoring | `metrics.py` — strict IOB2, hand-rolled, cross-checked against seqeval |
| The one decode path | `inference.py` — shared by API, benchmark and parity check |
| Corpus loading | `data.py` |
| Casing guard | `tokenization.py` |
| Sweep definition | `sweep.py::default_grid` |
| HTTP surface | `api/main.py`, shapes in `api/schemas.py` |
| Qdrant | `retrieval/index.py`, `retrieval/search.py` |
| Unattended orchestration | `pipeline.py` |

## Making changes safely

**Changing span decoding** (`inference.decode_entities`) changes what `/extract`
returns *and* what the ONNX parity check compares — they are the same code, on
purpose. Run `test_inference.py` and `test_onnx.py`.

**Changing scoring** (`metrics.py`) changes every F1 in the table.
`test_matches_seqeval_on_random_sequences` is the safety net. Note that scoring
is deliberately stricter than serving on orphan `I-` tags — see
[ADR-004](DECISIONS.md#adr-004). A tidy-up that unifies them will fail a test
that exists to prevent exactly that.

**Changing the label set or its order** changes `id2label` in the exported
config, which is what the container reads. Re-export and re-index.

**Changing the embedding model** requires `build_index --recreate`. Old vectors
are in a different space and would rank nonsensically.

**Adding a sweep arm** — edit `default_grid`. `test_cli.py::TestSweepGrid`
asserts the grid's properties (≥6 arms, unique names, varied hyperparameters,
seed repeats), so a typo is caught immediately rather than six GPU-hours later.

**Never hand-edit a results table.** `docs/sweep_table.md`,
`docs/benchmark.md` and the four marked README blocks are generated. Regenerate
them.

## Reference environment

RTX 3050 Ti Laptop (**4 GB VRAM**), Ryzen 7 5800H, 16 GB RAM, Windows 11,
Python 3.13, torch 2.6.0+cu124. The 4 GB is why `max_per_device_batch` exists.
Full details and troubleshooting in [ENVIRONMENT.md](ENVIRONMENT.md).

Everything works on CPU, just slowly. Nothing requires this specific machine.

## Current state

- All twelve known defects fixed; none open.
- Pipeline run end to end; every documented number measured on the machine
  above. Full detail in [RESULTS.md](RESULTS.md); the run narrative in
  [PROGRESS.md](PROGRESS.md).
- 200 tests passing, offline, in about 35 seconds.
- Next work, prioritised, in [TODO.md](TODO.md). The most valuable item is
  measuring latency under concurrency — the current benchmark is serial and
  says nothing about behaviour under load.

## Honest caveats

Not defects — scope decisions, documented in [PRD.md](PRD.md):

- One entity type as shipped (`Disease`). BC5CDR would add `Chemical` but has
  not been trained. No genes in either corpus.
- Trained on abstracts, not clinical notes. Performance on real notes would be
  substantially worse and nothing here measures it.
- No negation detection. "No evidence of carcinoma" returns `carcinoma`.
- No entity linking. "MI" and "myocardial infarction" are unrelated strings.
- `score` orders predictions; it is not a calibrated probability.
- Retrieval over a few thousand abstracts is a demonstration, not a literature
  search.
- **Not a medical device.** Not validated for clinical use.

## If something is broken

1. `.\make.ps1 test` — narrows it to plumbing versus environment fast.
2. [ENVIRONMENT.md](ENVIRONMENT.md#troubleshooting) — a symptom-to-cause table
   covering every failure hit on this machine.
3. `GET /health` — for anything about the running service, `detail` names the
   underlying error rather than making you guess.
