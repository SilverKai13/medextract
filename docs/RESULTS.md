# Results

Every measurement this project has produced, in full, with the command that
produced it. This is the reference; [PROGRESS.md](PROGRESS.md) is the dated
narrative of the run that generated it.

**Measured 2026-08-20** on the reference machine ([ENVIRONMENT.md](ENVIRONMENT.md)):
AMD Ryzen 7 5800H (8C/16T), 16 GB RAM, NVIDIA RTX 3050 Ti Laptop 4 GB, driver
566.07 / CUDA 12.7, Windows 11 build 26200, Python 3.13.5, torch 2.6.0+cu124,
onnxruntime 1.26.0.

Nothing below was typed by hand. Each section names the artefact it was read
from and the command that regenerates it.

---

## 1. Headline

| Claim | Target | Measured | Verdict |
|---|---|---|---|
| Entity F1, NCBI-Disease test | ≥ 0.85 | **0.8838** best, 0.8803 mean over 9 arms | Met by every arm |
| Entity F1, validation | ≥ 0.85 | **0.8818** best | Met by every arm |
| ONNX ≡ PyTorch, logits | < 1e-4 | **1.48e-05** | 6.8× inside tolerance |
| ONNX ≡ PyTorch, argmax | 100% | **1.0000** | Exact |
| ONNX ≡ PyTorch, decoded spans | identical | **identical** | Exact |
| ONNX speedup, batch 1 | — | **2.72×** | Measured |
| ONNX speedup, batch 16 | — | **1.32×** | Measured |
| Serving image | < 1.5 GB | **395 MB** | 26% of budget |
| PyTorch in serving image | absent | **absent** | Verified inside the image |
| Configurations compared | ≥ 6 | **9** | With 3 seed repeats |
| Test suite | offline, fast | **202 passed, 1 skipped, 26 s** | No network, no GPU |

---

## 2. Training

### 2.1 Setup

| | |
|---|---|
| Corpus | NCBI-Disease |
| Source that answered | `hub-parquet (ncbi/ncbi_disease@refs/convert/parquet)` |
| Splits | 5,433 train / 924 validation / 941 test sentences |
| Gold entity spans | 787 validation / 960 test |
| Label set | `O`, `B-Disease`, `I-Disease` |
| Scoring | Strict entity-level IOB2, micro-averaged |
| Precision | fp16 on all arms (CUDA auto-detected) |
| Wall clock | ~40 min for all nine arms |

The gold-span counts matter for reading everything below: **787 validation
spans means one entity is worth 0.13% of recall.** Differences of a few
thousandths of F1 are differences of a handful of entities.

```powershell
.\make.ps1 sweep          # artifacts/sweep/sweep_results.json + mlflow.db
.\make.ps1 report         # docs/sweep_table.md
```

### 2.2 Full results

Sorted by validation F1. Generated table: [sweep_table.md](sweep_table.md).

| arm | base | lr | batch | ep | seed | subwords | val P | val R | **val F1** | val loss |
|---|---|---|---|---|---|---|---|---|---|---|
| pubmedbert_lr3e-5_bs16_ep3 | PubMedBERT | 3e-5 | 16 | 3 | 42 | first | 0.8730 | 0.8907 | **0.8818** | 0.0520 |
| lr3e-5_bs16_ep3_seed2024 | BioBERT | 3e-5 | 16 | 3 | 2024 | first | 0.8534 | 0.8729 | **0.8631** | 0.0432 |
| lr3e-5_bs16_ep5 | BioBERT | 3e-5 | 16 | 5 | 42 | first | 0.8548 | 0.8679 | **0.8613** | 0.0622 |
| lr3e-5_bs16_ep3_seed1337 | BioBERT | 3e-5 | 16 | 3 | 1337 | first | 0.8516 | 0.8679 | **0.8597** | 0.0469 |
| lr3e-5_bs16_ep3_all-subwords | BioBERT | 3e-5 | 16 | 3 | 42 | **all** | 0.8483 | 0.8666 | **0.8573** | 0.0637 |
| lr3e-5_bs16_ep3 | BioBERT | 3e-5 | 16 | 3 | 42 | first | 0.8500 | 0.8640 | **0.8570** | 0.0561 |
| lr3e-5_bs32_ep5 | BioBERT | 3e-5 | 32 | 5 | 42 | first | 0.8518 | 0.8615 | **0.8566** | 0.0451 |
| lr5e-5_bs16_ep3 | BioBERT | 5e-5 | 16 | 3 | 42 | first | 0.8405 | 0.8640 | **0.8521** | 0.0592 |
| lr2e-5_bs16_ep3 | BioBERT | 2e-5 | 16 | 3 | 42 | first | 0.8309 | 0.8742 | **0.8520** | 0.0409 |

Same arms, held-out test split:

| arm | test P | test R | **test F1** | test loss |
|---|---|---|---|---|
| lr3e-5_bs16_ep3_seed1337 | 0.8802 | 0.8875 | **0.8838** | 0.0527 |
| lr3e-5_bs32_ep5 | 0.8913 | 0.8708 | **0.8809** | 0.0458 |
| lr3e-5_bs16_ep3_seed2024 | 0.8723 | 0.8896 | **0.8809** | 0.0436 |
| lr2e-5_bs16_ep3 | 0.8645 | 0.8969 | **0.8804** | 0.0425 |
| lr3e-5_bs16_ep3_all-subwords | 0.8682 | 0.8854 | **0.8767** | 0.0708 |
| lr3e-5_bs16_ep3 | 0.8650 | 0.8875 | **0.8761** | 0.0520 |
| lr5e-5_bs16_ep3 | 0.8600 | 0.8896 | **0.8746** | 0.0564 |
| pubmedbert_lr3e-5_bs16_ep3 | 0.8655 | 0.8781 | **0.8718** | 0.0533 |
| lr3e-5_bs16_ep5 | 0.8643 | 0.8760 | **0.8702** | 0.0660 |

### 2.3 The same results as entity counts

Percentages hide how few decisions separate these arms. Derived from precision,
recall and the 787 gold validation spans:

| arm | true pos | predicted | false pos | false neg |
|---|---|---|---|---|
| pubmedbert_lr3e-5_bs16_ep3 | **701** | 803 | **102** | **86** |
| lr2e-5_bs16_ep3 | 688 | 828 | 140 | 99 |
| lr3e-5_bs16_ep3_seed2024 | 687 | 805 | 118 | 100 |
| lr3e-5_bs16_ep3_seed1337 | 683 | 802 | 119 | 104 |
| lr3e-5_bs16_ep5 | 683 | 799 | 116 | 104 |
| lr3e-5_bs16_ep3_all-subwords | 682 | 804 | 122 | 105 |
| lr3e-5_bs16_ep3 | 680 | 800 | 120 | 107 |
| lr5e-5_bs16_ep3 | 680 | 809 | 129 | 107 |
| lr3e-5_bs32_ep5 | 678 | 796 | 118 | 109 |

The whole BioBERT table spans **678 to 688 true positives** — ten entities.

### 2.4 Run-to-run variance

Three seeds, everything else identical (`lr3e-5_bs16_ep3`):

| seed | val F1 | val TP | test F1 |
|---|---|---|---|
| 42 | 0.8570 | 680 | 0.8761 |
| 1337 | 0.8597 | 683 | 0.8838 |
| 2024 | 0.8631 | 687 | 0.8809 |
| **mean** | **0.8599** | 683 | **0.8803** |
| **sd** | **0.0031** | | **0.0039** |
| **spread** | **0.0061** | **7 entities** | **0.0077** |

Nothing varies but weight initialisation and data ordering. **Seven entities out
of 787.** That is the resolution limit of this benchmark at one seed per arm,
and it is the number every other comparison has to be read against.

### 2.5 What is distinguishable, and what is not

| Comparison | Δ val F1 | vs seed spread (0.0061) | Verdict |
|---|---|---|---|
| Subword convention (all vs first) | **0.0004** | 0.07× | **Indistinguishable** |
| Batch 16 vs 32 (at ep5) | 0.0047 | 0.77× | Indistinguishable |
| 3 vs 5 epochs (at lr3e-5) | 0.0043 | 0.70× | Indistinguishable |
| lr 2e-5 vs 3e-5 vs 5e-5 | ≤ 0.0050 | ≤ 0.82× | Indistinguishable |
| All 8 BioBERT arms, full range | 0.0111 | 1.8× | Indistinguishable |
| **PubMedBERT vs best BioBERT** | **0.0187** | **3.1×** | **Real** |

**The subword convention arm settles its own question.** It exists to test
[ADR-002](DECISIONS.md#adr-002) rather than argue about it, and the answer is
0.0004 — a fifteenth of the noise, three entities. For this corpus and model
the convention does not matter. Worth having run precisely because the
intuition says it should.

**No hyperparameter in the BioBERT grid is separable from noise.** Reporting
that is the useful outcome. Presenting the 0.0111 top-to-bottom range as a
tuning result would be reading seed variance as signal.

### 2.6 PubMedBERT is the one real effect

+0.0187 validation F1 over the best BioBERT arm, ~3 seed-widths, and unusually
it improves **both** sides of the trade:

| | PubMedBERT | BioBERT (seed mean) | Δ |
|---|---|---|---|
| val precision | 0.8730 | 0.8517 | +0.0213 |
| val recall | 0.8907 | 0.8683 | +0.0224 |
| false positives | **102** | 119 | −17 |
| false negatives | **86** | 104 | −18 |

Plausible mechanism, consistent with what the tokenizers do. PubMedBERT's
vocabulary is built from PubMed rather than adapted from general-domain BERT,
so biomedical terms survive whole:

| term | PubMedBERT | BioBERT |
|---|---|---|
| `adenomatous` | `adenomatous` (1) | `Aden ##oma ##to ##us` (4) |
| `egfr` | `egfr` (1) | `E ##G ##F ##R` (4) |
| `brca1` | `brca1` (1) | `BR ##CA ##1` (3) |

Under the first-subword convention, a term that is one token gets one
supervised position covering the whole word; a term that is four tokens gets
one supervised position covering a quarter of it. Fewer pieces means less to
get wrong.

### 2.7 Two cautions on reading this table

**Validation and test F1 are negatively correlated: r = −0.399.**

PubMedBERT is first on validation and **eighth of nine** on test.
`lr3e-5_bs16_ep5` is third on validation and last on test. With ~900 sentences
and ~900 gold spans per split, and one seed per arm, this is what noise looks
like — not a ranking that inverts for some interesting reason.

`select_best_arm` picks by validation F1. That is a reproducible rule chosen in
advance, not a claim that the selected arm generalises best. Distinguishing
these arms would need more seeds per arm, not more arms.

**Validation loss carries no information about entity F1: r = −0.001.**

| arm | val loss | val F1 |
|---|---|---|
| lr2e-5_bs16_ep3 | **0.0409** (best) | **0.8520** (worst) |
| lr3e-5_bs16_ep3_all-subwords | 0.0637 (worst) | 0.8573 |
| pubmedbert | 0.0520 | **0.8818** (best) |

The arm with the lowest loss has the lowest F1. Token-level cross-entropy and
strict entity-level F1 are close to orthogonal here, which is exactly why
`metric_for_best_model="f1"` and not loss. Checkpoint selection on loss would
have picked differently and worse.

---

## 3. ONNX export and parity

```powershell
.\make.ps1 export        # artifacts/onnx/ + parity.json + artifacts/embedder/
```

Exported arm: `pubmedbert_lr3e-5_bs16_ep3` (highest validation F1).

| Check | Tolerance | Measured |
|---|---|---|
| Max absolute logit difference | `atol=1e-4`, `rtol=1e-3` | **1.48e-05** |
| Argmax agreement (per token) | — | **1.0000** |
| Decoded entity spans | identical | **identical** |
| Probe texts | — | 5 |

Independently reproduced on a second arm (`lr2e-5_bs16_ep3`) while de-risking:
**7.87e-06**, argmax 1.0000, spans identical.

Bit equality is not on offer — ONNX Runtime and PyTorch fuse and schedule
kernels differently. 1e-4 is tight enough that neither argmax nor the softmax
confidences move, which the second and third checks confirm directly rather
than assume.

| Artefact | Size |
|---|---|
| NER graph `artifacts/onnx/model.onnx` | 435.9 MB, opset 14, dynamic batch + sequence |
| Query encoder `artifacts/embedder/model.onnx` | 435.8 MB, mean pooling, 768-dim |

Encoder pooling was read from the model's own `1_Pooling/config.json`, not
assumed — getting it wrong produces embeddings that are subtly worse rather
than obviously broken.

---

## 4. Latency

```powershell
.\make.ps1 benchmark     # docs/benchmark.json + docs/benchmark.md
```

100 iterations after 10 warmup, CPU only, `OMP_NUM_THREADS=8`,
`torch_num_threads=8`. Tokenisation is done once and hoisted out of the timing
loop: it is identical across backends, so including it would dilute the
difference being measured.

| backend | batch | mean | **p50** | p90 | p95 | p99 | min | max | throughput |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch eager | 1 | 50.73 | **50.09** | 54.80 | 55.82 | 56.91 | 46.87 | 59.30 | 19.97 seq/s |
| ONNX Runtime | 1 | 18.54 | **18.39** | 19.57 | 19.96 | 20.93 | 16.98 | 21.06 | 54.39 seq/s |
| PyTorch eager | 16 | 189.31 | **187.12** | 198.57 | 201.14 | 217.54 | 176.92 | 232.42 | 85.50 seq/s |
| ONNX Runtime | 16 | 142.75 | **142.19** | 152.62 | 155.80 | 158.78 | 124.58 | 162.87 | 112.53 seq/s |

All figures in milliseconds.

**Speedup at p50: 2.72× at batch 1, 1.32× at batch 16.**

Two things worth reading off this table beyond the headline:

**The gain shrinks with batch size.** At batch 1 the work per call is small
enough that fixed per-call overhead dominates, and that is what ONNX Runtime
mainly removes. At batch 16 the matrix multiplications already saturate the
CPU, so there is proportionally less overhead per sequence left to recover. The
README previously said "expect 2× to 4×, do not quote a number from that
range" — batch 1 lands inside it, batch 16 does not. That is precisely why the
instruction was to measure.

**ONNX Runtime's tail is tighter, not just its median.** Spread from min to
p99 is 4.1 ms against PyTorch's 10.0 ms at batch 1, and 34.2 ms against 40.6 ms
at batch 16. For a service where the tail is what a caller notices, the more
predictable distribution is worth as much as the lower median.

Single-machine CPU numbers. No GPU was involved and no GPU numbers are
reported.

---

## 5. Retrieval

```powershell
.\make.ps1 fetch         # data/pubmed.jsonl
.\make.ps1 index         # Qdrant collection
```

### 5.1 Corpus

| | |
|---|---|
| Records fetched | **3,000** |
| Unique PMIDs | 3,000 (no duplicates) |
| File | `data/pubmed.jsonl`, 5.7 MB |
| Abstract length | min 4, median **1,660**, mean 1,659, max 5,446 chars |
| Publication years | 2023–2026, 26 records with no year (stored `null`) |
| Distinct journals | 691 |
| Structured abstracts | **1,191** reassembled with their section labels |
| Records with no abstract | dropped at parse — 0 empty records indexed |
| Missing titles | 0 |

Query: `(neoplasms[MeSH] OR "drug therapy"[MeSH] OR genetics[MeSH]) AND
hasabstract AND english[lang] AND 2015:2025[dp]`, fetched through E-utilities
at the published rate limit with a contact address attached.

The 1,191 structured abstracts are the ones that would have been silently
truncated to their first section by a naive parser — 40% of the corpus.

### 5.2 Index

| | |
|---|---|
| Collection | `pubmed_abstracts` |
| Points | **3,000** |
| Dimension | 768 (read from the encoder, never hardcoded) |
| Distance | Cosine (vectors L2-normalised at encode time) |
| Status | green |
| Point ids | UUIDv5 of the PMID — re-indexing is idempotent |
| Encoder | `pritamdeka/S-PubMedBert-MS-MARCO` via ONNX; hashing fallback refused |

### 5.3 Retrieval quality

Not a recall benchmark — there are no relevance judgements for this corpus.
Three queries with their top hits, as a sanity check that the vectors mean
something:

**`targeted therapy for EGFR mutant lung cancer`** — 702 ms, query entities
`[('EGFR mutant lung cancer', 'Disease')]`

| score | title |
|---|---|
| 0.9291 | Case Report: Ivonescimab in EGFR-mutant lung cancer with baseline malignant pleural effusion |
| 0.9264 | Advancements in targeted therapies for acute myeloid leukemia |
| 0.9258 | Frontiers in thoracic oncology: new breakthroughs in molecular targets and immunotherapy |

**`BRCA1 germline mutation breast cancer screening`** — 908 ms, query entities
`[('breast cancer', 'Disease')]`

| score | title |
|---|---|
| 0.9266 | Germline BRCA1/2 Variants in Polish Patients with Family History of Breast and Ovarian Cancer |
| 0.9178 | Geographic Variability in Germline Genetic Testing for Prostate Cancer |
| 0.9092 | Utilization of Genetic Testing and Surgical Implications in an Ethnically Diverse cohort |

**`cisplatin induced nephrotoxicity`** — 1075 ms, query entities
`[('nephrotoxicity', 'Disease')]`

| score | title |
|---|---|
| 0.9260 | Nephrotoxicity and associated risk factors in patients treated with pemetrexed |
| 0.9144 | The association between DNA repair gene polymorphisms and cisplatin-induced ototoxicity |
| 0.9115 | Pro-apoptotic effects of metformin and cisplatin in non-small cell Lung Cancer |

Top hit is a direct topical match in all three. Note the **compressed score
range** — 0.90 to 0.93 across quite different queries — which is characteristic
of a domain-specific encoder on a single-domain corpus: everything is somewhat
similar to everything. Absolute scores are not a relevance threshold; the
ranking is what carries information.

`took_ms` of 0.7–1.1 s is dominated by running the NER model over every hit,
not by the vector search. `extract_entities: false` skips that.

---

## 6. Serving container

```powershell
.\make.ps1 docker && .\make.ps1 docker-size
```

| | Measured |
|---|---|
| Image size | **395 MB** (budget 1,500 MB — **26%**) |
| Base | `python:3.11-slim-bookworm` |
| `torch` / `datasets` / `mlflow` | **absent**, verified by import probe inside the image |
| `onnxruntime` / `transformers` / `fastapi` / `qdrant-client` | present |
| User | non-root, uid 10001 |
| Healthcheck | hits `/health`, not the port |

The README predicted ~560 MB by adding a 390 MB dependency measurement to a
~150 MB base. That double-counts: the 390 MB venv sits on a Python runtime the
base image already supplies. Corrected in place with the reasoning kept, rather
than quietly adjusted.

### 6.1 Live endpoint verification

Against the running container:

| Check | Result |
|---|---|
| `/health`, no model mounted | 200, `status: degraded`, reason in `detail` |
| `/extract`, no model mounted | 503, names the missing path and the fix |
| `/health`, model mounted | 200, `ok`, labels `["O","B-Disease","I-Disease"]`, retrieval available |
| `/extract`, README example | `Familial adenomatous polyposis` [0:30] @ **0.9994** |
| Offsets, non-ASCII | 65-char / 70-byte input; offsets index **characters** |
| `/extract/batch` vs `/extract` | byte-identical results |
| `/search` | ranked hits, entities tagged per hit |
| Oversized input (25 kB) | **413** |
| Empty text / `min_score=5` / missing field | **422** |
| `X-Request-ID` | echoed when supplied, generated otherwise |
| Logs | JSON, one object per line, **no request bodies** |

The README's documented example response is now literally what the service
returns — start 0, end 30 — rather than an illustration.

---

## 7. Test suite

```powershell
.\make.ps1 test
```

**202 passed, 1 skipped, 26 s.** Offline: no network, no checkpoint download,
no GPU. Started this work at 163 (all 10 modules failing to import).

| Module | Tests |
|---|---|
| `test_retrieval.py` | 30 |
| `test_corpus_sources.py` | 31 |
| `test_inference.py` | 28 |
| `test_cli.py` | 25 |
| `test_api.py` | 23 |
| `test_metrics.py` | 15 |
| `test_pipeline.py` | 14 |
| `test_labels.py` | 12 |
| `test_data.py` | 12 |
| `test_onnx.py` | 7 |
| `test_readme_sync.py` | 6 |

The skip is the seqeval cross-check, which runs only when seqeval is
importable — it does not install on current setuptools.

`ruff check` and `ruff format --check` clean across 38 files.

---

## 8. Where the raw numbers live

| Artefact | Contents |
|---|---|
| `mlflow.db` | Every run: 24 metrics × 9 arms, all hyperparameters, corpus provenance |
| `artifacts/sweep/sweep_results.json` | Per-arm summary, written incrementally |
| `artifacts/onnx/parity.json` | The parity result quoted in §3 |
| `docs/benchmark.json` | Full latency distribution plus hardware and thread settings |
| `docs/sweep_table.md` | §2.2, generated by `report.py` |
| `docs/benchmark.md` | §4, generated by `benchmark.py` |
| `data/pubmed.jsonl` | The 3,000 abstracts |

Inspect the tracking store directly:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 9. What these results do not show

Stated plainly, because the numbers above are easy to over-read:

- **No clinical-note performance.** Both corpora are edited PubMed abstracts.
  Performance on real notes would be substantially worse and nothing here
  measures it.
- **No concurrency profile.** §4 is serial, one request in flight. It says
  nothing about p99 under load. The most valuable open item in
  [TODO.md](TODO.md).
- **No retrieval recall metric.** §5.3 is a sanity check, not an evaluation.
  There are no relevance judgements for this corpus.
- **No calibration.** `score` orders predictions usefully; it is not a
  probability of correctness, and `min_score` invites treating it as one.
- **One entity type.** The shipped model emits `Disease` only. BC5CDR loads and
  would add `Chemical`, but no arm has been trained on it.
- **Single machine, single run.** Every number is one machine, and — apart from
  the three seed repeats — one seed. §2.4 is the honest estimate of how much
  that matters.
- **Not a medical device.** Nothing here is validated for clinical use.
