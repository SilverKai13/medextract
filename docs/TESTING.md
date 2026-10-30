# Testing

```powershell
.\make.ps1 test          # pytest -q
.\make.ps1 cov           # with coverage
.\make.ps1 test -k retrieval
```

Linux / CI: `make test`, `make cov`.

## The strategy in one sentence

The suite verifies that the plumbing is correct; the sweep and the benchmark
measure how well it works.

That split is deliberate and it is why the suite has the shape it does. Mixing
the two would make `pytest` slow, networked, GPU-dependent and
non-deterministic — which in practice means a suite nobody runs. So there is a
hard rule: **the test suite never touches the network, never downloads a
checkpoint, and never needs a GPU.** It runs in under a minute.

What that costs: no test asserts an F1 number. Model quality is the sweep's
job, recorded in [RESULTS.md](RESULTS.md).

## The stub model

Instead of pulling BioBERT, `tests/conftest.py` builds a genuinely tiny BERT —
2 layers, hidden size 32, 2 attention heads — with random weights and a
hand-written WordPiece vocabulary, saved in the normal HuggingFace layout.

Its predictions are meaningless. That is fine: what is under test is label
alignment, span decoding, ONNX export and parity, request handling and
retrieval, none of which care whether the logits are any good.

The vocabulary is built deliberately so biomedical terms fragment into several
subwords (`aden ##oma ##tous`, `poly ##posis`, `cis ##plat ##in`), because that
is precisely the case the alignment code has to get right. A stub vocabulary
where every word was one token would test nothing.

Session-scoped fixtures: `tiny_model_dir`, `tiny_tokenizer`, `tiny_onnx_dir`
(exported through the project's own exporter, not a fixture-only shortcut),
`settings_for`, `client`.

## Coverage by area

| Module | Tests | What is pinned |
|---|---|---|
| `test_labels.py` | 12 | Exact label *positions* against a real fast tokenizer, both subword conventions, alignment length invariance |
| `test_metrics.py` | 15 | Strict IOB2 span extraction, PRF arithmetic, the seqeval cross-check, the deliberate strict/lenient split |
| `test_data.py` | 17 | Batched tokenise-and-align, truncation safety, string and integer tags, the capability guard, type restriction |
| `test_corpus_sources.py` | 31 | CoNLL and JSONL parsers, three-source fallback order, tokenizer casing guard, effective-batch arithmetic, tracking URI |
| `test_inference.py` | 28 | Span decoding edge cases, windowing, dedupe, character offsets, actionable load errors |
| `test_onnx.py` | 13 | Export shape, logit tolerance, decoded-span agreement, parity failure detection, benchmark percentiles, the integrity record and a tampered graph |
| `test_api.py` | 26 | Every endpoint, every error status, request ids, unicode offsets, batch/single equivalence, the hash on `/health` |
| `test_retrieval.py` | 30 | PubMed XML parsing, pagination, retries, rate limiting, embedders, Qdrant indexing and search |
| `test_pipeline.py` | 16 | Best-arm selection, resume logic, Docker detection, `.env` editing, the honest not-done wording |
| `test_cli.py` | 32 | Flag wiring, the sweep grid, per-type report columns, type restriction flags |
| `test_readme_sync.py` | 9 | Marker replacement, idempotence, loud failure on a missing marker, the optional load-test block |
| `test_loadtest.py` | 9 | The concurrency harness end to end through an ASGI transport: every level run, errors counted not raised, a degraded service refused |
| `test_calibration.py` | 12 | Gold spans as character offsets, reliability bins, ECE weighting, threshold table |

**250 tests**, one skipped (the seqeval cross-check when seqeval is not
importable). The same suite passes on transformers 4.x and 5.x; CI runs
both.

## Tests worth knowing about

These carry more weight than their line count suggests.

**`test_metrics.py::test_predicting_nothing_scores_zero_not_high`** — asserts
the failure mode that motivates entity-level scoring. A model predicting all
`O` scores above 90% on token accuracy; here it scores zero.

**`test_metrics.py::test_matches_seqeval_on_random_sequences`** — cross-checks
the hand-rolled scorer against seqeval across 25 randomised trials, whenever
seqeval happens to be importable. It is how the strict/lenient distinction was
found in the first place. Skipped rather than failed when seqeval is absent,
because seqeval 1.2.2 does not install on current setuptools.

**`test_metrics.py::test_strict_scoring_and_lenient_serving_differ_on_purpose`**
— pins both sides of the one case where evaluation and serving disagree, so
neither drifts into the other. See [DECISIONS.md](DECISIONS.md#adr-004).

**`test_api.py::test_unicode_offsets_are_character_offsets`** — the offsets
contract, for non-ASCII input. Python character offsets, not UTF-8 byte
offsets.

**`test_api.py::test_batch_matches_single_calls`** — batching is an
optimisation, so it must not change answers.

**`test_onnx.py::test_both_backends_decode_identical_entities`** — the parity
claim at the level a caller cares about, not just logit tolerance.

**`test_onnx.py::test_check_parity_fails_when_tolerance_is_impossible`** —
tests the *gate*, not just the happy path. A parity check that cannot fail is
not a check.

**`test_corpus_sources.py::test_a_cased_checkpoint_is_not_lowercased`** —
reproduces the BioBERT failure mode on a checkpoint the test controls: saved
without a `tokenizer_config.json`, exactly as BioBERT ships, so `AutoTokenizer`
has to invent `do_lower_case` and gets it wrong.

**`test_inference.py::TestSubwordAggregation`** — pins the fix for the one
defect no unit test could have caught, because every earlier decode test
constructed one token per word. It reproduces the exact fragmentation seen on a
real model: `B-Disease, O, I-Disease` across the six pieces of
`nephrotoxicity`, which used to decode to `"ne"` and `"rot"`.

**`test_corpus_sources.py::TestTextEncoding`** — walks the AST of every source
and test file and fails if any `read_text`/`write_text` omits `encoding`.
Behaviour cannot catch this: on a UTF-8 machine the default is already correct,
and it only breaks for someone else, later.

**`test_inference.py::test_long_text_is_windowed_not_truncated`** — an entity
past the model's context limit is still found.

**`test_onnx.py::TestArtefactIntegrity::test_a_tampered_graph_is_refused_at_load`**
— one byte appended to `model.onnx` after the parity check, and the engine
refuses it with both hashes. Its API-level twin asserts the service still
answers `/health`, degraded, with the reason in `/extract`'s 503.

**`test_data.py::test_the_guard_is_decided_by_capability_not_by_the_is_fast_flag`**
— a tokenizer whose `is_fast` says one thing and whose behaviour says
another. The guard believes the behaviour. This is the test that replaced a
library version pin.

**`test_loadtest.py::test_server_errors_are_counted_not_raised`** — a 500
mid-run is a data point about the server, not a crash of the harness.

**`test_readme_sync.py::test_real_readme_has_all_four_marker_pairs`** — runs
against the actual `README.md`, so removing a marker fails CI rather than
silently leaving stale numbers.

## Testing philosophy

**Assert behaviour, not absence of exceptions.** Three of the defects this
project has hit fail *silently* — wrong tokenizer casing, misaligned labels, a
hashing-vector index. A test that only checks "it did not throw" would pass on
all three. Each therefore has a test asserting the specific observable outcome.

**Test the failure path too.** `test_check_parity_fails_when_tolerance_is_impossible`,
`test_missing_collection_fails_loudly`, `test_gives_up_after_retries`,
`test_empty_results_raise_with_a_clear_message`. A guard that has never been
seen to fire is not known to work.

**Error messages are part of the contract.** Several tests match on message
content (`match="fast tokenizer"`, `match="label_names override"`,
`match="Could not load"`) because an unactionable error costs real time.

**Pure logic gets isolated so it can be tested cheaply.** `labels.py` has no
heavy imports specifically so alignment can be tested without `datasets`.

## What is not tested, and why

| Not covered | Why | Covered instead by |
|---|---|---|
| Model quality / F1 | Needs a GPU, the corpus, ~40 min | The sweep → [RESULTS.md §2](RESULTS.md#2-training) |
| Real latency | Machine-dependent by definition | `make benchmark` → [RESULTS.md §4](RESULTS.md#4-latency) |
| Image size | Needs a Docker daemon | CI `build` job, hard-fails over 1.5 GB |
| Live E-utilities | Network, rate limits, flaky by nature | Mocked transports; the real client is exercised by `make fetch` |
| Concurrency / load | Machine-dependent | The harness is tested here; `make loadtest` → [RESULTS.md §4.3](RESULTS.md#43-latency-under-concurrency) |
| Score reliability | Needs the real model and corpus | `make calibrate` → [calibration.md](calibration.md) |
| Real embedding quality | Needs the encoder downloaded | Nothing — the hashing stub tests the plumbing only |

The `slow` marker exists in `pyproject.toml` for tests needing network or a
real model. Nothing currently uses it; it is there so that adding such a test
does not mean weakening the default run.

## CI

`.github/workflows/ci.yml`:

| Job | Does |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `test` | pytest with coverage on Python 3.11 and 3.12, each against transformers `<5` and `>=5.10`, `fail-fast: false` |
| `audit` | `pip-audit --strict` on what the image installs; an advisory audit of the training extras |
| `build` | Build the image, enforce the 1.5 GB budget, scan it with Trivy, then start it **with no model mounted** and assert `/health` answers as degraded rather than crash-looping |
| `publish` | On a `v*` tag only, push to GHCR |

The test job installs CPU-only torch from the PyTorch index so the runner does
not pull ~2.5 GB of CUDA wheels it cannot use. The four-way matrix is what
makes the "both transformers lines" claim a tested one rather than a stated
one.

That smoke test is the one worth noting: it verifies a *degraded* start, which
is the behaviour that makes a broken deployment diagnosable.

## Adding a test

1. Offline. No network, no checkpoint download, no GPU.
2. Use the existing fixtures rather than building another stub.
3. Assert the observable outcome, not that nothing raised.
4. If it needs the network or a real model, mark it `slow` and say why.
5. Name it as a sentence about behaviour —
   `test_invalid_continuation_is_discarded_not_repaired`, not `test_spans_2`.
