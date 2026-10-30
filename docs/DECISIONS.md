# Decision Record

Decisions that a reader is likely to question, with the actual reasoning and
what each one costs. Newest last.

---

<a id="adr-001"></a>

## ADR-001 — Fine-tune an encoder rather than prompt an LLM

**Status:** accepted

**Context.** Biomedical NER can be done by fine-tuning a domain-pretrained
encoder or by prompting a general LLM.

**Decision.** Fine-tune BioBERT for token classification.

**Why.** The output contract requires exact character offsets. A token
classifier produces them structurally — every prediction is attached to a token
whose offsets the tokenizer already knows. An LLM produces text that then has
to be aligned back to the source, and that alignment fails on repeated
substrings, near-misses in surface form, and hallucinated spans. Also: single-
digit-millisecond CPU inference, no per-token cost, no third-party data egress,
and a benchmark that published baselines exist for.

**Cost.** Two entity types, fixed at training time. Adding a type means a new
corpus and a new run, not a prompt edit.

---

<a id="adr-002"></a>

## ADR-002 — Label only the first subword of each word

**Status:** accepted, and swept

**Context.** Labels are per word; the model sees subwords. `Adenomatous`
becomes four pieces. Something must decide what the other three get.

**Decision.** Default to labelling only the first subword; the rest get `-100`.
The alternative — label every subword, converting `B-` to `I-` after the first
— is implemented and run as a sweep arm.

**Why.** The first-subword convention is the standard HuggingFace recipe and
what published NCBI-Disease baselines use, so the reported F1 is comparable to
them. Making the alternative an arm rather than an argument means the choice is
evidenced rather than asserted.

**Cost.** Fewer supervised positions per epoch.

---

<a id="adr-003"></a>

## ADR-003 — Hand-rolled entity scoring instead of seqeval

**Status:** accepted

**Context.** seqeval is the conventional library for IOB2 entity scoring.

**Decision.** Implement strict IOB2 scoring in `metrics.py`, about sixty lines.

**Why.** seqeval 1.2.2 ships only an sdist whose `setup.py` no longer builds
against current setuptools — it fails at metadata generation, which would break
both CI and the Docker build. An unbuildable dependency in a container image is
worse than sixty lines of well-tested code.

The shortcut is verified rather than assumed:
`test_matches_seqeval_on_random_sequences` cross-checks against seqeval across
25 randomised trials whenever seqeval *is* importable.

**Cost.** Code to maintain. Mitigated by the cross-check — which is also how
ADR-004 was discovered.

---

<a id="adr-004"></a>

## ADR-004 — Scoring is stricter than serving, on exactly one case

**Status:** accepted

**Context.** An `I-Disease` tag with no `B-Disease` before it is a malformed
prediction. Models emit them.

**Decision.** Evaluation discards it (strict IOB2). Serving recovers it into a
span.

**Why.** They are answering different questions.

- *Scoring* is asked "how does this compare to the published baseline?"
  Baselines score strictly; scoring leniently would inflate the number against
  the target the brief sets, and make it incomparable.
- *Serving* is asked "what is in this text?" A caller wants the entity found.
  Refusing to return one because the model emitted a malformed prefix helps
  nobody.

**Cost.** Two code paths that must not converge. Both are pinned by tests —
`test_strict_scoring_and_lenient_serving_differ_on_purpose` exists specifically
so a future tidy-up that unifies them fails.

---

<a id="adr-005"></a>

## ADR-005 — `torch.onnx.export` rather than `optimum`

**Status:** accepted; `optimum` retained as an option

**Decision.** Default to `torch.onnx.export`; `--exporter optimum` is
implemented and selectable.

**Why.** The native path pins input names, output names, dynamic axes and opset
explicitly rather than inheriting a converter's defaults, and it works across
torch versions. `optimum`'s version compatibility with torch is narrower, and
the combination in this environment hits a bug in its external-data handling.
`dynamo=False` for a related reason: torch 2.6's dynamo exporter pulls in
onnxscript and mishandles external data for small graphs.

**Cost.** No ORTOptimizer graph fusions by default. `--exporter optimum
--optimize` is there when they are wanted.

---

<a id="adr-006"></a>

## ADR-006 — Export is gated on a three-level parity check

**Status:** accepted

**Decision.** Compare logits (`atol=1e-4`), argmax agreement, and decoded
entity spans. Any mismatch aborts the export.

**Why.** A benchmark of a graph that quietly computes something different is a
benchmark of the wrong thing. The three levels escalate toward what a caller
actually experiences: logits are the numerical claim, argmax is the prediction,
identical spans are the promise `/extract` makes.

Bit equality is not on offer — ONNX Runtime and PyTorch fuse and schedule
kernels differently. 1e-4 is tight enough that neither argmax nor the softmax
confidences move.

**Cost.** A few seconds per export, and an export that can fail. Both intended.

---

<a id="adr-007"></a>

## ADR-007 — No PyTorch in the serving container

**Status:** accepted

**Decision.** `requirements-serve.txt` excludes torch, datasets, mlflow and the
training stack. Serving runs ONNX graphs only.

**Why.** Torch is needed to *produce* the graph, which happens offline, and
never to run it. The torch wheel alone is larger than the entire resulting
image, so dropping it is the single biggest saving.

**Cost.** The query encoder also had to be exported to ONNX — hence
`retrieval/export_embedder.py` — because `sentence-transformers` would have
dragged torch back in. Pooling and normalisation moved into numpy as a result.

---

<a id="adr-008"></a>

## ADR-008 — The service starts without a model

**Status:** accepted

**Decision.** Model and retrieval load failures are caught at startup. The
process comes up, `/health` returns 200 with `status: degraded` and the error,
and the affected endpoints return 503 with the same detail.

**Why.** A container that crash-loops cannot tell you why. This is diagnosable
from `docker ps` and one curl. CI asserts it explicitly by starting the image
with no model mounted.

**Cost.** A misconfigured deployment looks "up" to a naive port check. Mitigated
by the `HEALTHCHECK` hitting `/health` rather than the port.

---

<a id="adr-009"></a>

## ADR-009 — Retrieval, not RAG

**Status:** accepted

**Decision.** `/search` returns ranked passages with entities tagged. Nothing
generates an answer.

**Why.** Calling it RAG would claim a generation step that does not exist. The
retrieval and extraction layers are the parts with verifiable behaviour;
bolting on generation would add the one component whose output cannot be
checked against a ground truth here.

**Cost.** Callers wanting a synthesised answer must add that themselves.

---

<a id="adr-010"></a>

## ADR-010 — Refuse the hashing-embedder fallback when indexing

**Status:** accepted

**Decision.** The *service* falls back to hash embeddings if the real encoder
cannot load, so it keeps answering. `build_index` does not:
`allow_fallback=False`, overridable with `--allow-hashing-fallback`.

**Why.** An index built from hash vectors looks fine, retrieves badly, and
costs a full pass over the corpus to rebuild. Failing loudly is cheaper than
discovering it from bad search results weeks later. Asymmetry between the two
paths is the point: a degraded answer now is acceptable; a degraded artefact
persisted to disk is not.

---

<a id="adr-011"></a>

## ADR-011 — Corpus loading has three sources

**Status:** accepted

**Context.** `load_dataset("ncbi_disease")` fails on any current install. The
Hub repository contains a loader script and no data files, and `datasets` 3.0
removed script execution.

**Decision.** Try, in order: the Hub's `refs/convert/parquet` branch, a plain
Hub load, then direct download of the upstream corpus files the official loader
script itself fetches. Log which answered and record it in MLflow.

**Why.** Pinning `datasets<3` to keep script loading would freeze a core
dependency to reach a deprecated code path. The parquet branch works today but
is Hub-generated infrastructure that is documented as subject to change. The
upstream files are the actual ground truth and are stable. Having all three
means no single mechanism disappearing breaks training.

**Cost.** More code than one `load_dataset` call, and one behavioural
difference: the direct parser drops the empty end-of-file example the official
script emits, so split sizes differ by one (5432/923/940 versus
5433/924/941). An example with no tokens has no gold or predicted spans and
cannot move entity-level F1.

---

<a id="adr-012"></a>

## ADR-012 — Tokenizer casing is detected from the vocabulary

**Status:** accepted

**Context.** `dmis-lab/biobert-base-cased-v1.2` ships no
`tokenizer_config.json`. `AutoTokenizer` therefore falls back to the BERT class
default of `do_lower_case=True` — against a *cased* vocabulary. `APC` becomes
`a ##p ##c`; every gene symbol shatters into single characters. Training
converges, the loss curve looks healthy, and the F1 is quietly several points
lower.

**Decision.** `tokenization.load_fast_tokenizer` decides casing from the
vocabulary itself — the ratio of content tokens containing uppercase, special
and bracketed tokens excluded — then verifies with a probe string and raises if
case still did not survive.

**Why.** The checkpoints that get this wrong are exactly the ones whose
metadata is missing, so metadata cannot be the signal. Inferring from the model
name would work for `biobert-base-cased` and fail on the next checkpoint named
carelessly. The vocabulary is an observation about the artefact and cannot lie
about itself. Raising rather than warning because a run that trains on shredded
input is far more expensive to discover later.

**Cost.** A heuristic with a threshold. Mitigated by there being no checkpoint
between 0% and 20% uppercase, and by the probe check catching a
misclassification.

---

<a id="adr-013"></a>

## ADR-013 — `batch_size` means effective batch size

**Status:** accepted

**Context.** The reference GPU has 4 GB. The sweep contains a `bs32` arm.

**Decision.** `batch_size` is the effective batch size. `resolve_batching`
splits it into a per-device batch and gradient accumulation steps whose product
always equals it. `sweep.run_sweep` retries an OOM arm once at half the
per-device batch.

**Why.** The alternative — silently training on whatever fits — would make the
`bs32` row in the comparison table a lie, and make results incomparable across
machines. This way an arm means the same thing on a laptop card as on a
datacentre GPU, and the split is recorded in MLflow so the hardware reality is
visible without being confused with the hyperparameter.

**Cost.** More wall-clock time for large-batch arms.

---

<a id="adr-014"></a>

## ADR-014 — MLflow tracks to SQLite, not `./mlruns`

**Status:** accepted

**Decision.** Default tracking URI is `sqlite:///mlflow.db`, from
`config.mlflow_tracking_uri()`.

**Why.** MLflow 3 put the filesystem store into maintenance mode and raises
rather than opening one, so it is not a working default. Independently, the
file store writes thousands of very small files — one per metric and per
parameter per run — which is poor behaviour inside a synced folder.

Routing through one helper also fixes a Windows trap: interpolating a path into
a `file://` prefix yields backwards separators and only two slashes, so MLflow
reads the drive letter as a hostname and writes runs somewhere nobody looks.

**Cost.** `mlruns/` directories from earlier runs are not read. `mlflow migrate-filestore`
exists if that matters.

---

<a id="adr-015"></a>

## ADR-015 — Import-time environment guards in `__init__.py`

**Status:** accepted

**Context.** Two failures on the reference machine happen before any project
code runs. Importing pyarrow after torch and transformers have initialised
kills the process with an access violation — a segfault, not a catchable
exception. And a TensorFlow install anywhere on the machine makes
`transformers.Trainer` raise at import, because its integrations reach for
`TFPreTrainedModel` and Keras 3 is unsupported.

**Decision.** `medextract/__init__.py` preloads pyarrow and sets `USE_TF=0`.

**Why.** `__init__.py` is the one module every entry point goes through, so the
guards cannot drift out of sync with the places that need them. Both are
no-ops where they do not apply — the serving container has neither pyarrow nor
torch. For TensorFlow, the alternative is `pip install tf-keras`, a
compatibility shim installed purely to satisfy an import path never taken;
declining the path is smaller and makes behaviour identical on machines with
and without TF. `setdefault` keeps it overridable.

**Cost.** Import side effects, which are normally worth avoiding. Accepted
because the alternative is a segfault with no stack trace.

---

<a id="adr-016"></a>

## ADR-016 — Checkpoints pruned, weights not logged to MLflow

**Status:** accepted

**Decision.** `keep_checkpoints=False` removes the Trainer's per-epoch
checkpoints after the final save. `log_model_artifact=False` logs
`config.json`, the tokenizer files and a `model_dir` tag instead of ~430 MB of
weights per arm.

**Why.** Nine arms would otherwise leave several gigabytes of optimiser state
plus a second full copy of every model. The checkpoints have already served
their purpose by the time `load_best_model_at_end` has run, and the weights are
on disk under `artifacts/` with a tag pointing at them.

**Cost.** No resume-from-checkpoint after an arm finishes, and MLflow cannot
serve the model directly. Both flags flip it back.

---

<a id="adr-017"></a>

## ADR-017 — Decode at word granularity, not subword

**Status:** accepted

**Context.** Under the default alignment convention (ADR-002) only a word's
first subword is supervised; continuations get `IGNORE_INDEX` and contribute
nothing to the loss. Their predictions were therefore never trained.

`decode_entities` originally ran the BIO state machine over every subword
position, including those. On a real fine-tuned model this shredded output:
`nephrotoxicity` predicted `B-Disease, O, I-Disease, O, O, O` across its six
pieces decoded to the two fragments `"ne"` and `"rot"`, with characters
silently skipped between them.

**Decision.** Collapse subwords into words before decoding. A word's label and
score come from its first subword; its span runs from that subword's start to
the last subword's end.

**Why.** It is the only decoding consistent with how the model was trained —
reading an unsupervised position treats noise as evidence. It is also the
standard choice (`aggregation_strategy="first"` in HuggingFace's token
classification pipeline).

Confirmed not to be fixable by training instead: the `all-subwords` arm, where
continuations *are* supervised, fragmented too. One stray mid-word `O` still
breaks a span, and biomedical words fragment into many pieces, so it happens
constantly.

**Why it went unnoticed.** Evaluation and serving used different code at
different granularities. `metrics.decode_predictions` drops positions whose
gold label is `IGNORE_INDEX`, so scoring only ever saw first subwords and was
correct throughout. The reported F1 was honest about the convention it
measured while the product returned fragments — the worst kind of divergence,
because the headline number looks fine.

**Consequence, and the real win.** Evaluation and serving now agree on
granularity, so the reported F1 finally predicts what a caller receives. Before
this they were measuring different things and nothing said so.

**Cost.** `decode_entities` needs `word_ids` from the tokenizer. Passing
`word_ids=None` keeps the old per-token path for callers already working at
token level, which is what the existing token-level tests exercise.

## ADR-018 — Tokenizer capability is probed, not inferred from a flag

**Context.** Label alignment needs `word_ids()`; character spans need an
offset mapping. Both came only from the Rust-backed tokenizers, and `is_fast`
was the proxy. transformers 5 removed the pure-Python implementations, so every
tokenizer now reports `is_fast=True`. The guard could no longer fail, the test
that proved it could no longer construct a slow tokenizer, and the project had
pinned `transformers<5` to keep both.

**Decision.** `tokenization.supports_word_alignment` tokenises a two-word
probe and asks for `word_ids()` and the offset mapping. If either raises, the
guard raises with a message naming what is missing. The tests feed it an object
whose flag lies.

**Why.** A version pin is a statement about a library; a probe is a statement
about the requirement. The requirement did not change between 4.x and 5.x and
the probe gives the same answer on both, which is what let the pin go.

**Cost.** One extra tokeniser call at load time, measured in microseconds.

## ADR-019 — The serving image ships transformers 5; the measurements stay on 4

**Context.** The first dependency audit found nine published advisories
against 4.57.6, every fix on the 5.x line. Every number in
[RESULTS.md](RESULTS.md) was produced on 4.57.6.

**Decision.** `requirements-serve.txt` pins `transformers>=5.10,<6`. The
`train` and `serve` extras in `pyproject.toml` allow `>=4.40,<6`. CI tests the
suite on both lines.

**Why not just move everything?** Because re-running a forty-minute sweep to
reproduce numbers that would not change is not evidence of anything, and
claiming the old numbers under a new pin without saying so would be worse. The
honest shape is: the image runs the line with no advisories; the record says
which line produced each measurement; the thing that actually crosses from
one to the other — the tokenizer — was checked to be identical, byte for byte,
on 940 test sentences and 3,000 abstracts ([SECURITY.md](SECURITY.md#why-the-image-is-on-transformers-5)).

**Why not argue the advisories away?** Each one is in a code path the
container never runs, and that argument is in SECURITY.md. But a permanently
red `--strict` audit trains everyone to ignore it, and the next advisory that
*does* matter would land in the same red list. A clean audit is worth a pin.

**Cost.** Two lines to keep working. The export needed a 5.x-specific path
(ADR-021); nothing else did.

## ADR-020 — The parity check writes the hash; the engine checks it

**Context.** Nothing verified that the `model.onnx` being served was the one
the parity check had compared against PyTorch. A truncated copy or a partial
mount would have surfaced as strange predictions, which is the most expensive
possible way to find out.

**Decision.** `check_parity` writes the graph's SHA-256 into `parity.json`,
only after the check passes. `NerEngine` hashes the file before opening it and
refuses a mismatch with both hashes in the message. `/health` reports the hash
and whether it was verified.

**Why there and not in a deploy script.** The parity check is the one moment
the graph is known to be right, so that is where the claim should be written.
A hash computed anywhere else would be a hash of whatever happened to be on
disk.

**Why a missing record is a warning, not a refusal.** An older export, or a
graph that arrived some other way, is a weaker claim rather than a
contradiction. It loads, logs, and reports `integrity_verified: false`.
Refusing it would make the field redundant.

**Cost.** About a second at startup to stream 436 MB through SHA-256.

## ADR-021 — On transformers 5 the export builds its own attention mask

**Context.** transformers 5's mask utilities read the shape of a position
tensor that is a scalar under TorchScript tracing; `torch.onnx.export` dies
with `IndexError` inside `sdpa_mask`. The dynamo exporter is not available on
the pinned torch in a form that handles external data for small graphs, and
upgrading torch to chase an exporter is a larger change than the problem.

**Decision.** When the library's major version is 5 or above, the export
wrapper converts the 2-D padding mask into the prepared 4-D additive mask
BERT uses internally, which the utilities return untouched. 4.x is unchanged.

**Why this is safe.** The graph is compared against the PyTorch forward
afterwards exactly as before, and the 5.x export of the shipped checkpoint
matches the 4.x export to 4.6e-6 on every real token. The mask the wrapper
builds is the mask the model would have built.

**Cost.** A version branch in the exporter, and the obligation to re-check it
when transformers 6 arrives.

## ADR-022 — The load test is closed-loop, end to end, and reported apart from the benchmark

**Context.** `benchmark.py` times the model call in-process with one request
in flight. It is the right instrument for comparing backends and says nothing
about what a caller sees under load.

**Decision.** `loadtest.py` drives the running service over HTTP with `c`
concurrent callers, each waiting for its response before sending the next,
for a fixed request budget per level. It reports client-observed percentiles
and throughput at each level and refuses a degraded service.

**Why closed-loop.** It models a fixed pool of upstream callers, which is what
most consumers of an internal extraction service are, and it cannot overload
the server into a queueing collapse that would measure the queue rather than
the service. An open-loop arrival-rate test answers a different question.

**Why the payload is abstracts, not the parity sentences.** The parity
sentences are short. A caller sends abstracts. Timing the easy input would
flatter every number.

**Why the two tables are never merged.** The benchmark isolates the backend;
the load test includes everything from the socket to the JSON. Putting them
side by side invites reading one as the other.

## ADR-023 — `score` stays uncalibrated, and the miscalibration is published

**Context.** `score` is a mean softmax probability over the span's tokens.
`min_score` invited callers to read it as a probability of correctness. Nothing
had measured whether it was one.

**Decision.** Measure it (`calibration.py`), publish the reliability bins and
what each `min_score` threshold buys in precision and recall, and say in the
API documentation that the score orders predictions rather than estimating
their probability. Do not fit a calibrator.

**Why not fit one.** Fitting isotonic or temperature scaling on the validation
split would spend the only held-out data the sweep selection has not already
touched, and would make the served score depend on a second artefact that
could drift from the model it was fitted to. The measured curve is monotone,
so the ordering is sound; a caller who needs a probability can apply the
published table to their own threshold.

**Consequence.** The limitation is a number now — an expected calibration
error on the validation split — rather than a sentence.
