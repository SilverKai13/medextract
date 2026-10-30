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
