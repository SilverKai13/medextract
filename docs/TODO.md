# TODO

Remaining work, prioritised. Completed work is in [TASKS.md](TASKS.md); fixed
defects are in [BUGS.md](BUGS.md).

Nothing here blocks the project being complete. The first pass ran the
pipeline end to end; the second pass measured latency under concurrency and
score reliability, answered the BC5CDR question, added dependency and image
scanning, pinned the base by digest, and made the served graph verify itself.
What follows is what a third iteration would pick up.

## Priority 1 — gaps in what is currently claimed

### Hosted deployment

Everything runs on one machine behind `localhost`. The compose file, the
image budget and the smoke test are all there; what is missing is a target.
A single small VM or a managed container service with TLS termination and
an API key at the ingress would turn "local deployment" into a URL. The
README says plainly there is no public URL; this is the item that changes
that sentence.

### Authentication and rate limiting

Deliberately out of the application and at the ingress ([SECURITY.md](SECURITY.md)),
which means that until there is an ingress there is nothing. Goes with the
item above.

### Integrity for the query encoder

Only the NER graph carries a hash. `artifacts/embedder/model.onnx` is trusted
as mounted. The same record-at-export, check-at-load pattern applies; it was
not done because the embedder's export path is a different script and the
retrieval index would need a matching provenance note to be worth it.

## Priority 2 — worth doing next

### Fit a calibrator, or do not, deliberately

The score's miscalibration is measured ([calibration.md](calibration.md)) and
the decision not to fit one is recorded ([ADR-023](DECISIONS.md#adr-023--score-stays-uncalibrated-and-the-miscalibration-is-published)).
If a downstream consumer needs a probability, isotonic regression on a held-out
split that the sweep never selected on — which means carving one — is the
route, and the reliability table here is the baseline to beat.

### Thread limits and workers

The load test shows one request saturating all sixteen threads and every
further caller queueing ([RESULTS.md §4.3](RESULTS.md#43-latency-under-concurrency)).
`NerEngine` already takes `intra_op_threads`; the experiment is several
uvicorn workers each pinned to a few threads, against the one-process
default, on the same harness. It may trade single-request latency for
throughput, or it may not; it is a measurement, not a decision.

### Open-loop load test

`loadtest.py` is closed-loop by design ([ADR-022](DECISIONS.md#adr-022--the-load-test-is-closed-loop-end-to-end-and-reported-apart-from-the-benchmark)):
it answers "what do N waiting callers see". An arrival-rate test answers "at
what request rate does the queue grow without bound", which a capacity plan
also wants. Different harness, same reporting.

### Serve the BC5CDR model by default?

It emits two types and its PubMedBERT arm is the best-supported result in the
project. The default stays NCBI-Disease because that is the corpus the target
was set on. Switching the default is a one-line change and a README rewrite;
the question is whether the headline should be the brief's number or the
more useful model.

## Priority 3 — larger scope

### Negation and assertion detection

"No evidence of carcinoma" returns `carcinoma`. For clinical use this is the
single most consequential gap, and it is the one most likely to cause harm if a
downstream consumer treats extracted entities as asserted findings. Currently
documented as out of scope in [PRD.md](PRD.md); doing it properly needs an
assertion corpus and is its own project.

### Entity linking

Normalising spans to MeSH or UMLS so "MI", "myocardial infarction" and "heart
attack" resolve to one concept. Without it, any cross-document aggregation
undercounts.

### Gene entities

Neither configured corpus annotates genes. Would need a third corpus (JNLPBA or
BC2GM), not a config change.

### Evaluate on clinical notes

Both corpora are edited academic abstracts. Performance on real notes —
abbreviated, ungrammatical, full of local shorthand — would be substantially
worse and nothing here measures it. Even a small annotated sample would turn an
acknowledged unknown into a number.

## Deliberately not doing

Reasoning in [PRD.md](PRD.md) and [DECISIONS.md](DECISIONS.md).

| | Why |
|---|---|
| Answer generation | `/search` is retrieval plus extraction. Adding generation adds the one component whose output cannot be checked against a ground truth here |
| Full-PubMed scale | A few thousand abstracts is a demonstration, not a literature search over 35M records |
| Built-in auth / rate limiting | Belongs at the ingress, not in the application |
| Async workers or a queue | Extraction is a synchronous forward pass of a few milliseconds; a broker would be more moving parts than the latency justifies |
| Clinical validation | Not a medical device |
| Re-running the NCBI sweep on transformers 5 | The numbers would not change and the record says which line produced them ([ADR-019](DECISIONS.md#adr-019--the-serving-image-ships-transformers-5-the-measurements-stay-on-4)) |
