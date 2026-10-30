# TODO

Remaining work, prioritised. Completed work is in [TASKS.md](TASKS.md); fixed
defects are in [BUGS.md](BUGS.md).

Nothing here blocks the project being complete — the pipeline runs end to end
and every documented number is measured. These are the things a next iteration
should pick up.

## Priority 1 — gaps in what is currently claimed

### Measure latency under concurrency

`benchmark.py` times serial calls with tokenisation hoisted out of the loop.
That is the right measurement for comparing backends, and the wrong one for
answering "what will a caller see under load?" A p99 measured with one request
in flight says nothing about a p99 with sixteen.

Needs a small concurrent harness — `httpx` with an async client against the
running container, ramping concurrency, reporting throughput and latency at
each level. Report it separately from the backend comparison; do not conflate
the two.

### Add dependency vulnerability scanning to CI

`pip-audit` or Trivy in the `build` job, failing on high severity. Listed as a
known gap in [SECURITY.md](SECURITY.md#known-gaps). Cheap to add and currently
the most obvious hole in an otherwise careful build.

### Pin base image by digest

`python:3.11-slim-bookworm` is a moving tag. Pinning the digest makes builds
reproducible and makes a base-image change a visible commit rather than a
silent one.

## Priority 2 — worth doing next

### Train on BC5CDR and compare

BC5CDR is registered in `config.DATASETS` with a working loader and a verified
label ordering, but no arm has been run against it. Two open questions:

- Does a combined disease + chemical model match two specialised ones?
- Does the extra chemical supervision help or hurt disease F1?

This is a `--dataset bc5cdr` sweep and a table, not new code.

### Report per-type metrics in the sweep table

`metrics.entity_scores` already computes per-type precision, recall and F1 and
logs them to MLflow. `report.py` only surfaces the micro figures. On a
single-type corpus that is the same number, but the moment BC5CDR is swept it
stops being.

### Calibrate confidence, or stop implying it is calibrated

`score` is a mean softmax value. It orders predictions usefully; it is not a
probability of correctness, and `min_score` invites callers to treat it as one.
Either fit a calibration on the validation split and report reliability, or say
plainly in the API docs that it is an ordering signal only.

### Artefact integrity check at load

Nothing verifies that `model.onnx` is the graph the parity check passed. A
hash written into `parity.json` at export and checked at startup would close
it, and would surface a partial mount or a truncated copy immediately rather
than as strange predictions.

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
