# Product Requirements

## Problem

Teams working with biomedical literature and clinical text need to know which
diseases and drugs a document mentions, and *where* — precisely enough to
highlight the span in a UI, feed it to a coder, or use it as a retrieval
filter. General-domain NER does this badly: biomedical vocabulary fragments in
ways general tokenizers were not trained for, multi-word disease names are
routinely split, and gene symbols look like noise.

Off-the-shelf options force an unattractive trade. Cloud clinical-NLP APIs mean
sending text to a third party, which is a non-starter for anything
patient-derived. Research code from the papers is usually a training script
with no serving story, no latency characterisation, and no way to tell whether
the model you deployed computes what the model you evaluated computed.

## Users and jobs

| User | Job to be done |
|---|---|
| Application engineer | Get entities with offsets from a text, over HTTP, fast enough to call per keystroke-batch, without hosting PyTorch |
| Data scientist | Compare fine-tuning configurations on a fixed benchmark and defend the choice |
| Literature analyst | Ask a question and get back relevant abstracts with the entities already marked |
| Platform operator | Deploy a small, non-root container, know when it is unhealthy and why |

## Success metrics

These are the numbers the project is judged on. Measured values are in
[RESULTS.md](RESULTS.md#1-headline).

| # | Metric | Target | Rationale |
|---|---|---|---|
| M1 | Entity F1 on NCBI-Disease test | ≥ 0.85 | Roughly in line with published baselines. A number well below it means something is broken, not that the task is hard. |
| M2 | Scoring method | Strict entity-level IOB2 | The convention the baselines use. Token accuracy is meaningless here — most tokens are `O`, so predicting nothing scores above 90%. |
| M3 | ONNX/PyTorch agreement | Identical decoded spans; max logit diff < 1e-4 | A latency win on a graph that computes something different is not a win. |
| M4 | Serving image size | < 1.5 GB | Enforced in CI, so a regression fails the PR rather than the deploy. |
| M5 | Configurations compared | ≥ 6, with seed repeats | So a 0.005 F1 gap can be read against run-to-run variance instead of over-interpreted. |
| M6 | Offsets | `text[start:end]` equals the returned surface form, including non-ASCII | Character offsets, not byte offsets. Asserted in the test suite. |
| M7 | Startup with no model | Container starts, `/health` returns 200 `degraded` with the reason | A crash-looping container cannot tell you why it is broken. |

## Requirements summary

Full numbered list with test traceability in [REQUIREMENTS.md](REQUIREMENTS.md);
endpoint-level behaviour in [SPEC.md](SPEC.md).

**Must have**

- Extract disease and chemical spans with type, character offsets, confidence.
- Batch extraction in one call.
- Semantic search over an indexed PubMed corpus, with entities tagged in hits.
- Health endpoint reporting model, backend, label set and retrieval state.
- Reproducible training with tracked hyperparameters and results.
- ONNX export gated on a parity check.
- Container that serves without PyTorch.

**Should have**

- Request tracing through an `X-Request-ID` that reaches library log lines.
- Windowed inference so entities past the model's context limit are still found.
- Confidence filtering per request.
- A one-command unattended pipeline from empty checkout to live service.

**Will not have**

| Not building | Why |
|---|---|
| Gene entities | Neither NCBI-Disease nor BC5CDR annotates them. Would need a third corpus, not a config change. |
| Negation / assertion | "No evidence of carcinoma" yields `carcinoma`. For clinical use this is the whole ballgame; doing it properly is its own project. |
| Entity linking | Spans are typed but not normalised, so "MI" and "myocardial infarction" stay unrelated strings. |
| Relation extraction | BC5CDR carries chemical-induced-disease relations; only the entity layer is used. |
| Answer generation | `/search` is retrieval plus extraction. Nothing generates text. |
| Full-PubMed recall | A few thousand abstracts is a demonstration, not a literature search over 35M records. |
| Clinical validation | Not a medical device. Nothing here is validated for decisions about anyone's care. |

## Constraints

- **No third-party inference.** Everything runs locally or in the operator's own
  container.
- **Training and serving are separate.** Serving must never require the training
  stack; that is what keeps the image small.
- **Measured or absent.** A table either holds a number produced by a script on
  this hardware or it says so. No estimates presented as results.

## Open questions

Tracked in [TODO.md](TODO.md). The substantive ones: whether a combined
NCBI-Disease + BC5CDR model beats two specialised ones, and what the true
latency profile looks like under concurrency rather than the serial benchmark
currently reported.
