# PROJECT

## What it is

MedExtract takes free biomedical or clinical prose and returns the diseases and
chemicals in it, each tagged with its type, its character offsets in the
submitted text, and a confidence score. A second endpoint takes a question and
returns the most relevant PubMed abstracts from a local vector index, with the
same tagging applied to each hit.

It is a deliberately small but complete version of the extraction layer a
clinical NLP product needs: a fine-tuned model, an optimised runtime, an API, a
vector index, and the CI to keep them working.

## Why it exists

Biomedical text is dense with entities that general-domain NLP handles badly.
"Familial adenomatous polyposis" is one disease, not three words; `APC` is a
gene, not a typo; "no evidence of carcinoma" contains a disease mention that a
naive pipeline will report as a finding. Getting the entity layer right is the
precondition for anything downstream — cohort selection, literature triage,
coding support — and it is the layer where the failure modes are subtlest.

The project exists to demonstrate that layer end to end, with every claim
measured rather than asserted. That constraint drives most of the design: the
result tables in `README.md` are written by scripts, the ONNX export refuses to
ship a graph that disagrees with PyTorch, and the sweep runs nine arms so the
reported number is a choice out of a table rather than one lucky run.

## Scope

**In scope**

- Disease and chemical entity extraction with character offsets and scores.
- A reproducible fine-tuning sweep with tracked hyperparameters and results.
- An ONNX runtime path with a verified parity guarantee and measured latency.
- A REST API with health reporting, request tracing, and batch support.
- Semantic retrieval over a few thousand PubMed abstracts.
- A container that runs the whole serving path without PyTorch, and CI that
  enforces its size budget.

**Explicitly out of scope**

- Gene entities. Neither configured corpus annotates them.
- Clinical notes. Both corpora are edited academic abstracts.
- Negation and assertion detection.
- Entity linking to MeSH, UMLS or any ontology.
- Relation extraction.
- Answer generation. `/search` returns passages and what was found in them;
  nothing writes prose. Calling it RAG would claim a generation step that does
  not exist.

See [PRD.md](PRD.md) for the reasoning behind these boundaries and
[../README.md#limitations](../README.md#limitations) for what they cost.

## Status

The pipeline is complete, tested and has been run end to end on real data.
**Every measurement is in [RESULTS.md](RESULTS.md)**; the narrative of the run
that produced them is in [PROGRESS.md](PROGRESS.md).

| Area | State |
|---|---|
| Training, alignment, sweep, MLflow | Run; nine arms recorded |
| ONNX export and parity | Run; parity enforced |
| Latency benchmark | Measured on this machine's CPU |
| API and schemas | Complete, tested |
| Retrieval and indexing | Run against a live Qdrant |
| Container and CI | Built and size-checked |

## Who it is for

- **A reviewer** asking whether the numbers are real and how they were
  produced. Start at [../README.md](../README.md), then [DECISIONS.md](DECISIONS.md).
- **An engineer** extending it. Start at [ARCHITECTURE.md](ARCHITECTURE.md),
  then [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md).
- **An operator** running it. Start at [DEPLOYMENT.md](DEPLOYMENT.md) and
  [ENVIRONMENT.md](ENVIRONMENT.md).

## Related documents

Everything else is indexed in [README.md](README.md).
