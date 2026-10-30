# Security

## Scope and posture

MedExtract is a stateless inference service. It holds no user accounts, no
credentials, no session state, and no persistent record of what it was asked.
That removes most of the usual attack surface and concentrates what remains in
three places: what you send it, what it sends outward, and what ships in the
container.

> **Not a medical device.** Nothing here is validated for clinical use and it
> must not be used to make decisions about anyone's care. See
> [Clinical safety](#clinical-safety).

## Threat model

| # | Threat | Exposure | Mitigation |
|---|---|---|---|
| T1 | Resource exhaustion via oversized input | Anyone who can reach `/extract` | 20,000-char cap → `413`; 64-text cap on batch; windowing is bounded by input length |
| T2 | Malformed input causing a crash | Same | Pydantic validation → `422`; a catch-all middleware turns any unhandled exception into a `500` with a request id, never a stack trace |
| T3 | Sensitive text leaking into logs | Operators, log aggregation | Request/response **bodies are never logged**. Only path, method, status, duration, character count and entity count |
| T4 | Data egress to third parties at inference | Anyone submitting text | None: extraction and search are entirely local. No outbound call is made while serving |
| T5 | Data egress during corpus fetch | Operator-initiated only | `retrieval/pubmed.py` contacts NCBI E-utilities and sends the configured contact email. Offline path only, never in the serving container |
| T6 | Supply-chain risk in the image | Deploy | Narrow dependency set with no compilers, build caches or PyTorch; base image pinned by digest; `pip-audit --strict` on what the image installs and Trivy on the built image, both in CI |
| T7 | Container escape / privilege abuse | Deploy | Non-root `uid 10001`; no capabilities added; artefacts mounted read-only |
| T8 | Model artefact tampering | Anyone who can write `artifacts/` | Mounted `:ro`, and the graph's SHA-256 is checked at load against the hash recorded when it passed the parity check. A mismatch refuses the load; `/health` reports the hash and whether it was verified |
| T9 | Unauthenticated access | Network reachability | **None built in** — see below |

## Authentication and authorisation

**There is none, by design.** No API keys, no tokens, no user model. The
service is intended to sit behind an ingress that terminates TLS and enforces
whatever policy the deployment requires.

**Consequence:** binding it to a public interface exposes it to anyone. The
compose file publishes `8000` on the host, which is right for local
development and wrong for anything else. In a real deployment, put it behind a
gateway and do not publish the port.

Rate limiting is the same story: the input caps bound the cost of a *single*
request but nothing bounds request *volume*.

## Handling sensitive text

The API accepts free clinical prose, which in a real deployment may be
patient-derived.

**What the service does with it:**

- Holds it in memory for the duration of the request.
- Echoes it back in `ExtractResponse.text` — so it is in the response body, and
  therefore in anything that caches or logs response bodies.
- Writes **nothing** to disk.
- Sends **nothing** anywhere.

**What it logs:** `path`, `method`, `status`, `took_ms`, `request_id`, `chars`,
`entities`. Character and entity *counts*, never content. If you extend the
logging, keep it that way.

**What it cannot do for you:** de-identification. There is no PHI detection
here. Text arriving with identifiers in it leaves with them in the echoed
`text` field.

## Outbound network

| When | To | What is sent |
|---|---|---|
| Serving | Nothing | — |
| `make fetch` | `eutils.ncbi.nlm.nih.gov` | Query terms, `tool=medextract`, contact email, optional API key |
| Training / export | `huggingface.co` | Model and dataset downloads |
| Corpus fallback | `github.com` | The NCBI-Disease CoNLL files |

The serving container sets `HF_HUB_OFFLINE=1`, so even an accidental Hub call
in the serving path fails rather than reaching out.

**The contact email is deliberate.** NCBI asks for one on E-utilities traffic
and publishes rate limits (3 req/s without a key, 10 with). Both are respected
in `PubMedClient` — getting a project's IP blocked from E-utilities is slow to
undo. It is a real address sent to a third party; it lives in `.env`, which is
gitignored.

## Secrets

| Secret | Where | Notes |
|---|---|---|
| `MEDEXTRACT_NCBI_API_KEY` | `.env` | Optional; only raises the E-utilities rate limit |
| `MEDEXTRACT_NCBI_EMAIL` | `.env` | Contact address, not a credential, but still personal data |
| `GITHUB_TOKEN` | CI | Provided by Actions for GHCR publish; never checked in |

`.env` is in `.gitignore`; `.env.example` carries empty placeholders. Nothing
else in the project needs a secret, and no secret is baked into an image.

## Container hardening

From the `Dockerfile`:

- **Multi-stage build.** `build-essential` and pip's caches exist only in the
  builder. The runtime stage copies the finished virtualenv and nothing else.
- **Non-root.** `useradd --uid 10001 medextract`, `USER medextract`. The app
  cannot write outside what it is given.
- **Read-only artefacts.** Compose mounts `./artifacts:/app/artifacts:ro`.
- **Narrow runtime deps.** `requirements-serve.txt` excludes torch, datasets,
  mlflow and the training stack. Fewer packages is less to patch.
- **`curl` is the only added OS package**, and only for the `HEALTHCHECK`.
- **Real healthcheck.** Hits `/health` rather than probing the port, so a
  process that is up but broken is reported as such.

- **Base image pinned by digest.** `ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:…`
  in the Dockerfile, so a change to the base is a visible commit rather than
  a silent rebuild. Refreshing it is a deliberate act: inspect the tag,
  paste the new index digest, commit.
- **Scanned.** CI runs Trivy over the built image — OS packages and Python
  packages — and fails on HIGH or CRITICAL findings that have a fix.
  Unfixed findings are reported, because nothing can be done about them
  except know.

Not done, and worth doing in a real deployment: a read-only root filesystem,
`--cap-drop ALL`, and a seccomp profile.

## Dependency posture

`requirements-serve.txt` pins compatible ranges rather than exact versions.
Two scanners run in CI on every push:

| Scanner | Scope | Policy |
|---|---|---|
| `pip-audit --strict` | What `requirements-serve.txt` resolves to — the Python packages the image installs | Any known vulnerability fails the build |
| Trivy | The built image: Debian packages and Python packages | HIGH/CRITICAL with a fix available fails the build; unfixed are reported |
| `pip-audit` (advisory) | The `train`, `serve` and `dev` extras from `pyproject.toml` | Reported, never blocking — see below |

### Why the image is on transformers 5

The first audit, run on the `transformers>=4.40,<5` pin the project shipped
with, found nine advisories against 4.57.6, the last 4.x release:

| Advisory | What it is | Fixed in |
|---|---|---|
| PYSEC-2025-217 | Deserialisation in the X-CLIP checkpoint conversion script | no 4.x fix |
| PYSEC-2026-2288 | `Trainer._load_rng_state` calls `torch.load` without `weights_only` | 5.0.0 |
| PYSEC-2026-2289 | `config.json` can point `_attn_implementation_internal` at a Hub repo and load code from it | 5.3.0 |
| PYSEC-2026-2290 | `trust_remote_code` overridden by serialised config in the LightGlue loading path | 5.5.0 |
| PYSEC-2026-3929 | Path traversal through `chat_template` keys in `save_pretrained` | 5.10.0 |
| PYSEC-2026-4174 | `load_custom_generate` fetches remote code before the consent check | after 5.8.1 |

Not one of those code paths runs in the container: it loads a local
tokenizer with `HF_HUB_OFFLINE=1`, never calls `from_pretrained` on a model,
never saves anything, and has no `Trainer`. Arguing each one away would
have been defensible. It would also have left a `--strict` audit permanently
red and a reviewer with a list to re-check every time a new advisory
landed. The 5.x line has none of them, so the better question was whether
the 5.x tokenizer is the same tokenizer — and that is checkable. Input ids
and offset mappings were compared on every sentence of the NCBI-Disease test
split and every indexed abstract, for the NER tokenizer and the query
encoder's tokenizer, across 4.57.6 and 5.8.1: byte-identical. The pin moved
to `>=5.10,<6`, the first release with every fix, and the audit is clean.

The training stack still allows 4.x, because that is what every measurement
in [RESULTS.md](RESULTS.md) was produced on. It runs offline on a developer's
machine against local checkpoints and local corpus files, so the advisories
above are accepted there, and the advisory audit in CI exists so that
acceptance is visible rather than forgotten. Both lines are tested on every
push.

## Input validation summary

| Endpoint | Bound | Response |
|---|---|---|
| `/extract` | `text` ≤ 20,000 chars, non-empty | `413` / `422` |
| `/extract` | `0.0 ≤ min_score ≤ 1.0` | `422` |
| `/extract/batch` | 1–64 texts, each ≤ 20,000 chars | `422` / `413` |
| `/search` | non-empty query, `1 ≤ top_k ≤ 50` | `422` |

All enforced declaratively in the Pydantic models, so the OpenAPI document and
the runtime checks are the same definitions.

## Clinical safety

Beyond information security, the failure modes that matter clinically:

- **No negation detection.** "No evidence of carcinoma" returns `carcinoma` as
  a disease entity. Any downstream use that treats extracted entities as
  *asserted findings* will be wrong, and wrong in the dangerous direction.
- **No entity linking.** "MI", "myocardial infarction" and "heart attack" are
  three unrelated strings. Counting entities across documents will undercount.
- **Trained on abstracts, not notes.** Both corpora are edited academic prose.
  Performance on real clinical text — abbreviated, ungrammatical, full of local
  shorthand — would be substantially worse, and nothing here measures that.
- **Confidence is not calibrated, and that is measured.** `score` is a mean
  softmax value. On the validation split it is overconfident across every
  bin — a mean score of 0.97 against a precision of 0.86 — so `min_score`
  is a ranking cut-off, not a probability. [calibration.md](calibration.md)
  has the bins and what each threshold actually buys.

These are scope decisions, documented in [PRD.md](PRD.md), not defects.

## Known gaps

Honest list of what a production deployment would still need:

1. Authentication, authorisation and rate limiting at the ingress.
2. TLS — the service speaks plain HTTP and expects termination upstream.
3. Audit logging, if the deployment needs to answer "who asked what, when".
4. Read-only root filesystem and dropped capabilities.
5. Integrity of the *query encoder* graph. Only the NER graph carries a
   hash; `artifacts/embedder/model.onnx` is trusted as mounted.

Closed since the first release: dependency scanning (both scanners in CI),
base-image digest pinning, and NER artefact integrity at load.

## Reporting

This is a demonstration project with no production deployment and no security
contact. Anything found should be raised as a GitHub issue.
