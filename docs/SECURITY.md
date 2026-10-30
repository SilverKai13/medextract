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
| T6 | Supply-chain risk in the image | Deploy | Runtime carries a deliberately narrow dependency set; no compilers, no build caches, no PyTorch |
| T7 | Container escape / privilege abuse | Deploy | Non-root `uid 10001`; no capabilities added; artefacts mounted read-only |
| T8 | Model artefact tampering | Anyone who can write `artifacts/` | Mounted `:ro`. Integrity of the mount is the deployer's responsibility — see [gaps](#known-gaps) |
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

Not done, and worth doing in a real deployment: a read-only root filesystem,
`--cap-drop ALL`, a seccomp profile, and pinned base-image digests rather than
the `python:3.11-slim-bookworm` tag.

## Dependency posture

`requirements-serve.txt` pins compatible ranges rather than exact versions,
with `transformers>=4.40,<5` bounded for a functional reason documented in
`pyproject.toml` — 5.x makes every tokenizer report `is_fast=True`, defeating a
guard the label alignment depends on.

CI runs lint and the full suite on every push and both Python 3.11 and 3.12
before the image is built.

**No automated vulnerability scanning is configured.** Adding `pip-audit` or
Trivy to the `build` job is the obvious next step; it is listed in
[TODO.md](TODO.md).

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
- **Confidence is not calibrated.** `score` is a mean softmax value. It orders
  predictions usefully; it is not a probability of correctness.

These are scope decisions, documented in [PRD.md](PRD.md), not defects.

## Known gaps

Honest list of what a production deployment would still need:

1. Authentication, authorisation and rate limiting at the ingress.
2. TLS — the service speaks plain HTTP and expects termination upstream.
3. Automated dependency vulnerability scanning in CI.
4. Base-image digest pinning.
5. Artefact integrity verification — nothing checks that `model.onnx` is the
   graph the parity check passed.
6. Audit logging, if the deployment needs to answer "who asked what, when".
7. Read-only root filesystem and dropped capabilities.

## Reporting

This is a demonstration project with no production deployment and no security
contact. Anything found should be raised as a GitHub issue.
