# API Specification

Behavioural contract for every endpoint. Shapes come from
`src/medextract/api/schemas.py`, which is also what generates the OpenAPI
document at `/docs` — there is no second, hand-maintained description that can
drift.

Base URL in the default compose deployment: `http://localhost:8000`.

## Conventions

**Request id.** Every response carries `X-Request-ID`. Send your own to
correlate across systems; otherwise a 16-character hex id is generated. The id
travels in a `ContextVar`, so it appears on log lines emitted by library code
that knows nothing about the web layer.

**Offsets.** `start` and `end` are Python character offsets into the exact
string submitted. `text[start:end] == entity.text` holds, including for
non-ASCII input. They are not UTF-8 byte offsets.

**Scores.** `score` is the mean softmax probability across the tokens making up
the span, in `[0, 1]`, rounded to four decimals.

**Errors.** `422` comes from Pydantic validation and carries its standard
detail structure. `413`, `503` and `500` carry `{"detail": "..."}`. The `500`
handler also includes `request_id`.

---

## `POST /extract`

Extract entities from one text.

**Request**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `text` | string | yes | `min_length=1`, at most `MEDEXTRACT_MAX_INPUT_CHARS` (default 20,000) |
| `min_score` | float | no | `0.0 ≤ x ≤ 1.0`, default `0.0` |

```json
{"text": "Familial adenomatous polyposis is caused by germline mutations in the APC gene."}
```

**Response `200`**

```json
{
  "entities": [
    {"text": "Familial adenomatous polyposis", "label": "Disease",
     "start": 0, "end": 30, "score": 0.9871}
  ],
  "text": "Familial adenomatous polyposis is caused by germline mutations in the APC gene.",
  "model_version": "ncbi-disease-v1",
  "took_ms": 8.4
}
```

`text` is echoed so a caller can confirm the offsets refer to the string it
thinks they do without keeping its own copy. `took_ms` times inference only,
not transport.

Entities are returned in document order, deduplicated across overlapping
windows keeping the highest score for any repeated span.

**Errors**

| Status | Cause |
|---|---|
| `413` | `text` longer than the configured limit. Detail names the actual length and the limit. |
| `422` | `text` missing, empty, not a string, or `min_score` out of range; malformed JSON. |
| `503` | No model loaded. Detail carries the underlying load error. |

**Behaviour worth knowing**

- Long inputs are windowed with overlap (`max_sequence_length` 256, `stride`
  32), not truncated. An entity at character 8,000 is found.
- An `I-` tag with no preceding `B-` *does* open a span here. Evaluation
  discards it; serving recovers it. This disagreement is deliberate — see
  [DECISIONS.md](DECISIONS.md#adr-004).

---

## `POST /extract/batch`

Same extraction, several texts, one model call.

**Request**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `texts` | string[] | yes | 1–64 items |
| `min_score` | float | no | `0.0 ≤ x ≤ 1.0`, default `0.0` |

**Response `200`**

```json
{
  "results": [[{"text": "...", "label": "Disease", "start": 0, "end": 30, "score": 0.98}], []],
  "model_version": "ncbi-disease-v1",
  "took_ms": 41.2
}
```

`results[i]` corresponds to `texts[i]` and is identical to what `/extract`
would return for that text alone. Asserted by
`test_api.py::test_batch_matches_single_calls`.

**Errors**: `413` if any single text exceeds the limit; `422` for an empty list
or more than 64 items; `503` with no model.

---

## `POST /search`

Rank indexed PubMed abstracts against a query, then tag entities in the hits.

**This is retrieval plus extraction, not RAG.** Nothing generates an answer. It
returns passages and what was found in them.

**Request**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query` | string | yes | `min_length=1` |
| `top_k` | int | no | `1 ≤ k ≤ 50`, default `5` |
| `extract_entities` | bool | no | default `true` |

```json
{"query": "targeted therapy for EGFR mutant lung cancer", "top_k": 5}
```

**Response `200`**

```json
{
  "query": "targeted therapy for EGFR mutant lung cancer",
  "query_entities": [{"text": "lung cancer", "label": "Disease", "start": 32, "end": 43, "score": 0.99}],
  "hits": [
    {
      "pmid": "12345678",
      "title": "...",
      "abstract": "...",
      "journal": "Journal of Clinical Oncology",
      "year": 2021,
      "score": 0.8123,
      "entities": [{"text": "...", "label": "Disease", "start": 10, "end": 21, "score": 0.97}]
    }
  ],
  "took_ms": 128.5
}
```

`score` is cosine similarity. Vectors are L2-normalised at encode time and the
collection uses cosine distance, so scores are directly comparable across
queries. Hit `entities` offsets index that hit's `abstract`, not the query.

`extract_entities: false` skips the per-hit model calls and returns `entities`
as `[]`. `query_entities` is always computed — it is one short text.

**Errors**

| Status | Cause |
|---|---|
| `422` | Empty query, or `top_k` outside 1–50. |
| `503` | Retrieval unavailable — Qdrant unreachable or the collection was never built. Detail says which, and names `medextract-index`. |
| `503` | No model loaded (entity tagging cannot run). |

`top_k` larger than the corpus is not an error; you get what exists.

---

## `GET /health`

Liveness plus what is actually loaded.

**Response `200`**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "ncbi-disease-v1",
  "backend": "onnx",
  "labels": ["O", "B-Disease", "I-Disease"],
  "retrieval_available": true,
  "uptime_seconds": 1284.31,
  "version": "0.1.0"
}
```

| Field | Meaning |
|---|---|
| `status` | `ok` when a model is loaded, `degraded` otherwise |
| `labels` | The model's own label set, ordered by id — what it can actually emit |
| `retrieval_available` | Whether `/search` will work |

**Returns `200` even when degraded.** The container is running and can say why
it is not useful, which is more diagnosable than a dead endpoint or a crash
loop. The Docker `HEALTHCHECK` hits this path rather than probing the port, so
a process that is up with no model is reported as running-but-degraded.

---

## `GET /docs`, `GET /openapi.json`

Generated OpenAPI documentation. Every endpoint above appears with the request
and response models the handlers are validated against.

---

## Not provided

| Not there | Why |
|---|---|
| Authentication | Deployment concern; put it in the ingress. Nothing here handles credentials. See [SECURITY.md](SECURITY.md). |
| Rate limiting | Same. |
| Pagination on `/search` | `top_k` caps at 50; a demonstration corpus does not need deeper paging. |
| Streaming responses | Extraction is a single forward pass; there is nothing to stream. |
| Entity linking / normalisation | Out of scope, see [PRD.md](PRD.md). |
| A generation endpoint | Deliberately absent. |
