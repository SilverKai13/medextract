# Deployment

## The shape of a deployment

One stateless API container plus one Qdrant container. Model artefacts are
**mounted, not baked in**: they change on a different cadence to the code, and
putting a few hundred megabytes of weights into every image build makes each
one slow to push.

The consequence is that the image is useful without them. With no artefacts
present the service still starts, `/health` returns 200 with `status:
degraded` and the reason, and `/extract` returns 503 with the same detail.

## Local

```bash
docker compose up --build -d
curl localhost:8000/health
```

Compose brings up both services. `depends_on: qdrant: condition:
service_healthy` means the API does not start until Qdrant answers.

Without a model, expect `degraded`. To get a real one:

```powershell
.\make.ps1 sweep         # or: .\make.ps1 train
.\make.ps1 export        # writes artifacts/onnx and artifacts/embedder
.\make.ps1 fetch
.\make.ps1 index
.\make.ps1 up
```

Or all of it unattended: `.\make.ps1 all`.

### Without Docker

```powershell
.\make.ps1 serve
```

Set `MEDEXTRACT_QDRANT_URL` to a filesystem path and `build_client` opens an
embedded on-disk Qdrant instead of connecting to a server. `pipeline.py`
selects this automatically when `docker info` fails, so the one-command path
works on a machine with no daemon.

## The image

Multi-stage, `python:3.11-slim-bookworm`, pinned by digest:

```dockerfile
ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
FROM ${BASE_IMAGE} AS builder
```

A tag is a moving target: the same build on two days can produce two
different images with nothing in the repository to show why. With the
digest, a base change is a commit. To refresh it, inspect the tag and paste
the index digest:

```bash
docker buildx imagetools inspect python:3.11-slim-bookworm
```

**Build stage** has `build-essential` and pip's caches, installs
`requirements-serve.txt` into `/opt/venv`, then installs the package with
`--no-deps`. Dependencies are copied and installed *before* the source, so
editing code does not invalidate the dependency layer.

**Runtime stage** copies the finished virtualenv and nothing else. No
compilers, no caches, no `.pyc`, no vendored test directories.

```dockerfile
USER medextract          # uid 10001
HEALTHCHECK CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "medextract.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Why no PyTorch

The model runs as an ONNX graph. Torch is only needed to *produce* that graph,
which happens offline. Dropping it is the single largest saving in the image —
the torch wheel alone is larger than the whole image.

The consequence is that the *query encoder* had to be exported to ONNX too,
which is why `retrieval/export_embedder.py` exists: `sentence-transformers`
would have dragged torch straight back in.

`requirements-serve.txt` therefore excludes torch, datasets, mlflow and the
training stack. `transformers` is present only for its fast tokenizer, on
the 5.x line — [SECURITY.md](SECURITY.md#dependency-posture) explains why
the image and the training stack differ there, and what was checked before
they did.

### Size budget

**Under 1.5 GB**, enforced in CI so a regression fails the PR rather than the
deploy. The measured size appears in the workflow summary.

| | |
|---|---|
| Runtime dependencies, measured | 390 MB (clean venv from `requirements-serve.txt`, torch absent) |
| `python:3.11-slim-bookworm` base | ~150 MB |
| Expected image | ~560 MB |
| Actually built | **395 MB** — see [RESULTS.md §6](RESULTS.md#6-serving-container) |

The 390 MB is a real measurement, not an estimate: a fresh virtualenv built
from `requirements-serve.txt`, with the serving path then exercised inside it
and PyTorch confirmed absent. That last check is the one that matters — a
runtime dependency set that is 390 MB but incomplete would be worse than a
larger one that works.

### Environment in the image

```
MEDEXTRACT_MODEL_DIR=/app/artifacts/onnx
MEDEXTRACT_EMBEDDING_ONNX_DIR=/app/artifacts/embedder
MEDEXTRACT_BACKEND=onnx
TRANSFORMERS_NO_ADVISORY_WARNINGS=1
HF_HUB_OFFLINE=1
```

`HF_HUB_OFFLINE=1` means an accidental Hub call in the serving path fails
rather than reaching out. The serving container should never talk to the
internet.

## Compose

```yaml
volumes:
  - ./artifacts:/app/artifacts:ro     # read-only
```

Qdrant persists to a named volume `qdrant_storage`, so the index survives
`docker compose down`. Its healthcheck probes with bash's `/dev/tcp` rather
than curl, because the Qdrant image ships neither curl nor wget and adding a
package purely to health-check it is not worth the layer.

## CI/CD

`.github/workflows/ci.yml`, on every push and PR:

| Job | Runs |
|---|---|
| `lint` | `ruff check`, `ruff format --check` |
| `test` | pytest with coverage on Python 3.11 and 3.12, each against transformers 4.x and 5.x |
| `audit` | `pip-audit --strict` on `requirements-serve.txt`; an advisory audit of the training extras |
| `build` | Build image → enforce 1.5 GB budget → Trivy scan → smoke test |
| `publish` | **`v*` tags only** — push to GHCR |

The smoke test starts the container **with no model mounted** and asserts it
comes up and answers `/health`. It verifies the degraded path specifically,
because that is the behaviour that makes a broken deployment diagnosable.

Image names are lowercased explicitly (`tr '[:upper:]' '[:lower:]'`) because
GHCR requires lowercase repository names and `github.repository` preserves the
owner's actual case.

Branch builds stop at `build`. Only a tagged release publishes.

### Releasing

```bash
git tag v1.0.0
git push origin v1.0.0
```

`docker/metadata-action` tags the image `{version}`, `{major}.{minor}` and
`sha`.

## Operating it

### Health

`GET /health` is the single source of truth:

| Field | Watch for |
|---|---|
| `status` | `degraded` means no model loaded |
| `model_loaded` | |
| `retrieval_available` | `false` means `/search` returns 503 |
| `labels` | What the model can actually emit — catches a wrong model mounted |
| `model_sha256` | Which graph is loaded — catches a stale mount |
| `integrity_verified` | `false` means the graph arrived without a parity record; a *mismatch* never gets this far, it is a load failure |
| `uptime_seconds` | Restart detection |

### Logs

JSON, one object per line, on stdout. Anything logged *while a request is being
handled* carries its `request_id`, including lines from library code that knows
nothing about the web layer — the id travels in a `ContextVar` rather than being
threaded through call signatures.

uvicorn's own access line is the exception and shows `request_id: "-"`. It is
emitted after the ASGI app has returned, by which point the middleware has
already reset the ContextVar. The application's own `"message": "request"` line
carries the real id and the same status and duration, so nothing is lost —
correlate on that one.

Logged per request: `path`, `method`, `status`, `took_ms`, `request_id`, and
for extraction `chars` and `entities`. **Request and response bodies are never
logged** — see [SECURITY.md](SECURITY.md).

### Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `/health` `degraded` | No model at `MEDEXTRACT_MODEL_DIR` | Check the mount; `detail` names the error |
| `/health` `degraded`, `/extract` 503 with `does not match` | `model.onnx` differs from the hash in its `parity.json` — truncated copy, partial mount, swapped file | Re-run the export, or restore the artefact directory as a unit |
| `/extract` 503 | Same | Same |
| `/search` 503 | Qdrant down, or collection never built | `docker compose ps qdrant`, then `make index` |
| `/search` returns nothing sensible | Index built with hashing fallback vectors | Rebuild with `--recreate`; `allow_fallback` is off by default so this needs an explicit override |
| Container restarting | Something other than a load failure — loads are caught | `docker compose logs api` |

### Updating the model

Weights are not in the image, so no rebuild is needed:

```bash
.\make.ps1 export                  # writes artifacts/onnx
docker compose restart api         # reloads at startup
```

The model loads once in the lifespan handler. Loading per request would
dominate latency — a BioBERT-sized graph takes hundreds of milliseconds to
initialise against single-digit milliseconds to run.

### Changing the embedding model

The collection must be recreated. Old vectors are in a different space and
would rank nonsensically against new queries; the dimension may also differ.

```bash
.\make.ps1 export
.\.venv\Scripts\python -m medextract.retrieval.build_index --recreate
```

## Scaling

The API is stateless apart from a read-only mount, so replicas behind a load
balancer would work. Nothing here demonstrates replicas, but the single
container's behaviour under concurrent load is measured: `make loadtest`
ramps closed-loop concurrency from 1 to 16 against a running service and
reports client-observed percentiles and throughput at each level —
[RESULTS.md §4.3](RESULTS.md#43-latency-under-concurrency). That is the
number a capacity plan starts from.

Qdrant is a single node with a local volume. Its own clustering would be the
route beyond that.

## Not configured

Deliberately left to the deployment: TLS termination, authentication, rate
limiting, log shipping, metrics/tracing export, autoscaling, and a read-only
root filesystem with dropped capabilities. See
[SECURITY.md](SECURITY.md#known-gaps).
