# Environment

## Setting up from a clean checkout

### Windows

```powershell
.\make.ps1 venv          # creates .venv and installs everything
.\make.ps1 test          # ~35s, should be all green
```

`make.ps1 venv` creates the virtualenv with `--system-site-packages`. That is
deliberate: it reuses an already-installed CUDA build of PyTorch instead of
re-downloading roughly 2.5 GB of wheels. Packages installed *into* the venv
still take precedence over the system ones, which is how `transformers<5` gets
pinned for this project without disturbing anything else on the machine.

If there is no system PyTorch, install one first:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Linux / macOS

```bash
pip install -e ".[dev,serve,train]"
make test
```

### Configuration

```powershell
Copy-Item .env.example .env
# then set MEDEXTRACT_NCBI_EMAIL if you intend to fetch from PubMed
```

## Environment variables

All settings are read by `config.Settings` from the environment or `.env`, with
the `MEDEXTRACT_` prefix. Unknown `MEDEXTRACT_*` variables are ignored rather
than fatal.

### Model and runtime

| Variable | Default | Notes |
|---|---|---|
| `MEDEXTRACT_MODEL_DIR` | `artifacts/model` | Point at `artifacts/onnx` for the exported graph, or a training output dir to serve PyTorch |
| `MEDEXTRACT_BACKEND` | `onnx` | `onnx` or `torch`. The container only has `onnx` |
| `MEDEXTRACT_MODEL_VERSION` | `dev` | Free-form; echoed in every response |
| `MEDEXTRACT_LOG_LEVEL` | `INFO` | |

### Request limits

| Variable | Default | Notes |
|---|---|---|
| `MEDEXTRACT_MAX_INPUT_CHARS` | `20000` | Beyond this, `413` |
| `MEDEXTRACT_MAX_SEQUENCE_LENGTH` | `256` | Tokens per window |
| `MEDEXTRACT_STRIDE` | `32` | Window overlap, so entities on a boundary are not lost |

### Retrieval

| Variable | Default | Notes |
|---|---|---|
| `MEDEXTRACT_RETRIEVAL_ENABLED` | `true` | `false` makes `/search` return 503 and skips loading the encoder |
| `MEDEXTRACT_QDRANT_URL` | `http://localhost:6333` | An `http(s)://` URL, `:memory:`, or a filesystem path for embedded on-disk mode |
| `MEDEXTRACT_QDRANT_COLLECTION` | `pubmed_abstracts` | |
| `MEDEXTRACT_EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | |
| `MEDEXTRACT_EMBEDDING_ONNX_DIR` | `artifacts/embedder` | Preferred over sentence-transformers when present |

### E-utilities

| Variable | Default | Notes |
|---|---|---|
| `MEDEXTRACT_NCBI_EMAIL` | *(empty)* | NCBI asks for a contact address. Sent to a third party — see [SECURITY.md](SECURITY.md) |
| `MEDEXTRACT_NCBI_API_KEY` | *(empty)* | Optional; raises the rate limit from 3/s to 10/s |

### Not prefixed

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | Overrides the default `sqlite:///mlflow.db` |
| `HF_HOME` | HuggingFace cache location. `make.ps1` sets `C:\hf-cache` if unset — the checkpoints total over a gigabyte and the repository is in a synced folder |
| `USE_TF` | Set to `0` by `medextract/__init__.py`; `setdefault`, so still overridable |
| `PYTHONUTF8` | Set to `1` by `make.ps1`. Windows still defaults stdio to the ANSI codepage, which turns an em-dash in a generated table into a crash |
| `OMP_NUM_THREADS` | Recorded in benchmark output; set it to make CPU timings comparable |

## Reference machine

Every measurement in [PROGRESS.md](PROGRESS.md) and
[benchmark.md](benchmark.md) was produced here.

| | |
|---|---|
| OS | Windows 11 Home Single Language, 10.0.26200 |
| CPU | AMD Ryzen 7 5800H, 8 cores / 16 threads |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, **4 GB VRAM** |
| Driver / CUDA | 566.07 / 12.7 |
| Python | 3.13.5 |
| PyTorch | 2.6.0+cu124 |
| Docker | 20.10.11, compose v2.2.1 |

**The 4 GB matters.** It is why `max_per_device_batch` exists and why the
`bs32` arm runs as 16 × 2 accumulation rather than 32 on the device. See
[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md#9-batch-resolution-trainpy).

## Pinned versions and why

| Package | Constraint | Reason |
|---|---|---|
| `transformers` | `>=4.40,<5` | 5.x makes every tokenizer report `is_fast=True`, defeating the guard in `data.py:81` that label alignment depends on |
| `datasets` | `>=2.18` | Any modern version works — the corpus loader no longer depends on script execution |
| `mlflow` | `>=2.12` | 3.x required for the SQLite backend default; the file store raises in 3.x |
| `torch` | `>=2.2` | Training and export only. Never in the serving image |
| `onnxruntime` | `>=1.17` | The serving runtime |

## Known environment hazards

Each of these bit during setup on this machine. Full write-ups in
[BUGS.md](BUGS.md).

**Global `transformers` 5.8.1.** Present system-wide and incompatible. The venv
shadows it; do not run the project with the system Python.

**pyarrow / torch DLL order.** Importing pyarrow after torch and transformers
have initialised kills the process with an access violation — a segfault, exit
code 139, not a catchable exception. `medextract/__init__.py` preloads pyarrow
to fix the order. Symptom: `Windows fatal exception: access violation` inside
`pyarrow/__init__.py`.

**TensorFlow + Keras 3.** A TensorFlow install anywhere on the machine makes
`transformers.Trainer` fail at import, because its integrations reach for
`TFPreTrainedModel`. `USE_TF=0` is set in `medextract/__init__.py`. Symptom:
`ValueError: Your currently installed version of Keras is Keras 3`.

**cp1252 stdio.** Reading a UTF-8 file with `Path.read_text()` and no explicit
encoding fails on Windows. Symptom:
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x90`. Always pass
`encoding="utf-8"`.

**OneDrive.** The repository lives in a synced folder. `HF_HOME` points
outside it, checkpoints are pruned after each arm, and MLflow uses a single
SQLite file rather than a tree of thousands of tiny ones. Sweep output is still
about 3.7 GB — see [DATABASE.md](DATABASE.md#filesystem-artefacts).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'medextract'` | Package not installed | `.\make.ps1 venv` |
| `Windows fatal exception: access violation` | pyarrow/torch load order | Ensure `import medextract` happens first |
| `Your currently installed version of Keras is Keras 3` | TF on the machine | `USE_TF=0` — set automatically; check it is not overridden |
| `UnicodeDecodeError: 'charmap' codec` | Missing `encoding="utf-8"` | Add it; set `PYTHONUTF8=1` |
| `filesystem tracking backend ... maintenance mode` | MLflow 3 rejecting `./mlruns` | Use `config.mlflow_tracking_uri()`, which returns SQLite |
| `Cannot determine label names for ...` | Corpus loaded without a `ClassLabel` | Add `label_names` to the spec in `config.DATASETS` |
| `CUDA out of memory` | Batch too large for 4 GB | Lower `--max-per-device-batch`; the sweep retries once automatically |
| `Qdrant collection ... does not exist` | Index never built | `.\make.ps1 fetch` then `.\make.ps1 index` |
| `/health` reports `degraded` | No model at `MEDEXTRACT_MODEL_DIR` | Check the path; the detail field names the underlying error |
| `No module named 'make'` / `make: command not found` | No GNU make on Windows | Use `.\make.ps1` |

## Verifying a setup

```powershell
.\make.ps1 test                                    # 192 passed, 1 skipped
.\make.ps1 lint                                    # clean
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
