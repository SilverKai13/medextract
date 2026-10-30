# Data Model

Every structure that crosses a boundary — HTTP, disk, the vector store, the
tracking store. Wire-level behaviour is in [SPEC.md](SPEC.md); the vector store
itself is in [DATABASE.md](DATABASE.md).

## Label vocabularies

BIO tagging. `B-X` begins an entity of type `X`, `I-X` continues one, `O` is
outside. **Integer order is part of the data**, not an implementation detail:
it is the encoding in the corpus and it becomes `id2label` in the exported
`config.json`, which is what the serving container reads.

**NCBI-Disease** (`ncbi/ncbi_disease`) — the default corpus:

| id | label |
|---|---|
| 0 | `O` |
| 1 | `B-Disease` |
| 2 | `I-Disease` |

**BC5CDR** (`tner/bc5cdr`) — registered, adds chemicals. The order is not
alphabetical and not guessable; it comes from the mirror's own
`dataset/label.json`:

| id | label |
|---|---|
| 0 | `O` |
| 1 | `B-Chemical` |
| 2 | `B-Disease` |
| 3 | `I-Disease` |
| 4 | `I-Chemical` |

`labels.IGNORE_INDEX = -100` marks positions excluded from the loss: special
tokens, and continuation subwords under the default convention.

## Corpus example

After loading, before tokenisation. Identical shape from all three sources.

| Field | Type | Notes |
|---|---|---|
| `id` | `string` | Sequential within the split |
| `tokens` | `Sequence(Value("string"))` | Pre-tokenised words, as the corpus ships them |
| `ner_tags` | `Sequence(ClassLabel(names=...))` | Integers indexing the vocabulary above. Column is `tags` on the BC5CDR mirror. |

```json
{"id": "0",
 "tokens": ["Identification", "of", "APC2", ",", "a", "homologue", "of", "the",
            "adenomatous", "polyposis", "coli", "tumour", "suppressor", "."],
 "ner_tags": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0]}
```

After tokenisation, columns are replaced by `input_ids`, `attention_mask`,
`token_type_ids` and `labels`, all the same length per example.

## `Entity` (`inference.py`)

The core output type. Frozen dataclass; `as_dict()` is what reaches the wire.

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | Surface form, sliced from the input |
| `label` | `str` | Entity type with the BIO prefix stripped — `Disease`, not `B-Disease` |
| `start` | `int` | Character offset, inclusive |
| `end` | `int` | Character offset, exclusive |
| `score` | `float` | Mean softmax confidence over the span's tokens, rounded to 4dp in `as_dict()` |

**Invariant:** `submitted_text[start:end] == text`. Python character offsets,
not UTF-8 byte offsets. Asserted for non-ASCII input in
`test_api.py::test_unicode_offsets_are_character_offsets`.

## `Abstract` (`retrieval/pubmed.py`)

One PubMed record, as fetched and as stored on disk.

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `pmid` | `str` | no | PubMed id, the natural key |
| `title` | `str` | no | Flattened from `ArticleTitle`, including nested markup |
| `abstract` | `str` | no | All `AbstractText` sections joined; labelled sections prefixed `LABEL: ` |
| `journal` | `str \| None` | yes | |
| `year` | `int \| None` | yes | Absent or non-numeric `PubDate/Year` yields `None` rather than an error |

`text` (property) is `f"{title}\n\n{abstract}"` — what gets embedded and what
gets tagged.

Persisted as JSON Lines at `data/pubmed.jsonl`, one record per line, UTF-8, not
ASCII-escaped. Records with no abstract are dropped at parse time; they cannot
be embedded usefully.

## API schemas (`api/schemas.py`)

These generate the OpenAPI document, so there is no second description to drift.

### Requests

| Model | Field | Type | Constraint |
|---|---|---|---|
| `ExtractRequest` | `text` | `str` | `min_length=1` |
| | `min_score` | `float` | `0.0 ≤ x ≤ 1.0`, default `0.0` |
| `BatchExtractRequest` | `texts` | `list[str]` | `min_length=1`, `max_length=64` |
| | `min_score` | `float` | `0.0 ≤ x ≤ 1.0` |
| `SearchRequest` | `query` | `str` | `min_length=1` |
| | `top_k` | `int` | `1 ≤ k ≤ 50`, default `5` |
| | `extract_entities` | `bool` | default `true` |

### Responses

| Model | Fields |
|---|---|
| `EntityOut` | `text`, `label`, `start`, `end`, `score` |
| `ExtractResponse` | `entities: list[EntityOut]`, `text`, `model_version`, `took_ms` |
| `BatchExtractResponse` | `results: list[list[EntityOut]]`, `model_version`, `took_ms` |
| `SearchHit` | `pmid`, `title`, `abstract`, `score`, `journal?`, `year?`, `entities: list[EntityOut]` |
| `SearchResponse` | `query`, `query_entities`, `hits: list[SearchHit]`, `took_ms` |
| `HealthResponse` | `status: "ok"\|"degraded"`, `model_loaded`, `model_version`, `backend`, `labels`, `retrieval_available`, `uptime_seconds`, `version`, `model_sha256`, `integrity_verified` |
| `ErrorResponse` | `detail`, `request_id?` |

`ExtractResponse.text` echoes the input so a caller can confirm the offsets
refer to the string it thinks they do without keeping its own copy.

## `RunConfig` (`train.py`)

Everything that affects a training run, so the whole record lands in MLflow
rather than in someone's shell history.

| Field | Default | Notes |
|---|---|---|
| `base_model` | `dmis-lab/biobert-base-cased-v1.2` | |
| `dataset` | `ncbi_disease` | Key into `config.DATASETS` |
| `learning_rate` | `3e-5` | |
| `batch_size` | `16` | **Effective** batch size, not per-device |
| `epochs` | `4.0` | |
| `weight_decay` | `0.01` | |
| `warmup_ratio` | `0.1` | |
| `max_length` | `256` | |
| `seed` | `42` | |
| `label_all_subwords` | `False` | The alignment convention |
| `fp16` | `False` | Auto-enabled when CUDA is present |
| `max_per_device_batch` | `16` | Device ceiling; the rest comes from accumulation |
| `keep_checkpoints` | `False` | Trainer checkpoints removed after the final save |
| `log_model_artifact` | `False` | Log metadata, not ~430 MB of weights per arm |
| `run_name` | `None` | |

## `DatasetSpec` (`config.py`)

| Field | Notes |
|---|---|
| `hf_id` | Hub repository |
| `tokens_column` / `tags_column` | Differ between mirrors, so they are config not constants |
| `config_name` | Hub config, usually `None` |
| `label_names` | Authoritative ordering. Required for the direct path — a raw corpus file carries no `ClassLabel`. |
| `validation_split` / `test_split` | A corpus without a validation split gets 10% of train held out, seed 13 |
| `direct_urls` | `{split: url}` for the fallback source |
| `direct_format` | `conll` or `jsonl` |

## `PreparedData` (`data.py`)

| Field | Notes |
|---|---|
| `tokenized` | `DatasetDict` of model inputs |
| `label_names`, `label2id`, `id2label` | The vocabulary, three ways |
| `source` | Which of the three loaders answered — recorded in MLflow |
| `split_sizes` | Per-split counts — also recorded |

The last two exist because "which F1?" is only answerable if the provenance of
the data is attached to the run.

## Tracking store

SQLite at `mlflow.db`. Experiment `medextract-ner`, one run per sweep arm.

- **Params** — every `RunConfig` field, plus `num_labels`, `train_examples`,
  `per_device_batch_size`, `gradient_accumulation_steps`, `corpus_source`,
  `split_sizes`, `tokenizer_lowercases`, `platform`, `torch_threads`.
- **Metrics** — `eval_*` and `test_*`: `precision`, `recall`, `f1`, `support`,
  per-type variants, `macro_f1`, plus the Trainer's runtime figures.
- **Tags** — `model_dir` pointing at the saved weights.
- **Artifacts** — `config.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `run_config.json`. Not the weights.

`report.py` reads this to generate [sweep_table.md](sweep_table.md).

## On-disk layout

```
artifacts/
  sweep/<arm-name>/model/    one saved model per arm (~412 MB)
  sweep/sweep_results.json   incremental results, written after each arm
  run/model/                 single-run output
  onnx/                      model.onnx + tokenizer + config + parity.json
  onnx-bc5cdr/               the same for the Disease + Chemical model
  embedder/                  model.onnx + tokenizer + embedder_config.json
  qdrant_local/              embedded Qdrant, when Docker is not used
data/
  pubmed.jsonl               fetched abstracts
  corpora/<dataset>/         cached corpus files for the direct source
mlflow.db                    tracking store
docs/benchmark.json          latency results with hardware
docs/loadtest.json           latency under concurrency, with the client's hardware
docs/calibration.json        reliability bins and threshold table
```

All of it is reproducible from the scripts and all of it is gitignored.

## `parity.json`

Written by `export_onnx.check_parity` next to `model.onnx`, and only after the
check passes, so its presence means exactly one thing.

```json
{
  "max_abs_logit_diff": 1.4781951904296875e-05,
  "atol": 0.0001,
  "rtol": 0.001,
  "argmax_agreement": 1.0,
  "entity_spans_match": true,
  "n_texts": 5,
  "model_file": "model.onnx",
  "model_bytes": 435856965,
  "model_sha256": "d48b71b7a0b64db82d598409aa85289d01c9620a9fad6df905206e08b323528d",
  "exported_with": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "onnx": "1.22.0", "onnxruntime": "1.26.0"}
}
```

`NerEngine` reads `model_sha256` at load and refuses a graph that does not
match it (`integrity.IntegrityError`). A missing record is logged and the graph
is served with `integrity_verified: false`; a contradicting one is a load
failure. See [TECHNICAL_SPEC.md §14](TECHNICAL_SPEC.md#14-artefact-integrity-integritypy).
