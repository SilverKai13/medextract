# Requirements

Numbered requirements with the test that proves each one. A requirement without
a test is an intention, so every row here names a real test node; run them with
`.\make.ps1 test`.

Behavioural detail for the API requirements is in [SPEC.md](SPEC.md); the
product reasoning behind them is in [PRD.md](PRD.md).

## Functional

### Extraction

| ID | Requirement | Verified by |
|---|---|---|
| FR-1 | `POST /extract` returns every disease and chemical span in the submitted text, each with type, start offset, end offset and confidence. | `test_api.py::test_happy_path` |
| FR-2 | Offsets are Python character offsets into the exact string submitted, so `text[start:end]` reproduces the returned surface form — including for non-ASCII input. | `test_api.py::test_unicode_offsets_are_character_offsets`, `test_inference.py::test_offsets_index_into_the_original_text` |
| FR-3 | A span covering several WordPiece subwords is returned as one entity spanning the full surface form. | `test_inference.py::test_multi_subword_entity_spans_full_surface_form` |
| FR-4 | Confidence is the mean softmax probability over the tokens in the span. | `test_inference.py::test_score_is_mean_over_span_tokens` |
| FR-5 | `min_score` drops entities scoring below the given threshold. | `test_api.py::test_min_score_filters_results` |
| FR-6 | A `B-` tag always starts a new entity, and a type change always splits one. | `test_inference.py::test_b_tag_starts_a_second_entity`, `test_inference.py::test_type_switch_splits_the_span` |
| FR-7 | Input longer than the model's context is windowed with overlap, not truncated; an entity past the limit is still found. | `test_inference.py::test_long_text_is_windowed_not_truncated` |
| FR-8 | Spans found in more than one window are merged, keeping the highest score, and returned in document order. | `test_inference.py::test_results_are_sorted_and_deduplicated` |
| FR-8a | A word split into several subwords is returned as one entity, never as fragments: its label comes from the first subword, the only one training supervises. | `test_inference.py::TestSubwordAggregation` (7), `::test_predicted_spans_never_start_mid_word` |
| FR-9 | `POST /extract/batch` accepts up to 64 texts and returns one result list per text, identical to calling `/extract` on each. | `test_api.py::test_returns_one_result_per_text`, `test_api.py::test_batch_matches_single_calls` |

### Retrieval

| ID | Requirement | Verified by |
|---|---|---|
| FR-10 | `POST /search` ranks indexed abstracts against a query by cosine similarity and returns the top `k`. | `test_retrieval.py::test_search_returns_requested_number_of_hits`, `test_retrieval.py::test_lexically_close_query_ranks_the_right_abstract_first` |
| FR-11 | Each hit carries its PMID, title, abstract, journal, year and score. | `test_retrieval.py::test_hits_carry_payload_and_scores` |
| FR-12 | Entities are extracted from each hit unless `extract_entities` is false. | `test_retrieval.py::test_search_returns_hits_with_entities`, `test_retrieval.py::test_extract_entities_false_skips_tagging` |
| FR-13 | Re-indexing the same corpus is idempotent: point ids derive from PMIDs, so a second pass updates rather than duplicates. | `test_retrieval.py::test_reindexing_is_idempotent`, `test_retrieval.py::test_point_ids_are_stable_for_a_pmid` |
| FR-14 | Structured abstracts are reassembled across all their `AbstractText` sections with labels retained, not truncated to the first. | `test_retrieval.py::test_structured_abstract_sections_are_joined_with_labels` |
| FR-15 | Records with no abstract are dropped rather than indexed empty. | `test_retrieval.py::test_records_without_abstracts_are_dropped` |
| FR-16 | The E-utilities client paginates to the requested maximum, sends a contact address, retries transient failures and respects the published rate limit. | `test_retrieval.py::test_search_paginates_until_max`, `::test_contact_details_are_sent`, `::test_transient_failure_is_retried`, `::test_rate_limiter_spaces_requests` |
| FR-17 | Retrieval works without Docker: a filesystem path opens an embedded on-disk Qdrant. | `test_retrieval.py::test_build_client_accepts_a_local_path_for_docker_free_retrieval` |

### Operations

| ID | Requirement | Verified by |
|---|---|---|
| FR-18 | `GET /health` reports status, whether the model loaded, the version, backend, label set, retrieval availability and uptime. | `test_api.py::test_reports_ok_with_a_model_loaded` |
| FR-19 | With no model present the service still starts, `/health` returns 200 with `status: degraded` and the load error. | `test_api.py::test_reports_degraded_without_a_model` |
| FR-20 | Every response carries `X-Request-ID`, echoing a caller-supplied one when present and generating a unique one otherwise. | `test_api.py::test_request_id_is_returned`, `::test_caller_supplied_request_id_is_echoed`, `::test_request_ids_differ_between_calls` |
| FR-21 | OpenAPI documentation is generated from the same schemas the handlers validate against. | `test_api.py::test_schema_documents_the_endpoints` |

### Training and evaluation

| ID | Requirement | Verified by |
|---|---|---|
| FR-22 | Word-level labels are projected onto subwords via `word_ids()`; only the first subword of a word is supervised by default and special tokens never are. | `test_labels.py::test_only_first_subword_is_labelled_by_default`, `::test_special_tokens_are_ignored`, `::test_no_supervised_position_is_a_special_token` |
| FR-23 | The alternative convention — label every subword, `B-` becoming `I-` after the first — is selectable and swept, not assumed. | `test_labels.py::test_label_all_subwords_converts_b_to_i_on_continuations`, `test_cli.py::test_both_subword_conventions_are_covered` |
| FR-24 | Truncation never desynchronises labels from input ids. | `test_data.py::test_truncation_does_not_desynchronise_labels`, `test_labels.py::test_alignment_length_always_matches_input` |
| FR-25 | Scoring is strict entity-level IOB2: exact span and type, invalid continuations discarded rather than repaired. | `test_metrics.py::test_invalid_continuation_is_discarded_not_repaired`, `::test_orphan_i_tag_yields_no_span`, `::test_partial_span_match_counts_as_wrong` |
| FR-26 | A model that predicts nothing scores near zero, not the >90% a token-accuracy metric would give it. | `test_metrics.py::test_predicting_nothing_scores_zero_not_high` |
| FR-27 | Scoring and serving disagree on orphan `I-` tags deliberately, and both behaviours are pinned. | `test_metrics.py::test_strict_scoring_and_lenient_serving_differ_on_purpose` |
| FR-28 | The hand-rolled scorer agrees with seqeval across randomised sequences whenever seqeval is importable. | `test_metrics.py::test_matches_seqeval_on_random_sequences` |
| FR-29 | The sweep compares at least six configurations, varying learning rate, batch size, epochs, subword convention and base checkpoint, with seed repeats on one arm. | `test_cli.py::TestSweepGrid` (7 tests) |
| FR-30 | The corpus loads from a Hub parquet conversion, a plain Hub load, or the upstream corpus files, in that order, and reports which answered. | `test_corpus_sources.py::test_falls_through_to_direct_download_when_the_hub_fails`, `::test_exhausting_every_source_names_all_of_them` |
| FR-31 | A cased checkpoint is never tokenised with lowercasing, even when it ships no `tokenizer_config.json`. | `test_corpus_sources.py::test_a_cased_checkpoint_is_not_lowercased` |
| FR-32 | An effective batch size larger than the device can hold is reached by gradient accumulation, never by silently training on less. | `test_corpus_sources.py::test_the_effective_batch_size_is_always_preserved` |

### Export

| ID | Requirement | Verified by |
|---|---|---|
| FR-33 | The exported ONNX graph accepts dynamic batch and sequence axes with pinned input/output names. | `test_onnx.py::test_export_produces_a_graph` |
| FR-34 | Export is gated on a parity check covering logits, argmax and decoded spans; failure aborts rather than warns. | `test_onnx.py::test_check_parity_reports_and_passes`, `::test_check_parity_fails_when_tolerance_is_impossible`, `::test_both_backends_decode_identical_entities` |
| FR-35 | The best arm is selected by validation F1, ties broken by test F1, and an empty result set fails with a clear message. | `test_pipeline.py::test_picks_the_highest_validation_f1`, `::test_ties_are_broken_by_test_f1`, `::test_empty_results_raise_with_a_clear_message` |
| FR-36 | README result tables are rewritten from generated output between marker pairs, and a missing marker fails loudly. | `test_readme_sync.py` (6 tests) |

## Non-functional

| ID | Requirement | Target | Verified by |
|---|---|---|---|
| NFR-1 | Test-set entity F1 on NCBI-Disease | ≥ 0.85 | [RESULTS.md §2](RESULTS.md#2-training) — measured 0.8838 |
| NFR-2 | ONNX logits match PyTorch | `atol=1e-4`, `rtol=1e-3` | `test_onnx.py::test_logits_match_within_tolerance` |
| NFR-3 | Serving image size | < 1.5 GB | [RESULTS.md §6](RESULTS.md#6-serving-container) — measured 395 MB; CI `build` job enforces it |
| NFR-4 | Serving carries no PyTorch | `requirements-serve.txt` excludes torch, datasets, mlflow | [DEPLOYMENT.md](DEPLOYMENT.md) |
| NFR-5 | Model loads once at startup, not per request | Lifespan handler | `api/state.py`, `test_api.py::test_reports_ok_with_a_model_loaded` |
| NFR-6 | Test suite runs offline in under a minute | No network, no downloads | `.\make.ps1 test` |
| NFR-7 | Container runs as a non-root user | uid 10001 | `Dockerfile` |
| NFR-8 | Input size is bounded | 20,000 chars, 413 beyond it | `test_api.py::test_oversized_input_is_rejected_with_413` |
| NFR-9 | Batch size is bounded | 64 texts | `test_api.py::test_empty_list_is_rejected`, schema `max_length` |
| NFR-10 | Malformed input is rejected with 422, not a 500 | Pydantic validation | `test_api.py::test_malformed_json_is_rejected`, `::test_missing_field_is_rejected` |
| NFR-11 | Latency is reported as percentiles with the hardware recorded | p50/p90/p95/p99 | [RESULTS.md §4](RESULTS.md#4-latency), `test_onnx.py::test_benchmark_runs_and_reports_percentiles` |
| NFR-12 | An index is never built from fallback hashing vectors by accident | `allow_fallback=False` by default | `test_cli.py::test_fallback_is_off_by_default` |
| NFR-13 | Every text read and write names its encoding, so generated files are valid UTF-8 on any platform | No implicit locale encoding | `test_corpus_sources.py::TestTextEncoding` (3) |

## Traceability gaps

Requirements that are *not* covered by an automated test, and why:

- **NFR-1** needs a GPU, the corpus and roughly 40 minutes. It is measured by
  the sweep and recorded in [RESULTS.md](RESULTS.md), not by pytest.
- **NFR-3** needs a Docker daemon. Enforced in the CI `build` job.
- **NFR-11** is machine-dependent by definition. The harness is tested; the
  numbers are recorded, not asserted.

This split is deliberate: the test suite verifies that the plumbing is correct,
and the sweep and benchmark measure how well it works. Mixing them would make
`pytest` slow, networked and non-deterministic. See [TESTING.md](TESTING.md).
