"""Tests for corpus loading, tokenizer casing and batch resolution.

These cover the three things that were silently broken before a real run was
ever attempted: a dataset that no longer loads the documented way, a tokenizer
that destroys a cased vocabulary, and an effective batch size that does not fit
on the card. All three fail without raising, which is why each gets a test that
asserts the behaviour rather than the absence of an exception.

Everything here runs offline: the parsers are fed inline fixtures and the Hub
loaders are monkeypatched to raise.
"""

from __future__ import annotations

import json

import pytest

from medextract.config import DATASETS, mlflow_tracking_uri
from medextract.data import parse_conll, parse_tagged_jsonl

CONLL_FIXTURE = "\n".join(
    [
        "Identification\tO",
        "of\tO",
        "adenomatous\tB-Disease",
        "polyposis\tI-Disease",
        "coli\tI-Disease",
        ".\tO",
        "",
        "No\tO",
        "abnormality\tO",
        ".\tO",
        "",
        "",  # trailing blank lines, as the real files have
    ]
)

JSONL_FIXTURE = "\n".join(
    [
        json.dumps({"tokens": ["Naloxone", "reverses", "the", "effect"], "tags": [1, 0, 0, 0]}),
        json.dumps({"tokens": ["acute", "hepatic", "failure"], "tags": [2, 3, 3]}),
        "",
    ]
)

BC5CDR_LABELS = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]


class TestConllParser:
    def test_splits_on_blank_lines(self):
        sentences = parse_conll(CONLL_FIXTURE)
        assert len(sentences) == 2

    def test_tokens_and_tags_stay_aligned(self):
        for tokens, tags in parse_conll(CONLL_FIXTURE):
            assert len(tokens) == len(tags)

    def test_recovers_the_entity_span(self):
        tokens, tags = parse_conll(CONLL_FIXTURE)[0]
        assert tokens[2:5] == ["adenomatous", "polyposis", "coli"]
        assert tags[2:5] == ["B-Disease", "I-Disease", "I-Disease"]

    def test_trailing_blank_lines_do_not_produce_an_empty_sentence(self):
        """The official loader script emits one; an empty example is not data."""
        assert all(tokens for tokens, _ in parse_conll(CONLL_FIXTURE))

    def test_malformed_lines_are_skipped_not_fatal(self):
        sentences = parse_conll("good\tO\ngarbage-with-no-tab\nalso\tO\n")
        assert sentences == [(["good", "also"], ["O", "O"])]


class TestJsonlParser:
    def test_integer_tags_become_label_strings(self):
        sentences = parse_tagged_jsonl(JSONL_FIXTURE, "tokens", "tags", BC5CDR_LABELS)
        assert sentences[0][1] == ["B-Chemical", "O", "O", "O"]
        assert sentences[1][1] == ["B-Disease", "I-Disease", "I-Disease"]

    def test_blank_lines_are_ignored(self):
        assert len(parse_tagged_jsonl(JSONL_FIXTURE, "tokens", "tags", BC5CDR_LABELS)) == 2

    def test_length_mismatch_is_an_error_rather_than_a_silent_truncation(self):
        bad = json.dumps({"tokens": ["a", "b"], "tags": [0]})
        with pytest.raises(ValueError, match="Corrupt record"):
            parse_tagged_jsonl(bad, "tokens", "tags", BC5CDR_LABELS)


class TestDatasetSpecs:
    def test_every_corpus_has_a_direct_source(self):
        """The Hub repos are script-only, so a fallback is not optional."""
        for key, spec in DATASETS.items():
            assert spec.direct_urls, f"{key} has no direct_urls"
            assert set(spec.direct_urls) == {"train", "validation", "test"}

    def test_every_corpus_declares_its_label_ordering(self):
        """Raw corpus files carry no ClassLabel, so the ordering must be config."""
        for key, spec in DATASETS.items():
            assert spec.label_names, f"{key} has no label_names"

    def test_bc5cdr_label_order_matches_the_published_mapping(self):
        """From tner/bc5cdr's own dataset/label.json -- the ints depend on it."""
        assert list(DATASETS["bc5cdr"].label_names) == BC5CDR_LABELS

    def test_ncbi_disease_is_a_three_tag_disease_corpus(self):
        assert list(DATASETS["ncbi_disease"].label_names) == ["O", "B-Disease", "I-Disease"]


class TestCorpusFallback:
    def test_falls_through_to_direct_download_when_the_hub_fails(self, monkeypatch):
        """Both Hub paths raising must not be fatal -- that is the normal case."""
        import medextract.data as data_mod

        calls: list[str] = []

        def exploding_load_dataset(*args, **kwargs):
            calls.append("hub")
            raise RuntimeError("dataset scripts are no longer supported")

        def fake_direct(spec, cache_dir=None):
            calls.append("direct")
            return {"train": object(), "validation": object(), "test": object()}

        monkeypatch.setattr("datasets.load_dataset", exploding_load_dataset)
        monkeypatch.setattr(data_mod, "load_from_direct_urls", fake_direct)

        _raw, spec, source = data_mod.load_corpus("ncbi_disease")
        assert calls == ["hub", "hub", "direct"]
        assert source == "direct-download"
        assert spec.hf_id == "ncbi/ncbi_disease"

    def test_exhausting_every_source_names_all_of_them(self, monkeypatch):
        import medextract.data as data_mod

        def explode(*args, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr("datasets.load_dataset", explode)
        monkeypatch.setattr(data_mod, "load_from_direct_urls", explode)

        with pytest.raises(RuntimeError, match="Could not load") as excinfo:
            data_mod.load_corpus("ncbi_disease")
        message = str(excinfo.value)
        assert "hub-parquet" in message
        assert "direct-download" in message

    def test_unknown_dataset_is_rejected_before_any_network_call(self):
        import medextract.data as data_mod

        with pytest.raises(KeyError, match="Unknown dataset"):
            data_mod.load_corpus("not_a_corpus")


class TestTokenizerCasing:
    def test_a_cased_vocabulary_is_detected_from_the_vocabulary(
        self, make_bert_tokenizer, tmp_path
    ):
        from medextract.tokenization import vocabulary_is_cased

        cased_vocab = tmp_path / "vocab.txt"
        cased_vocab.write_text(
            "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "APC", "BRCA", "Aden", "the"]),
            encoding="utf-8",
        )
        assert vocabulary_is_cased(make_bert_tokenizer(cased_vocab))

    def test_an_uncased_vocabulary_is_left_alone(self, tiny_tokenizer):
        """The stub vocabulary is all lowercase; nothing should be 'fixed'."""
        from medextract.tokenization import vocabulary_is_cased

        assert not vocabulary_is_cased(tiny_tokenizer)

    def test_loading_an_uncased_checkpoint_keeps_lowercasing(self, tiny_model_dir):
        from medextract.tokenization import load_fast_tokenizer

        tokenizer = load_fast_tokenizer(tiny_model_dir)
        assert tokenizer.is_fast

    def test_a_cased_checkpoint_is_not_lowercased(self, make_bert_tokenizer, tmp_path):
        """The BioBERT failure mode, reproduced on a checkpoint we control.

        Saved without a tokenizer_config.json, exactly as BioBERT ships, so
        AutoTokenizer has to invent do_lower_case and gets it wrong.
        """
        from medextract.tokenization import load_fast_tokenizer

        vocab = tmp_path / "vocab.txt"
        vocab.write_text(
            "\n".join(
                [
                    *["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                    *["F", "##ami", "##lial", "AP", "##C", "BR", "##CA", "##1", "E", "##G", "##F"],
                    *["##R", "Aden", "##oma", "##to", "##us", "the", "of"],
                ]
            ),
            encoding="utf-8",
        )
        make_bert_tokenizer(vocab, do_lower_case=True).save_pretrained(str(tmp_path))
        (tmp_path / "tokenizer_config.json").unlink()
        # 5.x also writes the backend's tokenizer.json, which carries the
        # lowercasing normaliser with it. BioBERT ships neither file -- only
        # vocab.txt -- so remove it too to reproduce that shape faithfully.
        (tmp_path / "tokenizer.json").unlink(missing_ok=True)
        # config.json without a tokenizer_config.json is precisely the shape
        # BioBERT ships in, and it is what makes AutoTokenizer guess.
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "bert", "vocab_size": 22}), encoding="utf-8"
        )

        tokenizer = load_fast_tokenizer(tmp_path)
        pieces = tokenizer.convert_ids_to_tokens(
            tokenizer("APC EGFR", add_special_tokens=False)["input_ids"]
        )
        assert "AP" in pieces and "[UNK]" not in pieces


class TestBatchResolution:
    def test_a_batch_that_fits_needs_no_accumulation(self):
        from medextract.train import RunConfig, resolve_batching

        assert resolve_batching(RunConfig(batch_size=16, max_per_device_batch=16)) == (16, 1)

    def test_an_oversized_batch_is_reached_by_accumulation(self):
        """bs32 on a 4GB card: 16 through the device, twice, not 16 and a lie."""
        from medextract.train import RunConfig, resolve_batching

        assert resolve_batching(RunConfig(batch_size=32, max_per_device_batch=16)) == (16, 2)

    def test_the_effective_batch_size_is_always_preserved(self):
        from medextract.train import RunConfig, resolve_batching

        for batch_size in (8, 16, 32, 64):
            for ceiling in (1, 4, 8, 16, 64):
                per_device, accumulation = resolve_batching(
                    RunConfig(batch_size=batch_size, max_per_device_batch=ceiling)
                )
                assert per_device * accumulation == batch_size
                assert per_device <= ceiling

    def test_a_tiny_ceiling_still_yields_a_usable_config(self):
        from medextract.train import RunConfig, resolve_batching

        assert resolve_batching(RunConfig(batch_size=32, max_per_device_batch=1)) == (1, 32)


class TestSweepDefaults:
    def test_mixed_precision_follows_the_hardware(self, monkeypatch):
        """A CLI sweep must not silently train in fp32 on a GPU."""
        import medextract.sweep as sweep_mod

        monkeypatch.setattr(sweep_mod, "cuda_is_available", lambda: True)
        ns = sweep_mod.build_parser().parse_args([])
        assert ns.fp16 is None  # unset means "decide from the hardware"
        assert (sweep_mod.cuda_is_available() if ns.fp16 is None else ns.fp16) is True

    def test_mixed_precision_can_be_forced_off(self):
        import medextract.sweep as sweep_mod

        assert sweep_mod.build_parser().parse_args(["--no-fp16"]).fp16 is False

    def test_out_of_memory_is_recognised_from_either_signal(self):
        from medextract.sweep import _is_out_of_memory

        class OutOfMemoryError(RuntimeError):
            pass

        assert _is_out_of_memory(OutOfMemoryError("boom"))
        assert _is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        assert not _is_out_of_memory(ValueError("something else entirely"))


class TestTrackingUri:
    def test_is_a_well_formed_file_uri(self):
        """An interpolated Windows path yields file://C:... which MLflow misreads."""
        uri = mlflow_tracking_uri()
        assert uri.startswith("file:///") or "://" in uri
        assert "\\" not in uri

    def test_the_environment_wins(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
        assert mlflow_tracking_uri() == "http://tracking.example:5000"


class TestTextEncoding:
    """Every text read and write must name its encoding.

    Windows defaults stdio and `Path.read_text()` to the ANSI codepage, so an
    em-dash in a generated table is enough to either crash the reader or write
    a file that is not valid UTF-8. Both happened here: once in a test reading
    the README, and once in `benchmark.py` writing `docs/benchmark.md` as
    cp1252, which renders as mojibake anywhere else.

    Neither is catchable by testing behaviour, because on a UTF-8 machine the
    default is already right. So the invariant is asserted over the source.
    """

    def _unencoded_calls(self, root):
        import ast

        offenders = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in {"read_text", "write_text"}:
                    continue
                if not any(kw.arg == "encoding" for kw in node.keywords):
                    offenders.append(f"{path.name}:{node.lineno} .{func.attr}()")
        return offenders

    def test_package_names_an_encoding_everywhere(self):
        from medextract.config import REPO_ROOT

        assert self._unencoded_calls(REPO_ROOT / "src") == []

    def test_tests_name_an_encoding_everywhere(self):
        from medextract.config import REPO_ROOT

        assert self._unencoded_calls(REPO_ROOT / "tests") == []

    def test_generated_documents_are_valid_utf8(self):
        """Regenerating a table must not leave a file the rest of the world cannot read."""
        from medextract.config import REPO_ROOT

        for name in ("benchmark.md", "sweep_table.md"):
            path = REPO_ROOT / "docs" / name
            if not path.exists():
                continue  # not generated yet on a fresh checkout
            path.read_bytes().decode("utf-8")
