"""Phase 2: export a fine-tuned model to ONNX and verify it agrees with PyTorch.

The parity check runs first and hard-fails on mismatch. A benchmark against a
graph that quietly produces different logits is a benchmark of the wrong thing,
so the numbers are not worth reading until this passes.

    python -m medextract.export_onnx --model-dir artifacts/run/model \
        --output-dir artifacts/onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from medextract.config import REPO_ROOT

log = logging.getLogger(__name__)

# Sentences with the kind of vocabulary that fragments into many subwords --
# a parity check on "hello world" proves very little about this model.
PARITY_TEXTS = [
    "Familial adenomatous polyposis is caused by germline mutations in the APC gene.",
    "The patient developed acute hepatic failure after acetaminophen overdose.",
    "Loss of BRCA1 function increases susceptibility to breast and ovarian carcinoma.",
    "No abnormality detected.",
    "Treatment with cisplatin induced nephrotoxicity and ototoxicity in 12% of cases.",
]

# ONNX Runtime and PyTorch differ in kernel implementations and fusion, so bit
# equality is not on offer. 1e-4 on logits is tight enough that argmax and the
# softmax confidences are unaffected in practice.
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3


DEFAULT_OPSET = 14


def _copy_sidecar_files(model_dir: Path, output_dir: Path) -> None:
    """Put the tokenizer and config next to the graph.

    The serving image loads everything from this one directory and never sees
    the PyTorch checkpoint, so anything the runtime needs has to be here.
    """
    import shutil

    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "run_config.json",
    ):
        source = model_dir / name
        if source.exists():
            shutil.copy(source, output_dir / name)


def export_with_torch(model_dir: Path, output_dir: Path, *, opset: int = DEFAULT_OPSET) -> Path:
    """Export via torch.onnx.export.

    This is the default because it has the fewest moving parts and pins the
    things that matter -- input names, output names, which axes are dynamic,
    and the opset -- explicitly rather than inheriting them from a converter's
    defaults. Batch size and sequence length are both dynamic so one graph
    serves single requests and batches.
    """
    import torch
    from transformers import AutoModelForTokenClassification

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    takes_token_type_ids = "token_type_ids" in model.forward.__code__.co_varnames
    input_names = ["input_ids", "attention_mask"]
    if takes_token_type_ids:
        input_names.append("token_type_ids")

    class LogitsOnly(torch.nn.Module):
        """Unwrap the ModelOutput so the graph has one clean output tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            return self.inner(**kwargs).logits

    dummy_shape = (2, 8)
    dummy = [torch.ones(dummy_shape, dtype=torch.long), torch.ones(dummy_shape, dtype=torch.long)]
    if takes_token_type_ids:
        dummy.append(torch.zeros(dummy_shape, dtype=torch.long))

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        LogitsOnly(model),
        tuple(dummy),
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "sequence"} for name in [*input_names, "logits"]},
        opset_version=opset,
        do_constant_folding=True,
        # torch 2.6's dynamo exporter pulls in onnxscript and mishandles
        # external data for small graphs; the TorchScript path is stable here.
        dynamo=False,
    )
    _copy_sidecar_files(model_dir, output_dir)
    return onnx_path


def export_with_optimum(model_dir: Path, output_dir: Path, *, optimize: bool = False) -> Path:
    """Export via optimum.onnxruntime.

    Kept as an alternative for the graph-level fusions ORTOptimizer applies,
    which torch.onnx.export does not do. Its version compatibility with torch
    is narrower, so it is not the default.
    """
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    ort_model = ORTModelForTokenClassification.from_pretrained(str(model_dir), export=True)
    ort_model.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(str(model_dir), use_fast=True).save_pretrained(str(output_dir))

    if optimize:
        from optimum.onnxruntime import ORTOptimizer
        from optimum.onnxruntime.configuration import OptimizationConfig

        ORTOptimizer.from_pretrained(ort_model).optimize(
            save_dir=str(output_dir),
            optimization_config=OptimizationConfig(optimization_level=99),
        )
        log.info("wrote graph-optimised variant")

    return output_dir / "model.onnx"


def export(
    model_dir: Path,
    output_dir: Path,
    *,
    exporter: str = "torch",
    optimize: bool = False,
    opset: int = DEFAULT_OPSET,
) -> Path:
    log.info("exporting %s -> %s (%s)", model_dir, output_dir, exporter)
    if exporter == "torch":
        onnx_path = export_with_torch(model_dir, output_dir, opset=opset)
    elif exporter == "optimum":
        onnx_path = export_with_optimum(model_dir, output_dir, optimize=optimize)
    else:
        raise ValueError(f"Unknown exporter {exporter!r}; expected 'torch' or 'optimum'.")

    log.info("ONNX graph: %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def check_parity(
    model_dir: Path,
    onnx_dir: Path,
    texts: list[str] | None = None,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict:
    """Compare raw logits, then compare the decoded entities.

    Logit tolerance is the numerical claim; matching entity spans is the claim
    that actually matters to a caller of /extract.
    """
    from medextract.inference import NerEngine

    texts = texts or PARITY_TEXTS
    torch_engine = NerEngine(model_dir, backend="torch")
    onnx_engine = NerEngine(onnx_dir, backend="onnx")

    # Same tokenisation into both backends, so any difference is the graph.
    encoded = torch_engine.encode(texts)
    model_inputs = {
        k: np.asarray(v)
        for k, v in encoded.items()
        if k in {"input_ids", "attention_mask", "token_type_ids"}
    }

    torch_logits = torch_engine.logits(model_inputs)
    onnx_logits = onnx_engine.logits(model_inputs)

    max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    argmax_agreement = float(np.mean(torch_logits.argmax(-1) == onnx_logits.argmax(-1)))

    torch_entities = torch_engine.predict(texts)
    onnx_entities = onnx_engine.predict(texts)
    spans_match = all(
        [(e.start, e.end, e.label) for e in a] == [(e.start, e.end, e.label) for e in b]
        for a, b in zip(torch_entities, onnx_entities, strict=True)
    )

    result = {
        "max_abs_logit_diff": max_abs_diff,
        "atol": atol,
        "rtol": rtol,
        "argmax_agreement": argmax_agreement,
        "entity_spans_match": spans_match,
        "n_texts": len(texts),
    }

    np.testing.assert_allclose(onnx_logits, torch_logits, atol=atol, rtol=rtol)
    if not spans_match:
        raise AssertionError(
            "ONNX and PyTorch logits agree within tolerance but decode to different "
            "entity spans. Inspect the boundary cases before trusting the export."
        )

    log.info(
        "parity ok: max|diff|=%.2e, argmax agreement=%.4f, spans identical",
        max_abs_diff,
        argmax_agreement,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a fine-tuned NER model to ONNX.")
    p.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "run" / "model")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "onnx")
    p.add_argument("--exporter", choices=["torch", "optimum"], default="torch")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument(
        "--optimize",
        action="store_true",
        help="apply ORTOptimizer graph fusions (requires --exporter optimum)",
    )
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--skip-parity", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)

    export(
        ns.model_dir,
        ns.output_dir,
        exporter=ns.exporter,
        optimize=ns.optimize,
        opset=ns.opset,
    )

    if ns.skip_parity:
        log.warning("parity check skipped by flag; benchmark numbers are unverified")
        return 0

    result = check_parity(ns.model_dir, ns.output_dir, atol=ns.atol, rtol=ns.rtol)
    (ns.output_dir / "parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
