"""Export a sentence-transformers encoder to ONNX.

The serving image runs on onnxruntime and carries no PyTorch -- that is most of
how it stays under the size target. The query encoder therefore has to be an
ONNX graph too, not just the NER model.

Pooling is read from the sentence-transformers module config rather than
assumed, because getting it wrong produces embeddings that are subtly wrong
rather than obviously broken: search still returns results, they are just worse.

    python -m medextract.retrieval.export_embedder \
        --model pritamdeka/S-PubMedBert-MS-MARCO --output-dir artifacts/embedder
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from medextract.config import REPO_ROOT, get_settings

log = logging.getLogger(__name__)

POOLING_CONFIG_NAME = "1_Pooling/config.json"


def detect_pooling(model_name_or_path: str) -> str:
    """Return "mean" or "cls", from the model's own pooling config if present."""
    try:
        from huggingface_hub import hf_hub_download

        local = Path(model_name_or_path) / POOLING_CONFIG_NAME
        path = (
            local
            if local.exists()
            else Path(hf_hub_download(model_name_or_path, POOLING_CONFIG_NAME))
        )
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("could not read pooling config (%s); assuming mean pooling", exc)
        return "mean"

    if config.get("pooling_mode_cls_token"):
        return "cls"
    if config.get("pooling_mode_mean_tokens", True):
        return "mean"
    log.warning("unrecognised pooling config %s; assuming mean", config)
    return "mean"


def export_embedder(model_name: str, output_dir: Path, *, opset: int = 14) -> Path:
    import torch
    from transformers import AutoModel

    from medextract.tokenization import load_fast_tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Same loader the NER side uses, so a query encoder whose checkpoint is
    # missing its tokenizer config cannot end up casing queries differently
    # from the documents it was indexed against.
    tokenizer = load_fast_tokenizer(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    class TokenEmbeddings(torch.nn.Module):
        """Emit last_hidden_state; pooling happens in numpy at query time."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            return self.inner(**kwargs).last_hidden_state

    takes_token_type_ids = "token_type_ids" in model.forward.__code__.co_varnames
    input_names = ["input_ids", "attention_mask"]
    dummy = [torch.ones((2, 8), dtype=torch.long), torch.ones((2, 8), dtype=torch.long)]
    if takes_token_type_ids:
        input_names.append("token_type_ids")
        dummy.append(torch.zeros((2, 8), dtype=torch.long))

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        TokenEmbeddings(model),
        tuple(dummy),
        str(onnx_path),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes={
            **{name: {0: "batch", 1: "sequence"} for name in input_names},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "embedder_config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "pooling": detect_pooling(model_name),
                "dimension": int(model.config.hidden_size),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("embedder exported to %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Export the query encoder to ONNX.")
    p.add_argument("--model", default=None, help="defaults to MEDEXTRACT_EMBEDDING_MODEL")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "embedder")
    p.add_argument("--opset", type=int, default=14)
    ns = p.parse_args(argv)

    export_embedder(ns.model or get_settings().embedding_model, ns.output_dir, opset=ns.opset)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
