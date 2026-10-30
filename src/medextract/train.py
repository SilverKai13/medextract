"""Phase 1: fine-tune a BioBERT variant for biomedical NER.

Every run is recorded in MLflow with the hyperparameters that produced it and
the entity-level scores that came out, so the README table is generated from
the tracking store rather than from notes.

    python -m medextract.train --lr 3e-5 --batch-size 16 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from medextract.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET,
    REPO_ROOT,
    mlflow_tracking_uri,
)

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "medextract-ner"

# Files worth keeping in the tracking store: enough to identify and re-load a
# run, none of the weights. See `log_model_artifact` below.
MODEL_METADATA_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "run_config.json",
)


@dataclass
class RunConfig:
    base_model: str = DEFAULT_BASE_MODEL
    dataset: str = DEFAULT_DATASET
    learning_rate: float = 3e-5
    # The *effective* batch size -- what the optimiser steps on, and what the
    # comparison table means by "batch". When it exceeds what the GPU holds it
    # is reached by accumulation rather than by quietly training on less.
    batch_size: int = 16
    epochs: float = 4.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 256
    seed: int = 42
    label_all_subwords: bool = False
    fp16: bool = False
    # How many examples actually go through the device at once. 16 fits
    # BERT-base at 256 tokens on a 4GB card with fp16; larger cards can raise
    # it, and the sweep lowers it on its own if an arm runs out of memory.
    max_per_device_batch: int = 16
    # The Trainer's per-epoch checkpoints are only needed while the run is in
    # flight -- load_best_model_at_end consumes them, then save_model writes
    # the model that matters. Nine arms' worth left behind is several GB for
    # nothing.
    keep_checkpoints: bool = False
    # Copying ~430MB of weights into mlruns/ per arm duplicates what is already
    # on disk under artifacts/. The metadata and a pointer are logged instead.
    log_model_artifact: bool = False
    run_name: str | None = None


def resolve_batching(config: RunConfig) -> tuple[int, int]:
    """Split an effective batch size into per-device batch and accumulation.

    Returns (per_device_batch, gradient_accumulation_steps). The product is the
    effective batch size, so an arm means the same thing whatever hardware it
    lands on -- the number in the table stays comparable across machines.
    """
    per_device = max(1, min(config.batch_size, config.max_per_device_batch))
    accumulation = max(1, round(config.batch_size / per_device))
    return per_device, accumulation


def train(config: RunConfig, output_dir: Path, *, experiment: str = DEFAULT_EXPERIMENT) -> dict:
    import mlflow
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from medextract import data as data_mod
    from medextract.metrics import build_compute_metrics
    from medextract.tokenization import load_fast_tokenizer

    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Not AutoTokenizer directly: BioBERT ships no tokenizer_config.json, and
    # the default that gets invented for it lowercases a cased vocabulary.
    tokenizer = load_fast_tokenizer(config.base_model)
    prepared = data_mod.prepare(
        tokenizer,
        config.dataset,
        max_length=config.max_length,
        label_all_subwords=config.label_all_subwords,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=len(prepared.label_names),
        id2label=prepared.id2label,
        label2id=prepared.label2id,
    )

    per_device, accumulation = resolve_batching(config)
    log.info(
        "effective batch %d = %d per device x %d accumulation steps",
        per_device * accumulation,
        per_device,
        accumulation,
    )

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        # Evaluation is a no-grad forward pass, but on a small card the
        # activations still dominate; matching the train batch keeps a run that
        # trains from failing three epochs later on its first eval.
        per_device_eval_batch_size=per_device,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=config.seed,
        fp16=config.fp16,
        logging_steps=50,
        report_to=[],  # MLflow is driven explicitly below, not via the integration
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=prepared.tokenized["train"],
        eval_dataset=prepared.tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(prepared.id2label),
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(asdict(config))
        mlflow.log_params(
            {
                "num_labels": len(prepared.label_names),
                "train_examples": len(prepared.tokenized["train"]),
                "per_device_batch_size": per_device,
                "gradient_accumulation_steps": accumulation,
                # Which of the three corpus sources answered. "Which F1?" is
                # only answerable if the provenance of the data is in the run.
                "corpus_source": prepared.source,
                "split_sizes": json.dumps(prepared.split_sizes or {}),
                "tokenizer_lowercases": bool(getattr(tokenizer, "do_lower_case", False)),
                "platform": platform.platform(),
                "torch_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            }
        )

        trainer.train()

        val_metrics = trainer.evaluate(prepared.tokenized["validation"])
        test_metrics = trainer.evaluate(prepared.tokenized["test"], metric_key_prefix="test")

        # MLflow rejects non-numeric metric values; runtime/samples-per-second
        # entries from the Trainer are numeric, but guard anyway.
        loggable = {
            k: float(v)
            for k, v in {**val_metrics, **test_metrics}.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(loggable)

        model_dir = output_dir / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        (model_dir / "run_config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )

        if config.log_model_artifact:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
        else:
            for name in MODEL_METADATA_FILES:
                candidate = model_dir / name
                if candidate.exists():
                    mlflow.log_artifact(str(candidate), artifact_path="model")
        mlflow.set_tag("model_dir", str(model_dir))

        run_id = mlflow.active_run().info.run_id

    if not config.keep_checkpoints:
        # The best weights are already in model_dir by this point; what is left
        # under checkpoints/ is a duplicate plus optimiser state.
        shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)

    summary = {
        "run_id": run_id,
        "model_dir": str(model_dir),
        "corpus_source": prepared.source,
        "validation": {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "test": {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))},
    }
    log.info(
        "run %s finished: validation f1=%.4f test f1=%.4f",
        run_id,
        val_metrics.get("eval_f1", float("nan")),
        test_metrics.get("test_f1", float("nan")),
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune a BioBERT variant for biomedical NER.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--lr", type=float, default=3e-5, dest="learning_rate")
    p.add_argument("--batch-size", type=int, default=16, help="effective batch size")
    p.add_argument(
        "--max-per-device-batch",
        type=int,
        default=16,
        help="examples per device step; the rest of the batch is reached by accumulation",
    )
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-all-subwords", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--log-model-artifact", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "run")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)
    config = RunConfig(
        base_model=ns.base_model,
        dataset=ns.dataset,
        learning_rate=ns.learning_rate,
        batch_size=ns.batch_size,
        max_per_device_batch=ns.max_per_device_batch,
        epochs=ns.epochs,
        weight_decay=ns.weight_decay,
        warmup_ratio=ns.warmup_ratio,
        max_length=ns.max_length,
        seed=ns.seed,
        label_all_subwords=ns.label_all_subwords,
        fp16=ns.fp16,
        keep_checkpoints=ns.keep_checkpoints,
        log_model_artifact=ns.log_model_artifact,
        run_name=ns.run_name,
    )
    summary = train(config, ns.output_dir, experiment=ns.experiment)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
