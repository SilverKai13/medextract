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
from dataclasses import asdict, dataclass
from pathlib import Path

from medextract.config import DEFAULT_BASE_MODEL, DEFAULT_DATASET, REPO_ROOT

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "medextract-ner"


@dataclass
class RunConfig:
    base_model: str = DEFAULT_BASE_MODEL
    dataset: str = DEFAULT_DATASET
    learning_rate: float = 3e-5
    batch_size: int = 16
    epochs: float = 4.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 256
    seed: int = 42
    label_all_subwords: bool = False
    fp16: bool = False
    run_name: str | None = None


def train(config: RunConfig, output_dir: Path, *, experiment: str = DEFAULT_EXPERIMENT) -> dict:
    import mlflow
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from medextract import data as data_mod
    from medextract.metrics import build_compute_metrics

    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
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

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=max(config.batch_size, 32),
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

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file://{REPO_ROOT / 'mlruns'}"))
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(asdict(config))
        mlflow.log_params(
            {
                "num_labels": len(prepared.label_names),
                "train_examples": len(prepared.tokenized["train"]),
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
        (model_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2))
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    summary = {
        "run_id": run_id,
        "model_dir": str(model_dir),
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-all-subwords", action="store_true")
    p.add_argument("--fp16", action="store_true")
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
        epochs=ns.epochs,
        weight_decay=ns.weight_decay,
        warmup_ratio=ns.warmup_ratio,
        max_length=ns.max_length,
        seed=ns.seed,
        label_all_subwords=ns.label_all_subwords,
        fp16=ns.fp16,
        run_name=ns.run_name,
    )
    summary = train(config, ns.output_dir, experiment=ns.experiment)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
