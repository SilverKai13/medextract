"""Phase 1: run the comparison sweep.

The brief asks for at least six configurations so the reported number is a
choice out of a table rather than one lucky run. The grid below varies the two
things that actually move token-classification F1 (learning rate and epochs),
holds a seed fixed for comparability, then re-runs the best-looking arm under
two more seeds so the spread is visible.

    python -m medextract.sweep --output-root artifacts/sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from medextract.config import ALT_BASE_MODEL, DEFAULT_BASE_MODEL, DEFAULT_DATASET, REPO_ROOT
from medextract.train import DEFAULT_EXPERIMENT, RunConfig, resolve_batching, train

log = logging.getLogger(__name__)


def default_grid(base: RunConfig) -> list[RunConfig]:
    arms = [
        replace(base, learning_rate=2e-5, batch_size=16, epochs=3, run_name="lr2e-5_bs16_ep3"),
        replace(base, learning_rate=3e-5, batch_size=16, epochs=3, run_name="lr3e-5_bs16_ep3"),
        replace(base, learning_rate=5e-5, batch_size=16, epochs=3, run_name="lr5e-5_bs16_ep3"),
        replace(base, learning_rate=3e-5, batch_size=16, epochs=5, run_name="lr3e-5_bs16_ep5"),
        replace(base, learning_rate=3e-5, batch_size=32, epochs=5, run_name="lr3e-5_bs32_ep5"),
        # Subword labelling convention as an arm rather than an assumption.
        replace(
            base,
            learning_rate=3e-5,
            batch_size=16,
            epochs=3,
            label_all_subwords=True,
            run_name="lr3e-5_bs16_ep3_all-subwords",
        ),
        # Different pretraining corpus, same recipe.
        replace(
            base,
            base_model=ALT_BASE_MODEL,
            learning_rate=3e-5,
            batch_size=16,
            epochs=3,
            run_name="pubmedbert_lr3e-5_bs16_ep3",
        ),
    ]
    # Seed repeats on the middle-of-the-road arm, so the table shows run-to-run
    # variance and a 0.005 F1 gap can be read for what it is.
    arms += [
        replace(
            base,
            learning_rate=3e-5,
            batch_size=16,
            epochs=3,
            seed=seed,
            run_name=f"lr3e-5_bs16_ep3_seed{seed}",
        )
        for seed in (1337, 2024)
    ]
    return arms


def suffix_arms(arms: list[RunConfig], keep_types: tuple[str, ...]) -> list[RunConfig]:
    """Name restricted arms after their types.

    So a Disease-only arm and the combined arm it is compared with can live in
    one experiment without colliding on run name, and the table says which is
    which without a reader having to open the params.
    """
    suffix = "_" + "+".join(t.lower() for t in keep_types)
    return [replace(a, run_name=f"{a.run_name}{suffix}") for a in arms]


def cuda_is_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:  # pragma: no cover - torch is a training-only dependency
        return False


def _is_out_of_memory(exc: BaseException) -> bool:
    """Recognise a CUDA OOM without importing torch just to catch it."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def run_sweep(arms: list[RunConfig], output_root: Path, experiment: str) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i, arm in enumerate(arms, start=1):
        name = arm.run_name or f"arm{i}"
        log.info("[%d/%d] %s", i, len(arms), name)
        try:
            summary = train(arm, output_root / name, experiment=experiment)
        except Exception as exc:
            if _is_out_of_memory(exc):
                # Halve what goes through the device and make it up with
                # accumulation. The effective batch size -- the thing the table
                # compares -- is unchanged, so the arm stays meaningful; it
                # just runs slower. Losing the arm entirely would be worse.
                per_device, _ = resolve_batching(arm)
                retry = replace(arm, max_per_device_batch=max(1, per_device // 2))
                log.warning(
                    "arm %s ran out of GPU memory at %d per device; retrying at %d",
                    name,
                    per_device,
                    retry.max_per_device_batch,
                )
                try:
                    summary = train(retry, output_root / name, experiment=experiment)
                except Exception:
                    log.exception("arm %s failed again after the OOM retry; continuing", name)
                    continue
            else:
                # One bad arm should not throw away the arms already finished.
                log.exception("arm %s failed; continuing", name)
                continue
        results.append({"run_name": name, **summary})
        (output_root / "sweep_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the MedExtract hyperparameter sweep.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "sweep")
    p.add_argument(
        "--max-per-device-batch",
        type=int,
        default=16,
        help="examples per device step; the rest of each batch comes from accumulation",
    )
    # Mirrors what pipeline.py decides for itself: fp16 on when there is a GPU
    # to run it on. Without this a CLI sweep silently trains in fp32 and takes
    # about twice as long as the same sweep launched from the pipeline.
    p.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="force mixed precision (default: on when CUDA is available)",
    )
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument(
        "--keep-types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="restrict every arm to these entity types; run names get a suffix",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only the named arms (by run_name), e.g. --only lr3e-5_bs16_ep3",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)

    fp16 = cuda_is_available() if ns.fp16 is None else ns.fp16
    log.info("mixed precision: %s", "on" if fp16 else "off")

    keep_types = tuple(ns.keep_types) if ns.keep_types else None
    base = RunConfig(
        base_model=ns.base_model,
        dataset=ns.dataset,
        fp16=fp16,
        max_per_device_batch=ns.max_per_device_batch,
        keep_checkpoints=ns.keep_checkpoints,
        keep_types=keep_types,
    )
    arms = default_grid(base)
    if ns.only:
        wanted = set(ns.only)
        arms = [a for a in arms if a.run_name in wanted]
        if not arms:
            raise SystemExit(f"No arms matched {sorted(wanted)}")
    if keep_types:
        arms = suffix_arms(arms, keep_types)

    results = run_sweep(arms, ns.output_root, ns.experiment)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
