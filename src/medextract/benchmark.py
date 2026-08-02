"""Phase 2: latency benchmark, PyTorch eager vs ONNX Runtime.

Reports percentiles rather than means. Mean latency hides the tail, and the
tail is what a caller notices. Hardware and thread settings are recorded in the
output because a speedup figure without them is not a claim anyone can check.

    python -m medextract.benchmark --torch-dir artifacts/run/model \
        --onnx-dir artifacts/onnx --iterations 100
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from medextract.config import REPO_ROOT
from medextract.export_onnx import PARITY_TEXTS

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = (1, 16)
DEFAULT_ITERATIONS = 100
DEFAULT_WARMUP = 10


def hardware_info() -> dict[str, Any]:
    import os

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(float(np.percentile(ordered, 50)), 3),
        "p90_ms": round(float(np.percentile(ordered, 90)), 3),
        "p95_ms": round(float(np.percentile(ordered, 95)), 3),
        "p99_ms": round(float(np.percentile(ordered, 99)), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def time_backend(
    engine,
    texts: list[str],
    batch_size: int,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, Any]:
    """Time the model call only.

    Tokenisation is done once and hoisted out of the loop: it is identical
    across backends, so including it would dilute the very difference being
    measured. End-to-end request latency is what /extract's own logging reports.
    """
    batch = [texts[i % len(texts)] for i in range(batch_size)]
    encoded = engine.encode(batch)
    model_inputs = {
        k: np.asarray(v)
        for k, v in encoded.items()
        if k in {"input_ids", "attention_mask", "token_type_ids"}
    }

    for _ in range(warmup):
        engine.logits(model_inputs)

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        engine.logits(model_inputs)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    stats = percentiles(samples_ms)
    stats["batch_size"] = batch_size
    stats["iterations"] = iterations
    stats["throughput_seq_per_s"] = round(batch_size / (stats["p50_ms"] / 1000.0), 2)
    return stats


def run_benchmark(
    torch_dir: Path,
    onnx_dir: Path,
    *,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    texts: list[str] | None = None,
) -> dict[str, Any]:
    from medextract.inference import NerEngine

    texts = texts or PARITY_TEXTS
    engines = {
        "pytorch_eager": NerEngine(torch_dir, backend="torch"),
        "onnxruntime": NerEngine(onnx_dir, backend="onnx"),
    }

    results: dict[str, Any] = {"hardware": hardware_info(), "backends": {}}
    for name, engine in engines.items():
        results["backends"][name] = {
            str(bs): time_backend(engine, texts, bs, iterations=iterations, warmup=warmup)
            for bs in batch_sizes
        }

    results["speedup_p50"] = {
        str(bs): round(
            results["backends"]["pytorch_eager"][str(bs)]["p50_ms"]
            / results["backends"]["onnxruntime"][str(bs)]["p50_ms"],
            2,
        )
        for bs in batch_sizes
    }
    return results


def to_markdown(results: dict[str, Any]) -> str:
    lines = [
        "| backend | batch | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | throughput (seq/s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for backend, by_batch in results["backends"].items():
        for batch, stats in by_batch.items():
            lines.append(
                f"| {backend} | {batch} | {stats['p50_ms']} | {stats['p90_ms']} | "
                f"{stats['p95_ms']} | {stats['p99_ms']} | {stats['throughput_seq_per_s']} |"
            )

    speedups = ", ".join(f"batch {bs}: {v}x" for bs, v in results["speedup_p50"].items())
    hardware = results["hardware"]
    lines += [
        "",
        f"ONNX Runtime speedup over PyTorch eager at p50 — {speedups}.",
        "",
        f"Measured on {hardware['platform']}, {hardware['cpu_count']} vCPU, "
        f"torch {hardware.get('torch', 'n/a')}, onnxruntime "
        f"{hardware.get('onnxruntime', 'n/a')}. CPU only, no GPU involved.",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark PyTorch eager against ONNX Runtime.")
    p.add_argument("--torch-dir", type=Path, default=REPO_ROOT / "artifacts" / "run" / "model")
    p.add_argument("--onnx-dir", type=Path, default=REPO_ROOT / "artifacts" / "onnx")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--json-out", type=Path, default=REPO_ROOT / "docs" / "benchmark.json")
    p.add_argument("--markdown-out", type=Path, default=REPO_ROOT / "docs" / "benchmark.md")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)

    results = run_benchmark(
        ns.torch_dir,
        ns.onnx_dir,
        batch_sizes=tuple(ns.batch_sizes),
        iterations=ns.iterations,
        warmup=ns.warmup,
    )

    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    ns.json_out.write_text(json.dumps(results, indent=2))
    markdown = to_markdown(results)
    ns.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    ns.markdown_out.write_text(markdown + "\n")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
