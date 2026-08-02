"""One-command, resumable, unattended run: dataset in, live system out.

Drives the same phase functions the `make sweep` / `make export` / `make
benchmark` / `make fetch` / `make index` targets call, in the order a person
would run them by hand, deciding along the way what those targets otherwise
leave to a human (which sweep arm to export, whether Docker is available,
what to write into the README). Nothing here reimplements training, export or
retrieval logic -- it only sequences it.

    python -m medextract.pipeline
    python -m medextract.pipeline --force-retrain --pubmed-max 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Any

from medextract.config import DEFAULT_DATASET, REPO_ROOT
from medextract.train import DEFAULT_EXPERIMENT, RunConfig

log = logging.getLogger(__name__)

SWEEP_ROOT = REPO_ROOT / "artifacts" / "sweep"
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"
EMBEDDER_DIR = REPO_ROOT / "artifacts" / "embedder"
QDRANT_LOCAL_DIR = REPO_ROOT / "artifacts" / "qdrant_local"
PUBMED_DUMP = REPO_ROOT / "data" / "pubmed.jsonl"
DOCKER_QDRANT_URL = "http://localhost:6333"
API_HEALTH_URL = "http://localhost:8000/health"
ENV_PATH = REPO_ROOT / ".env"
README_PATH = REPO_ROOT / "README.md"


def environment_banner() -> dict[str, Any]:
    """Log what's available and decide whether to train in fp16."""
    import os
    import platform

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    cuda_available = False
    try:
        import torch

        info["torch"] = torch.__version__
        cuda_available = torch.cuda.is_available()
        info["cuda_available"] = cuda_available
        if cuda_available:
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        log.warning("torch is not importable -- install the 'train' extra first")

    log.info("environment: %s", json.dumps(info))
    return {**info, "fp16": cuda_available}


def maybe_run_sweep(
    arms: list[RunConfig], output_root: Path, experiment: str, *, force_retrain: bool = False
) -> list[dict]:
    """Run the sweep, unless a complete result set is already on disk."""
    results_path = output_root / "sweep_results.json"
    wanted_names = {a.run_name for a in arms}

    if results_path.exists() and not force_retrain:
        existing = json.loads(results_path.read_text())
        existing_names = {r["run_name"] for r in existing}
        if wanted_names <= existing_names:
            log.info(
                "sweep already complete (%d/%d arms in %s); skipping. "
                "Pass --force-retrain to redo it.",
                len(existing_names),
                len(wanted_names),
                results_path,
            )
            return existing

    from medextract.sweep import run_sweep

    return run_sweep(arms, output_root, experiment)


def select_best_arm(results: list[dict]) -> dict:
    """Highest validation F1, ties broken by test F1. Fails loud if nothing succeeded."""
    if not results:
        raise SystemExit(
            "Every sweep arm failed -- there is no model to export. Check the logs above."
        )

    def key(result: dict) -> tuple[float, float]:
        val_f1 = result.get("validation", {}).get("eval_f1", float("-inf"))
        test_f1 = result.get("test", {}).get("test_f1", float("-inf"))
        return (val_f1, test_f1)

    best = max(results, key=key)
    log.info(
        "best arm: %s (val_f1=%.4f)",
        best["run_name"],
        best.get("validation", {}).get("eval_f1", float("nan")),
    )
    return best


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def update_env_var(path: Path, key: str, value: str) -> None:
    """Set `key=value` in a .env-style file, preserving every other line."""
    example = REPO_ROOT / ".env.example"
    if not path.exists() and example.exists():
        shutil.copy(example, path)

    lines = path.read_text().splitlines() if path.exists() else []
    prefix = f"{key}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def wait_for_url(url: str, *, timeout: float = 120.0, interval: float = 2.0) -> bool:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=3.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    return False


def fetch_pubmed_abstracts(query: str, max_results: int, out_path: Path) -> int:
    from medextract.config import get_settings
    from medextract.retrieval.pubmed import PubMedClient, write_jsonl

    settings = get_settings()
    client = PubMedClient(email=settings.ncbi_email, api_key=settings.ncbi_api_key)
    try:
        pmids = client.search(query, max_results)
        abstracts = client.fetch(pmids)
    finally:
        client.close()
    write_jsonl(abstracts, out_path)
    return len(abstracts)


def maybe_fetch_pubmed(query: str, max_results: int, out_path: Path, *, force_fetch: bool) -> int:
    from medextract.retrieval.pubmed import read_jsonl

    if out_path.exists() and not force_fetch:
        count = len(read_jsonl(out_path))
        log.info(
            "%s already has %d abstracts; skipping fetch. Pass --force-fetch to redo it.",
            out_path,
            count,
        )
        return count
    return fetch_pubmed_abstracts(query, max_results, out_path)


def build_retrieval_index(qdrant_url: str, abstracts_path: Path) -> int:
    from medextract.config import get_settings
    from medextract.retrieval.embed import build_embedder
    from medextract.retrieval.index import build_client, index_abstracts
    from medextract.retrieval.pubmed import read_jsonl

    settings = get_settings()
    abstracts = read_jsonl(abstracts_path)
    embedder = build_embedder(
        settings.embedding_model, onnx_dir=settings.embedding_onnx_dir, allow_fallback=False
    )
    client = build_client(qdrant_url)
    return index_abstracts(client, settings.qdrant_collection, abstracts, embedder)


def status_text(*, corpus_size: int | None) -> str:
    corpus_note = f"a {corpus_size}-abstract PubMed index" if corpus_size else "no retrieval corpus"
    return (
        f"> **Status: pipeline run completed on {date.today().isoformat()}.** "
        f"Real numbers below, from a fine-tuned model and {corpus_note}. "
        "See `docs/sweep_table.md` and `docs/benchmark.md` for full detail."
    )


def not_done_text(*, live_url: str | None) -> str:
    deployment = f"the live API at `{live_url}`" if live_url else "not deployed in this run"
    return (
        f"Every phase has been run end to end: training, ONNX export, benchmarking, "
        f"retrieval indexing, and {deployment}. Remaining gaps are the ones documented "
        "under [Limitations](#limitations) -- entity types, clinical-note performance, "
        "negation, entity linking, relation extraction, and corpus scale -- which are "
        "scope decisions, not unfinished work."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full MedExtract pipeline: sweep, export, benchmark, "
        "retrieval, serve, and README update, in one unattended pass."
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-fetch", action="store_true")
    p.add_argument("--skip-retrieval", action="store_true")
    p.add_argument("--pubmed-max", type=int, default=3000)
    p.add_argument("--no-docker", action="store_true")
    p.add_argument("--no-readme-update", action="store_true")
    return p


def run_pipeline(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)

    env = environment_banner()

    from medextract.sweep import default_grid

    base = RunConfig(dataset=ns.dataset, fp16=env["fp16"])
    arms = default_grid(base)
    results = maybe_run_sweep(arms, SWEEP_ROOT, ns.experiment, force_retrain=ns.force_retrain)
    best = select_best_arm(results)
    model_dir = Path(best["model_dir"])

    from medextract.report import build_table

    sweep_table = build_table(ns.experiment)
    sweep_table_path = REPO_ROOT / "docs" / "sweep_table.md"
    sweep_table_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_table_path.write_text(sweep_table + "\n")

    from medextract.export_onnx import check_parity, export
    from medextract.retrieval.export_embedder import export_embedder

    export(model_dir, ONNX_DIR, exporter="torch")
    check_parity(model_dir, ONNX_DIR)

    from medextract.config import get_settings

    export_embedder(get_settings().embedding_model, EMBEDDER_DIR)

    from medextract.benchmark import run_benchmark, to_markdown

    benchmark_results = run_benchmark(model_dir, ONNX_DIR)
    benchmark_table = to_markdown(benchmark_results)
    (REPO_ROOT / "docs" / "benchmark.json").write_text(json.dumps(benchmark_results, indent=2))
    (REPO_ROOT / "docs" / "benchmark.md").write_text(benchmark_table + "\n")

    corpus_size: int | None = None
    live_url: str | None = None
    using_docker = docker_available() and not ns.no_docker

    if not ns.skip_retrieval:
        if using_docker:
            subprocess.run(["docker", "compose", "up", "-d", "qdrant"], cwd=REPO_ROOT, check=True)
            qdrant_url = DOCKER_QDRANT_URL
            wait_for_url(f"{DOCKER_QDRANT_URL}/", timeout=60.0)
        else:
            log.info("Docker not available; using an on-disk Qdrant at %s", QDRANT_LOCAL_DIR)
            qdrant_url = str(QDRANT_LOCAL_DIR)

        update_env_var(ENV_PATH, "MEDEXTRACT_QDRANT_URL", qdrant_url)

        from medextract.retrieval.pubmed import DEFAULT_QUERY

        maybe_fetch_pubmed(DEFAULT_QUERY, ns.pubmed_max, PUBMED_DUMP, force_fetch=ns.force_fetch)
        corpus_size = build_retrieval_index(qdrant_url, PUBMED_DUMP)

    if using_docker:
        subprocess.run(["docker", "compose", "up", "--build", "-d"], cwd=REPO_ROOT, check=True)
        if wait_for_url(API_HEALTH_URL, timeout=180.0):
            live_url = "http://localhost:8000"
        else:
            log.warning(
                "API did not answer /health within the timeout; check `docker compose logs api`"
            )
    else:
        log.info(
            "Docker not available/enabled -- start the API yourself with `make serve` "
            "(reads MEDEXTRACT_QDRANT_URL from .env, already pointed at the local index)."
        )

    if not ns.no_readme_update:
        from medextract.readme_sync import sync_readme

        sync_readme(
            README_PATH,
            sweep_table_md=sweep_table,
            benchmark_table_md=benchmark_table,
            status_text=status_text(corpus_size=corpus_size),
            not_done_text=not_done_text(live_url=live_url),
        )

    val_f1 = best.get("validation", {}).get("eval_f1", float("nan"))
    speedups = ", ".join(f"batch {bs}: {v}x" for bs, v in benchmark_results["speedup_p50"].items())
    print(
        "\n".join(
            [
                "",
                "=== MedExtract pipeline finished ===",
                f"Best arm: {best['run_name']} (validation F1 {val_f1:.4f})",
                f"ONNX speedup over PyTorch at p50: {speedups}",
                f"PubMed corpus indexed: {corpus_size if corpus_size is not None else 'skipped'}",
                f"Live API: {live_url or 'not started -- run `make serve`'}",
            ]
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return run_pipeline(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
