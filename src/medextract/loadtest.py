"""Latency under concurrency, measured from the outside.

`benchmark.py` times the model call in-process with one request in flight.
That is the right measurement for comparing backends and the wrong one for
answering "what will a caller see under load?" -- a p99 measured with one
request in flight says nothing about a p99 with sixteen.

This drives a running API over HTTP at a ramp of concurrency levels and
reports client-observed latency and throughput at each. Everything a caller
pays for is inside the timing: connection, JSON, tokenisation, windowing, the
model call, serialisation. It is reported separately from the backend
comparison on purpose; the two answer different questions.

    python -m medextract.loadtest --url http://localhost:8000 \
        --levels 1 2 4 8 16 --requests 200 --corpus data/pubmed.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from medextract.benchmark import hardware_info, percentiles
from medextract.config import REPO_ROOT
from medextract.export_onnx import PARITY_TEXTS

log = logging.getLogger(__name__)

DEFAULT_LEVELS = (1, 2, 4, 8, 16)
DEFAULT_REQUESTS = 200
DEFAULT_TIMEOUT_S = 60.0


def sample_corpus(path: Path, n: int, *, seed: int = 0) -> list[str]:
    """Pick abstracts from a fetched PubMed dump, so the payload is the real thing.

    Abstracts are what a caller of /extract actually sends. The parity
    sentences are short and would flatter every number here.
    """
    from medextract.retrieval.pubmed import read_jsonl

    abstracts = [a.abstract for a in read_jsonl(path) if a.abstract]
    if not abstracts:
        raise ValueError(f"{path} holds no abstracts with text")
    rng = random.Random(seed)
    return rng.sample(abstracts, min(n, len(abstracts)))


async def _worker(
    client: httpx.AsyncClient,
    texts: list[str],
    counter: dict[str, int],
    total: int,
    samples_ms: list[float],
    errors: list[str],
) -> None:
    """Send requests back to back until the shared budget is spent.

    A worker never has more than one request in flight, so `c` workers is
    exactly `c` concurrent requests -- a closed-loop model, which is what a
    fixed pool of upstream callers looks like.
    """
    while True:
        i = counter["next"]
        if i >= total:
            return
        counter["next"] = i + 1

        text = texts[i % len(texts)]
        started = time.perf_counter()
        try:
            response = await client.post("/extract", json={"text": text})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.status_code != 200:
                errors.append(f"HTTP {response.status_code}")
                continue
        except httpx.HTTPError as exc:
            errors.append(type(exc).__name__)
            continue
        samples_ms.append(elapsed_ms)


async def run_level(
    client: httpx.AsyncClient, texts: list[str], concurrency: int, total: int
) -> dict[str, Any]:
    counter = {"next": 0}
    samples_ms: list[float] = []
    errors: list[str] = []

    wall_started = time.perf_counter()
    await asyncio.gather(
        *(_worker(client, texts, counter, total, samples_ms, errors) for _ in range(concurrency))
    )
    wall_s = time.perf_counter() - wall_started

    stats: dict[str, Any] = percentiles(samples_ms) if samples_ms else {}
    stats.update(
        {
            "concurrency": concurrency,
            "requests": total,
            "completed": len(samples_ms),
            "errors": len(errors),
            "error_kinds": sorted(set(errors)),
            "wall_s": round(wall_s, 3),
            "throughput_req_per_s": round(len(samples_ms) / wall_s, 2) if wall_s else 0.0,
        }
    )
    return stats


async def run_loadtest(
    client: httpx.AsyncClient,
    texts: list[str],
    *,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    requests_per_level: int = DEFAULT_REQUESTS,
    warmup: int = 10,
) -> dict[str, Any]:
    """Ramp concurrency against an already-running service.

    `/health` is checked first and a degraded service is refused: timing a
    service that answers 503 would measure the error path, not the model.
    """
    health = (await client.get("/health")).json()
    if not health.get("model_loaded"):
        raise RuntimeError(f"service is degraded, refusing to time it: {health}")

    # Warm the server (first-call allocations, thread pool spin-up) and the
    # client's connection pool before anything is recorded.
    for i in range(warmup):
        await client.post("/extract", json={"text": texts[i % len(texts)]})

    results: dict[str, Any] = {
        "endpoint": "/extract",
        "model_version": health.get("model_version"),
        "backend": health.get("backend"),
        "requests_per_level": requests_per_level,
        "payload": {
            "texts": len(texts),
            "mean_chars": round(statistics.fmean(len(t) for t in texts), 1),
            "max_chars": max(len(t) for t in texts),
        },
        "client": hardware_info(),
        "levels": {},
    }
    for concurrency in levels:
        log.info("concurrency %d: %d requests", concurrency, requests_per_level)
        results["levels"][str(concurrency)] = await run_level(
            client, texts, concurrency, requests_per_level
        )
    return results


def to_markdown(results: dict[str, Any]) -> str:
    lines = [
        "| concurrency | requests | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) "
        "| throughput (req/s) | errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for concurrency, stats in results["levels"].items():
        lines.append(
            f"| {concurrency} | {stats['requests']} | {stats.get('p50_ms', '-')} | "
            f"{stats.get('p90_ms', '-')} | {stats.get('p95_ms', '-')} | "
            f"{stats.get('p99_ms', '-')} | {stats.get('max_ms', '-')} | "
            f"{stats['throughput_req_per_s']} | {stats['errors']} |"
        )

    payload = results["payload"]
    client = results["client"]
    lines += [
        "",
        f"Client-observed, end to end over HTTP against `{results['endpoint']}` on the "
        f"`{results['backend']}` backend, {payload['texts']} distinct texts cycled, "
        f"mean {payload['mean_chars']} characters (max {payload['max_chars']}). "
        "Closed loop: each concurrent caller waits for its response before sending "
        "the next.",
        "",
        f"Client on {client['platform']}, {client['cpu_count']} vCPU. Server and client "
        "share the machine, so these numbers are a ceiling on what this box can serve, "
        "not a measurement of the server alone.",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drive a running MedExtract API at rising concurrency.")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="per level")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="a fetched PubMed JSONL; abstracts are sampled as the payload",
    )
    p.add_argument("--sample", type=int, default=64, help="abstracts to sample from --corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--json-out", type=Path, default=REPO_ROOT / "docs" / "loadtest.json")
    p.add_argument("--markdown-out", type=Path, default=REPO_ROOT / "docs" / "loadtest.md")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns = build_parser().parse_args(argv)

    texts = sample_corpus(ns.corpus, ns.sample, seed=ns.seed) if ns.corpus else list(PARITY_TEXTS)

    async def _run() -> dict[str, Any]:
        pool = max(ns.levels)
        limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
        async with httpx.AsyncClient(base_url=ns.url, timeout=ns.timeout, limits=limits) as client:
            return await run_loadtest(
                client,
                texts,
                levels=tuple(ns.levels),
                requests_per_level=ns.requests,
                warmup=ns.warmup,
            )

    results = asyncio.run(_run())
    results["url"] = ns.url

    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    ns.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = to_markdown(results)
    ns.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    ns.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
