"""Process-wide resources for the API.

The model is loaded once at startup and held here. Loading per request would
dominate latency -- a BioBERT-sized graph takes hundreds of milliseconds to
initialise, against single-digit milliseconds to run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from medextract.config import Settings

log = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    engine: Any | None = None
    searcher: Any | None = None
    started_at: float = field(default_factory=time.time)
    load_error: str | None = None
    retrieval_error: str | None = None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def model_loaded(self) -> bool:
        return self.engine is not None

    @property
    def retrieval_available(self) -> bool:
        return self.searcher is not None

    @property
    def labels(self) -> list[str]:
        if self.engine is None:
            return []
        return [self.engine.id2label[i] for i in sorted(self.engine.id2label)]


def load_state(settings: Settings) -> AppState:
    """Build application state, tolerating a missing model or vector store.

    A hard failure at import time would mean the container cannot start at all
    and /health cannot explain why. Instead the process comes up, /health
    reports "degraded" with the reason, and the affected endpoints return 503.
    """
    state = AppState(settings=settings)

    from medextract.inference import NerEngine

    try:
        started = time.perf_counter()
        state.engine = NerEngine(
            settings.model_dir,
            backend=settings.backend,  # type: ignore[arg-type]
            max_length=settings.max_sequence_length,
            stride=settings.stride,
        )
        log.info(
            "model loaded",
            extra={
                "model_dir": str(settings.model_dir),
                "backend": settings.backend,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
                "labels": state.labels,
            },
        )
    except Exception as exc:
        state.load_error = f"{type(exc).__name__}: {exc}"
        log.error("model failed to load", extra={"error": state.load_error})

    if settings.retrieval_enabled:
        try:
            from medextract.retrieval.search import Searcher

            state.searcher = Searcher.from_settings(settings)
            log.info("retrieval ready", extra={"collection": settings.qdrant_collection})
        except Exception as exc:
            state.retrieval_error = f"{type(exc).__name__}: {exc}"
            log.warning("retrieval unavailable", extra={"error": state.retrieval_error})

    return state
