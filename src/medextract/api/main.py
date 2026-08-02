"""FastAPI application: entity extraction and semantic retrieval."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from medextract import __version__
from medextract.api.logging_config import configure_logging, request_id_var
from medextract.api.schemas import (
    BatchExtractRequest,
    BatchExtractResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from medextract.api.state import AppState, load_state
from medextract.config import Settings, get_settings

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = getattr(app.state, "settings_override", None) or get_settings()
    configure_logging(settings.log_level)
    log.info("starting MedExtract", extra={"version": __version__})
    app.state.ctx = load_state(settings)
    yield
    log.info("shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="MedExtract",
        version=__version__,
        summary="Biomedical named entity extraction with semantic retrieval over PubMed.",
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings_override = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id, time the call, log the outcome."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "unhandled error",
                extra={"path": request.url.path, "method": request.method},
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        finally:
            request_id_var.reset(token)

        took_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "took_ms": took_ms,
                "request_id": request_id,
            },
        )
        return response

    register_routes(app)
    return app


def get_ctx(request: Request) -> AppState:
    return request.app.state.ctx


def require_engine(ctx: AppState = Depends(get_ctx)):
    if ctx.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {ctx.load_error or 'unknown reason'}",
        )
    return ctx.engine


def _check_length(text: str, settings: Settings) -> None:
    if len(text) > settings.max_input_chars:
        raise HTTPException(
            # Literal rather than a starlette constant: the name for 413 was
            # renamed between versions and this pins the wire behaviour.
            status_code=413,
            detail=(
                f"Input is {len(text)} characters; the limit is "
                f"{settings.max_input_chars}. Split the document and post the parts."
            ),
        )


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(ctx: AppState = Depends(get_ctx)) -> HealthResponse:
        """Liveness plus what is actually loaded.

        Returns 200 even when degraded: the container is running and can say
        why it is not useful, which is more diagnosable than a dead endpoint.
        """
        return HealthResponse(
            status="ok" if ctx.model_loaded else "degraded",
            model_loaded=ctx.model_loaded,
            model_version=ctx.settings.model_version,
            backend=ctx.settings.backend,
            labels=ctx.labels,
            retrieval_available=ctx.retrieval_available,
            uptime_seconds=round(ctx.uptime_seconds, 2),
            version=__version__,
        )

    @app.post("/extract", response_model=ExtractResponse, tags=["extraction"])
    def extract(
        payload: ExtractRequest,
        ctx: AppState = Depends(get_ctx),
        engine=Depends(require_engine),
    ) -> ExtractResponse:
        _check_length(payload.text, ctx.settings)
        started = time.perf_counter()
        entities = engine.predict_one(payload.text)
        took_ms = round((time.perf_counter() - started) * 1000, 2)

        kept = [e.as_dict() for e in entities if e.score >= payload.min_score]
        log.info(
            "extract",
            extra={"chars": len(payload.text), "entities": len(kept), "infer_ms": took_ms},
        )
        return ExtractResponse(
            entities=kept,
            text=payload.text,
            model_version=ctx.settings.model_version,
            took_ms=took_ms,
        )

    @app.post("/extract/batch", response_model=BatchExtractResponse, tags=["extraction"])
    def extract_batch(
        payload: BatchExtractRequest,
        ctx: AppState = Depends(get_ctx),
        engine=Depends(require_engine),
    ) -> BatchExtractResponse:
        for text in payload.texts:
            _check_length(text, ctx.settings)
        started = time.perf_counter()
        batches = engine.predict(payload.texts)
        took_ms = round((time.perf_counter() - started) * 1000, 2)
        return BatchExtractResponse(
            results=[
                [e.as_dict() for e in entities if e.score >= payload.min_score]
                for entities in batches
            ],
            model_version=ctx.settings.model_version,
            took_ms=took_ms,
        )

    @app.post("/search", response_model=SearchResponse, tags=["retrieval"])
    def search(
        payload: SearchRequest,
        ctx: AppState = Depends(get_ctx),
        engine=Depends(require_engine),
    ) -> SearchResponse:
        """Rank indexed abstracts against a query, then tag entities in the hits.

        This is retrieval plus extraction, not generation -- nothing here
        writes an answer, it returns the passages and what was found in them.
        """
        if ctx.searcher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Retrieval unavailable: {ctx.retrieval_error or 'not configured'}. "
                    "Bring up Qdrant and build the index with medextract-index."
                ),
            )

        started = time.perf_counter()
        hits = ctx.searcher.search(payload.query, top_k=payload.top_k)

        query_entities = [e.as_dict() for e in engine.predict_one(payload.query)]
        if payload.extract_entities and hits:
            per_hit = engine.predict([h["abstract"] for h in hits])
            for hit, entities in zip(hits, per_hit, strict=True):
                hit["entities"] = [e.as_dict() for e in entities]

        took_ms = round((time.perf_counter() - started) * 1000, 2)
        log.info("search", extra={"top_k": payload.top_k, "hits": len(hits), "took_ms": took_ms})
        return SearchResponse(
            query=payload.query,
            query_entities=query_entities,
            hits=hits,
            took_ms=took_ms,
        )


app = create_app()
