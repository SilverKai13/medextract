"""Request and response schemas.

Response shapes are pinned here rather than built ad hoc in the handlers, so
the OpenAPI document is generated from the same definitions the handlers are
validated against.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": (
                    "Familial adenomatous polyposis is caused by germline mutations "
                    "in the APC gene."
                )
            }
        }
    )

    text: str = Field(..., min_length=1, description="Free biomedical or clinical prose.")
    min_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Drop entities scoring below this confidence."
    )


class BatchExtractRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class EntityOut(BaseModel):
    text: str = Field(..., description="The matched surface form.")
    label: str = Field(..., description="Entity type, e.g. Disease or Chemical.")
    start: int = Field(..., description="Character offset into the request text, inclusive.")
    end: int = Field(..., description="Character offset into the request text, exclusive.")
    score: float = Field(..., description="Mean softmax confidence over the span's tokens.")


class ExtractResponse(BaseModel):
    entities: list[EntityOut]
    # Echoed so a caller can confirm the offsets refer to the text it thinks
    # they do, without keeping its own copy.
    text: str
    model_version: str
    took_ms: float


class BatchExtractResponse(BaseModel):
    results: list[list[EntityOut]]
    model_version: str
    took_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    backend: str
    labels: list[str]
    retrieval_available: bool
    uptime_seconds: float
    version: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "targeted therapy for EGFR mutant lung cancer", "top_k": 5}
        }
    )

    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    # Entity extraction over hits costs a model call per abstract; make it
    # optional so a caller who only wants ranking does not pay for it.
    extract_entities: bool = True


class SearchHit(BaseModel):
    pmid: str
    title: str
    abstract: str
    score: float = Field(..., description="Cosine similarity from the vector index.")
    journal: str | None = None
    year: int | None = None
    entities: list[EntityOut] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    query_entities: list[EntityOut]
    hits: list[SearchHit]
    took_ms: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
