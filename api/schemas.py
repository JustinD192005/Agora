"""Response schemas for the read-side API used by the frontend.

Kept separate from api/db.py (ORM models) and api/main.py's request models so
the frontend has a stable, narrow contract that doesn't leak DB columns.
"""
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ---------- list ----------

class RunListItem(BaseModel):
    id: UUID
    question: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    expected_researchers: int | None
    duration_seconds: float | None


class RunListResponse(BaseModel):
    items: list[RunListItem]
    total: int
    limit: int
    offset: int


# ---------- full state ----------

class PlanSubQuestion(BaseModel):
    question: str
    approach: str | None = None
    rationale: str | None = None


class PlanOut(BaseModel):
    interpretation: str | None = None
    sub_questions: list[PlanSubQuestion] = Field(default_factory=list)


class CitationOut(BaseModel):
    url: str
    quote: str
    supports: str | None = None


class ResearcherReportOut(BaseModel):
    sub_question_index: int
    sub_question: str
    approach: str | None = None
    status: str
    iterations: int
    terminated_reason: str
    summary: str
    confidence_notes: str | None = None
    citations: list[CitationOut]
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None = None


class CoverageNoteOut(BaseModel):
    sub_question: str
    coverage: str
    note: str


class SourceDisagreementOut(BaseModel):
    topic: str
    claim_a: str
    claim_b: str
    sources_a: list[str]
    sources_b: list[str]
    notes: str | None = None


class SynthesisOut(BaseModel):
    answer: str
    citations: list[CitationOut]
    coverage: list[CoverageNoteOut]
    source_disagreements: list[SourceDisagreementOut]
    caveats: str | None = None


class RunFullResponse(BaseModel):
    id: UUID
    question: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None
    expected_researchers: int | None
    completed_researchers: int
    failed_researchers: int
    plan: PlanOut | None
    researchers: list[ResearcherReportOut]
    synthesis: SynthesisOut | None
    final_answer: str | None


# ---------- events ----------

class EventOut(BaseModel):
    id: int
    run_id: UUID
    task_id: UUID | None
    ts: datetime
    kind: str
    payload: dict[str, Any]


class EventsResponse(BaseModel):
    items: list[EventOut]
    next_since_id: int | None
