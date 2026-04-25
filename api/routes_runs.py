"""Read-side endpoints used by the frontend.

The existing api/main.py exposes POST /runs, GET /runs/{id}, and GET /health.
This router adds the richer reads a UI needs:

- GET /runs                 paginated list with optional status filter
- GET /runs/{id}/full       full state — plan, researchers, synthesis, citations
- GET /runs/{id}/events     paginated event log with a since_id cursor
- GET /runs/{id}/stream     SSE stream of new events until the run is terminal

Polling-based SSE: we poll the events table once per second by id cursor.
A LISTEN/NOTIFY upgrade is possible later but would require touching every
INSERT site, which we're avoiding for now.
"""
import asyncio
import json
from datetime import datetime
from typing import AsyncIterator
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, func, select

from api.db import Event, Run, SessionLocal, Task
from api.schemas import (
    CitationOut,
    CoverageNoteOut,
    EventOut,
    EventsResponse,
    PlanOut,
    PlanSubQuestion,
    ResearcherReportOut,
    RunFullResponse,
    RunListItem,
    RunListResponse,
    SourceDisagreementOut,
    SynthesisOut,
)

log = structlog.get_logger()

router = APIRouter(prefix="/runs", tags=["runs"])


VALID_STATUSES = {
    "pending",
    "planning",
    "researching",
    "synthesizing",
    "completed",
    "failed",
}

TERMINAL_STATUSES = ("completed", "failed")


# ============================================================
# helpers
# ============================================================

def _duration(created_at: datetime, completed_at: datetime | None) -> float | None:
    if completed_at is None:
        return None
    return round((completed_at - created_at).total_seconds(), 2)


def _citation(d: dict) -> CitationOut:
    return CitationOut(
        url=d.get("url", ""),
        quote=d.get("quote", ""),
        supports=d.get("supports"),
    )


def _sse_frame(event: str, data: dict) -> bytes:
    """Format a single SSE frame. `data` is JSON-encoded."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode("utf-8")


# ============================================================
# GET /runs — list
# ============================================================

@router.get("", response_model=RunListResponse)
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="Filter by run status"),
) -> RunListResponse:
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Must be one of: {sorted(VALID_STATUSES)}",
        )

    async with SessionLocal() as session:
        list_stmt = select(Run)
        count_stmt = select(func.count()).select_from(Run)
        if status is not None:
            list_stmt = list_stmt.where(Run.status == status)
            count_stmt = count_stmt.where(Run.status == status)

        total = (await session.execute(count_stmt)).scalar_one()
        result = await session.execute(
            list_stmt.order_by(desc(Run.created_at)).limit(limit).offset(offset)
        )
        runs = list(result.scalars().all())

    items = [
        RunListItem(
            id=r.id,
            question=r.user_question,
            status=r.status,
            created_at=r.created_at,
            completed_at=r.completed_at,
            expected_researchers=r.expected_researchers,
            duration_seconds=_duration(r.created_at, r.completed_at),
        )
        for r in runs
    ]
    return RunListResponse(items=items, total=total, limit=limit, offset=offset)


# ============================================================
# GET /runs/{id}/full — rich state for a single run
# ============================================================

@router.get("/{run_id}/full", response_model=RunFullResponse)
async def get_run_full(run_id: UUID) -> RunFullResponse:
    async with SessionLocal() as session:
        run_result = await session.execute(select(Run).where(Run.id == run_id))
        run = run_result.scalar_one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        task_result = await session.execute(
            select(Task)
            .where(Task.run_id == run_id)
            .order_by(Task.started_at.asc().nulls_last())
        )
        tasks = list(task_result.scalars().all())

    planner_task = next((t for t in tasks if t.kind == "planner"), None)
    synth_task = next((t for t in tasks if t.kind == "synthesizer"), None)
    researcher_tasks = [t for t in tasks if t.kind == "researcher"]

    # --- plan ---
    plan: PlanOut | None = None
    if planner_task and planner_task.output:
        out = planner_task.output
        plan = PlanOut(
            interpretation=out.get("interpretation"),
            sub_questions=[
                PlanSubQuestion(
                    question=sq.get("question", ""),
                    approach=sq.get("approach"),
                    rationale=sq.get("rationale"),
                )
                for sq in out.get("sub_questions", [])
            ],
        )

    # --- researchers ---
    # Multiple task rows can exist for the same sub_question_index if a job was
    # ever retried. Keep the latest terminal one per index.
    by_index: dict[int, Task] = {}
    for t in researcher_tasks:
        idx = (t.input or {}).get("sub_question_index")
        if idx is None:
            continue
        existing = by_index.get(idx)
        if existing is None:
            by_index[idx] = t
            continue
        new_done = t.completed_at
        old_done = existing.completed_at
        if new_done and (old_done is None or new_done > old_done):
            by_index[idx] = t

    researchers: list[ResearcherReportOut] = []
    for idx in sorted(by_index.keys()):
        t = by_index[idx]
        inp = t.input or {}
        out = t.output or {}
        researchers.append(ResearcherReportOut(
            sub_question_index=idx,
            sub_question=inp.get("sub_question", "(unknown)"),
            approach=inp.get("approach"),
            status=t.status,
            iterations=out.get("iterations", 0),
            terminated_reason=out.get(
                "terminated_reason",
                "error" if t.status == "failed" else "unknown",
            ),
            summary=out.get("summary", ""),
            confidence_notes=out.get("confidence_notes"),
            citations=[_citation(c) for c in out.get("citations", [])],
            started_at=t.started_at,
            completed_at=t.completed_at,
            error=t.error,
        ))

    completed_count = sum(
        1 for r in researchers if r.status in ("completed", "completed_partial")
    )
    failed_count = sum(1 for r in researchers if r.status == "failed")

    # --- synthesis ---
    synthesis: SynthesisOut | None = None
    if synth_task and synth_task.output:
        out = synth_task.output
        synthesis = SynthesisOut(
            answer=out.get("answer", ""),
            citations=[_citation(c) for c in out.get("citations", [])],
            coverage=[
                CoverageNoteOut(
                    sub_question=c.get("sub_question", ""),
                    coverage=c.get("coverage", ""),
                    note=c.get("note", ""),
                )
                for c in out.get("coverage", [])
            ],
            source_disagreements=[
                SourceDisagreementOut(
                    topic=d.get("topic", ""),
                    claim_a=d.get("claim_a", ""),
                    claim_b=d.get("claim_b", ""),
                    sources_a=d.get("sources_a", []),
                    sources_b=d.get("sources_b", []),
                    notes=d.get("notes"),
                )
                for d in out.get("source_disagreements", [])
            ],
            caveats=out.get("caveats"),
        )

    return RunFullResponse(
        id=run.id,
        question=run.user_question,
        status=run.status,
        created_at=run.created_at,
        completed_at=run.completed_at,
        duration_seconds=_duration(run.created_at, run.completed_at),
        expected_researchers=run.expected_researchers,
        completed_researchers=completed_count,
        failed_researchers=failed_count,
        plan=plan,
        researchers=researchers,
        synthesis=synthesis,
        final_answer=run.final_answer,
    )


# ============================================================
# GET /runs/{id}/events — paginated event log
# ============================================================

@router.get("/{run_id}/events", response_model=EventsResponse)
async def list_events(
    run_id: UUID,
    since_id: int = Query(0, ge=0, description="Return events with id > since_id"),
    limit: int = Query(200, ge=1, le=1000),
) -> EventsResponse:
    async with SessionLocal() as session:
        run_check = await session.execute(select(Run.id).where(Run.id == run_id))
        if run_check.scalar_one_or_none() is None:
            raise HTTPException(status_code=404, detail="Run not found")

        result = await session.execute(
            select(Event)
            .where(Event.run_id == run_id, Event.id > since_id)
            .order_by(Event.id.asc())
            .limit(limit)
        )
        events = list(result.scalars().all())

    items = [
        EventOut(
            id=e.id,
            run_id=e.run_id,
            task_id=e.task_id,
            ts=e.ts,
            kind=e.kind,
            payload=e.payload,
        )
        for e in events
    ]
    next_since_id = items[-1].id if items else (since_id or None)
    return EventsResponse(items=items, next_since_id=next_since_id)


# ============================================================
# GET /runs/{id}/stream — SSE
# ============================================================

SSE_POLL_SECONDS = 1.0
SSE_IDLE_TIMEOUT_SECONDS = 600  # cap a single connection at 10 min of silence


@router.get("/{run_id}/stream")
async def stream_events(run_id: UUID, request: Request, since_id: int = 0):
    """Stream events as Server-Sent Events. Closes when the run reaches a
    terminal status (completed/failed) or the client disconnects.

    Frame schema:
      event: status   data: {"id": "...", "status": "researching"}
      event: event    data: {"id": 42, "kind": "...", "payload": {...}, ...}
      event: close    data: {"reason": "terminal" | "idle_timeout"}
    """

    async def gen() -> AsyncIterator[bytes]:
        # confirm run exists up front
        async with SessionLocal() as session:
            r = await session.execute(select(Run.status).where(Run.id == run_id))
            initial_status = r.scalar_one_or_none()

        if initial_status is None:
            yield _sse_frame("error", {"detail": "Run not found"})
            return

        cursor = since_id
        idle_seconds = 0.0

        # initial status push so the client has something to render immediately
        yield _sse_frame("status", {"id": str(run_id), "status": initial_status})

        while True:
            if await request.is_disconnected():
                return

            async with SessionLocal() as session:
                ev_result = await session.execute(
                    select(Event)
                    .where(Event.run_id == run_id, Event.id > cursor)
                    .order_by(Event.id.asc())
                    .limit(200)
                )
                new_events = list(ev_result.scalars().all())

                run_result = await session.execute(
                    select(Run.status).where(Run.id == run_id)
                )
                run_status = run_result.scalar_one_or_none()

            if new_events:
                idle_seconds = 0.0
                for e in new_events:
                    yield _sse_frame("event", {
                        "id": e.id,
                        "task_id": str(e.task_id) if e.task_id else None,
                        "ts": e.ts.isoformat(),
                        "kind": e.kind,
                        "payload": e.payload,
                    })
                cursor = new_events[-1].id
            else:
                idle_seconds += SSE_POLL_SECONDS
                # heartbeat — keeps proxies + the browser EventSource alive
                yield b": keep-alive\n\n"

            if run_status in TERMINAL_STATUSES:
                yield _sse_frame("status", {"id": str(run_id), "status": run_status})
                yield _sse_frame("close", {"reason": "terminal"})
                return

            if idle_seconds >= SSE_IDLE_TIMEOUT_SECONDS:
                yield _sse_frame("close", {"reason": "idle_timeout"})
                return

            await asyncio.sleep(SSE_POLL_SECONDS)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering of the stream
        },
    )
