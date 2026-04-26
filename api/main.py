"""FastAPI application — accepts run submissions and enqueues planner jobs."""
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID
from fastapi.responses import Response
import asyncio 
import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select

from api.config import get_settings
from api.db import Event, Run, SessionLocal, Task

log = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    app.state.arq = await create_pool(redis_settings)
    log.info("api.startup", redis=settings.redis_url)

    # Optionally spawn arq worker inside this process (set START_WORKER=1)
    worker_task = None
    if os.getenv("START_WORKER") == "1":
        from arq.worker import create_worker
        from worker.main import WorkerSettings
        log.info("worker.starting_in_process")
        worker = create_worker(WorkerSettings)
        worker_task = asyncio.create_task(worker.async_run())
        app.state.worker = worker
        app.state.worker_task = worker_task

    yield

    if worker_task is not None:
        log.info("worker.stopping")
        await app.state.worker.close()
        worker_task.cancel()

    await app.state.arq.aclose()
    log.info("api.shutdown")


app = FastAPI(title="Agora", lifespan=lifespan)

import os

# Comma-separated list from env var, e.g. "http://localhost:3000,https://agora-app.vercel.app"
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request/response schemas ----------

class SubmitRunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    bust_cache: bool = Field(default=False)


class RunResponse(BaseModel):
    id: UUID
    question: str
    status: str
    final_answer: str | None

    @classmethod
    def from_orm(cls, run: Run) -> "RunResponse":
        return cls(
            id=run.id,
            question=run.user_question,
            status=run.status,
            final_answer=run.final_answer,
        )


class TaskResponse(BaseModel):
    id: UUID
    kind: str
    status: str
    input: dict
    output: dict | None
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None


class RunDetailResponse(BaseModel):
    id: UUID
    question: str
    status: str
    final_answer: str | None
    expected_researchers: int | None
    created_at: datetime
    completed_at: datetime | None
    tasks: list[TaskResponse]


# ---------- Endpoints ----------

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.post("/runs", response_model=RunResponse, status_code=201)
async def submit_run(payload: SubmitRunRequest) -> RunResponse:
    async with SessionLocal() as session:
        run = Run(
            user_question=payload.question,
            status="pending",
            metadata_={"bust_cache": payload.bust_cache},
        )
        session.add(run)
        await session.flush()

        session.add(Event(
            run_id=run.id,
            kind="run_created",
            payload={"question": payload.question, "bust_cache": payload.bust_cache},
        ))
        await session.commit()
        await session.refresh(run)

    await app.state.arq.enqueue_job("run_planner", str(run.id))

    log.info("run.submitted", run_id=str(run.id), question=payload.question)
    return RunResponse.from_orm(run)


@app.get("/runs", response_model=list[RunResponse])
async def list_runs() -> list[RunResponse]:
    """Return the 20 most recent runs for the history page."""
    async with SessionLocal() as session:
        result = await session.execute(
            select(Run).order_by(Run.created_at.desc()).limit(20)
        )
        runs = result.scalars().all()
        return [RunResponse.from_orm(r) for r in runs]


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: UUID) -> RunResponse:
    async with SessionLocal() as session:
        result = await session.execute(select(Run).where(Run.id == run_id))
        run = result.scalar_one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunResponse.from_orm(run)


@app.get("/runs/{run_id}/details", response_model=RunDetailResponse)
async def get_run_details(run_id: UUID) -> RunDetailResponse:
    """Return full run details including all task statuses.

    Used by the dashboard to show the pipeline state in real time.
    The frontend polls this every 2-3 seconds while a run is in progress.
    """
    async with SessionLocal() as session:
        run_result = await session.execute(select(Run).where(Run.id == run_id))
        run = run_result.scalar_one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        tasks_result = await session.execute(
            select(Task)
            .where(Task.run_id == run_id)
            .order_by(Task.started_at)
        )
        tasks = tasks_result.scalars().all()

    return RunDetailResponse(
        id=run.id,
        question=run.user_question,
        status=run.status,
        final_answer=run.final_answer,
        expected_researchers=run.expected_researchers,
        created_at=run.created_at,
        completed_at=run.completed_at,
        tasks=[
            TaskResponse(
                id=t.id,
                kind=t.kind,
                status=t.status,
                input=t.input or {},
                output=t.output,
                started_at=t.started_at,
                completed_at=t.completed_at,
                error=t.error,
            )
            for t in tasks
        ],
    )

@app.get("/runs/{run_id}/report.pdf")
async def download_report(run_id: UUID):
    """Generate and return a PDF research report for the given run."""
    async with SessionLocal() as session:
        run_result = await session.execute(select(Run).where(Run.id == run_id))
        run = run_result.scalar_one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        tasks_result = await session.execute(
            select(Task).where(Task.run_id == run_id).order_by(Task.started_at)
        )
        tasks = tasks_result.scalars().all()

    from api.pdf import generate_pdf

    run_data = {
        "id": str(run.id),
        "question": run.user_question,
        "status": run.status,
        "final_answer": run.final_answer,
        "tasks": [
            {
                "kind": t.kind,
                "status": t.status,
                "input": t.input or {},
                "output": t.output,
            }
            for t in tasks
        ],
    }

    pdf_bytes = generate_pdf(run_data)
    filename = f"agora-report-{str(run_id)[:8]}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )