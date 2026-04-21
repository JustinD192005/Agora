"""FastAPI application — accepts run submissions and enqueues planner jobs."""
from contextlib import asynccontextmanager
from uuid import UUID

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from api.config import get_settings
from api.db import Event, Run, SessionLocal

log = structlog.get_logger()
settings = get_settings()


# ---------- Lifespan: create the arq Redis pool once at startup ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    app.state.arq = await create_pool(redis_settings)
    log.info("api.startup", redis=settings.redis_url)
    yield
    await app.state.arq.aclose()
    log.info("api.shutdown")


app = FastAPI(title="Agora", lifespan=lifespan)


# ---------- Request/response schemas ----------

class SubmitRunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)


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


# ---------- Endpoints ----------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/runs", response_model=RunResponse, status_code=201)
async def submit_run(payload: SubmitRunRequest) -> RunResponse:
    async with SessionLocal() as session:
        run = Run(user_question=payload.question, status="pending")
        session.add(run)
        await session.flush()

        # Log a run_created event
        session.add(Event(
            run_id=run.id,
            kind="run_created",
            payload={"question": payload.question},
        ))
        await session.commit()
        await session.refresh(run)

    # Enqueue planner job
    await app.state.arq.enqueue_job("run_planner", str(run.id))

    log.info("run.submitted", run_id=str(run.id), question=payload.question)
    return RunResponse.from_orm(run)


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: UUID) -> RunResponse:
    async with SessionLocal() as session:
        result = await session.execute(select(Run).where(Run.id == run_id))
        run = result.scalar_one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunResponse.from_orm(run)