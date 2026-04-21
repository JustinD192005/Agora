"""Arq worker — consumes jobs from Redis and processes them."""
from datetime import datetime, timezone
from uuid import UUID

import structlog
from arq.connections import RedisSettings
from sqlalchemy import select

from api.config import get_settings
from api.db import Event, Run, SessionLocal, Task
from worker.planner import generate_plan

log = structlog.get_logger()
settings = get_settings()


async def run_planner(ctx: dict, run_id: str) -> None:
    """Generate a research plan for a submitted question.

    Flow:
    1. Load the run, mark it as 'planning'
    2. Call Gemini via the planner module
    3. Write the plan as a Task row (kind='planner', output=plan_json)
    4. Mark the run as 'researching' — ready for Day 3's researcher fan-out

    Errors are caught and marked on the run; the worker doesn't crash
    on a bad LLM response.
    """
    run_uuid = UUID(run_id)
    log.info("planner.start", run_id=run_id)

    # Load the run
    async with SessionLocal() as session:
        result = await session.execute(select(Run).where(Run.id == run_uuid))
        run = result.scalar_one_or_none()
        if run is None:
            log.error("planner.run_not_found", run_id=run_id)
            return

        question = run.user_question
        run.status = "planning"
        session.add(Event(
            run_id=run_uuid,
            kind="planner_started",
            payload={"question": question},
        ))
        await session.commit()

    # Call the LLM (outside the DB session — don't hold a connection during network I/O)
    try:
        plan = await generate_plan(question)
    except Exception as e:
        log.exception("planner.failed", run_id=run_id, error=str(e))
        async with SessionLocal() as session:
            result = await session.execute(select(Run).where(Run.id == run_uuid))
            run = result.scalar_one()
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            session.add(Event(
                run_id=run_uuid,
                kind="planner_failed",
                payload={"error": str(e)},
            ))
            await session.commit()
        return

    # Persist the plan
    plan_dict = plan.model_dump(mode="json")
    log.info(
        "planner.plan_ready",
        run_id=run_id,
        num_sub_questions=len(plan.sub_questions),
    )

    async with SessionLocal() as session:
        # Reload the run inside this session
        result = await session.execute(select(Run).where(Run.id == run_uuid))
        run = result.scalar_one()

        # Create a Task row for the planner itself, with the plan as its output
        planner_task = Task(
            run_id=run_uuid,
            kind="planner",
            input={"question": question},
            output=plan_dict,
            status="completed",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        session.add(planner_task)

        # Log a plan_created event with the full plan for observability
        session.add(Event(
            run_id=run_uuid,
            kind="plan_created",
            payload=plan_dict,
        ))

        # Advance run status — Day 3 will trigger researchers from here
        run.status = "researching"
        await session.commit()

    log.info("planner.done", run_id=run_id)


class WorkerSettings:
    """Arq worker configuration — arq imports this class by path."""
    functions = [run_planner]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10
    job_timeout = 60