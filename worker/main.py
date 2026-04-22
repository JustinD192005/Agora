"""Arq worker — consumes jobs from Redis and processes them."""
from datetime import datetime, timezone
from uuid import UUID

import structlog
from arq.connections import RedisSettings
from sqlalchemy import select

from api.config import get_settings
from api.db import Event, Run, SessionLocal, Task
from worker.planner import generate_plan
from worker.researcher import run_research_loop

log = structlog.get_logger()
settings = get_settings()


# ============================================================
# Planner (unchanged from Day 2)
# ============================================================

async def run_planner(ctx: dict, run_id: str) -> None:
    """Generate a research plan for a submitted question."""
    run_uuid = UUID(run_id)
    log.info("planner.start", run_id=run_id)

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
                payload={"error": str(e)[:500]},
            ))
            await session.commit()
        return

    plan_dict = plan.model_dump(mode="json")
    log.info("planner.plan_ready", run_id=run_id, num_sub_questions=len(plan.sub_questions))

    async with SessionLocal() as session:
        result = await session.execute(select(Run).where(Run.id == run_uuid))
        run = result.scalar_one()

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

        session.add(Event(
            run_id=run_uuid,
            kind="plan_created",
            payload=plan_dict,
        ))

        run.status = "researching"
        await session.commit()

    log.info("planner.done", run_id=run_id)


# ============================================================
# Researcher (new for Day 3)
# ============================================================

async def run_researcher(ctx: dict, run_id: str, sub_question_index: int) -> None:
    """Answer one sub-question from a run's plan.

    Loads the plan from the planner's Task row, picks the sub-question
    at the given index, runs the ReAct loop, and persists the mini-report
    as its own Task row plus trace events.
    """
    run_uuid = UUID(run_id)
    log.info("researcher.task.start", run_id=run_id, sub_q_index=sub_question_index)

    # --- Load the plan ---
    async with SessionLocal() as session:
        result = await session.execute(
            select(Task).where(
                Task.run_id == run_uuid,
                Task.kind == "planner",
            )
        )
        planner_task = result.scalar_one_or_none()
        if planner_task is None:
            log.error("researcher.no_plan", run_id=run_id)
            return

        sub_questions = planner_task.output.get("sub_questions", [])
        if sub_question_index >= len(sub_questions):
            log.error(
                "researcher.bad_index",
                run_id=run_id,
                index=sub_question_index,
                available=len(sub_questions),
            )
            return

        sub_q = sub_questions[sub_question_index]
        sub_question_text = sub_q["question"]

    # --- Create a Task row for this researcher ---
    researcher_task_id = None
    async with SessionLocal() as session:
        task = Task(
            run_id=run_uuid,
            kind="researcher",
            input={
                "sub_question_index": sub_question_index,
                "sub_question": sub_question_text,
                "approach": sub_q.get("approach"),
            },
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        session.add(task)
        session.add(Event(
            run_id=run_uuid,
            kind="researcher_started",
            payload={"sub_question_index": sub_question_index, "sub_question": sub_question_text},
        ))
        await session.commit()
        await session.refresh(task)
        researcher_task_id = task.id

    # --- Run the loop (no DB session held during network I/O) ---
    try:
        report = await run_research_loop(sub_question_text)
    except Exception as e:
        log.exception("researcher.crashed", run_id=run_id, sub_q_index=sub_question_index)
        async with SessionLocal() as session:
            result = await session.execute(select(Task).where(Task.id == researcher_task_id))
            task = result.scalar_one()
            task.status = "failed"
            task.error = str(e)[:500]
            task.completed_at = datetime.now(timezone.utc)
            session.add(Event(
                run_id=run_uuid,
                task_id=researcher_task_id,
                kind="researcher_crashed",
                payload={"error": str(e)[:500]},
            ))
            await session.commit()
        return

    # --- Persist the mini-report and trace events ---
    report_dict = report.model_dump(mode="json")
    log.info(
        "researcher.task.done",
        run_id=run_id,
        sub_q_index=sub_question_index,
        iterations=report.iterations,
        terminated=report.terminated_reason,
        num_citations=len(report.citations),
    )

    async with SessionLocal() as session:
        result = await session.execute(select(Task).where(Task.id == researcher_task_id))
        task = result.scalar_one()
        task.output = report_dict
        task.status = "completed" if report.terminated_reason == "finish" else "completed_partial"
        task.completed_at = datetime.now(timezone.utc)

        # Persist each trace event as its own Event row for the dashboard
        for ev in report.trace:
            session.add(Event(
                run_id=run_uuid,
                task_id=researcher_task_id,
                kind=f"researcher.{ev['kind']}",
                payload=ev["payload"],
            ))

        session.add(Event(
            run_id=run_uuid,
            task_id=researcher_task_id,
            kind="researcher_finished",
            payload={
                "sub_question_index": sub_question_index,
                "terminated_reason": report.terminated_reason,
                "iterations": report.iterations,
                "num_citations": len(report.citations),
            },
        ))
        await session.commit()


# ============================================================
# Arq config
# ============================================================

class WorkerSettings:
    """Arq worker configuration — arq imports this class by path."""
    functions = [run_planner, run_researcher]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10
    job_timeout = 300  # researchers can take a while: 8 iterations × ~20s = ~160s worst case