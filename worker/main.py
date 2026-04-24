"""Arq worker — consumes jobs from Redis and processes them."""
from datetime import datetime, timezone, timedelta
from uuid import UUID

import structlog
from arq.connections import RedisSettings
from sqlalchemy import select

from api.config import get_settings
from api.db import Event, Run, SessionLocal, Task
from worker.fan_in import try_enqueue_synthesizer
from worker.planner import generate_plan
from worker.researcher import run_research_loop
from worker.synthesizer import SubQuestionResult, synthesize

log = structlog.get_logger()
settings = get_settings()

# --- Pacing config ---
# Spacing (in seconds) between researcher enqueue times. Each researcher gets
# its start deferred by idx * RESEARCHER_STAGGER_SECONDS, so opening LLM calls
# are spread over a window rather than bursting in a single second.
RESEARCHER_STAGGER_SECONDS = 3


# ============================================================
# Planner (Day 2 + Day 4 fan-out)
# ============================================================

async def run_planner(ctx: dict, run_id: str) -> None:
    """Generate a research plan for a submitted question, then fan out researchers."""
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
    num_sub_questions = len(plan.sub_questions)
    log.info("planner.plan_ready", run_id=run_id, num_sub_questions=num_sub_questions)

    # --- Persist plan, advance run status, and record expected researcher count ---
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
        run.expected_researchers = num_sub_questions
        await session.commit()

    # Fan out with pacing: stagger researcher starts by RESEARCHER_STAGGER_SECONDS
    # per job to avoid burst-hammering the LLM's rate limiter. Researchers still
    # run in parallel overall, but their opening LLM calls don't all hit the
    # upstream API in the same second.
    redis = ctx["redis"]
    enqueued_jobs: list[str] = []
    for idx in range(num_sub_questions):
        job = await redis.enqueue_job(
            "run_researcher",
            run_id,
            idx,
            _defer_by=timedelta(seconds=idx * RESEARCHER_STAGGER_SECONDS),
        )
        enqueued_jobs.append(job.job_id)

    log.info(
        "planner.fanned_out",
        run_id=run_id,
        num_researchers=num_sub_questions,
        job_ids=enqueued_jobs,
    )

    async with SessionLocal() as session:
        session.add(Event(
            run_id=run_uuid,
            kind="researchers_enqueued",
            payload={
                "num_researchers": num_sub_questions,
                "job_ids": enqueued_jobs,
            },
        ))
        await session.commit()

    log.info("planner.done", run_id=run_id)


# ============================================================
# Researcher (Day 3 + Day 4 fan-in trigger)
# ============================================================

async def run_researcher(ctx: dict, run_id: str, sub_question_index: int) -> None:
    """Answer one sub-question from a run's plan."""
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

    # --- Run the loop ---
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
        try:
            await try_enqueue_synthesizer(run_id, ctx["redis"])
        except Exception as fan_in_exc:
            log.exception("researcher.fan_in_failed_after_crash", run_id=run_id, error=str(fan_in_exc))
        return

    # --- Persist the mini-report ---
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

    # --- Fan-in check ---
    try:
        await try_enqueue_synthesizer(run_id, ctx["redis"])
    except Exception as e:
        log.exception("researcher.fan_in_failed", run_id=run_id, error=str(e))


# ============================================================
# Synthesizer (Day 4 Phase 3 — real implementation)
# ============================================================

async def run_synthesizer(ctx: dict, run_id: str) -> None:
    """Generate the final synthesized answer from all mini-reports.

    Reads the planner task (for the plan + interpretation) and every
    researcher task (for mini-reports). Calls Gemini to produce the
    final answer, persists it as the run's final_answer, and marks
    the run completed.
    """
    run_uuid = UUID(run_id)
    log.info("synthesizer.start", run_id=run_id)

    # --- Load everything we need: run, planner task, researcher tasks ---
    async with SessionLocal() as session:
        run_result = await session.execute(select(Run).where(Run.id == run_uuid))
        run = run_result.scalar_one()

        planner_result = await session.execute(
            select(Task).where(Task.run_id == run_uuid, Task.kind == "planner")
        )
        planner_task = planner_result.scalar_one()

        researcher_result = await session.execute(
            select(Task)
            .where(Task.run_id == run_uuid, Task.kind == "researcher")
            .order_by(Task.started_at)
        )
        researcher_tasks = list(researcher_result.scalars().all())

    question = run.user_question
    plan = planner_task.output or {}
    interpretation = plan.get("interpretation", "")

    # --- Convert researcher task rows into SubQuestionResult objects ---
    results: list[SubQuestionResult] = []
    for t in researcher_tasks:
        output = t.output or {}
        input_ = t.input or {}
        results.append(SubQuestionResult(
            sub_question=input_.get("sub_question", "(unknown)"),
            approach=input_.get("approach"),
            summary=output.get("summary", "(no summary — researcher did not complete)"),
            citations=output.get("citations", []),
            terminated_reason=output.get("terminated_reason", "error"),
            iterations=output.get("iterations", 0),
        ))

    log.info(
        "synthesizer.inputs_loaded",
        run_id=run_id,
        num_mini_reports=len(results),
        num_successful=sum(1 for r in results if r.terminated_reason == "finish"),
    )

    # --- Call Gemini to synthesize ---
    try:
        report = await synthesize(
            question=question,
            interpretation=interpretation,
            results=results,
        )
    except Exception as e:
        log.exception("synthesizer.failed", run_id=run_id, error=str(e))
        async with SessionLocal() as session:
            run_result = await session.execute(select(Run).where(Run.id == run_uuid))
            run = run_result.scalar_one()
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            session.add(Event(
                run_id=run_uuid,
                kind="synthesizer_failed",
                payload={"error": str(e)[:500]},
            ))
            await session.commit()
        return

    report_dict = report.model_dump(mode="json")
    log.info(
        "synthesizer.report_ready",
        run_id=run_id,
        answer_length=len(report.answer),
        num_citations=len(report.citations),
    )

    # --- Persist: a Task row for observability, final_answer on the run ---
    async with SessionLocal() as session:
        run_result = await session.execute(select(Run).where(Run.id == run_uuid))
        run = run_result.scalar_one()

        synth_task = Task(
            run_id=run_uuid,
            kind="synthesizer",
            input={"num_mini_reports": len(results)},
            output=report_dict,
            status="completed",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        session.add(synth_task)

        run.status = "completed"
        run.final_answer = report.answer
        run.completed_at = datetime.now(timezone.utc)

        session.add(Event(
            run_id=run_uuid,
            kind="synthesis_complete",
            payload={
                "answer_length": len(report.answer),
                "num_citations": len(report.citations),
                "coverage_summary": [
                    {"sub_question": c.sub_question[:80], "coverage": c.coverage}
                    for c in report.coverage
                ],
            },
        ))
        await session.commit()

    log.info("synthesizer.done", run_id=run_id)


# ============================================================
# Arq config
# ============================================================

class WorkerSettings:
    """Arq worker configuration — arq imports this class by path."""
    functions = [run_planner, run_researcher, run_synthesizer]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10
    job_timeout = 300