"""Eval harness — runs Agora on benchmark questions and scores results.

Usage:
    .venv\\Scripts\\python.exe -m benchmarks.run_eval
    .venv\\Scripts\\python.exe -m benchmarks.run_eval --category comparison
    .venv\\Scripts\\python.exe -m benchmarks.run_eval --questions redis-vs-kafka,what-is-cap

The harness:
1. Loads questions from benchmarks/questions.yaml
2. For each question, submits to Agora's API (so we exercise the real pipeline)
3. Polls until the run completes (or fails or times out)
4. Pulls Agora's synthesizer output from the database
5. Calls the LLM judge to score the output
6. Writes a JSON report and a Markdown summary to benchmarks/reports/<timestamp>/

Design notes:
- Uses HTTP against a running Agora API server (default localhost:8000) rather
  than calling worker functions directly. Validates the user-facing pipeline,
  not just the internal code paths.
- Submits with bust_cache=False — we WANT cache hits for repeat eval runs.
  Cache hits make iterating on the judge prompt nearly free.
- Polls every 3 seconds with a 5-minute per-question timeout.
- Each eval run is its own timestamped directory so we can compare runs over time.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx
import structlog
import yaml
from sqlalchemy import select

from api.db import Run, SessionLocal, Task
from benchmarks.judge import EvalScores, score_run

log = structlog.get_logger()


# ============================================================
# Config
# ============================================================

API_BASE_URL = "http://localhost:8000"
QUESTIONS_PATH = Path("benchmarks/questions.yaml")
REPORTS_DIR = Path("benchmarks/reports")
POLL_INTERVAL_SECONDS = 3
RUN_TIMEOUT_SECONDS = 300  # 5 min per question

# Status values that mean "done, look at the result"
TERMINAL_RUN_STATUSES = ("completed", "failed")


# ============================================================
# Question loading
# ============================================================

def load_questions(
    questions_path: Path = QUESTIONS_PATH,
    category_filter: str | None = None,
    id_filter: list[str] | None = None,
) -> list[dict]:
    """Load benchmark questions, optionally filtered by category or id list."""
    with open(questions_path) as f:
        data = yaml.safe_load(f)

    questions = data["questions"]

    if category_filter:
        questions = [q for q in questions if q["category"] == category_filter]
    if id_filter:
        id_set = set(id_filter)
        questions = [q for q in questions if q["id"] in id_set]

    return questions


# ============================================================
# Agora API client
# ============================================================

async def submit_to_agora(client: httpx.AsyncClient, question: str) -> str:
    """Submit a question to Agora's API. Returns the run id."""
    resp = await client.post(
        f"{API_BASE_URL}/runs",
        json={"question": question, "bust_cache": False},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def wait_for_completion(run_id: str) -> str:
    """Poll the database until the run reaches a terminal status. Returns the final status."""
    run_uuid = UUID(run_id)
    elapsed = 0
    while elapsed < RUN_TIMEOUT_SECONDS:
        async with SessionLocal() as session:
            result = await session.execute(select(Run).where(Run.id == run_uuid))
            run = result.scalar_one_or_none()
            if run is None:
                raise RuntimeError(f"Run {run_id} disappeared from DB")
            if run.status in TERMINAL_RUN_STATUSES:
                return run.status

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    raise TimeoutError(f"Run {run_id} did not complete within {RUN_TIMEOUT_SECONDS}s")


async def fetch_synthesizer_output(run_id: str) -> dict | None:
    """Pull the synthesizer's output dict from the database."""
    run_uuid = UUID(run_id)
    async with SessionLocal() as session:
        result = await session.execute(
            select(Task).where(
                Task.run_id == run_uuid,
                Task.kind == "synthesizer",
            )
        )
        task = result.scalar_one_or_none()
        if task is None:
            return None
        return task.output


# ============================================================
# Per-question evaluation
# ============================================================

async def evaluate_question(
    client: httpx.AsyncClient,
    question_data: dict,
) -> dict:
    """Run Agora on one question and score the output. Returns a dict with results."""
    qid = question_data["id"]
    question_text = question_data["question"]

    log.info("eval.question.start", id=qid, category=question_data["category"])

    started_at = datetime.now(timezone.utc)
    result: dict = {
        "id": qid,
        "category": question_data["category"],
        "difficulty": question_data["expected_difficulty"],
        "question": question_text,
        "started_at": started_at.isoformat(),
        "run_id": None,
        "status": "pending",
        "error": None,
        "agora_output": None,
        "scores": None,
        "duration_seconds": None,
    }

    try:
        # 1. Submit to Agora
        run_id = await submit_to_agora(client, question_text)
        result["run_id"] = run_id

        # 2. Wait for completion
        final_status = await wait_for_completion(run_id)

        # 3. Pull the synthesizer output
        agora_output = await fetch_synthesizer_output(run_id)
        if agora_output is None:
            result["status"] = "no_output"
            result["error"] = (
                f"Run reached status={final_status} but synthesizer task has no output"
            )
            return result

        result["agora_output"] = agora_output

        # 4. Score it via the judge
        scores: EvalScores = await score_run(
            question=question_text,
            category=question_data["category"],
            difficulty=question_data["expected_difficulty"],
            expected_aspects=question_data["expected_aspects"],
            agora_output=agora_output,
        )
        result["scores"] = scores.model_dump()
        result["status"] = "scored"

    except Exception as exc:
        log.exception("eval.question.failed", id=qid, error=str(exc))
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    finally:
        ended_at = datetime.now(timezone.utc)
        result["duration_seconds"] = (ended_at - started_at).total_seconds()
        log.info(
            "eval.question.done",
            id=qid,
            status=result["status"],
            duration=result["duration_seconds"],
        )

    return result


# ============================================================
# Aggregation
# ============================================================

def aggregate(results: list[dict]) -> dict:
    """Compute aggregate stats from per-question results."""
    scored = [r for r in results if r["status"] == "scored"]
    n = len(scored)
    if n == 0:
        return {
            "n_total": len(results),
            "n_scored": 0,
            "n_failed": len(results),
            "averages": None,
        }

    def avg(field: str) -> float:
        return sum(r["scores"][field]["score"] for r in scored) / n

    averages = {
        "faithfulness": round(avg("faithfulness"), 2),
        "citation_quality": round(avg("citation_quality"), 2),
        "coverage": round(avg("coverage"), 2),
        "synthesis": round(avg("synthesis"), 2),
    }
    averages["overall"] = round(
        sum(averages.values()) / 4,
        2,
    )

    # Per-category averages
    by_category: dict[str, dict[str, float]] = {}
    categories = sorted({r["category"] for r in scored})
    for cat in categories:
        cat_scored = [r for r in scored if r["category"] == cat]
        cn = len(cat_scored)
        if cn == 0:
            continue
        cat_avg = {
            "faithfulness": round(sum(r["scores"]["faithfulness"]["score"] for r in cat_scored) / cn, 2),
            "citation_quality": round(sum(r["scores"]["citation_quality"]["score"] for r in cat_scored) / cn, 2),
            "coverage": round(sum(r["scores"]["coverage"]["score"] for r in cat_scored) / cn, 2),
            "synthesis": round(sum(r["scores"]["synthesis"]["score"] for r in cat_scored) / cn, 2),
            "n": cn,
        }
        cat_avg["overall"] = round(
            (cat_avg["faithfulness"] + cat_avg["citation_quality"]
             + cat_avg["coverage"] + cat_avg["synthesis"]) / 4,
            2,
        )
        by_category[cat] = cat_avg

    return {
        "n_total": len(results),
        "n_scored": n,
        "n_failed": len(results) - n,
        "averages": averages,
        "by_category": by_category,
    }


# ============================================================
# Report generation
# ============================================================

def render_markdown(report: dict) -> str:
    """Render the aggregate report as readable Markdown."""
    agg = report["aggregates"]
    lines: list[str] = []

    lines.append(f"# Agora Eval Report — {report['timestamp']}")
    lines.append("")
    lines.append(f"**Questions evaluated:** {agg['n_total']}")
    lines.append(f"**Successfully scored:** {agg['n_scored']}")
    lines.append(f"**Failed runs:** {agg['n_failed']}")
    lines.append("")

    if agg["averages"] is None:
        lines.append("## No questions scored — all runs failed.")
        return "\n".join(lines)

    avgs = agg["averages"]
    lines.append("## Aggregate scores (1-10 scale)")
    lines.append("")
    lines.append(f"- **Overall:** {avgs['overall']}")
    lines.append(f"- **Faithfulness:** {avgs['faithfulness']}")
    lines.append(f"- **Citation quality:** {avgs['citation_quality']}")
    lines.append(f"- **Coverage:** {avgs['coverage']}")
    lines.append(f"- **Synthesis:** {avgs['synthesis']}")
    lines.append("")

    if agg.get("by_category"):
        lines.append("## By category")
        lines.append("")
        lines.append("| Category | n | Overall | Faith | Cit | Cov | Synth |")
        lines.append("|---|---|---|---|---|---|---|")
        for cat, c in sorted(agg["by_category"].items()):
            lines.append(
                f"| {cat} | {c['n']} | {c['overall']} | {c['faithfulness']} "
                f"| {c['citation_quality']} | {c['coverage']} | {c['synthesis']} |"
            )
        lines.append("")

    lines.append("## Per-question results")
    lines.append("")
    for r in report["results"]:
        lines.append(f"### `{r['id']}` ({r['category']}, {r['difficulty']})")
        lines.append(f"**Question:** {r['question']}")
        lines.append("")
        if r["status"] != "scored":
            lines.append(f"**Status:** {r['status']}")
            if r.get("error"):
                lines.append(f"**Error:** `{r['error']}`")
            lines.append("")
            continue
        s = r["scores"]
        lines.append(
            f"**Scores:** overall: "
            f"{round((s['faithfulness']['score'] + s['citation_quality']['score'] + s['coverage']['score'] + s['synthesis']['score']) / 4, 2)} "
            f"| faith {s['faithfulness']['score']} | cit {s['citation_quality']['score']} "
            f"| cov {s['coverage']['score']} | synth {s['synthesis']['score']}"
        )
        lines.append(f"**Duration:** {r['duration_seconds']:.1f}s")
        if s.get("overall_notes"):
            lines.append(f"**Notes:** {s['overall_notes']}")
        lines.append("")

    return "\n".join(lines)


def write_report(results: list[dict], output_dir: Path) -> dict:
    """Build the report dict, write json + md, return the dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate(results)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregates": aggregates,
        "results": results,
    }

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    log.info("eval.report.written", json=str(json_path), md=str(md_path))
    return report


# ============================================================
# Main
# ============================================================

async def main(args: argparse.Namespace) -> None:
    questions = load_questions(
        category_filter=args.category,
        id_filter=args.questions.split(",") if args.questions else None,
    )
    if not questions:
        print("No questions matched the filter.", file=sys.stderr)
        sys.exit(1)

    log.info("eval.start", n_questions=len(questions))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = REPORTS_DIR / timestamp

    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(questions, 1):
            log.info("eval.progress", i=i, total=len(questions), id=q["id"])
            r = await evaluate_question(client, q)
            results.append(r)

            # Write incremental report after each question — if eval crashes
            # halfway, we still have partial data.
            write_report(results, output_dir)

    report = write_report(results, output_dir)
    agg = report["aggregates"]
    print()
    print("=" * 60)
    print(f"Eval complete: {agg['n_scored']}/{agg['n_total']} scored")
    if agg["averages"]:
        print(f"Overall: {agg['averages']['overall']} / 10")
        print(f"  Faith: {agg['averages']['faithfulness']}")
        print(f"  Cit:   {agg['averages']['citation_quality']}")
        print(f"  Cov:   {agg['averages']['coverage']}")
        print(f"  Synth: {agg['averages']['synthesis']}")
    print(f"Report: {output_dir}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Agora eval harness")
    p.add_argument(
        "--category",
        choices=["comparison", "definitional", "analytical", "time_sensitive", "adversarial"],
        help="Only run questions of this category",
    )
    p.add_argument(
        "--questions",
        help="Comma-separated list of question ids to run",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))