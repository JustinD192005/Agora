"""Manually enqueue a researcher job for testing Day 3.

Usage:
    uv run python scripts/enqueue_researcher.py <run_id> <sub_question_index>

Example:
    uv run python scripts/enqueue_researcher.py a6da6d29-6328-425a-8ef3-3138077ffdb5 0
"""
import asyncio
import sys
from arq import create_pool
from arq.connections import RedisSettings

from api.config import get_settings


async def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: enqueue_researcher.py <run_id> <sub_question_index>")
        sys.exit(1)

    run_id = sys.argv[1]
    sub_question_index = int(sys.argv[2])

    settings = get_settings()
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        job = await redis.enqueue_job("run_researcher", run_id, sub_question_index)
        print(f"Enqueued researcher job {job.job_id} for run={run_id} sub_q={sub_question_index}")
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())