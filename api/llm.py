"""LLM client — wraps Gemini with instructor for Pydantic-validated structured outputs.

All LLM calls in Agora go through this module. This gives us one place to:
- handle rate limits gracefully with provider-aware retry
- add caching / replay (Week 5)
- track tokens and cost (Week 5)
- swap providers (Week 2 adds Groq for researchers)
"""
import re
from typing import TypeVar

import instructor
import structlog
from google import genai
from google.genai.errors import ClientError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from api.config import get_settings

log = structlog.get_logger()
_settings = get_settings()


# ---------- Client setup ----------

# One shared Gemini client per process. The instructor wrapper gives us
# structured outputs — we pass a Pydantic model as response_model and get
# back a validated instance, with automatic retry on parse failure.
_gemini_raw = genai.Client(api_key=_settings.gemini_api_key)
gemini = instructor.from_genai(_gemini_raw, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)


# ---------- Rate-limit-aware exception classification ----------

def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient Gemini errors: 429 (rate limit) and 5xx (server).

    4xx errors other than 429 (bad requests, auth failures, schema issues)
    are NOT retryable — retrying will just burn quota with the same outcome.
    """
    if isinstance(exc, ClientError):
        # ClientError covers 4xx. We only retry 429.
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg
    # Network errors, timeouts, 5xx — retry
    return True


# ---------- Generic call helper ----------

T = TypeVar("T", bound=BaseModel)


@retry(
    stop=stop_after_attempt(4),
    # On 429, Gemini's free tier typically wants ~6s. Start at 8s, cap at 60s.
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type(Exception),  # we filter with _is_retryable below
    reraise=True,
)
async def call_structured(
    *,
    model: str,
    prompt: str,
    response_model: type[T],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> T:
    """Call an LLM and get back a validated Pydantic model.

    Retries transient failures (rate limits, network errors, 5xx) with
    exponential backoff. Pydantic validation failures are handled inside
    instructor — it re-prompts the model with the validation error.
    """
    log.info("llm.call.start", model=model, response_model=response_model.__name__)

    try:
        result = gemini.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
    except Exception as exc:
        if not _is_retryable(exc):
            log.error(
                "llm.call.non_retryable",
                model=model,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            raise
        # Extract the suggested retry delay if Gemini told us one — we log it
        # so it's visible in the worker output; tenacity's backoff handles the wait.
        delay = _extract_retry_delay(exc)
        log.warning(
            "llm.call.retryable_error",
            model=model,
            error_type=type(exc).__name__,
            suggested_delay_s=delay,
        )
        raise

    log.info("llm.call.done", model=model, response_model=response_model.__name__)
    return result


def _extract_retry_delay(exc: BaseException) -> float | None:
    """Parse Gemini's 'retryDelay' field from a 429 error, if present."""
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
    if match:
        return float(match.group(1))
    return None


# ---------- Model name constants ----------

# Centralized so we can swap models in one place.
GEMINI_FLASH = "gemini-2.5-flash"