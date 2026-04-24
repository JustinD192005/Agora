"""LLM client — wraps Gemini and Groq with instructor for Pydantic-validated outputs.

All LLM calls in Agora go through this module. This gives us one place to:
- handle rate limits gracefully with provider-aware retry
- add caching / replay (Week 5)
- track tokens and cost (Week 5)
- route different stages to different providers (planner=Gemini, researcher=Groq)
"""
import re
from typing import TypeVar

import instructor
import structlog
from google import genai
from google.genai.errors import ClientError as GeminiClientError
from openai import OpenAI
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

# Gemini (used by the planner and synthesizer — high-quality structured outputs)
_gemini_raw = genai.Client(api_key=_settings.gemini_api_key)
gemini = instructor.from_genai(_gemini_raw, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

# Groq (used by the researcher — fast, good at tool use, generous free tier)
_groq_raw = OpenAI(
    api_key=_settings.groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
groq = instructor.from_openai(_groq_raw, mode=instructor.Mode.TOOLS)


# ---------- Rate-limit-aware retry predicate ----------

def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient errors: 429 (rate limit) and 5xx (server)."""
    if isinstance(exc, GeminiClientError):
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "timeout" in msg or "connection" in msg:
        return True
    return False


# ---------- Generic call helpers ----------

T = TypeVar("T", bound=BaseModel)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def call_structured_gemini(
    *,
    model: str,
    prompt: str,
    response_model: type[T],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> T:
    """Call Gemini with a Pydantic response_model. Used by planner + synthesizer."""
    log.info("llm.gemini.start", model=model, response_model=response_model.__name__)

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
            log.error("llm.gemini.non_retryable", error_type=type(exc).__name__)
            raise
        log.warning("llm.gemini.retryable_error", error_type=type(exc).__name__)
        raise

    log.info("llm.gemini.done", model=model, response_model=response_model.__name__)
    return result


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def call_structured_groq(
    *,
    model: str,
    messages: list[dict],
    response_model: type[T],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> T:
    """Call Groq with a Pydantic response_model. Used by the researcher loop."""
    log.info("llm.groq.start", model=model, response_model=response_model.__name__)

    try:
        result = groq.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        if not _is_retryable(exc):
            log.error("llm.groq.non_retryable", error_type=type(exc).__name__, error=str(exc)[:2000])
            raise
        log.warning("llm.groq.retryable_error", error_type=type(exc).__name__)
        raise

    log.info("llm.groq.done", model=model, response_model=response_model.__name__)
    return result


# ---------- Model name constants ----------

GEMINI_FLASH = "gemini-2.5-flash"
GROQ_LLAMA = "llama-3.3-70b-versatile"


# ---------- Backward-compat alias ----------
call_structured = call_structured_gemini