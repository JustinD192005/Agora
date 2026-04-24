"""LLM client — wraps Gemini and Groq with instructor for Pydantic-validated outputs.

All LLM calls in Agora go through this module. This gives us one place to:
- handle rate limits gracefully with provider-aware retry
- cache responses for deterministic replay
- track tokens and cost (future)
- route different stages to different providers (planner=Gemini, researcher=Groq)

Caching design:
- Every LLM call first checks the llm_cache table, keyed by a hash of the
  input (prompt/messages + model + temperature + max_tokens + response_model).
- Cache hit: return cached output, no network call.
- Cache miss: make the real call (with retry), store result, return.
- Pass use_cache=False to force a real call (used by the bust_cache endpoint flag).
- The `kind` parameter ("planner" | "synthesizer" | "researcher") segregates
  cache entries so coincidentally-identical prompts don't leak across stages.
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

from api.cache import LLM_TTL, get_cached, hash_input, set_cached
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


# ---------- Generic type var for response models ----------

T = TypeVar("T", bound=BaseModel)


# ============================================================
# Gemini — public (caching) + private (retry) layers
# ============================================================

async def call_structured_gemini(
    *,
    model: str,
    prompt: str,
    response_model: type[T],
    kind: str = "llm",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    use_cache: bool = True,
) -> T:
    """Call Gemini with a Pydantic response_model. Used by planner + synthesizer.

    If use_cache=True (default), checks the cache before calling Gemini.
    Cache key is the hash of (prompt, response_model name, temperature, max_tokens).
    Cache hit returns the cached output parsed back into the response_model.
    Cache miss makes the real call, stores the result, and returns it.

    Passing kind="planner" or kind="synthesizer" segregates cache entries by
    caller so coincidentally-identical prompts don't share cache across stages.
    """
    cache_payload = {
        "prompt": prompt,
        "response_model": response_model.__name__,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    input_hash = hash_input(cache_payload)

    if use_cache:
        cached = await get_cached(input_hash, kind=kind, model=model)
        if cached is not None:
            log.info(
                "llm.gemini.cache_hit",
                model=model,
                response_model=response_model.__name__,
                kind=kind,
            )
            return response_model.model_validate(cached)

    result = await _call_gemini_with_retry(
        model=model,
        prompt=prompt,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if use_cache:
        await set_cached(
            input_hash,
            kind=kind,
            model=model,
            output=result.model_dump(mode="json"),
            ttl=LLM_TTL,
        )

    return result


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _call_gemini_with_retry(
    *,
    model: str,
    prompt: str,
    response_model: type[T],
    temperature: float,
    max_tokens: int,
) -> T:
    """The actual Gemini call with tenacity retry. Separated from caching
    logic so retries only apply to real network calls, not cache lookups."""
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


# ============================================================
# Groq — public (caching) + private (retry) layers
# ============================================================

async def call_structured_groq(
    *,
    model: str,
    messages: list[dict],
    response_model: type[T],
    kind: str = "researcher",
    temperature: float = 0.2,
    max_tokens: int = 800,
    use_cache: bool = True,
) -> T:
    """Call Groq with a Pydantic response_model. Used by the researcher loop.

    If use_cache=True (default), checks the cache before calling Groq.
    Cache key is the hash of (messages, response_model name, temperature,
    max_tokens). Full conversation history is part of the key, so the same
    sub-question with identical tool-observation chain hits the cache.
    """
    cache_payload = {
        "messages": messages,
        "response_model": response_model.__name__,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    input_hash = hash_input(cache_payload)

    if use_cache:
        cached = await get_cached(input_hash, kind=kind, model=model)
        if cached is not None:
            log.info(
                "llm.groq.cache_hit",
                model=model,
                response_model=response_model.__name__,
                kind=kind,
            )
            return response_model.model_validate(cached)

    result = await _call_groq_with_retry(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if use_cache:
        await set_cached(
            input_hash,
            kind=kind,
            model=model,
            output=result.model_dump(mode="json"),
            ttl=LLM_TTL,
        )

    return result


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _call_groq_with_retry(
    *,
    model: str,
    messages: list[dict],
    response_model: type[T],
    temperature: float,
    max_tokens: int,
) -> T:
    """The actual Groq call with tenacity retry."""
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
            log.error("llm.groq.non_retryable", error_type=type(exc).__name__, error=str(exc)[:200])
            raise
        log.warning("llm.groq.retryable_error", error_type=type(exc).__name__)
        raise

    log.info("llm.groq.done", model=model, response_model=response_model.__name__)
    return result


# ---------- Model name constants ----------

GEMINI_FLASH = "gemini-2.5-flash"
GROQ_LLAMA = "llama-3.3-70b-versatile"


# ---------- Backward-compat alias ----------
# worker/planner.py still imports `call_structured`. Keep the old name working.
call_structured = call_structured_gemini