"""Tools available to the researcher agent.

Each tool is a plain async function with:
- a Pydantic input schema (what the LLM must provide to call it)
- a Pydantic output schema (what the tool returns)
- a docstring (what the LLM sees to decide when to use it)

The input schemas double as the LLM's "function signatures" — instructor
generates the function-calling spec from them automatically.

Design principles:
- Each tool returns a structured result, never raises.
  Errors become a failure observation the agent can reason about.
- Hard caps on everything: search result count, fetch size, fetch timeout.
- Calculator is intentionally limited — no eval, no arbitrary code.
"""
import ast
import operator
from typing import Literal

import httpx
import structlog
import trafilatura
from pydantic import BaseModel, Field

from api.config import get_settings

log = structlog.get_logger()
_settings = get_settings()


# ============================================================
# web_search — Tavily
# ============================================================

class WebSearchInput(BaseModel):
    """Search the web for information."""
    query: str = Field(
        description="A focused search query. 3-8 words typically works best. "
                    "Use natural language, not operators."
    )


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    status: Literal["ok", "error"] = "ok"
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    error: str | None = None


async def web_search(input: WebSearchInput) -> WebSearchOutput:
    """Call Tavily's search API. Returns up to 5 results."""
    log.info("tool.web_search", query=input.query)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": _settings.tavily_api_key,
                    "query": input.query,
                    "max_results": 5,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:400],  # cap snippet length
            )
            for r in data.get("results", [])
        ]
        return WebSearchOutput(status="ok", query=input.query, results=results)

    except Exception as exc:
        log.warning("tool.web_search.failed", query=input.query, error=str(exc)[:200])
        return WebSearchOutput(
            status="error",
            query=input.query,
            error=f"Search failed: {type(exc).__name__}: {str(exc)[:200]}",
        )


# ============================================================
# web_fetch — httpx + trafilatura
# ============================================================

class WebFetchInput(BaseModel):
    """Fetch and extract the main text content from a URL."""
    url: str = Field(description="A full https:// URL to fetch.")


class WebFetchOutput(BaseModel):
    status: Literal["ok", "error"] = "ok"
    url: str
    content: str = ""
    content_length: int = 0
    truncated: bool = False
    error: str | None = None


# Hard cap on extracted text: 50KB. Most research needs don't exceed this,
# and longer content blows up LLM context and cost.
MAX_CONTENT_CHARS = 8000


async def web_fetch(input: WebFetchInput) -> WebFetchOutput:
    """Fetch a URL and extract main content using trafilatura."""
    log.info("tool.web_fetch", url=input.url)

    if not input.url.startswith(("http://", "https://")):
        return WebFetchOutput(
            status="error",
            url=input.url,
            error="URL must start with http:// or https://",
        )

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Agora Research Agent)"},
        ) as client:
            resp = await client.get(input.url)
            resp.raise_for_status()
            html = resp.text

        # trafilatura extracts the main article content, stripping nav/ads/etc
        extracted = trafilatura.extract(html) or ""

        truncated = len(extracted) > MAX_CONTENT_CHARS
        content = extracted[:MAX_CONTENT_CHARS]

        if not content:
            return WebFetchOutput(
                status="error",
                url=input.url,
                error="Page fetched but no readable content could be extracted",
            )

        return WebFetchOutput(
            status="ok",
            url=input.url,
            content=content,
            content_length=len(content),
            truncated=truncated,
        )

    except httpx.TimeoutException:
        return WebFetchOutput(status="error", url=input.url, error="Fetch timed out after 10s")
    except httpx.HTTPStatusError as exc:
        return WebFetchOutput(
            status="error",
            url=input.url,
            error=f"HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        log.warning("tool.web_fetch.failed", url=input.url, error=str(exc)[:200])
        return WebFetchOutput(
            status="error",
            url=input.url,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


# ============================================================
# calculator — safe expression eval
# ============================================================

class CalculatorInput(BaseModel):
    """Evaluate a mathematical expression."""
    expression: str = Field(
        description="A Python-style arithmetic expression. "
                    "Supported: + - * / ** % and parentheses. "
                    "Examples: '(100 * 1.08) ** 5', '2**20'"
    )


class CalculatorOutput(BaseModel):
    status: Literal["ok", "error"] = "ok"
    expression: str
    result: float | None = None
    error: str | None = None


# Whitelist of safe operators — nothing else is allowed.
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Walk an AST and evaluate only whitelisted arithmetic nodes."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


async def calculator(input: CalculatorInput) -> CalculatorOutput:
    """Evaluate an arithmetic expression using a whitelisted AST walker.

    This is NOT eval() — it's an explicit AST walker that only permits
    numeric literals and basic arithmetic operators. Function calls,
    attribute access, name lookups, etc. all raise ValueError.
    """
    log.info("tool.calculator", expression=input.expression)

    try:
        tree = ast.parse(input.expression, mode="eval")
        result = _safe_eval(tree.body)
        return CalculatorOutput(status="ok", expression=input.expression, result=result)
    except Exception as exc:
        return CalculatorOutput(
            status="error",
            expression=input.expression,
            error=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# finish — terminal tool
# ============================================================

class Citation(BaseModel):
    url: str = Field(description="The source URL this citation comes from.")
    quote: str = Field(
        description="A short verbatim quote from the source (< 200 chars) that "
                    "supports a claim in the summary.",
        max_length=400,
    )


class FinishInput(BaseModel):
    """Call this when you have enough information to answer the sub-question.

    Your summary should directly answer the sub-question in 2-5 sentences,
    grounded in the sources you've fetched. Include at least one citation.
    """
    summary: str = Field(
        description="A 2-5 sentence answer to the sub-question, in your own words.",
        max_length=2000,
    )
    citations: list[Citation] = Field(
        min_length=1,
        max_length=8,
        description="At least one citation. Each must be a URL you actually "
                    "fetched and a short quote from its content.",
    )
    confidence_notes: str = Field(
        default="",
        description="Optional: brief notes about confidence level, "
                    "e.g. 'Sources agree' or 'Only one source found'.",
        max_length=300,
    )


class FinishOutput(BaseModel):
    """Not called by the agent — the loop just unpacks FinishInput when it sees this tool."""
    summary: str
    citations: list[Citation]
    confidence_notes: str