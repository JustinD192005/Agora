
"""Researcher agent — answers a single sub-question via a ReAct tool-use loop.

Flow per iteration:
  1. Ask Groq (Llama 3.3 70B) which tool to call next, given the conversation so far
  2. Dispatch to the tool
  3. Append the tool result as an observation
  4. If the tool was `finish`, return the mini-report
  5. Otherwise loop

Hard cap: 8 iterations. After that, we force-finish with whatever the agent has.

Design notes:
- We use instructor's TOOLS mode — Groq sees the tool input schemas as
  OpenAI-style function specs and chooses one per turn.
- Every LLM call and tool call is logged and returned as structured trace events,
  so the worker can persist them for the Week 4 dashboard.
- All tool errors become observations the agent can reason about — we never
  crash the loop on a flaky URL or a malformed query.
"""
from typing import Literal, Union

import structlog
from pydantic import BaseModel, Field

from api.llm import GROQ_LLAMA, call_structured_groq
from worker.tools import (
    CalculatorInput,
    FinishInput,
    WebFetchInput,
    WebSearchInput,
    calculator,
    web_fetch,
    web_search,
)

log = structlog.get_logger()


# ============================================================
# Tool choice schema — what the LLM outputs each turn
# ============================================================

# The LLM picks exactly one tool per iteration. We use a Pydantic
# discriminated union so instructor can generate the correct function-
# calling spec, and we get a typed result we can dispatch on.

class SearchAction(BaseModel):
    """Run web_search."""
    tool: Literal["web_search"] = "web_search"
    input: WebSearchInput


class FetchAction(BaseModel):
    """Run web_fetch."""
    tool: Literal["web_fetch"] = "web_fetch"
    input: WebFetchInput


class CalculatorAction(BaseModel):
    """Run calculator."""
    tool: Literal["calculator"] = "calculator"
    input: CalculatorInput


class FinishAction(BaseModel):
    """Terminal tool — call this when you have enough to answer."""
    tool: Literal["finish"] = "finish"
    input: FinishInput


# Discriminated union — instructor will expose all four as callable tools
# and return exactly one per LLM call.
ToolAction = Union[SearchAction, FetchAction, CalculatorAction, FinishAction]


class AgentChoice(BaseModel):
    """What the LLM returns on each turn.

    thought: brief reasoning (for observability, not shown to user)
    action: which tool to call + its input
    """
    thought: str = Field(
        description="One sentence: why you're choosing this tool right now. "
                    "Helps you reason step-by-step.",
        max_length=400,
    )
    action: ToolAction


# ============================================================
# Output schemas — what the researcher produces
# ============================================================

class MiniReport(BaseModel):
    """The final answer for one sub-question."""
    sub_question: str
    summary: str
    citations: list[dict]  # list of {url, quote} — using dict to keep serialization simple
    confidence_notes: str
    iterations: int
    terminated_reason: Literal["finish", "iteration_cap", "error"]
    trace: list[dict]  # list of {kind, payload} events for the dashboard


# ============================================================
# Prompts
# ============================================================

RESEARCHER_SYSTEM = """You are a Researcher agent in Agora, a multi-agent research system.

Your job: answer ONE focused sub-question using the tools available to you.

AVAILABLE TOOLS (pick exactly one per turn):
- web_search(query): find relevant URLs and snippets
- web_fetch(url): get the main text of a specific page
- calculator(expression): evaluate arithmetic
- finish(summary, citations, confidence_notes): terminal — call when ready to answer

STRATEGY:
1. Start with web_search to find candidate sources.
2. Pick 1-3 promising URLs and web_fetch them to read actual content.
3. Synthesize an answer grounded in what you actually read.
4. Call finish with a concise summary and citations pointing to URLs you fetched.

RULES:
- NEVER cite a URL you haven't fetched. Every citation must be a URL that appeared
  in a successful web_fetch result.
- Every citation quote must be a SHORT VERBATIM excerpt from fetched content.
  Don't paraphrase inside quote marks.
- If a search returns no useful results, try a different query before giving up.
- If a fetch fails, try a different URL. Don't get stuck on one source.
- Keep summaries to 2-5 sentences. Long summaries dilute useful information.
- You have at most 8 tool calls. Budget them. Don't search 5 times before fetching anything.
- Each turn, use the "thought" field to briefly justify your choice.

IMPORTANT TOOL FORMAT:

When selecting a tool, you MUST return:

{
  "thought": "...",
  "action": {
    "tool": "<one of: web_search | web_fetch | calculator | finish>",
    "input": { ... }   // MUST be an object matching the tool schema
  }
}

Examples:

web_search:
{
  "thought": "I need to find sources",
  "action": {
    "tool": "web_search",
    "input": { "query": "vector database comparison pinecone weaviate qdrant" }
  }
}

web_fetch:
{
  "thought": "I should read this article",
  "action": {
    "tool": "web_fetch",
    "input": { "url": "https://example.com" }
  }
}

finish:
{
  "thought": "I have enough information",
  "action": {
    "tool": "finish",
    "input": {
      "summary": "...",
      "citations": [...],
      "confidence_notes": "..."
    }
  }
}

STRICT RULES:
- ALWAYS include "input"
- NEVER flatten fields
- NEVER omit required fields
- Return EXACTLY this structure
"""


# ============================================================
# The loop
# ============================================================

MAX_ITERATIONS = 8


async def run_research_loop(sub_question: str) -> MiniReport:
    """Drive the agent through its tool-use loop until finish or cap."""
    log.info("researcher.start", sub_question=sub_question[:100])

    messages: list[dict] = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": f"Sub-question: {sub_question}"},
    ]

    trace: list[dict] = []
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        log.info("researcher.iteration", sub_question=sub_question[:80], iteration=iterations)

        # --- LLM step: ask Groq what to do next ---
        try:
            choice = await call_structured_groq(
                model=GROQ_LLAMA,
                messages=messages,
                response_model=AgentChoice,
            )
        except Exception as exc:
            log.exception("researcher.llm_failed", iteration=iterations)
            trace.append({
                "kind": "llm_failed",
                "payload": {"iteration": iterations, "error": str(exc)[:300]},
            })
            return _emergency_report(sub_question, iterations, trace, reason="error")

        trace.append({
            "kind": "llm_choice",
            "payload": {
                "iteration": iterations,
                "thought": choice.thought,
                "tool": choice.action.tool,
            },
        })

        # --- Tool dispatch ---
        action = choice.action

        if isinstance(action, FinishAction):
            log.info("researcher.finish", iterations=iterations)
            trace.append({"kind": "tool_finish", "payload": action.input.model_dump()})
            return MiniReport(
                sub_question=sub_question,
                summary=action.input.summary,
                citations=[c.model_dump() for c in action.input.citations],
                confidence_notes=action.input.confidence_notes,
                iterations=iterations,
                terminated_reason="finish",
                trace=trace,
            )

        # Run the chosen tool and get an observation
        observation_text = await _dispatch_tool(action, trace)

        # Add both assistant intent and tool observation to conversation history
        messages.append({
            "role": "assistant",
            "content": f"Thought: {choice.thought}\nTool: {action.tool}\nInput: {action.input.model_dump_json()}",
        })
        messages.append({
            "role": "user",
            "content": f"Observation from {action.tool}:\n{observation_text}",
        })

    # --- Iteration cap hit ---
    log.warning("researcher.iteration_cap", sub_question=sub_question[:80])
    trace.append({"kind": "iteration_cap", "payload": {"iterations": iterations}})
    return _emergency_report(sub_question, iterations, trace, reason="iteration_cap")


async def _dispatch_tool(action: ToolAction, trace: list[dict]) -> str:
    """Run the chosen tool and return a text observation for the LLM.

    We deliberately serialize observations as compact strings rather than
    structured JSON. LLMs reason better over readable text than nested JSON.
    """
    if isinstance(action, SearchAction):
        result = await web_search(action.input)
        trace.append({"kind": "tool_call", "payload": {
            "tool": "web_search",
            "input": action.input.model_dump(),
            "result_summary": f"{len(result.results)} results, status={result.status}",
        }})
        if result.status == "error":
            return f"Search failed: {result.error}"
        if not result.results:
            return "Search returned zero results. Try a different query."
        lines = [f"Found {len(result.results)} results for '{result.query}':"]
        for i, r in enumerate(result.results, 1):
            lines.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
        return "\n".join(lines)

    if isinstance(action, FetchAction):
        result = await web_fetch(action.input)
        trace.append({"kind": "tool_call", "payload": {
            "tool": "web_fetch",
            "input": action.input.model_dump(),
            "result_summary": f"status={result.status}, length={result.content_length}",
        }})
        if result.status == "error":
            return f"Fetch failed: {result.error}"
        marker = " (truncated)" if result.truncated else ""
        return f"Fetched {result.url}{marker} ({result.content_length} chars):\n\n{result.content}"

    if isinstance(action, CalculatorAction):
        result = await calculator(action.input)
        trace.append({"kind": "tool_call", "payload": {
            "tool": "calculator",
            "input": action.input.model_dump(),
            "result_summary": f"status={result.status}",
        }})
        if result.status == "error":
            return f"Calculator error: {result.error}"
        return f"{action.input.expression} = {result.result}"

    return f"Unknown tool: {action}"


def _emergency_report(
    sub_question: str, iterations: int, trace: list[dict], reason: str,
) -> MiniReport:
    """Fallback when the loop exits without a proper finish call.

    Returning a structured 'no answer' is better than raising — the synthesizer
    (Day 4) can still include it and note that the researcher didn't converge.
    """
    return MiniReport(
        sub_question=sub_question,
        summary=f"The researcher did not reach a confident answer within {iterations} iterations.",
        citations=[],
        confidence_notes=f"Agent terminated due to: {reason}",
        iterations=iterations,
        terminated_reason=reason,
        trace=trace,
    ) 

