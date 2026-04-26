"""Researcher agent — answers a single sub-question via a ReAct tool-use loop.

Flow per iteration:
  1. Ask the researcher LLM which tool to call next, given the conversation so far
  2. Dispatch to the tool
  3. Append the tool result as an observation
  4. If the tool was `finish`, return the mini-report
  5. Otherwise loop

Hard cap: MAX_ITERATIONS. After that, we force-finish with whatever the agent has.

Design notes:
- We use instructor's TOOLS mode — the LLM sees the tool input schemas as
  OpenAI-style function specs and chooses one per turn.
- The schema uses a FLAT shape (tool + per-tool-input fields) rather than a
  nested discriminated union, because Groq's tool validator requires flat
  parameter shapes. The LLM picks one tool via the `tool` literal, then
  populates the corresponding `*_input` field.
- Every LLM call and tool call is logged and returned as structured trace events.
- All tool errors become observations the agent can reason about — we never
  crash the loop on a flaky URL or a malformed query.
- Older tool observations are compacted to short breadcrumbs after 2 turns, so
  per-iteration token usage stays roughly flat instead of growing linearly.
"""
from typing import Literal

import structlog
from pydantic import BaseModel, Field, model_validator

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

ToolName = Literal["web_search", "web_fetch", "calculator", "finish"]


class AgentChoice(BaseModel):
    """Flat tool-choice schema compatible with Groq's tool validator.

    The LLM picks ONE tool via `tool`, then fills in ONLY the matching
    `*_input` field. The other input fields should be left as null.

    We validate after-the-fact that the correct input field is populated,
    rather than using a discriminated union (which Groq's validator rejects
    as nested-object).
    """
    thought: str = Field(
        description="One sentence: why you're choosing this tool right now. "
                    "Helps you reason step-by-step.",
        max_length=400,
    )
    tool: ToolName = Field(
        description="Which tool to call. Exactly one of: web_search, web_fetch, calculator, finish."
    )
    # Populate ONLY the one matching the `tool` field above. Leave the others null.
    search_input: WebSearchInput | None = Field(
        default=None,
        description="REQUIRED if tool='web_search', otherwise null.",
    )
    fetch_input: WebFetchInput | None = Field(
        default=None,
        description="REQUIRED if tool='web_fetch', otherwise null.",
    )
    calculator_input: CalculatorInput | None = Field(
        default=None,
        description="REQUIRED if tool='calculator', otherwise null.",
    )
    finish_input: FinishInput | None = Field(
        default=None,
        description="REQUIRED if tool='finish', otherwise null.",
    )

    @model_validator(mode="after")
    def check_input_matches_tool(self) -> "AgentChoice":
        """Ensure the correct input field is populated for the chosen tool."""
        mapping = {
            "web_search": self.search_input,
            "web_fetch": self.fetch_input,
            "calculator": self.calculator_input,
            "finish": self.finish_input,
        }
        expected = mapping[self.tool]
        if expected is None:
            raise ValueError(
                f"tool='{self.tool}' but the matching input field is null. "
                f"You must populate the input for the chosen tool."
            )
        return self


# ============================================================
# Output schemas — what the researcher produces
# ============================================================

class MiniReport(BaseModel):
    """The final answer for one sub-question."""
    sub_question: str
    summary: str
    citations: list[dict]
    confidence_notes: str
    iterations: int
    terminated_reason: Literal["finish", "iteration_cap", "error"]
    trace: list[dict]


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
- You have at most 5 tool calls. Budget them. Don't search 5 times before fetching anything.
- Each turn, use the "thought" field to briefly justify your choice.

OUTPUT FORMAT (CRITICAL):
Each turn, you output:
  - thought: one sentence of reasoning
  - tool: one of "web_search", "web_fetch", "calculator", "finish"
  - the matching input field populated for the chosen tool:
      * if tool=web_search → populate search_input
      * if tool=web_fetch  → populate fetch_input
      * if tool=calculator → populate calculator_input
      * if tool=finish     → populate finish_input
  - leave the other input fields as null.

NOTE on history: older observations in this conversation may appear as compact
breadcrumbs like "Observation from web_fetch: [compacted — original was N chars]".
That just means you already consumed that content earlier. Don't re-fetch the same URL.
"""


# ============================================================
# The loop
# ============================================================

MAX_ITERATIONS = 8


async def run_research_loop(sub_question: str, use_cache: bool = True) -> MiniReport:
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

        # --- LLM step: ask the researcher LLM what to do next ---
        try:
            choice = await call_structured_groq(
                model=GROQ_LLAMA,
                messages=messages,
                response_model=AgentChoice,
                use_cache=use_cache,
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
                "tool": choice.tool,
            },
        })

        # --- Finish is terminal ---
        if choice.tool == "finish":
            log.info("researcher.finish", iterations=iterations)
            finish = choice.finish_input  # guaranteed non-None by the model_validator
            trace.append({"kind": "tool_finish", "payload": finish.model_dump()})
            return MiniReport(
                sub_question=sub_question,
                summary=finish.summary,
                citations=[c.model_dump() for c in finish.citations],
                confidence_notes=finish.confidence_notes,
                iterations=iterations,
                terminated_reason="finish",
                trace=trace,
            )

        # --- Tool dispatch ---
        observation_text = await _dispatch_tool(choice, trace, use_cache=use_cache)

        # Add both assistant intent and tool observation to conversation history
        # Serialize just the populated input, not the whole AgentChoice object
        input_for_history = _input_for_tool(choice)
        messages.append({
            "role": "assistant",
            "content": f"Thought: {choice.thought}\nTool: {choice.tool}\nInput: {input_for_history.model_dump_json()}",
        })
        messages.append({
            "role": "user",
            "content": f"Observation from {choice.tool}:\n{observation_text}",
        })

        # --- Compact older observations to save tokens ---
        _compact_old_observations(messages)

    # --- Iteration cap hit ---
    log.warning("researcher.iteration_cap", sub_question=sub_question[:80])
    trace.append({"kind": "iteration_cap", "payload": {"iterations": iterations}})
    return _emergency_report(sub_question, iterations, trace, reason="iteration_cap")


def _input_for_tool(choice: AgentChoice) -> BaseModel:
    """Pull out the populated input field for the chosen tool."""
    if choice.tool == "web_search":
        return choice.search_input
    if choice.tool == "web_fetch":
        return choice.fetch_input
    if choice.tool == "calculator":
        return choice.calculator_input
    if choice.tool == "finish":
        return choice.finish_input
    raise ValueError(f"Unknown tool: {choice.tool}")


async def _dispatch_tool(choice: AgentChoice, trace: list[dict], use_cache: bool = True) -> str:
    """Run the chosen tool and return a text observation for the LLM."""
    if choice.tool == "web_search":
        inp = choice.search_input
        result = await web_search(inp, use_cache=use_cache)
        trace.append({"kind": "tool_call", "payload": {
            "tool": "web_search",
            "input": inp.model_dump(),
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

    if choice.tool == "web_fetch":
        inp = choice.fetch_input
        result = await web_fetch(inp, use_cache=use_cache)
        trace.append({"kind": "tool_call", "payload": {
            "tool": "web_fetch",
            "input": inp.model_dump(),
            "result_summary": f"status={result.status}, length={result.content_length}",
        }})
        if result.status == "error":
            return f"Fetch failed: {result.error}"
        marker = " (truncated)" if result.truncated else ""
        return f"Fetched {result.url}{marker} ({result.content_length} chars):\n\n{result.content}"

    if choice.tool == "calculator":
        inp = choice.calculator_input
        result = await calculator(inp)
        trace.append({"kind": "tool_call", "payload": {
            "tool": "calculator",
            "input": inp.model_dump(),
            "result_summary": f"status={result.status}",
        }})
        if result.status == "error":
            return f"Calculator error: {result.error}"
        return f"{inp.expression} = {result.result}"

    return f"Unknown tool: {choice.tool}"


def _compact_old_observations(messages: list[dict]) -> None:
    """Replace old tool observations with short breadcrumbs to save tokens.

    Strategy: keep the most recent 2 user messages (tool observations) verbatim.
    Replace anything older with a brief '[compacted — original was N chars]'
    note. The assistant's thoughts are kept intact — they're short and useful as
    reasoning history.

    Index 0 is the system prompt, index 1 is the original user question;
    both are always preserved verbatim. After that, messages alternate
    assistant (thought + tool choice) and user (tool observation).

    Mutates the messages list in place. Idempotent.
    """
    observation_indices = [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and i >= 2 and m["content"].startswith("Observation from")
    ]

    if len(observation_indices) <= 2:
        return

    to_compact = observation_indices[:-2]
    for idx in to_compact:
        content = messages[idx]["content"]
        first_line = content.split("\n", 1)[0]
        if "[compacted" in first_line:
            continue
        messages[idx] = {
            "role": "user",
            "content": f"{first_line} [compacted — original was {len(content)} chars]",
        }


def _emergency_report(
    sub_question: str, iterations: int, trace: list[dict], reason: str,
) -> MiniReport:
    """Fallback when the loop exits without a proper finish call."""
    return MiniReport(
        sub_question=sub_question,
        summary=f"The researcher did not reach a confident answer within {iterations} iterations.",
        citations=[],
        confidence_notes=f"Agent terminated due to: {reason}",
        iterations=iterations,
        terminated_reason=reason,
        trace=trace,
    )