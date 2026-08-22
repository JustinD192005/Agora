"""Researcher agent — answers a single sub-question via a ReAct tool-use loop.

Flow per iteration:
  1. Ask the researcher LLM which tool to call next, given the conversation so far
  2. Dispatch to the tool
  3. Append the tool result as an observation
  4. If the tool was `finish`, return the mini-report
  5. Otherwise loop

Hard cap: MAX_ITERATIONS. In the final turns we inject explicit pressure toward
`finish` so the agent commits to an answer instead of researching forever.

Design notes:
- We use instructor's JSON mode — the LLM returns a JSON object matching the
  AgentChoice schema directly, rather than OpenAI-style function calls. Newer
  Groq models (gpt-oss) misread TOOLS mode and try to invoke the tool names
  from the prompt as real functions.
- The schema uses FLAT SCALAR fields for simple tool inputs (query, url,
  expression) rather than nested objects. gpt-oss returns nested single-field
  objects as bare strings, which breaks validation. Flat scalars sidestep the
  ambiguity entirely; we reconstruct the typed input objects before dispatch.
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
    """Flat tool-choice schema compatible with Groq's JSON mode.

    The LLM picks ONE tool via `tool`, then fills in ONLY the matching
    input field. Simple tools use flat scalar strings; finish uses a
    structured object because it genuinely needs nested citations.
    """
    thought: str = Field(
        description="One sentence: why you're choosing this tool right now. "
                    "Helps you reason step-by-step.",
        max_length=400,
    )
    tool: ToolName = Field(
        description="Which tool to call. Exactly one of: web_search, web_fetch, calculator, finish."
    )
    # Flat scalar inputs — populate ONLY the one matching `tool` above.
    query: str | None = Field(
        default=None,
        description="REQUIRED if tool='web_search'. A focused search query, "
                    "3-8 words. Plain string, not an object.",
    )
    url: str | None = Field(
        default=None,
        description="REQUIRED if tool='web_fetch'. A full https:// URL. "
                    "Plain string, not an object.",
    )
    expression: str | None = Field(
        default=None,
        description="REQUIRED if tool='calculator'. An arithmetic expression. "
                    "Plain string, not an object.",
    )
    finish_input: FinishInput | None = Field(
        default=None,
        description="REQUIRED if tool='finish', otherwise null.",
    )

    @model_validator(mode="after")
    def check_input_matches_tool(self) -> "AgentChoice":
        """Ensure the correct input field is populated for the chosen tool."""
        mapping = {
            "web_search": self.query,
            "web_fetch": self.url,
            "calculator": self.expression,
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
- web_search: find relevant URLs and snippets — set `query` to a plain string
- web_fetch: get the main text of a specific page — set `url` to a plain string
- calculator: evaluate arithmetic — set `expression` to a plain string
- finish: terminal — set `finish_input` with summary, citations, confidence_notes

TARGET PATTERN (follow this unless something fails):
  Turn 1: web_search
  Turn 2: web_fetch the best result
  Turn 3: web_fetch one more source
  Turn 4: finish
That is FOUR turns. Anything beyond is a fallback for failures, not the plan.

RULES:
- BIAS HEAVILY TOWARD FINISHING. Two good sources is enough. A concise answer with
  2 real citations beats a perfect answer you never deliver.
- If you have fetched at least ONE page successfully, you have enough to call finish.
- Do NOT search again after a successful fetch unless that content was completely
  irrelevant to the sub-question.
- Never fetch the same URL twice. Never repeat a search query you already ran.
- NEVER cite a URL you haven't fetched. Every citation must be a URL that appeared
  in a successful web_fetch result.
- Every citation quote must be a SHORT VERBATIM excerpt from fetched content.
  Don't paraphrase inside quote marks.
- If a fetch fails, try ONE different URL, then finish with what you have.
- Keep summaries to 2-5 sentences. Long summaries dilute useful information.
- Each turn, use the "thought" field to briefly justify your choice.

OUTPUT FORMAT (CRITICAL):
Each turn, you output:
  - thought: one sentence of reasoning
  - tool: one of "web_search", "web_fetch", "calculator", "finish"
  - the matching input field populated for the chosen tool:
      * if tool=web_search → set `query` to a PLAIN STRING (e.g. "raft consensus algorithm")
      * if tool=web_fetch  → set `url` to a PLAIN STRING (e.g. "https://raft.github.io/raft.pdf")
      * if tool=calculator → set `expression` to a PLAIN STRING (e.g. "42 * 1.5")
      * if tool=finish     → set `finish_input` to an object with summary, citations, confidence_notes
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
    successful_fetches = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        log.info("researcher.iteration", sub_question=sub_question[:80], iteration=iterations)

        # --- Escalating pressure toward finish as the cap approaches ---
        remaining = MAX_ITERATIONS - iterations
        if remaining <= 1:
            messages.append({
                "role": "user",
                "content": (
                    "STOP RESEARCHING. This is your LAST turn. You MUST set tool='finish' "
                    "now and populate finish_input using what you already know. Cite the URLs "
                    "you already fetched. Do not search or fetch again."
                ),
            })
        elif remaining == 2:
            messages.append({
                "role": "user",
                "content": (
                    f"Only {remaining} turns left. Fetch at most one more source, then "
                    "call finish. Do not start a new search."
                ),
            })
        elif successful_fetches >= 2:
            messages.append({
                "role": "user",
                "content": (
                    f"You have already fetched {successful_fetches} sources successfully. "
                    "That is enough. Call finish now unless a fetch was unusable."
                ),
            })

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

        if choice.tool == "web_fetch" and not observation_text.startswith("Fetch failed"):
            successful_fetches += 1

        # Add both assistant intent and tool observation to conversation history
        input_for_history = _input_repr(choice)
        messages.append({
            "role": "assistant",
            "content": f"Thought: {choice.thought}\nTool: {choice.tool}\nInput: {input_for_history}",
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


def _input_repr(choice: AgentChoice) -> str:
    """Short string form of the populated input, for conversation history."""
    if choice.tool == "web_search":
        return choice.query or ""
    if choice.tool == "web_fetch":
        return choice.url or ""
    if choice.tool == "calculator":
        return choice.expression or ""
    if choice.tool == "finish":
        return choice.finish_input.model_dump_json() if choice.finish_input else ""
    return ""


async def _dispatch_tool(choice: AgentChoice, trace: list[dict], use_cache: bool = True) -> str:
    """Run the chosen tool and return a text observation for the LLM."""
    if choice.tool == "web_search":
        inp = WebSearchInput(query=choice.query)
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
        inp = WebFetchInput(url=choice.url)
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
        inp = CalculatorInput(expression=choice.expression)
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
    """Fallback when the loop exits without a proper finish call.

    Salvages whatever the agent actually read: we pull the URLs it fetched
    successfully out of the trace so the synthesizer at least knows which
    sources were consulted, even though the agent never committed to a summary.
    """
    fetched_urls = [
        ev["payload"]["input"].get("url")
        for ev in trace
        if ev.get("kind") == "tool_call"
        and ev["payload"].get("tool") == "web_fetch"
        and "status=ok" in ev["payload"].get("result_summary", "")
    ]
    fetched_urls = [u for u in fetched_urls if u]

    if fetched_urls:
        summary = (
            f"The researcher consulted {len(fetched_urls)} source(s) but did not commit "
            f"to a confident answer within {iterations} iterations. Sources reviewed: "
            + ", ".join(fetched_urls[:3])
        )
    else:
        summary = (
            f"The researcher did not reach a confident answer within {iterations} iterations."
        )

    return MiniReport(
        sub_question=sub_question,
        summary=summary,
        citations=[],
        confidence_notes=f"LOW CONFIDENCE — agent terminated due to: {reason}",
        iterations=iterations,
        terminated_reason=reason,
        trace=trace,
    )