"""Synthesizer agent — produces a final answer from all researcher mini-reports.

The synthesizer is the terminal stage of Agora's pipeline. It reads:
  - The original user question
  - The plan's interpretation of that question
  - Every researcher's mini-report (summary + citations)

And produces:
  - A final prose answer addressing the user's question
  - A consolidated citation list (deduplicated by URL)
  - Metadata flagging which sub-questions were well-supported vs. thin

Design notes:
- We use Gemini (same as planner) for quality. One call per run.
- Structured output via instructor — the final answer is a validated model.
- We surface when researchers failed, rather than hiding it. If 2 of 5
  researchers hit errors, the synthesizer should say so honestly.
"""
from pydantic import BaseModel, Field

from api.llm import GEMINI_FLASH, call_structured_gemini


# ---------- Input data shape (what the worker gives us) ----------

class SubQuestionResult(BaseModel):
    """One sub-question + the researcher's mini-report for it."""
    sub_question: str
    approach: str | None = None
    summary: str
    citations: list[dict]  # [{url, quote}, ...]
    terminated_reason: str  # "finish" | "iteration_cap" | "error"
    iterations: int


# ---------- Output schema (what Gemini must return) ----------

class FinalCitation(BaseModel):
    """A citation in the final synthesized answer."""
    url: str
    quote: str = Field(max_length=400)
    supports: str = Field(
        description="One sentence: what claim in the final answer this citation supports.",
        max_length=300,
    )


class CoverageNote(BaseModel):
    """Per-sub-question note on how well it was answered."""
    sub_question: str
    coverage: str = Field(
        description="One of: 'well-supported', 'thin', 'failed'. Use 'failed' only "
                    "if the researcher errored or hit iteration cap without useful output."
    )
    note: str = Field(
        description="One short sentence explaining the coverage rating.",
        max_length=200,
    )


class SynthesisReport(BaseModel):
    """The final synthesized answer to the user's question."""
    answer: str = Field(
        description="A direct prose answer to the user's question, 3-6 paragraphs. "
                    "Ground every claim in the researchers' findings. Do not invent "
                    "facts not present in the mini-reports.",
        max_length=4000,
    )
    citations: list[FinalCitation] = Field(
        min_length=0,
        max_length=15,
        description="Citations backing key claims in the answer. Dedupe by URL — "
                    "don't cite the same URL twice. Prefer citations that cover "
                    "different sub-questions. If NO researchers produced real "
                    "citations (all failed), return an empty list rather than "
                    "fabricating placeholder citations.",
    )
    coverage: list[CoverageNote] = Field(
        description="One note per sub-question, in order.",
    )
    caveats: str = Field(
        default="",
        description="Optional: honest caveats about the research. Mention if some "
                    "sub-questions failed, if sources conflicted, or if the answer "
                    "is based on limited evidence.",
        max_length=500,
    )


# ---------- Prompt ----------

SYNTHESIZER_SYSTEM_PROMPT = """You are the Synthesizer for Agora, a multi-agent research system.

Your job: read the original user question, the plan's interpretation, and every researcher's mini-report. Produce a final, well-grounded answer.

RULES:

1. GROUND IN EVIDENCE. Every factual claim must trace back to something in the mini-reports. Do not invent facts, statistics, or product features that weren't in the source material.

2. NO FABRICATED CITATIONS. Only cite URLs and quotes that appear in the researchers' citations. If a sub-question produced no citations (e.g. researcher failed), don't cite anything for that part — acknowledge the gap instead. If NO researchers produced any real citations (all failed), return an EMPTY citations list. Do not invent placeholder URLs like "example.com" or "about:blank".

3. INTEGRATE, DON'T CONCATENATE. Don't just paste the 5 sub-answers back-to-back. Find the connections, contrasts, and themes across sub-questions. The whole should be more coherent than the parts.

4. BE HONEST ABOUT GAPS. If researchers failed or produced thin results, say so in the `caveats` field. Don't pretend coverage you don't have.

5. FORMAT: 3-6 paragraphs of direct prose. No headings, no bullet lists inside the answer. The answer should read like a concise expert briefing.

6. COVERAGE NOTES: For each sub-question, rate coverage as 'well-supported' (solid mini-report with real citations), 'thin' (some data but limited or single-source), or 'failed' (researcher errored or produced nothing useful).

ORIGINAL USER QUESTION:
{question}

PLANNER'S INTERPRETATION:
{interpretation}

RESEARCHER MINI-REPORTS:
{mini_reports}

Produce the final synthesized answer as structured output.
"""


def _format_mini_reports(results: list[SubQuestionResult]) -> str:
    """Serialize mini-reports as readable text for the Gemini prompt.

    We format as prose-y blocks rather than JSON — LLMs reason better over
    prose than nested dicts, same principle as in the researcher loop.
    """
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        status_tag = {
            "finish": "[OK]",
            "iteration_cap": "[PARTIAL — hit iteration limit]",
            "error": "[FAILED — researcher errored]",
        }.get(r.terminated_reason, f"[{r.terminated_reason}]")

        citations_text = ""
        if r.citations:
            citations_text = "\nCitations:\n" + "\n".join(
                f"  - {c.get('url', 'no-url')}: \"{c.get('quote', '')}\""
                for c in r.citations
            )
        else:
            citations_text = "\nCitations: (none)"

        block = (
            f"--- Sub-question {i} {status_tag} ---\n"
            f"Question: {r.sub_question}\n"
            f"Approach: {r.approach or 'unspecified'}\n"
            f"Iterations used: {r.iterations}\n"
            f"Summary: {r.summary}"
            f"{citations_text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# ---------- Entry point called by the worker ----------

async def synthesize(
    question: str,
    interpretation: str,
    results: list[SubQuestionResult],
    use_cache: bool = True,
) -> SynthesisReport:
    """Call Gemini to produce the final synthesized answer."""
    mini_reports_text = _format_mini_reports(results)
    prompt = SYNTHESIZER_SYSTEM_PROMPT.format(
        question=question,
        interpretation=interpretation,
        mini_reports=mini_reports_text,
    )

    report = await call_structured_gemini(
        model=GEMINI_FLASH,
        prompt=prompt,
        response_model=SynthesisReport,
        kind="synthesizer",
        use_cache=use_cache,
        max_tokens=4096,
        temperature=0.3,
    )
    return report