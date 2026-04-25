"""LLM-as-judge for evaluating Agora's research outputs.

Given an Agora SynthesisReport and a benchmark question's expected aspects,
produces structured scores across four dimensions:
  - faithfulness: are claims grounded in cited sources?
  - citation_quality: are citations real, specific, and diverse?
  - coverage: does the answer address every part of the question?
  - synthesis: does the answer integrate findings or just concatenate?

Each dimension scored 1-10 with a one-sentence justification.

Design notes:
- We use Gemini 2.5 Flash for the judge. Same model class as the synthesizer
  for consistency, but a different role (evaluator instead of producer).
- The judge sees the FULL Agora output including coverage notes and caveats.
  This lets it credit Agora for honest gap-acknowledgment, which is part of
  what makes Agora's design distinctive.
- Scores are integers 1-10. Sub-integer precision is illusory for LLM judges.
- The rubric is explicit and per-dimension. Vague rubrics produce noisy scores.
"""
from pydantic import BaseModel, Field

from api.llm import GEMINI_FLASH, call_structured_gemini


# ============================================================
# Score schema
# ============================================================

class DimensionScore(BaseModel):
    """One scoring dimension."""
    score: int = Field(
        ge=1, le=10,
        description="Integer score from 1 (terrible) to 10 (excellent).",
    )
    justification: str = Field(
        max_length=400,
        description="One sentence explaining the score, citing specifics from the answer.",
    )


class EvalScores(BaseModel):
    """Full eval output for one Agora run."""
    faithfulness: DimensionScore = Field(
        description="Are all factual claims in the answer actually supported by the cited sources? "
                    "10 = every claim traces to a citation. 1 = answer is largely fabricated."
    )
    citation_quality: DimensionScore = Field(
        description="Are citations real (working URLs), specific (deep links not just homepages), "
                    "and diverse (multiple distinct sources)? "
                    "10 = excellent citations across multiple authoritative sources. "
                    "1 = no citations or fabricated/placeholder URLs."
    )
    coverage: DimensionScore = Field(
        description="Does the answer address every aspect listed in expected_aspects? "
                    "10 = all aspects covered substantively. 1 = answer mostly missed the question."
    )
    synthesis: DimensionScore = Field(
        description="Does the answer INTEGRATE findings into a coherent take, or just list things side-by-side? "
                    "10 = real synthesis with insight. 1 = bullet-list summary with no integration."
    )
    overall_notes: str = Field(
        default="",
        description="Optional 2-3 sentence overall assessment. Mention any standout strengths or "
                    "weaknesses not captured in the dimension scores.",
        max_length=600,
    )


# ============================================================
# Judge prompt
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring outputs from Agora, an AI research assistant.

Agora was given a research question and produced a synthesized answer with citations and coverage notes. Your job: score the answer across four dimensions.

You are scoring the ANSWER QUALITY, not the question difficulty. A short answer to a simple question can score well if it's accurate and well-cited.

SCORING RUBRIC:

FAITHFULNESS (1-10):
- 10: Every factual claim is directly supported by the cited sources. No invented numbers, dates, or quotes.
- 7-9: Most claims supported; minor unsupported assertions.
- 4-6: Substantial unsupported claims; reasonable framing but some fabrication.
- 1-3: Answer is largely invented; citations don't support the claims.

CITATION_QUALITY (1-10):
- 10: 4+ citations across distinct authoritative sources, deep-linked to specific pages.
- 7-9: 3+ real citations, mostly authoritative.
- 4-6: Some citations but weak (forum posts, single source, or thin coverage).
- 1-3: No citations, broken URLs, or placeholder citations like "example.com".

COVERAGE (1-10):
- 10: Every expected aspect is addressed substantively.
- 7-9: Most aspects covered; one or two thin or missing.
- 4-6: Half the aspects covered well; rest missed or shallow.
- 1-3: Most expected aspects missing.

SYNTHESIS (1-10):
- 10: Integrated answer with real insight (e.g. recognizing east-west vs north-south traffic distinctions, naming tradeoffs explicitly).
- 7-9: Coherent prose that connects findings, not just bullet-list summarization.
- 4-6: Mostly side-by-side summary; some attempt at integration.
- 1-3: Pure concatenation of sub-question answers with no synthesis.

IMPORTANT:
- Coverage caveats are GOOD. If Agora honestly acknowledges a gap in research, don't penalize coverage — penalize FAITHFULNESS for hallucinated content instead. Honest "I don't have evidence" wins over confident hallucination.
- For ADVERSARIAL questions (where reasonable people disagree), a good answer ACKNOWLEDGES the disagreement. Don't reward confident takes on contested issues.
- For TIME_SENSITIVE questions, score citation quality more leniently — recent sources are inherently scarcer.

INPUTS:

QUESTION CATEGORY: {category}
QUESTION DIFFICULTY: {difficulty}
ORIGINAL QUESTION:
{question}

EXPECTED ASPECTS (a good answer should address these):
{expected_aspects}

AGORA'S ANSWER:
{answer}

AGORA'S CITATIONS:
{citations}

AGORA'S COVERAGE NOTES (Agora's own assessment of each sub-question):
{coverage}

AGORA'S CAVEATS:
{caveats}

Produce structured eval scores.
"""


def _format_citations(citations: list[dict]) -> str:
    """Format Agora's citations for the judge prompt."""
    if not citations:
        return "(no citations)"
    lines = []
    for i, c in enumerate(citations, 1):
        url = c.get("url", "?")
        quote = c.get("quote", "")[:200]
        supports = c.get("supports", "")[:200]
        lines.append(f"  [{i}] {url}\n      quote: {quote}\n      supports: {supports}")
    return "\n".join(lines)


def _format_coverage(coverage: list[dict]) -> str:
    """Format Agora's coverage notes for the judge prompt."""
    if not coverage:
        return "(no coverage notes)"
    lines = []
    for c in coverage:
        sub_q = c.get("sub_question", "?")[:120]
        rating = c.get("coverage", "?")
        note = c.get("note", "")[:200]
        lines.append(f"  - [{rating}] {sub_q}\n      {note}")
    return "\n".join(lines)


def _format_aspects(aspects: list[str]) -> str:
    return "\n".join(f"  - {a}" for a in aspects)


# ============================================================
# Entry point
# ============================================================

async def score_run(
    *,
    question: str,
    category: str,
    difficulty: str,
    expected_aspects: list[str],
    agora_output: dict,
) -> EvalScores:
    """Score a single Agora run against the benchmark expectations.

    `agora_output` is the synthesizer's full output dict — answer, citations,
    coverage, caveats. This is what gets stored in the synthesizer Task row.
    """
    answer = agora_output.get("answer", "(no answer)")
    citations = agora_output.get("citations", [])
    coverage = agora_output.get("coverage", [])
    caveats = agora_output.get("caveats", "(none)")

    prompt = JUDGE_SYSTEM_PROMPT.format(
        category=category,
        difficulty=difficulty,
        question=question,
        expected_aspects=_format_aspects(expected_aspects),
        answer=answer,
        citations=_format_citations(citations),
        coverage=_format_coverage(coverage),
        caveats=caveats or "(none)",
    )

    scores = await call_structured_gemini(
        model=GEMINI_FLASH,
        prompt=prompt,
        response_model=EvalScores,
        kind="judge",
        max_tokens=2048,
        temperature=0.0,  # judge should be as deterministic as possible
    )
    return scores