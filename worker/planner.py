"""Planner agent — decomposes a user question into a research plan.

The planner takes a raw question like "What are the tradeoffs between
vector databases for RAG?" and produces a structured plan with 3-7
sub-questions that researchers will investigate in parallel.

Design notes:
- We ask for 3-7 sub-questions (not more) to keep parallel researcher
  fan-out bounded and costs predictable.
- Each sub-question carries an 'approach' hint so the researcher can
  bias its tool selection (e.g. quantitative questions want the calculator).
- The Pydantic schema is the contract with Gemini — structured outputs
  physically prevent it from deviating.
"""
from enum import Enum

from pydantic import BaseModel, Field

from api.llm import GEMINI_FLASH, call_structured


# ---------- Schema ----------

class Approach(str, Enum):
    """Hint to the researcher about which tools will be most useful."""
    WEB_SEARCH = "web_search"          # broad factual questions
    SPECIFIC_SITE = "specific_site"    # question targets a known domain/source
    QUANTITATIVE = "quantitative"      # needs numbers, calculations, comparisons
    DEFINITIONAL = "definitional"      # "what is X" style, often single-hop


class SubQuestion(BaseModel):
    question: str = Field(
        description="A focused, self-contained sub-question. Must be answerable "
                    "independently of the other sub-questions — no pronouns "
                    "referring to siblings, no 'and also'."
    )
    approach: Approach = Field(
        description="Best research approach for this sub-question."
    )
    rationale: str = Field(
        description="One sentence: why this sub-question matters for the user's goal.",
        max_length=200,
    )


class ResearchPlan(BaseModel):
    """Structured output from the planner."""
    interpretation: str = Field(
        description="One-sentence restatement of what the user is really asking, "
                    "resolving any ambiguity.",
        max_length=300,
    )
    sub_questions: list[SubQuestion] = Field(
        min_length=3,
        max_length=5,
        description="3-5 focused sub-questions whose answers together address "
                    "the user's question.",
    )


# ---------- Prompt ----------

PLANNER_SYSTEM_PROMPT = """You are the Planner for Agora, a multi-agent research system.

Your job: take a user's research question and decompose it into 3-5 focused sub-questions that independent researcher agents can investigate in parallel. The answers to the sub-questions, combined, should fully address the user's question.
RULES for good decomposition:

1. COVERAGE: Every aspect of the user's question must be covered by at least one sub-question. Nothing important should fall through the cracks.

2. INDEPENDENCE: Each sub-question must be answerable WITHOUT seeing the answers to the other sub-questions. No "following up on the previous point" — researchers run in parallel and don't see each other.

3. SPECIFICITY: Sub-questions should be concrete and searchable. "What is the history of X?" is too broad. "What year was X first used commercially, and by whom?" is better.

4. NO META-QUESTIONS: Don't include sub-questions like "What are the key considerations?" or "What should we think about?". Every sub-question must have a factual answer.

5. RIGHT-SIZE: For simple factual questions, 3 sub-questions. For moderate complexity, 4. For complex multi-part questions, 5. Don't pad. Prefer fewer, sharper sub-questions over many overlapping ones.

6. APPROACH TAG: Tag each sub-question with the best research approach:
   - web_search: broad factual questions best served by searching and reading multiple sources
   - specific_site: question targets a known authoritative source (docs, a specific paper, a standard)
   - quantitative: needs numbers, calculations, or numerical comparison
   - definitional: straightforward "what is X" where a single good source usually suffices

OUTPUT: Return a structured plan. Do not include any prose outside the structured output.

USER QUESTION:
{question}
"""


# ---------- Entry point called by the worker ----------

async def generate_plan(question: str, use_cache: bool = True) -> ResearchPlan:
    """Call Gemini to produce a structured research plan."""
    prompt = PLANNER_SYSTEM_PROMPT.format(question=question)
    plan = await call_structured(
        model=GEMINI_FLASH,
        prompt=prompt,
        response_model=ResearchPlan,
        kind="planner",
        use_cache=use_cache,
        max_tokens=2048,
        temperature=0.3,
    )
    return plan