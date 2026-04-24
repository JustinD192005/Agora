
## Key engineering decisions

**Fan-in via Postgres row-level locking.** When multiple researchers finish at the same millisecond, how do you guarantee exactly-once synthesizer execution without a separate coordinator service? Agora uses `SELECT FOR UPDATE` on the parent run row, checks whether the synthesizer has already been enqueued, and acts atomically before releasing the lock. Concurrent completions serialize through the lock; exactly one wins and fires the synthesizer.

**Graceful failure as a design principle.** Tools never raise. They return `status="error"` with a reason, which becomes an observation the agent reasons about. When the researcher loop exhausts its iteration budget or an LLM call hard-fails, an emergency report is emitted with `terminated_reason="error"` and `num_citations=0`. The synthesizer sees these and produces an honest "I couldn't answer this" output with explicit `coverage: "failed"` flags — instead of hallucinating a confident-sounding answer from nothing.

**Structured outputs everywhere.** Every LLM call uses [instructor](https://github.com/jxnl/instructor) with a Pydantic response model. The planner emits a `ResearchPlan`, each researcher turn emits an `AgentChoice`, and the synthesizer emits a `SynthesisReport`. Invalid outputs are rejected at the schema level before they touch the database.

**Request pacing across fan-out.** Researchers are enqueued with staggered `_defer_by` delays (3s per job), so their opening LLM calls don't all hit the upstream API in the same second. Parallelism is preserved; burst rate-limiting is avoided.

**Token-aware conversation management.** Older tool observations in the researcher's conversation history are compacted to short breadcrumbs after 2 turns, so per-iteration token usage stays roughly flat instead of growing linearly. Saves 40-65% tokens on longer runs.



## Tech stack

- **Python 3.14** with async/await throughout
- **FastAPI** for the HTTP API
- **arq + Redis** for the async task queue
- **PostgreSQL 16** with `SQLAlchemy 2.0` (async) and Alembic migrations
- **Pydantic + instructor** for schema-validated LLM outputs
- **Gemini 2.5 Flash** for planning and synthesis
- **Llama 3.3 70B** via Groq for the researcher ReAct loop
- **Tavily** for web search; **httpx + trafilatura** for content fetching
- **Docker Compose** for local infrastructure

---

## Run locally

Requires Docker Desktop, Python 3.14+, and API keys for Gemini, Groq, and Tavily.

```bash
# 1. Clone + install
git clone https://github.com/JustinD192005/Agora.git
cd Agora
uv sync

# 2. Copy the env template and fill in your API keys
cp .env.example .env
# edit .env to add GEMINI_API_KEY, GROQ_API_KEY, TAVILY_API_KEY

# 3. Start Postgres + Redis
docker compose up -d

# 4. Run migrations
uv run alembic upgrade head

# 5. Start the API (in one terminal)
.venv/Scripts/python -m uvicorn api.main:app --reload --port 8000

# 6. Start the worker (in another terminal)
.venv/Scripts/python -m arq worker.main.WorkerSettings

# 7. Submit a question
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"question":"How do Redis and Kafka compare for real-time systems?"}'
```

Check `docs/runs/` for sample outputs from real runs.

---

## Sample output

See [`docs/runs/synth-istio-v1-pretty.json`](docs/runs/synth-istio-v1-pretty.json) for a full end-to-end run on *"How do service mesh architectures like Istio compare to API gateway patterns for microservice communication?"*.

Summary of that run:
- Planner produced 5 sub-questions in ~8 seconds
- 5 researchers executed in parallel with staggered starts
- 4/5 researchers completed successfully with 1 hitting the iteration cap
- Synthesizer integrated findings using industry terminology (east-west vs north-south traffic)
- Final answer: 2,741 chars, 5 real citations across 4 domains
- Coverage notes honestly flagged the one thin sub-question in `caveats`


## Known limitations

- **Free-tier LLM quotas are the primary bottleneck during development.** Agora does 8-12 LLM calls per question in a ~60-second burst across parallel researchers, which exceeds what most free tiers are built for. Developer-tier access on Groq (roughly $0.02 per full run) is recommended.
- **No deterministic replay yet**: re-running the same question hits the LLM again. 
- **No dashboard yet** — run state is queryable via database but there's no UI.  
- **arq retry creates orphan task rows.** When a researcher job raises, arq re-enqueues it, and our code creates a new task row on every invocation. Not functionally broken (fan-in still works correctly) but wasteful. Fix is trivial — set `max_tries=1`.
