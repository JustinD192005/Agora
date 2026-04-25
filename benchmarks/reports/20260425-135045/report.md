# Agora Eval Report — 2026-04-25T14:04:22.156215+00:00

**Questions evaluated:** 16
**Successfully scored:** 16
**Failed runs:** 0

## Aggregate scores (1-10 scale)

- **Overall:** 7.62
- **Faithfulness:** 7.81
- **Citation quality:** 7.31
- **Coverage:** 6.81
- **Synthesis:** 8.56

## By category

| Category | n | Overall | Faith | Cit | Cov | Synth |
|---|---|---|---|---|---|---|
| adversarial | 2 | 7.62 | 9.5 | 6.0 | 6.5 | 8.5 |
| analytical | 4 | 7.31 | 6.5 | 7.5 | 6.75 | 8.5 |
| comparison | 5 | 8.2 | 8.4 | 8.6 | 7.2 | 8.6 |
| definitional | 3 | 7.25 | 8.33 | 5.33 | 6.67 | 8.67 |
| time_sensitive | 2 | 7.38 | 6.5 | 8.0 | 6.5 | 8.5 |

## Per-question results

### `redis-vs-kafka` (comparison, easy)
**Question:** What are the tradeoffs between Redis and Kafka for real-time systems?

**Scores:** overall: 8.5 | faith 8 | cit 9 | cov 8 | synth 9
**Duration:** 15.5s
**Notes:** The answer provides a clear and well-structured comparison, effectively highlighting the distinct strengths and use cases of Redis and Kafka, and correctly identifying their complementary roles. The primary area for improvement is the lack of specific quantitative details for Kafka's latency and throughput, which would have provided a more balanced comparison with Redis's sub-millisecond claims.

### `postgres-vs-mongodb` (comparison, medium)
**Question:** What are the key differences between PostgreSQL and MongoDB for a SaaS backend, and when should each be chosen?

**Scores:** overall: 7.0 | faith 8 | cit 7 | cov 5 | synth 8
**Duration:** 35.2s
**Notes:** The answer provides a clear and well-structured comparison of PostgreSQL and MongoDB's core differences and ideal use cases. A significant strength is the explicit and honest caveat regarding the lack of information on performance and scalability. However, the overall coverage is incomplete due to the missing details on performance, scalability, and multi-tenancy, which are critical for SaaS backend considerations.

### `graphql-vs-rest` (comparison, medium)
**Question:** What are the main architectural differences between GraphQL and REST APIs, and what are the tradeoffs for a mid-sized SaaS company?

**Scores:** overall: 9.25 | faith 10 | cit 9 | cov 9 | synth 9
**Duration:** 47.7s
**Notes:** The answer provides a clear, balanced, and well-structured comparison of GraphQL and REST APIs, specifically tailored to the needs of a mid-sized SaaS company. It effectively highlights the key architectural differences and their practical implications, making it a highly useful and informative response.

### `service-mesh-vs-api-gateway` (comparison, medium)
**Question:** How do service mesh architectures like Istio compare to API gateway patterns for microservice communication?

**Scores:** overall: 8.5 | faith 8 | cit 9 | cov 8 | synth 9
**Duration:** 68.1s
**Notes:** The answer provides a clear and well-structured comparison, effectively differentiating the primary concerns of API gateways and service meshes. It excels in explaining their distinct purposes and how they can complement each other for comprehensive traffic management.

### `monolith-vs-microservices` (comparison, medium)
**Question:** When should an early-stage startup choose a monolithic architecture over microservices?

**Scores:** overall: 7.75 | faith 8 | cit 9 | cov 6 | synth 8
**Duration:** 53.7s
**Notes:** The answer provides a clear and well-reasoned argument for choosing a monolithic architecture for early-stage startups, effectively highlighting its advantages and the pitfalls of premature microservices adoption. While the core argument is strong and well-supported by citations, the answer falls short on fully addressing all expected aspects, particularly regarding team size considerations and a detailed migration path.

### `what-is-raft` (definitional, easy)
**Question:** What is the Raft consensus algorithm and what problem does it solve?

**Scores:** overall: 8.25 | faith 10 | cit 4 | cov 10 | synth 9
**Duration:** 34.8s
**Notes:** The answer provides an excellent, well-structured explanation of the Raft consensus algorithm, clearly defining its purpose and key mechanisms. Its primary area for improvement is the depth and breadth of its citations, as it relies on a single source.

### `what-is-cap` (definitional, easy)
**Question:** What is the CAP theorem and how does it apply to distributed database design?

**Scores:** overall: 5.5 | faith 6 | cit 4 | cov 4 | synth 8
**Duration:** 50.8s
**Notes:** The answer provides a decent high-level overview of the CAP theorem and the CP/AP distinction but suffers significantly from a lack of detailed definitions for C, A, and P, and the complete absence of database examples. The reliance on a single, non-authoritative citation also weakens its credibility.

### `what-is-rag` (definitional, easy)
**Question:** What is retrieval-augmented generation and how does it differ from fine-tuning a language model?

**Scores:** overall: 8.0 | faith 9 | cit 8 | cov 6 | synth 9
**Duration:** 45.0s
**Notes:** The answer provides a clear and well-structured comparison of RAG and fine-tuning, effectively highlighting their core differences and practical tradeoffs. The transparency about the 'thin' research on fine-tuning is a strength. However, the omission of the embedding and vector store pipeline for RAG is a notable gap in coverage for a definitional question.

### `vector-db-tradeoffs` (analytical, hard)
**Question:** What are the key tradeoffs when choosing between Pinecone, Weaviate, and Qdrant for a production RAG system?

**Scores:** overall: 8.0 | faith 7 | cit 9 | cov 7 | synth 9
**Duration:** 64.1s
**Notes:** The answer provides a comprehensive comparison of the three vector databases across key dimensions like cost, deployment, and performance. While it excels in detailing pricing and hosting models, it could benefit from more specific information regarding ecosystem integrations. The initial paragraph contains a few qualitative claims that lack direct citation support, but the bulk of the factual content is well-cited.

### `kubernetes-readiness` (analytical, hard)
**Question:** When is a small engineering team ready to adopt Kubernetes, and what are the warning signs they're not?

**Scores:** overall: 7.0 | faith 6 | cit 8 | cov 6 | synth 8
**Duration:** 60.2s
**Notes:** The answer provides a comprehensive overview of Kubernetes readiness and warning signs for small teams, demonstrating good synthesis. However, it significantly misses the crucial aspect of simpler alternatives, which is a key consideration for small teams, and many claims lack direct citation support despite being generally accurate.

### `orm-tradeoffs` (analytical, medium)
**Question:** What are the tradeoffs of using an ORM versus writing raw SQL in a high-performance backend service?

**Scores:** overall: 5.5 | faith 5 | cit 4 | cov 5 | synth 8
**Duration:** 54.9s
**Notes:** The answer is well-written and clearly structured, effectively comparing ORMs and raw SQL on developer productivity, code complexity, and maintainability. A significant strength is its explicit acknowledgment of research limitations regarding performance, security, and fine-grained control. However, the answer suffers from a lack of diverse and authoritative citations, with only one `dev.to` article supporting a minor point. It also missed covering key expected aspects like N+1 problems and hybrid approaches, which were not identified as research limitations by Agora.

### `event-sourcing-fit` (analytical, hard)
**Question:** For what kinds of business domains is event sourcing a good architectural fit, and where does it create more problems than it solves?

**Scores:** overall: 8.75 | faith 8 | cit 9 | cov 9 | synth 9
**Duration:** 62.9s
**Notes:** The answer provides a clear and well-structured overview of event sourcing's applicability, effectively balancing its benefits with its complexities and anti-patterns. The self-assessment notes from Agora regarding 'thin' and 'failed' sub-questions do not fully reflect the quality of the final synthesized answer, which successfully addresses all the expected aspects.

### `postgres-recent-features` (time_sensitive, medium)
**Question:** What are the most significant new features added to PostgreSQL in versions 16 and 17?

**Scores:** overall: 7.75 | faith 6 | cit 8 | cov 8 | synth 9
**Duration:** 62.0s
**Notes:** The answer provides a good overview and structure, clearly separating features by version and offering a comparative summary. However, the faithfulness for PostgreSQL 16's specific features could be improved with more direct citation support.

### `ai-agent-frameworks-2026` (time_sensitive, hard)
**Question:** What are the leading open-source frameworks for building agentic AI systems as of 2026, and what differentiates them?

**Scores:** overall: 7.0 | faith 7 | cit 8 | cov 5 | synth 8
**Duration:** 53.2s
**Notes:** The answer provides a strong foundational understanding of agentic AI and identifies several key frameworks. However, it significantly misses one expected framework (DSPy) and, despite its honesty, fails to provide the requested detailed differentiation and architectural patterns, which were core to the question.

### `typescript-vs-javascript` (adversarial, medium)
**Question:** Is TypeScript worth adopting for a small team building a product MVP?

**Scores:** overall: 7.5 | faith 9 | cit 5 | cov 8 | synth 8
**Duration:** 52.4s
**Notes:** The answer excels in its honesty and appropriate framing for an adversarial question, clearly outlining what it knows and what it couldn't confidently ascertain. The main area for improvement is the authority and quantity of its citations.

### `rust-for-web-services` (adversarial, hard)
**Question:** Should a typical SaaS company use Rust for their web backend, or is Go or Python a better choice?

**Scores:** overall: 7.75 | faith 10 | cit 7 | cov 5 | synth 9
**Duration:** 56.1s
**Notes:** The answer provides a strong comparison between Rust and Python, clearly outlining their respective strengths and weaknesses for SaaS backends. Its honesty in acknowledging the lack of comprehensive information on Go is commendable. However, the complete omission of the 'hiring market' as a factor for language choice is a significant oversight for a typical SaaS company's decision-making process.
