# Interview Rubric — RAG Strategy & Execution (Finance Company)

> This rubric turns your assignments into structured interview criteria. Use it for phone screens, on-sites, or panel debriefs.
## Here is your assignment one — Strategy Memo

### Core Competencies
- Problem framing & stakeholder alignment
- Corpus readiness & ingestion strategy
- Retrieval design (chunking, filters, reranking)
- Prompting & grounding discipline (citations, redaction)
- Governance & risk controls (RBAC/ABAC, PII, audit)
- Evaluation & metrics (RAGAS-like, SME agreement)
- Architecture choices & trade-offs (build/buy/hybrid)
- Change management & adoption plan

### Interview Prompts
- Walk me through your recommended first use case and why it wins.
- How would you convince Legal/Compliance to sign off on this pilot?
- What metrics would you publish weekly to prove value?
- When would you **not** use RAG? What would you do instead?

### What Good Looks Like
- Names concrete corpora and owners; proposes freshness SLAs.
- Explains chunking rationale and trade-offs; mentions parent-document retrieval.
- Uses hybrid retrieval and/or rerankers with clear latency/quality trade-offs.
- Requires citations and defines a minimum-citation/confidence gate.
- Describes RBAC/ABAC & metadata tagging; mentions redaction and immutable logs.
- Defines offline and online metrics; includes SME review protocol.
- Acknowledges lock-in; justifies build vs. buy vs. hybrid with criteria.
- Outlines change management with champions, training, and feedback loops.

### Red Flags
- Talks only about prompting; ignores retrieval quality and data hygiene.
- No plan for access controls or PII handling.
- Vague metrics or no evaluation methodology.
- Suggests fine-tuning to fix stale knowledge without justification.
- No owners or freshness plan for the corpus.

### Scoring Rubric (1–5)
| Dimension | 1 - Poor | 2 - Limited | 3 - Adequate | 4 - Strong | 5 - Exceptional |
|---|---|---|---|---|---|
| Problem framing | Misunderstands business need | Partial; misses constraints | Understands use case | Clear value thesis; constraints | Strategic clarity; measurable impact |
| Corpus readiness | No sources/owners | Few sources; no plan | Lists sources; basic plan | Owners, freshness SLAs | Continuous ingestion; quality gates |
| Retrieval design | Hand-wavy prompt-only | Simple top-K only | Sensible vector search | Hybrid + reranker; filters | Tuned indexes; parent-doc; latency budgets |
| Prompting & grounding | No citations | Optional citations | Requires citations | Citation+confidence gates | Policy-aware prompts; robust fallbacks |
| Governance & risk | Ignores PII/RBAC | Mentions security | Basic RBAC/PII | Detailed RBAC/ABAC; redaction | Full audit trail; red teams, DLP |
| Evaluation & metrics | None | Ad-hoc checks | Some metrics | Comprehensive offline+online | RAGAS-style; SME blind review; dashboards |
| Architecture choices | N/A | Random picks | Reasonable stack | Trade-off rationale | Cost/latency modeling; portability plan |
| Change management | None | Email announcement | Training plan | Champions + feedback loop | Adoption playbook; staged rollout |


---

## Here is your assignment to design your ﬁrst rag rank pilot — Pilot Design

### Core Competencies
- Problem framing & stakeholder alignment
- Corpus readiness & ingestion strategy
- Retrieval design (chunking, filters, reranking)
- Prompting & grounding discipline (citations, redaction)
- Governance & risk controls (RBAC/ABAC, PII, audit)
- Evaluation & metrics (RAGAS-like, SME agreement)
- Architecture choices & trade-offs (build/buy/hybrid)
- Change management & adoption plan

### Interview Prompts
- How do you choose chunk size and overlap for our corpus?
- Explain how you’d implement two-stage retrieval and when to enable reranking.
- What metadata filters would you add for our products and jurisdictions?
- How would you instrument logs to audit any answer end-to-end?

### What Good Looks Like
- Names concrete corpora and owners; proposes freshness SLAs.
- Explains chunking rationale and trade-offs; mentions parent-document retrieval.
- Uses hybrid retrieval and/or rerankers with clear latency/quality trade-offs.
- Requires citations and defines a minimum-citation/confidence gate.
- Describes RBAC/ABAC & metadata tagging; mentions redaction and immutable logs.
- Defines offline and online metrics; includes SME review protocol.
- Acknowledges lock-in; justifies build vs. buy vs. hybrid with criteria.
- Outlines change management with champions, training, and feedback loops.

### Red Flags
- Talks only about prompting; ignores retrieval quality and data hygiene.
- No plan for access controls or PII handling.
- Vague metrics or no evaluation methodology.
- Suggests fine-tuning to fix stale knowledge without justification.
- No owners or freshness plan for the corpus.

### Scoring Rubric (1–5)
| Dimension | 1 - Poor | 2 - Limited | 3 - Adequate | 4 - Strong | 5 - Exceptional |
|---|---|---|---|---|---|
| Problem framing | Misunderstands business need | Partial; misses constraints | Understands use case | Clear value thesis; constraints | Strategic clarity; measurable impact |
| Corpus readiness | No sources/owners | Few sources; no plan | Lists sources; basic plan | Owners, freshness SLAs | Continuous ingestion; quality gates |
| Retrieval design | Hand-wavy prompt-only | Simple top-K only | Sensible vector search | Hybrid + reranker; filters | Tuned indexes; parent-doc; latency budgets |
| Prompting & grounding | No citations | Optional citations | Requires citations | Citation+confidence gates | Policy-aware prompts; robust fallbacks |
| Governance & risk | Ignores PII/RBAC | Mentions security | Basic RBAC/PII | Detailed RBAC/ABAC; redaction | Full audit trail; red teams, DLP |
| Evaluation & metrics | None | Ad-hoc checks | Some metrics | Comprehensive offline+online | RAGAS-style; SME blind review; dashboards |
| Architecture choices | N/A | Random picks | Reasonable stack | Trade-off rationale | Cost/latency modeling; portability plan |
| Change management | None | Email announcement | Training plan | Champions + feedback loop | Adoption playbook; staged rollout |


---

## Here is your assignment three Rag risk Readiness Memo — Strategy Memo

### Core Competencies
- Problem framing & stakeholder alignment
- Corpus readiness & ingestion strategy
- Retrieval design (chunking, filters, reranking)
- Prompting & grounding discipline (citations, redaction)
- Governance & risk controls (RBAC/ABAC, PII, audit)
- Evaluation & metrics (RAGAS-like, SME agreement)
- Architecture choices & trade-offs (build/buy/hybrid)
- Change management & adoption plan

### Interview Prompts
- Walk me through your recommended first use case and why it wins.
- How would you convince Legal/Compliance to sign off on this pilot?
- What metrics would you publish weekly to prove value?
- When would you **not** use RAG? What would you do instead?

### What Good Looks Like
- Names concrete corpora and owners; proposes freshness SLAs.
- Explains chunking rationale and trade-offs; mentions parent-document retrieval.
- Uses hybrid retrieval and/or rerankers with clear latency/quality trade-offs.
- Requires citations and defines a minimum-citation/confidence gate.
- Describes RBAC/ABAC & metadata tagging; mentions redaction and immutable logs.
- Defines offline and online metrics; includes SME review protocol.
- Acknowledges lock-in; justifies build vs. buy vs. hybrid with criteria.
- Outlines change management with champions, training, and feedback loops.

### Red Flags
- Talks only about prompting; ignores retrieval quality and data hygiene.
- No plan for access controls or PII handling.
- Vague metrics or no evaluation methodology.
- Suggests fine-tuning to fix stale knowledge without justification.
- No owners or freshness plan for the corpus.

### Scoring Rubric (1–5)
| Dimension | 1 - Poor | 2 - Limited | 3 - Adequate | 4 - Strong | 5 - Exceptional |
|---|---|---|---|---|---|
| Problem framing | Misunderstands business need | Partial; misses constraints | Understands use case | Clear value thesis; constraints | Strategic clarity; measurable impact |
| Corpus readiness | No sources/owners | Few sources; no plan | Lists sources; basic plan | Owners, freshness SLAs | Continuous ingestion; quality gates |
| Retrieval design | Hand-wavy prompt-only | Simple top-K only | Sensible vector search | Hybrid + reranker; filters | Tuned indexes; parent-doc; latency budgets |
| Prompting & grounding | No citations | Optional citations | Requires citations | Citation+confidence gates | Policy-aware prompts; robust fallbacks |
| Governance & risk | Ignores PII/RBAC | Mentions security | Basic RBAC/PII | Detailed RBAC/ABAC; redaction | Full audit trail; red teams, DLP |
| Evaluation & metrics | None | Ad-hoc checks | Some metrics | Comprehensive offline+online | RAGAS-style; SME blind review; dashboards |
| Architecture choices | N/A | Random picks | Reasonable stack | Trade-off rationale | Cost/latency modeling; portability plan |
| Change management | None | Email announcement | Training plan | Champions + feedback loop | Adoption playbook; staged rollout |


---

## Here is your assignment for you are preparing a one page internal strategy brieﬁng to your leadership — Internal Strategy Briefing

### Core Competencies
- Problem framing & stakeholder alignment
- Corpus readiness & ingestion strategy
- Retrieval design (chunking, filters, reranking)
- Prompting & grounding discipline (citations, redaction)
- Governance & risk controls (RBAC/ABAC, PII, audit)
- Evaluation & metrics (RAGAS-like, SME agreement)
- Architecture choices & trade-offs (build/buy/hybrid)
- Change management & adoption plan

### Interview Prompts
- Which parts of the stack do we own vs. buy, and why?
- How will you roll this out to Support and Compliance with minimal disruption?
- What does success look like after 3 months and 12 months?
- Where do you anticipate pushback, and how do you handle it?

### What Good Looks Like
- Names concrete corpora and owners; proposes freshness SLAs.
- Explains chunking rationale and trade-offs; mentions parent-document retrieval.
- Uses hybrid retrieval and/or rerankers with clear latency/quality trade-offs.
- Requires citations and defines a minimum-citation/confidence gate.
- Describes RBAC/ABAC & metadata tagging; mentions redaction and immutable logs.
- Defines offline and online metrics; includes SME review protocol.
- Acknowledges lock-in; justifies build vs. buy vs. hybrid with criteria.
- Outlines change management with champions, training, and feedback loops.

### Red Flags
- Talks only about prompting; ignores retrieval quality and data hygiene.
- No plan for access controls or PII handling.
- Vague metrics or no evaluation methodology.
- Suggests fine-tuning to fix stale knowledge without justification.
- No owners or freshness plan for the corpus.

### Scoring Rubric (1–5)
| Dimension | 1 - Poor | 2 - Limited | 3 - Adequate | 4 - Strong | 5 - Exceptional |
|---|---|---|---|---|---|
| Problem framing | Misunderstands business need | Partial; misses constraints | Understands use case | Clear value thesis; constraints | Strategic clarity; measurable impact |
| Corpus readiness | No sources/owners | Few sources; no plan | Lists sources; basic plan | Owners, freshness SLAs | Continuous ingestion; quality gates |
| Retrieval design | Hand-wavy prompt-only | Simple top-K only | Sensible vector search | Hybrid + reranker; filters | Tuned indexes; parent-doc; latency budgets |
| Prompting & grounding | No citations | Optional citations | Requires citations | Citation+confidence gates | Policy-aware prompts; robust fallbacks |
| Governance & risk | Ignores PII/RBAC | Mentions security | Basic RBAC/PII | Detailed RBAC/ABAC; redaction | Full audit trail; red teams, DLP |
| Evaluation & metrics | None | Ad-hoc checks | Some metrics | Comprehensive offline+online | RAGAS-style; SME blind review; dashboards |
| Architecture choices | N/A | Random picks | Reasonable stack | Trade-off rationale | Cost/latency modeling; portability plan |
| Change management | None | Email announcement | Training plan | Champions + feedback loop | Adoption playbook; staged rollout |


---

## We'll ﬁnish with your ﬁnal assignment, a vision paper for 2030 and a quiz to test your mastery — Vision Paper 2030

### Core Competencies
- Problem framing & stakeholder alignment
- Corpus readiness & ingestion strategy
- Retrieval design (chunking, filters, reranking)
- Prompting & grounding discipline (citations, redaction)
- Governance & risk controls (RBAC/ABAC, PII, audit)
- Evaluation & metrics (RAGAS-like, SME agreement)
- Architecture choices & trade-offs (build/buy/hybrid)
- Change management & adoption plan

### Interview Prompts
- Describe how agentic workflows evolve our current RAG system.
- What’s your plan for multimodal retrieval in a PII-heavy environment?
- Which governance mechanisms must remain invariant as we scale?
- What competitive moat do we build via RAG by 2030?

### What Good Looks Like
- Names concrete corpora and owners; proposes freshness SLAs.
- Explains chunking rationale and trade-offs; mentions parent-document retrieval.
- Uses hybrid retrieval and/or rerankers with clear latency/quality trade-offs.
- Requires citations and defines a minimum-citation/confidence gate.
- Describes RBAC/ABAC & metadata tagging; mentions redaction and immutable logs.
- Defines offline and online metrics; includes SME review protocol.
- Acknowledges lock-in; justifies build vs. buy vs. hybrid with criteria.
- Outlines change management with champions, training, and feedback loops.

### Red Flags
- Talks only about prompting; ignores retrieval quality and data hygiene.
- No plan for access controls or PII handling.
- Vague metrics or no evaluation methodology.
- Suggests fine-tuning to fix stale knowledge without justification.
- No owners or freshness plan for the corpus.

### Scoring Rubric (1–5)
| Dimension | 1 - Poor | 2 - Limited | 3 - Adequate | 4 - Strong | 5 - Exceptional |
|---|---|---|---|---|---|
| Problem framing | Misunderstands business need | Partial; misses constraints | Understands use case | Clear value thesis; constraints | Strategic clarity; measurable impact |
| Corpus readiness | No sources/owners | Few sources; no plan | Lists sources; basic plan | Owners, freshness SLAs | Continuous ingestion; quality gates |
| Retrieval design | Hand-wavy prompt-only | Simple top-K only | Sensible vector search | Hybrid + reranker; filters | Tuned indexes; parent-doc; latency budgets |
| Prompting & grounding | No citations | Optional citations | Requires citations | Citation+confidence gates | Policy-aware prompts; robust fallbacks |
| Governance & risk | Ignores PII/RBAC | Mentions security | Basic RBAC/PII | Detailed RBAC/ABAC; redaction | Full audit trail; red teams, DLP |
| Evaluation & metrics | None | Ad-hoc checks | Some metrics | Comprehensive offline+online | RAGAS-style; SME blind review; dashboards |
| Architecture choices | N/A | Random picks | Reasonable stack | Trade-off rationale | Cost/latency modeling; portability plan |
| Change management | None | Email announcement | Training plan | Champions + feedback loop | Adoption playbook; staged rollout |


---

## Here is your assignment ﬁve Vision paper — Vision Paper 2030

### Core Competencies
- Problem framing & stakeholder alignment
- Corpus readiness & ingestion strategy
- Retrieval design (chunking, filters, reranking)
- Prompting & grounding discipline (citations, redaction)
- Governance & risk controls (RBAC/ABAC, PII, audit)
- Evaluation & metrics (RAGAS-like, SME agreement)
- Architecture choices & trade-offs (build/buy/hybrid)
- Change management & adoption plan

### Interview Prompts
- Describe how agentic workflows evolve our current RAG system.
- What’s your plan for multimodal retrieval in a PII-heavy environment?
- Which governance mechanisms must remain invariant as we scale?
- What competitive moat do we build via RAG by 2030?

### What Good Looks Like
- Names concrete corpora and owners; proposes freshness SLAs.
- Explains chunking rationale and trade-offs; mentions parent-document retrieval.
- Uses hybrid retrieval and/or rerankers with clear latency/quality trade-offs.
- Requires citations and defines a minimum-citation/confidence gate.
- Describes RBAC/ABAC & metadata tagging; mentions redaction and immutable logs.
- Defines offline and online metrics; includes SME review protocol.
- Acknowledges lock-in; justifies build vs. buy vs. hybrid with criteria.
- Outlines change management with champions, training, and feedback loops.

### Red Flags
- Talks only about prompting; ignores retrieval quality and data hygiene.
- No plan for access controls or PII handling.
- Vague metrics or no evaluation methodology.
- Suggests fine-tuning to fix stale knowledge without justification.
- No owners or freshness plan for the corpus.

### Scoring Rubric (1–5)
| Dimension | 1 - Poor | 2 - Limited | 3 - Adequate | 4 - Strong | 5 - Exceptional |
|---|---|---|---|---|---|
| Problem framing | Misunderstands business need | Partial; misses constraints | Understands use case | Clear value thesis; constraints | Strategic clarity; measurable impact |
| Corpus readiness | No sources/owners | Few sources; no plan | Lists sources; basic plan | Owners, freshness SLAs | Continuous ingestion; quality gates |
| Retrieval design | Hand-wavy prompt-only | Simple top-K only | Sensible vector search | Hybrid + reranker; filters | Tuned indexes; parent-doc; latency budgets |
| Prompting & grounding | No citations | Optional citations | Requires citations | Citation+confidence gates | Policy-aware prompts; robust fallbacks |
| Governance & risk | Ignores PII/RBAC | Mentions security | Basic RBAC/PII | Detailed RBAC/ABAC; redaction | Full audit trail; red teams, DLP |
| Evaluation & metrics | None | Ad-hoc checks | Some metrics | Comprehensive offline+online | RAGAS-style; SME blind review; dashboards |
| Architecture choices | N/A | Random picks | Reasonable stack | Trade-off rationale | Cost/latency modeling; portability plan |
| Change management | None | Email announcement | Training plan | Champions + feedback loop | Adoption playbook; staged rollout |


---
