# Build vs Buy vs Hybrid — Scorecard

> Use this one-page scorecard to guide your decision. Score each row from **1 (poor)** to **5 (excellent)**, then sum the totals. Treat 30+ as a strong fit.

## How to score
- Be honest about **must-haves** (security, citations, logging).
- Weigh rows that matter more to your context (e.g., data sovereignty) by multiplying their score by 2.

| Category | What “Excellent” Looks Like | Score (1–5) |
|---|---|---|
| **Speed to Value** | Pilot usable in < 3 weeks; minimal glue code |  |
| **Retrieval Quality** | Hybrid (keyword + vector) + reranking; tunable |  |
| **Citations & Provenance** | Every answer cites doc + section; clickable |  |
| **Security at Search Time** | RBAC/ABAC filters on retrieval; private networking |  |
| **Observability** | Full logs: query → retrieved → answer; SIEM export |  |
| **Integration** | Connectors to SharePoint/Confluence/CRM; webhooks |  |
| **Portability (Anti-Lock-In)** | Export vectors, metadata, prompts; BYO models |  |
| **TCO & Pricing Clarity** | Transparent usage model; forecastable costs |  |
| **SLA & Support** | Clear uptime, response times, roadmap access |  |
| **Team Fit** | Skills match; realistic ops burden |  |

### Interpreting totals
- **Build** leans strong if: you need deep governance, unusual workflows, or strict data residency.
- **Buy** wins if: time-to-value dominates and vendor meets your must-haves.
- **Hybrid** works when: you want vendor UX now but retain control of retrieval, data, or models.
