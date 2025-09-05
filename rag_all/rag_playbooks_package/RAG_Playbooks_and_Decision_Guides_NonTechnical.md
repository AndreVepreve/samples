# RAG Playbooks & Decision Guides (Non-Technical Edition)

> This practical guide turns your **RAG_Master_Sections_1–6.md** into clear, step-by-step playbooks. It explains *what to do* and *why*, in plain language.

---

## How to use this guide

- Skim the **Table of Contents** and jump to what you need right now (e.g., “Build vs Buy” or “Vendor Checklist”).
- Each playbook includes:
  - **When to use it**
  - **Quick version** (what good looks like)
  - **Step-by-step**
  - **Outputs** (what you should have when you’re done)
  - **Pitfalls** (mistakes to avoid)

---

## Table of Contents

- [1) Use-Case Triage (Where to Start)](#1-use-case-triage-where-to-start)
- [2) Build vs Buy vs Hybrid (Decision Guide)](#2-build-vs-buy-vs-hybrid-decision-guide)
- [3) Vendor Selection Checklist (RAG platforms / vector DBs)](#3-vendor-selection-checklist-rag-platforms--vector-dbs)
- [4) Pilot Design & Rollout (8-Week Plan)](#4-pilot-design--rollout-8-week-plan)
- [5) Data Readiness & Ingestion (Make Your Docs “RAG-Ready”)](#5-data-readiness--ingestion-make-your-docs-rag-ready)
- [6) Retrieval Strategy (Simple but Strong)](#6-retrieval-strategy-simple-but-strong)
- [7) Prompt & Answer Policy (What the Assistant Must/May/Will Not Do)](#7-prompt--answer-policy-what-the-assistant-mustmaywill-not-do)
- [8) Governance & Risk (Run it like a real system)](#8-governance--risk-run-it-like-a-real-system)
- [9) Evaluation & Monitoring (Prove it works)](#9-evaluation--monitoring-prove-it-works)
- [10) Multi-Agent RAG (When tasks get complex)](#10-multi-agent-rag-when-tasks-get-complex)
- [11) Multimodal & Real-Time (Voice, images, and fresh data)](#11-multimodal--real-time-voice-images-and-fresh-data)
- [12) Change Management (Make adoption stick)](#12-change-management-make-adoption-stick)
- [Appendix A: Scenario Templates](#appendix-a-scenario-templates)
- [Appendix B: Key Terms (Plain English)](#appendix-b-key-terms-plain-english)

---

## 1) Use-Case Triage (Where to Start)

**When to use it:** You want your first RAG wins with minimal risk.

**Quick version:** Start with workflows where answers **live in documents** (policies, SOPs, contracts), pain is visible (time spent searching, lots of escalations), **risk is manageable**, and outcomes are easy to measure.

**Steps**
1. **List candidates**: document-heavy questions your teams answer daily.
2. **Score each** on:
   - User pain (search time, escalations)
   - Data readiness (format, freshness)
   - Risk (PII/regulatory)
   - Measurability (clear KPIs like first-contact resolution or handle time)
3. **Pick 1–2 lanes** with high pain, low/medium risk, and measurable outcomes.
4. **Write success criteria** (e.g., ≥85% answers show citations; −20% escalations).

**Outputs:** A 1-page “Pilot Charter” with the chosen lane, target metrics, owners, and timeline.
**Pitfalls:** Starting with a high-risk, low-documentation task.

---

## 2) Build vs Buy vs Hybrid (Decision Guide)

**When to use it:** You’re picking how to stand up RAG (platform and components).

**Quick decision tree**
- **Need results fast?** → Start **Buy** (SaaS) and plan an exit path.
- **Strict data controls / custom orchestration?** → **Build** core retrieval & governance.
- **Want speed now, control later?** → **Hybrid** (own back end; use a vendor UI or hosted vector DB).

**What to compare**
- Retrieval quality; Citations/provenance; Security at search time; Observability; Integrations; Portability.

---

## 3) Vendor Selection Checklist (RAG platforms / vector DBs)

**Must-haves**
- Search: keyword (BM25), semantic (vector), and **hybrid**; optional **reranking**.
- Indexes: ANN (HNSW / IVF-PQ); batch & streaming updates.
- Security at search time: RBAC/ABAC filters; encryption; private networking.
- Citations for every answer; Observability (end-to-end logs).
- Integrations (SharePoint/Confluence/CRM) and SSO; Portability (export vectors & metadata).

---

## 4) Pilot Design & Rollout (8-Week Plan)

- **W0–1:** Scope & data (select lane; 30–150 gold docs; owners).
- **W2–3:** Ingestion & search (normalize → chunk → hybrid search + reranker; filters).
- **W4–5:** UX & guardrails (citations; confidence gating; human review path).
- **W6–7:** Run in the wild (feedback; metrics; SME checks).
- **W8:** Decide (expand, iterate, or park; publish a 1-pager).

---

## 5) Data Readiness & Ingestion (Make Your Docs “RAG-Ready”)

- **Acquire** sources; **Normalize** to HTML/MD; **Chunk** 200–600 words; slight overlap.
- **Tag** metadata (owner, effective date, product, region, confidentiality, access).
- **Index** keyword (BM25) + vector (FAISS/OpenSearch) for **hybrid** search.
- **QA**: fix tables; dedupe; remove obsolete copies.

---

## 6) Retrieval Strategy (Simple but Strong)

1. Keyword (BM25) + Semantic (vectors) → combine.
2. Apply metadata **filters** (product/region/date).
3. **Rerank** top candidates for accuracy.
4. Use **parent sections** if needed for context.
5. Provide **3–8** passages to the model for an answer **with citations**.

---

## 7) Prompt & Answer Policy

- **Must:** cite sources; state uncertainty; avoid guessing; follow redaction.
- **May:** summarize or compare; suggest next steps with links.
- **Will not:** provide legal advice; expose PII; answer beyond approved corpus.

---

## 8) Governance & Risk (NIST-aligned)

- **GOVERN**: ownership for policies and quality checks.
- **MAP**: identify risks (bias, leaks, wrong advice); document them.
- **MEASURE**: define metrics (accuracy, citation rate, flags).
- **MANAGE**: access rules, logs, reviews, red-team cadence.

---

## 9) Evaluation & Monitoring

- **Offline**: curated Q&A; measure faithfulness & context precision/recall (RAGAS-like).
- **Online**: citation rate, SME agreement %, flagged answers, stale-doc %, adoption and handle time.

---

## 10) Multi-Agent RAG

- Use a **Planner → Searcher → Writer → Verifier** flow for multi-step tasks.
- ReAct-style loops (think → act → think) reduce guesswork and improve traceability.

---

## 11) Multimodal & Real-Time

- **Voice**: speech → question → search → speak.
- **Vision**: screenshot/photo → detect artifact → retrieve SOP/KB.
- **Real-time**: re-index key changes quickly.

---

## 12) Change Management

1. Co-design with a friendly team + SME.
2. Teach “how to ask” & “how to verify” (read citations).
3. Publish weekly wins; track usage & sentiment.
4. Keep humans in the loop for sensitive outputs.

---

## Appendix A: Scenario Templates

### A. L1/L2 Support Co-Pilot
- Users: Support agents in a ticketing tool
- Sources: Policies, product guides, KB
- Flow: Ask → hybrid search + rerank → answer with citations
- KPIs: −AHT; −Escalations; ≥85% answers with citations

### B. Contract Clause Finder (Legal/Compliance)
- Users: Legal/compliance analysts
- Sources: Contracts, NDAs, regulatory notes
- Flow: Search clauses; filter; show clause + similar; draft with citations
- KPIs: Review time; SME agreement; audit trail completeness

### C. Regulatory Change Radar
- Users: Compliance
- Sources: Regulator PDFs + internal policies
- Flow: Weekly diff; link affected pages; assign owners
- KPIs: Time-to-insight; missed changes

### D. Field Assistant (Voice/Vision)
- Users: Field/service staff
- Sources: SOPs & troubleshooting guides
- Flow: Voice/photo → detect issue → retrieve steps → read back + show links
- KPIs: Time to resolution; repeat visits
