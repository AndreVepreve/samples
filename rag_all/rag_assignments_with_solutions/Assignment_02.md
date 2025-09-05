# Here is your assignment to design your ﬁrst rag rank pilot

## Original Prompt
Here is your assignment to design your ﬁrst rag rank pilot.
Choose one business function e.g. HR, legal, sales, operations and design a mini pilot using this template.
Option one pilot objective.
What decision or workﬂow will rag improve?
Option two source material what documents will you use?
PDFs.
Policies.
Product specs.
Contracts.
Option three target users who will use it e.g. support agents, HR team ﬁeld sales reps.
Option four evaluation metrics.
How will you measure success?
E.g. response speed, accuracy, user satisfaction, reduced manual eﬀort, option ﬁve tool or platform chosen no code tool or vendor you would test this on.
Also, there is a bonus draft a short internal email explaining this rag pilot to your executive sponsor or department head in plain English.

### References

- FAISS docs: https://faiss.ai/index.html
- Lewis et al., 2020: https://arxiv.org/abs/2005.11401
- LlamaIndex docs: https://docs.llamaindex.ai/

## What This Assesses
- Ability to scope a RAG use case and articulate value.
- Understanding of retrieval/guardrails/governance trade-offs.
- Capability to define measurable success metrics and risks.

## Sample Solution (Finance Company)
**Assumptions (mid-size private finance company):**
- Private, regulated lender with ~1,200 employees; ~$8B AUM; B2C lending + wealth management.
- Knowledge sources: policy manuals (HR, InfoSec), product sheets, pricing matrices, call-center transcripts, loan origination SOPs, ticketing (Jira/ServiceNow), contracts/NDA templates.
- Tooling: Confluence + SharePoint, Salesforce, Snowflake, S3; Okta; SOC 2 Type II.


### Sample Solution — Pilot Design
**Objective:** Reduce L2 policy escalations by 30% (consumer lending); improve handle time by 15%.

**Users:** 60 L1/L2 agents (Zendesk); 8 compliance analysts.

**Ingestion:** Connectors to Confluence/SharePoint; normalize to Markdown/HTML; chunk 200–400 words w/ 15% overlap; tag `owner`, `version`, `effective_date`, `confidentiality`, `rbac_tags`.

**Retrieval:** Dense recall + cross-encoder rerank; metadata filters (product, jurisdiction, date). Parent-doc return for synthesis.

**Generation:** Instruction-tuned LLM; prompt enforces citations, redaction, confidence threshold; escalation workflow.

**Observability:** Log query→retrieved chunks→answer; dashboard (Metabase) with citation rate, SME agreement, flags.

**Timeline:** 2 weeks ingestion; 4 weeks pilot; 2 weeks hardening & rollout.
