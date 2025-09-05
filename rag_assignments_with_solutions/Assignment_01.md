# Here is your assignment one

## Original Prompt
Here is your assignment one.
Leadership in action Rag strategy memo.
Write a one page internal strategy memo 300 500 words answering the following.
We're evaluating gen AI solutions based on what we now understand about Rag.
When should we prefer Rag over ﬁne tuning or prompt engineering?
What organizational needs does Rag best serve in our context e.g. customer support?
Compliance, sales enablement?
Some of the optional enhancements you can make are mention risks avoided by using Rag.
Propose one business unit where Rag could be piloted.

### References

- LangChain docs: https://python.langchain.com/docs/introduction/
- Lewis et al., 2020: https://arxiv.org/abs/2005.11401

## What This Assesses
- Ability to scope a RAG use case and articulate value.
- Understanding of retrieval/guardrails/governance trade-offs.
- Capability to define measurable success metrics and risks.

## Sample Solution (Finance Company)
**Assumptions (mid-size private finance company):**
- Private, regulated lender with ~1,200 employees; ~$8B AUM; B2C lending + wealth management.
- Knowledge sources: policy manuals (HR, InfoSec), product sheets, pricing matrices, call-center transcripts, loan origination SOPs, ticketing (Jira/ServiceNow), contracts/NDA templates.
- Tooling: Confluence + SharePoint, Salesforce, Snowflake, S3; Okta; SOC 2 Type II.


### Sample Solution — Strategy Memo (Executive Brief)
**Subject:** RAG-first plan, target functions, and an 8-week pilot for Support & Compliance

**Problem**
Agents and analysts waste time searching across Confluence/SharePoint; answers are inconsistent and hard to audit.

**Why RAG**
- **Grounding & citations** enable audit-ready responses.
- **Freshness** via re-indexing instead of retraining.
- **Governance** with RBAC/ABAC and per-chunk metadata.

**Where to start**
- **Support (L1/L2)**: eligibility, fees, disclosures, password resets.  
- **Compliance/Legal**: clause lookup for NDAs, loan agreements, KYC/AML guidance.

**Pilot (8 weeks)**
- **Scope**: Support policy Q&A + compliance clause retrieval.  
- **Sources**: Confluence SOPs, pricing sheets (CSV/PDF), Jira tickets, anonymized call transcripts.  
- **Stack**: OpenSearch/Chroma + reranker; LangChain/LlamaIndex; GPT/Claude; Okta RBAC; logs → Snowflake.  
- **Metrics**: ≥85% citation rate; ≥25% fewer L2 escalations; ≥15% faster handle time; SME agreement ≥80%.

**Risks/Mitigations**
PII leakage → redaction + ACLs; stale docs → freshness SLA; hallucination → rerank + min-citation; audit → immutable logs.

**Asks**
Approve connectors; nominate content owners; allocate SME review time weekly.
