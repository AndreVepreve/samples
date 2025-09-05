# Vendor Selection Checklist — RAG Platforms / Vector Databases

> Bring this to vendor demos. Mark **Yes/No** and capture concrete evidence or links.

## A. Retrieval
- [ ] Keyword (BM25) available and tunable
- [ ] Vector search (HNSW and/or IVF-PQ)
- [ ] **Hybrid** (keyword + vector) in a single query
- [ ] Reranking available (cross-encoder or similar)
- [ ] Tunable K, filters, and scoring

## B. Indexing & Data
- [ ] Batch + streaming updates without downtime
- [ ] Supports parent-document linking (section context)
- [ ] Handles tables, PDFs, and long docs well
- [ ] Deduplication / version control features

## C. Security & Privacy
- [ ] RBAC/ABAC enforced **at retrieval time**
- [ ] Encryption in transit & at rest; private networking/VPC peering
- [ ] Fine-grained access tags per chunk/document
- [ ] PII redaction or integration point provided

## D. Observability & Operations
- [ ] Full logs: query → retrieved chunks → final answer
- [ ] Export to SIEM / data lake
- [ ] Dashboards for accuracy, citation %, and stale-doc %
- [ ] Clear backup/restore, retention, and migration paths

## E. Integrations
- [ ] SharePoint / Confluence connectors
- [ ] CRM / ticketing (e.g., Salesforce, Zendesk)
- [ ] Webhooks for re-indexing on updates
- [ ] SSO (SAML/OIDC), SCIM provisioning

## F. Portability & Cost
- [ ] Export vectors & metadata; BYO models allowed
- [ ] Transparent pricing & forecast tools
- [ ] No hard lock-in (documented migration steps)

## Notes / Evidence
- Vendor links, screenshots, and contract language belong here.
