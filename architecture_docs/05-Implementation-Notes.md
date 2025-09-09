
# Implementation Notes (Security, Observability, Performance, CI/CD) — 2025-09-09

## Security
- **Identity:** Cognito or OIDC; IRSA per pod; least-privilege IAM.
- **Encryption:** KMS CMKs for S3, Aurora, OpenSearch; TLS everywhere.
- **Policy:** OPA or Cedar pre-hooks; purpose binding; deny non-approved content.

## Observability
- **Tracing:** OpenTelemetry across MCP → domusDocs → Data Platform.
- **Decision Records:** inputs (redacted), tools used, outputs hash, citations, model IDs.
- **SLOs:** p95 latency per tool and per blueprint; index freshness lag.

## Performance and Cost
- **Hybrid retrieval:** BM25 prefilter then vector KNN then rerank; cap k for LLM context.
- **Caching:** per (claims_hash, query_hash) for 5m on read-only calls.
- **OpenSearch:** hot or warm tiers, ISM rollover; right-sized shards and replicas.
- **Batching:** embeddings in batches; reuse cached vectors.

## CI/CD and Change Management
- **Schema validation:** JSON Schema (ajv) and static policy checks in CI.
- **Signing:** cosign or KMS; MCP verifies signature before exposure.
- **Promotion:** draft → approved; canary via allowlist_groups; rollback to prior version.
- **Golden tests:** expected fields, latency budgets, citation presence.
