
# Blueprint Concept, Ecosystem, and Lifecycle — 2025-09-09

## 1. What is a Blueprint?
A **Blueprint** is a **versioned declarative artifact** (YAML or JSON) that specifies how to ingest, process, extract, classify, and store documents for a **specific business use case**. It encodes business rules, compliance gates, and downstream actions.

## 2. Why Blueprints?
- Document and domain variability (contracts vs policies vs HR handbooks).
- Governance and traceability with consistent, repeatable runs.
- Scale via configuration rather than ad hoc code.

## 3. Ecosystem Placement
- **domusDocs:** authoritative registry and runtime executor of document blueprints.
- **MCP:** registers approved blueprints and exposes capabilities to agents/products.
- **Data Platform:** receives curated outputs (extractions, metrics, lineage).

## 4. Lifecycle
1. **Design** → author spec in repo; PR review.
2. **Validate** → JSON Schema + static policy checks.
3. **Publish** → store in S3; index in Aurora; status draft or approved.
4. **Register** → MCP indexes approved versions for discovery.
5. **Execute** → Step Functions or Temporal pipelines run with lineage.
6. **Monitor** → OTEL traces, OpenLineage facets, Decision Records.
7. **Evolve** → new versions, canary flags, replay strategy.

## 5. Lifecycle Diagram
```mermaid
flowchart TD
  classDef step fill:#0ea5e9,stroke:#0a4,stroke-width:1,color:#fff,font-weight:700;
  classDef action fill:#f8fafc,stroke:#94a3b8,stroke-width:1,color:#0f172a;
  classDef gov fill:#fee2e2,stroke:#dc2626,stroke-width:1,color:#991b1b;

  D1["Definition"]:::step
  R1["Registration"]:::step
  E1["Execution"]:::step
  M1["Monitoring and Governance"]:::step
  EV1["Evolution"]:::step

  D1 -->|SME and Architect design| D2["Blueprint Spec YAML or JSON with rules, fields, labels, tags"]:::action
  D2 -->|Publish to MCP via domusDocs| R1
  R1 -->|Stored in| R2["Blueprint Registry Aurora and S3"]:::action
  R1 -->|Discoverable via| R3["MCP Tool Catalog"]:::action

  R1 --> E1
  E1 --> E2["DomusDocs Pipeline OCR then Redaction then Chunking then Embedding then Classification or Extraction"]:::action
  E1 --> E3["Outputs to Storage S3 Aurora Vector DB"]:::action
  E1 --> E4["Feeds Data Platform Data Lake and Core Models"]:::action

  E1 --> M1
  M1 --> M2["Governance RBAC ABAC HITL Redaction"]:::gov
  M1 --> M3["Audit and Lineage Decision Records Logs Metrics"]:::gov

  M1 --> EV1
  EV1 --> EV2["Blueprint Update new fields models policies"]:::action
  EV1 --> EV3["Versioned in Registry v1 to v2 to v3"]:::action
  EV1 --> D1
```

## 6. Artifact Ownership and Placement
- **Authored and stored** in domusDocs (S3 canonical, Aurora index). 
- **Registered and exposed** by MCP for discovery and governed execution. 
- **Executed** by domusDocs pipelines; MCP brokers access and collects audit.

## 7. Standards to Anchor On
- **JSON Schema** to validate the artifact.
- **ASL** (Amazon States Language) or **Temporal** to execute workflows.
- **OpenAPI/AsyncAPI** for tool contracts and events.
- **OPA or Cedar** for policies.
- **OTEL and OpenLineage** for traces and lineage.
