
# Blueprint Schema and Example — 2025-09-09

## 1. Minimal JSON Schema (Starter)
```json
{
  "$id": "https://org.example/schemas/blueprint.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "domusDocs Blueprint",
  "type": "object",
  "required": ["id","version","title","scope","sources","processing","storage","governance"],
  "properties": {
    "id": { "type": "string", "pattern": "^[a-z0-9-]{3,64}$" },
    "version": { "type": "integer", "minimum": 1 },
    "title": { "type": "string" },
    "description": { "type": "string" },
    "scope": {"type":"object"},
    "sources": {"type":"object"},
    "ingestion": {"type":"object"},
    "processing": {"type":"object"},
    "fusion": {"type":"object"},
    "visualization": {"type":"object"},
    "synthesis": {"type":"object"},
    "storage": {"type":"object"},
    "governance": {"type":"object"},
    "actions": {"type":"object"},
    "execution": {"type":"object"},
    "inputs": {"type":"object"},
    "contracts": {"type":"object"},
    "observability": {"type":"object"},
    "tests": {"type":"object"}
  },
  "additionalProperties": false
}
```

## 2. Example Blueprint (YAML)
See `blueprints/allocations/allocations-visual-v1.yaml` below.

```yaml
id: allocations-visual-v1
version: 1
title: Investor Allocation Visualization Regions
description: >
  Region allocation chart and narrative for an investor as of a date.
scope:
  tenant_mode: multi
  roles_allowed: [Advisor, Client, Ops]
  allowlist_groups: [Allocations-Canary]
sources:
  domusDocs:
    include_tags: [policy, allocation_limits, product]
    status: approved
ingestion:
  ocr: none
  parse: { tables: false, headers: false }
processing:
  chunking: { strategy: headings, max_tokens: 450, overlap_tokens: 50 }
  embeddings: { provider: bedrock, model: amazon.titan-embed-text-v2, dims: 1024 }
fusion:
  metrics:
    tool_id: data.metrics
    args_template:
      get: ["allocation_pct"]
      dims: ["region"]
      where: "investorId={inputs.investorId} AND as_of={inputs.asOf|now}"
visualization:
  type: bar
  library: quicksight|d3
  schema: { x: region, y: allocation_pct }
synthesis:
  format: "chart plus narrative plus citations"
storage:
  vectors: opensearch:index=alloc-vec-v1
  metadata: aurora:domusdocs
  blobs: s3://org-knowledge/derived/allocations/
governance:
  pii_redaction: true
  hitl_required: false
  retention_days: 3650
  audit_level: full
execution:
  mode: step_functions
  asl_s3: s3://blueprints/allocations/asl/allocations-visual-v1.asl.json
inputs:
  required: [investorId]
  optional: [asOf]
```

## 3. Minimal ASL (Step Functions) Stub
```json
{
  "Comment": "Allocations Visualization Regions",
  "StartAt": "ResolveInputs",
  "States": {
    "ResolveInputs": {"Type":"Pass","Next":"FetchMetrics"},
    "FetchMetrics": {"Type":"Task","Resource":"arn:aws:states:::lambda:invoke","Next":"Synthesize"},
    "Synthesize": {"Type":"Task","Resource":"arn:aws:states:::lambda:invoke","End":true}
  }
}
```
