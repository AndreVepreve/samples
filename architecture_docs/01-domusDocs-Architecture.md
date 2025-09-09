
# domusDocs Architecture (AWS + K8s + MCP Integration) — 2025-09-09

This document describes the production-grade architecture of **domusDocs** as an enterprise document-intelligence backbone. It covers components, data flow, governance, integration with the MCP server, and the data platform feed.

## 1. Goals
- Modular **LLM-based** ingestion, processing, and storage powered by **Blueprints**.
- Governed access (RBAC/ABAC), lineage, HITL approvals.
- Tight integration with **MCP** (tool catalog, policy, audit) and **Data Platform** (Data Lake/Lakehouse).
- Multi-source ingestion (APIs, frontends/apps, schedulers, external systems).

## 2. High-Level Diagram
```mermaid
flowchart LR
  classDef hdr fill:#0ea5e9,color:#fff,font-weight:700,stroke:#0ea5e9
  classDef box fill:#f8fafc,stroke:#94a3b8,color:#0f172a

  subgraph UI[Frontends and Producers]
    U1[Angular UI HITL and Ops]:::box
    U2[Partner or API Producers]:::box
    U3[Schedulers EventBridge CRON]:::box
    U4[Apps and Batch Importers]:::box
  end

  subgraph MCP[MCP Server Control Plane]
    M1[AuthZ Cognito plus OPA or Cedar]:::box
    M2[Tool Catalog and Policy Hooks]:::box
    M3[Audit and Telemetry OTEL to Grafana]:::box
  end

  subgraph DD[domusDocs on EKS]
    direction TB
    A1[API Gateway NestJS]:::box
    A2[Blueprint Registry Aurora plus S3]:::box
    A3[Ingestion Router Step Functions or Temporal]:::box
    A4[Connectors HTTP S3 SharePoint Email FTP]:::box
    P1[OCR and Parse Textract adapters]:::box
    P2[Redaction and Normalizer]:::box
    P3[Chunker]:::box
    P4[Embeddings Bedrock or Cohere]:::box
    P5[Classifier and Extractor LLM plus rules]:::box
    P6[Vector Writer OpenSearch or pgvector]:::box
    G1[Governance RBAC ABAC Lineage]:::box
  end

  subgraph STG[Storage]
    S3R[S3 Raw]:::box
    S3N[S3 Normalized or Derived]:::box
    DB[(Aurora Postgres metadata lineage)]:::box
    VEC[(Vector DB OpenSearch or pgvector)]:::box
  end

  subgraph DP[Data Platform]
    DL[S3 Data Lake Bronze Silver Gold]:::box
    CAT[Glue Catalog and Lake Formation]:::box
    DW[Redshift or Snowflake]:::box
  end

  subgraph RET[RAG and Retrieval]
    R1[Retriever API NestJS]:::box
  end

  UI-->A1
  U2-->A1
  U3-->A1
  U4-->A4

  A1-->A2
  A1-->A3
  A3-->A4-->P1-->P2-->P3-->P4-->P5-->P6

  P1-->S3N
  P2-->S3N
  P3-->S3N
  P4-->VEC
  P5-->DB
  P6-->VEC
  A4-->S3R
  A1-->DB
  G1---A1
  G1---R1

  MCP<-->A1
  MCP<-->R1

  S3N--->DL
  DB--->DL
  VEC--->DL
  DL-->CAT-->DW

  R1-->MCP
```

## 3. Responsibilities by Domain
- **API and Control Plane (NestJS on EKS):** ingress, MCP tool endpoints, policy enforcement, auditing.
- **Blueprint Registry:** versioned YAML or JSON; schema-validated; canary flags.
- **Ingestion Orchestration:** Step Functions or Temporal; idempotent workers; retries.
- **Processing:** Textract OCR, redaction, chunking, embeddings, classification/extraction.
- **Storage:** S3 (raw/normalized/derived), Aurora (metadata/extractions/lineage), Vector DB (OpenSearch or pgvector).
- **Governance:** RBAC, ABAC, redaction, lineage, HITL.
- **RAG Service:** hybrid retrieval BM25 then vector KNN then rerank; strict ABAC filters.
- **Data Platform Feed:** Bronze/Silver/Gold zones; Glue Catalog; Lake Formation policies.

## 4. Contracts
- **S3 layout:** raw, normalized, derived, manifest per document.
- **Aurora tables:** documents, extractions, lineage, decisions.
- **OpenSearch mapping:** embedding knn_vector and ABAC metadata fields.
- **MCP Tool Endpoints:** docs.search, docs.retrieve, docs.metadata; governed by OPA or Cedar.

## 5. Security, Reliability, Cost
- **Identity:** Cognito, IRSA; KMS encryption; private networking via VPC endpoints.
- **Observability:** OpenTelemetry traces; structured logs; SLOs.
- **Backpressure:** queues, admission controller; budget caps per blueprint.
- **Cost controls:** Intelligent-Tiering on S3; serverless Aurora; rightsize OpenSearch; batch embeddings.
