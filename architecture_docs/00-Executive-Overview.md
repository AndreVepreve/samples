
# Executive Overview: MCP-centered AI Transformation — 2025-09-09

- **domusDocs** provides modular LLM-based ingestion with governed retrieval and a blueprint registry.
- **MCP** is the control plane exposing governed tools and enforcing policy, audit, and observability.
- **Data Platform** consumes normalized outputs and powers analytics and models.
- **Agents** use MCP tools to retrieve, reason, and act (visualizations, tickets, workflows).

## Executive Diagram (Core Only)
```mermaid
graph TD
  classDef core fill:#0ea5e9,stroke:#0ea5e9,color:#fff,font-weight:700;
  classDef comp fill:#f8fafc,stroke:#94a3b8,color:#0f172a;

  DD["domusDocs"]:::core
  MCP["MCP Server"]:::core
  DP["Data Platform"]:::core
  AG["Agents"]:::core
  PR["Digital Products"]:::core

  DD --> MCP
  DP --> MCP
  MCP --> AG
  MCP --> PR
  DD --> DP
```
