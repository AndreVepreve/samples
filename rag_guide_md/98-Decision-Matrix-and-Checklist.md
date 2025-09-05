# Decision Matrix — Build vs Buy vs Hybrid

| Criterion | Build (LangChain/LlamaIndex + FAISS/Chroma/OpenSearch) | Buy (Hosted RAG/Search) | Hybrid |
|---|---|---|---|
| Control & sovereignty | Full — choose LLMs, indexes, metadata & ACL model | Limited; vendor choices & opaque internals | Own retrieval/governance; vendor UI |
| Time-to-value | Slower initial setup | Fast pilot | Medium |
| Governance & ACLs | Index-time filters; Kendra/OpenSearch ACLs | Varies by vendor | Strong (you own policies) |
| Performance tuning | Index type (HNSW/IVF, PQ), rerankers | Limited knobs | Key knobs owned |
| Portability/lock-in | High portability | Risk of lock-in | Reduced lock-in |
| Cost profile | Infra + ops; can optimize | Subscription; predictable | Mixed |

---

# Content Readiness Checklist (share with doc owners)

- **Format**: native text (not scans); if OCR, perform QA.
- **Structure**: headings, lists, tables; avoid wall-of-text.
- **Clarity**: remove duplicates/contradictions; resolve version conflicts.
- **Freshness**: versioned; `effective_date` & **owner** assigned.
- **Classification**: confidentiality tags; export controls.
- **Access**: RBAC/ABAC tags set; who can see what is explicit.
- **Metadata**: `source_uri`, `title`, `section`, `product`, `jurisdiction`.
- **Testing**: sample Q&A for each document; SME sign-off.
