# humetric-retrieval

Hybrid candidate generation. Four signals, fused via Reciprocal Rank Fusion:

1. **BM25** over profile text (`rank_bm25`).
2. **Dense** ANN over text embeddings (FAISS via `humetric-store`).
3. **Graph** ANN over LightGCN embeddings (only when the bundle is loaded).
4. **Hard filters** from a `ParsedQuery` — SQL WHERE pre-filter on persons + skills.

```python
from humetric_retrieval import SearchEngine
engine = SearchEngine.build(conn, text_index=..., text_encoder=...).unwrap()
cands = engine.search(parsed_query, k=50).unwrap()
```

Graph and two-tower branches are wired but optional: pass them at `build()` and they're added to the fusion automatically. Skipping either degrades quietly to text-only.
