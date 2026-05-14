from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from humetric_core import Err, Ok, Result
from humetric_embed import TextEncoder
from humetric_store import VectorIndex

from humetric_retrieval.errors import EmbedWrapped, RetrievalError, StoreWrapped


@dataclass(slots=True)
class DenseBranch:
    """Encode-and-search over a single VectorIndex (text or two-tower)."""

    encoder: TextEncoder
    index: VectorIndex

    def search(self, query: str, k: int) -> Result[list[tuple[str, float]], RetrievalError]:
        v_r = self.encoder.encode_one(query)
        if isinstance(v_r, Err):
            return Err(EmbedWrapped(cause=v_r.error))
        s_r = self.index.search(v_r.value, k)
        if isinstance(s_r, Err):
            return Err(StoreWrapped(cause=s_r.error))
        return Ok(s_r.value)


@dataclass(slots=True)
class StaticVectorBranch:
    """Precomputed person embedding (graph or tower) → cosine search.

    No encoder: the query side projects onto the same space by taking the
    mean of seed-person vectors (e.g. the recruiter's interest set).
    """

    index: VectorIndex

    def search_by_vector(
        self, vec: np.ndarray, k: int
    ) -> Result[list[tuple[str, float]], RetrievalError]:
        r = self.index.search(vec, k)
        if isinstance(r, Err):
            return Err(StoreWrapped(cause=r.error))
        return Ok(r.value)
