from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import psycopg
from humetric_core import Err, Ok, ParsedQuery, Result

from humetric_retrieval.bm25 import Bm25Index, build_bm25
from humetric_retrieval.dense import DenseBranch, StaticVectorBranch
from humetric_retrieval.errors import RetrievalError
from humetric_retrieval.filters import candidate_ids
from humetric_retrieval.fuse import rrf


@dataclass(frozen=True, slots=True)
class Candidate:
    person_id: str
    score: float


@dataclass(slots=True)
class SearchEngine:
    conn: psycopg.Connection
    bm25: Bm25Index
    text_branch: DenseBranch | None = None
    graph_branch: StaticVectorBranch | None = None
    tower_branch: StaticVectorBranch | None = None
    # Graph and tower vectors live in different spaces (different dims), so
    # each branch needs its own seed/interest vector. None disables the branch.
    interest_vector_graph: np.ndarray | None = None
    interest_vector_tower: np.ndarray | None = None
    per_branch_k: int = 200
    fuse_k: int = 60
    _candidate_cache: dict[str, set[str]] = field(default_factory=dict, init=False)

    def search(self, parsed: ParsedQuery, k: int = 50) -> Result[list[Candidate], RetrievalError]:
        # 1. Hard filter to a candidate id set. ``None`` means no filter was
        # specified; a (possibly empty) set means the filter was applied.
        allowed_r = candidate_ids(self.conn, parsed)
        if isinstance(allowed_r, Err):
            return allowed_r
        allowed = allowed_r.value

        def _restrict(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
            if allowed is None:
                return ranked
            return [p for p in ranked if p[0] in allowed]

        # 2. Gather rankings from each enabled branch.
        rankings: list[list[tuple[str, float]]] = []

        bm = self.bm25.search(parsed.free_text, self.per_branch_k)
        if bm:
            rankings.append(_restrict(bm))

        if self.text_branch is not None:
            dr = self.text_branch.search(parsed.free_text, self.per_branch_k)
            if isinstance(dr, Err):
                return dr
            rankings.append(_restrict(dr.value))

        if self.graph_branch is not None and self.interest_vector_graph is not None:
            gr = self.graph_branch.search_by_vector(self.interest_vector_graph, self.per_branch_k)
            if isinstance(gr, Err):
                return gr
            rankings.append(_restrict(gr.value))

        if self.tower_branch is not None and self.interest_vector_tower is not None:
            tr = self.tower_branch.search_by_vector(self.interest_vector_tower, self.per_branch_k)
            if isinstance(tr, Err):
                return tr
            rankings.append(_restrict(tr.value))

        # 3. Reciprocal Rank Fusion.
        fused = rrf(rankings, k=self.fuse_k, top_n=k)
        return Ok([Candidate(person_id=pid, score=score) for pid, score in fused])


def build_engine(
    conn: psycopg.Connection,
    *,
    bm25: Bm25Index | None = None,
    text_branch: DenseBranch | None = None,
    graph_branch: StaticVectorBranch | None = None,
    tower_branch: StaticVectorBranch | None = None,
    interest_vector_graph: np.ndarray | None = None,
    interest_vector_tower: np.ndarray | None = None,
) -> Result[SearchEngine, RetrievalError]:
    # Pass a pre-built/opened `bm25` to avoid re-tokenizing the corpus on
    # every startup. Callers that don't pass one get an in-memory rebuild.
    if bm25 is None:
        bm25_r = build_bm25(conn)
        if isinstance(bm25_r, Err):
            return bm25_r
        bm25 = bm25_r.value
    return Ok(
        SearchEngine(
            conn=conn,
            bm25=bm25,
            text_branch=text_branch,
            graph_branch=graph_branch,
            tower_branch=tower_branch,
            interest_vector_graph=interest_vector_graph,
            interest_vector_tower=interest_vector_tower,
        )
    )
