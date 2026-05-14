from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
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
    conn: sqlite3.Connection
    bm25: Bm25Index
    text_branch: DenseBranch | None = None
    graph_branch: StaticVectorBranch | None = None
    tower_branch: StaticVectorBranch | None = None
    interest_vector: np.ndarray | None = None  # for graph/tower branches
    per_branch_k: int = 200
    fuse_k: int = 60
    _candidate_cache: dict[str, set[str]] = field(default_factory=dict, init=False)

    def search(self, parsed: ParsedQuery, k: int = 50) -> Result[list[Candidate], RetrievalError]:
        # 1. Hard filter to a candidate id set.
        allowed_r = candidate_ids(self.conn, parsed)
        if isinstance(allowed_r, Err):
            return allowed_r
        allowed = allowed_r.value

        # 2. Gather rankings from each enabled branch.
        rankings: list[list[tuple[str, float]]] = []

        bm = self.bm25.search(parsed.free_text, self.per_branch_k)
        if bm:
            rankings.append([p for p in bm if not allowed or p[0] in allowed])

        if self.text_branch is not None:
            dr = self.text_branch.search(parsed.free_text, self.per_branch_k)
            if isinstance(dr, Err):
                return dr
            rankings.append([p for p in dr.value if not allowed or p[0] in allowed])

        if self.graph_branch is not None and self.interest_vector is not None:
            gr = self.graph_branch.search_by_vector(self.interest_vector, self.per_branch_k)
            if isinstance(gr, Err):
                return gr
            rankings.append([p for p in gr.value if not allowed or p[0] in allowed])

        if self.tower_branch is not None and self.interest_vector is not None:
            tr = self.tower_branch.search_by_vector(self.interest_vector, self.per_branch_k)
            if isinstance(tr, Err):
                return tr
            rankings.append([p for p in tr.value if not allowed or p[0] in allowed])

        # 3. Reciprocal Rank Fusion.
        fused = rrf(rankings, k=self.fuse_k, top_n=k)
        return Ok([Candidate(person_id=pid, score=score) for pid, score in fused])


def build_engine(
    conn: sqlite3.Connection,
    *,
    text_branch: DenseBranch | None = None,
    graph_branch: StaticVectorBranch | None = None,
    tower_branch: StaticVectorBranch | None = None,
    interest_vector: np.ndarray | None = None,
) -> Result[SearchEngine, RetrievalError]:
    bm25_r = build_bm25(conn)
    if isinstance(bm25_r, Err):
        return bm25_r
    return Ok(
        SearchEngine(
            conn=conn,
            bm25=bm25_r.value,
            text_branch=text_branch,
            graph_branch=graph_branch,
            tower_branch=tower_branch,
            interest_vector=interest_vector,
        )
    )
