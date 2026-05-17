from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import psycopg
from humetric_core import EntityType, Err, Ok, ParsedQuery, Result

from humetric_retrieval.bm25 import Bm25Index, build_bm25
from humetric_retrieval.dense import DenseBranch, StaticVectorBranch
from humetric_retrieval.errors import RetrievalError
from humetric_retrieval.filters import candidate_ids, organization_candidate_ids
from humetric_retrieval.fuse import rrf


@dataclass(frozen=True, slots=True)
class Candidate:
    entity_id: str
    entity_type: EntityType
    score: float


@dataclass(slots=True)
class TypeBranches:
    """All retrieval indices for one entity type (person or organization).

    Each branch is independent; missing branches just don't contribute to the
    RRF fusion. `interest_vector_*` are the query-side projections onto the
    graph/tower vector spaces — typically a centroid of the recruiter's
    recently-viewed entities.
    """

    bm25: Bm25Index
    text_branch: DenseBranch | None = None
    graph_branch: StaticVectorBranch | None = None
    tower_branch: StaticVectorBranch | None = None
    interest_vector_graph: np.ndarray | None = None
    interest_vector_tower: np.ndarray | None = None


@dataclass(slots=True)
class SearchEngine:
    conn: psycopg.Connection
    persons: TypeBranches | None = None
    organizations: TypeBranches | None = None
    per_branch_k: int = 200
    fuse_k: int = 60
    _candidate_cache: dict[str, set[str]] = field(default_factory=dict, init=False)

    def search(
        self,
        parsed: ParsedQuery,
        k: int = 50,
        entity_types: tuple[EntityType, ...] | None = None,
    ) -> Result[list[Candidate], RetrievalError]:
        """Search the enabled entity types and return per-type ranked blocks
        concatenated in `entity_types` (or `parsed.target_entity_types`) order.

        Each block is independently RRF-fused and capped at `k`. The CLI/
        orchestrator may group the resulting blocks by `entity_type` for
        rendering — they are not cross-type calibrated.
        """
        targets = entity_types or parsed.target_entity_types
        out: list[Candidate] = []
        for et in targets:
            branches = self._branches_for(et)
            if branches is None:
                continue
            sub = self._search_one_type(parsed, k, et, branches)
            if isinstance(sub, Err):
                return sub
            out.extend(sub.value)
        return Ok(out)

    def _branches_for(self, et: EntityType) -> TypeBranches | None:
        if et == "person":
            return self.persons
        if et == "organization":
            return self.organizations
        return None

    def _search_one_type(
        self,
        parsed: ParsedQuery,
        k: int,
        et: EntityType,
        branches: TypeBranches,
    ) -> Result[list[Candidate], RetrievalError]:
        if et == "person":
            allowed_r = candidate_ids(self.conn, parsed)
        else:
            allowed_r = organization_candidate_ids(self.conn, parsed)
        if isinstance(allowed_r, Err):
            return allowed_r
        allowed = allowed_r.value

        def _restrict(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
            if allowed is None:
                return ranked
            return [p for p in ranked if p[0] in allowed]

        rankings: list[list[tuple[str, float]]] = []

        bm = branches.bm25.search(parsed.free_text, self.per_branch_k)
        if bm:
            rankings.append(_restrict(bm))

        if branches.text_branch is not None:
            dr = branches.text_branch.search(parsed.free_text, self.per_branch_k)
            if isinstance(dr, Err):
                return dr
            rankings.append(_restrict(dr.value))

        if branches.graph_branch is not None and branches.interest_vector_graph is not None:
            gr = branches.graph_branch.search_by_vector(
                branches.interest_vector_graph, self.per_branch_k
            )
            if isinstance(gr, Err):
                return gr
            rankings.append(_restrict(gr.value))

        if branches.tower_branch is not None and branches.interest_vector_tower is not None:
            tr = branches.tower_branch.search_by_vector(
                branches.interest_vector_tower, self.per_branch_k
            )
            if isinstance(tr, Err):
                return tr
            rankings.append(_restrict(tr.value))

        fused = rrf(rankings, k=self.fuse_k, top_n=k)
        return Ok([Candidate(entity_id=eid, entity_type=et, score=score) for eid, score in fused])


def build_engine(
    conn: psycopg.Connection,
    *,
    persons: TypeBranches | None = None,
    organizations: TypeBranches | None = None,
    bm25: Bm25Index | None = None,
    text_branch: DenseBranch | None = None,
    graph_branch: StaticVectorBranch | None = None,
    tower_branch: StaticVectorBranch | None = None,
    interest_vector_graph: np.ndarray | None = None,
    interest_vector_tower: np.ndarray | None = None,
) -> Result[SearchEngine, RetrievalError]:
    """Build a SearchEngine.

    Two calling conventions:
    - Pass `persons=TypeBranches(...)` and/or `organizations=TypeBranches(...)`
      for the polymorphic shape.
    - Pass the loose `bm25=`, `text_branch=`, etc. for a person-only engine
      (preserves the v0.1 caller signature; auto-builds the person branches).
    """
    if persons is None and organizations is None:
        # Convenience path: assemble a single-type person engine from the
        # loose kwargs. Mirrors the v0.1 build_engine contract.
        if bm25 is None:
            bm25_r = build_bm25(conn, table="persons")
            if isinstance(bm25_r, Err):
                return bm25_r
            bm25 = bm25_r.value
        persons = TypeBranches(
            bm25=bm25,
            text_branch=text_branch,
            graph_branch=graph_branch,
            tower_branch=tower_branch,
            interest_vector_graph=interest_vector_graph,
            interest_vector_tower=interest_vector_tower,
        )

    return Ok(SearchEngine(conn=conn, persons=persons, organizations=organizations))
