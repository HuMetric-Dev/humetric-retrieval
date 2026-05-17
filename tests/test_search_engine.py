from __future__ import annotations

from typing import Any

import numpy as np
import psycopg
from humetric_core import Ok, ParsedQuery, Person, Skill
from humetric_embed import TextEncoder
from humetric_store import VectorIndex, upsert_person
from humetric_store.db import VECTOR_DIMS

from humetric_retrieval import DenseBranch, build_engine

_DIM = VECTOR_DIMS["graph"]


def _onehot(idx: int) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[idx] = 1.0
    return v


class _FakeEncoder(TextEncoder):
    """An encoder that maps single-word queries to a fixed direction.

    Useful for unit-testing the dense branch without downloading a model.
    """

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        super().__init__(model="fake")
        self._fake_dim = next(iter(mapping.values())).shape[0]
        self._dim = self._fake_dim
        self._mapping = mapping

        class _M:
            def encode(_self, texts: list[str], **_kw: Any) -> np.ndarray:
                rows: list[np.ndarray] = []
                for t in texts:
                    rows.append(self._mapping.get(t, np.zeros(self._fake_dim, dtype=np.float32)))
                return np.stack(rows).astype(np.float32)

            def get_sentence_embedding_dimension(_self) -> int:
                return self._fake_dim

        self._model = _M()


def _seed(conn: psycopg.Connection) -> None:
    people = [
        Person(
            id="gh:alice",
            source="github",
            name="Alice",
            headline="rust engineer building distributed systems",
            skills=(Skill.of("rust"),),
        ),
        Person(
            id="gh:bob",
            source="github",
            name="Bob",
            headline="frontend developer, react",
            skills=(Skill.of("react"),),
        ),
        Person(
            id="gh:carol",
            source="github",
            name="Carol",
            headline="rust + payments",
            skills=(Skill.of("rust"), Skill.of("payments")),
        ),
    ]
    for p in people:
        upsert_person(conn, p).unwrap()


def test_end_to_end_bm25_plus_dense(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    rust = _onehot(0)
    react = _onehot(1)
    payments = _onehot(2)

    vecs = VectorIndex(pg_conn, dim=_DIM, kind="graph")
    vecs.add_batch(
        [
            ("gh:alice", rust),
            ("gh:bob", react),
            ("gh:carol", rust + payments),
        ]
    ).unwrap()

    encoder = _FakeEncoder({"rust engineer": rust, "react dev": react})
    dense = DenseBranch(encoder=encoder, index=vecs)

    engine = build_engine(pg_conn, text_branch=dense).unwrap()
    cands = engine.search(ParsedQuery(free_text="rust engineer"), k=3).unwrap()
    ids = [c.entity_id for c in cands]
    assert ids[0] == "gh:alice"
    assert set(ids[:2]) == {"gh:alice", "gh:carol"}


def test_filter_intersects_with_dense_branch(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    rust = _onehot(0)
    payments = _onehot(2)

    vecs = VectorIndex(pg_conn, dim=_DIM, kind="graph")
    vecs.add_batch(
        [
            ("gh:alice", rust),
            ("gh:carol", rust + payments),
        ]
    ).unwrap()
    encoder = _FakeEncoder({"rust": rust})
    dense = DenseBranch(encoder=encoder, index=vecs)
    engine = build_engine(pg_conn, text_branch=dense).unwrap()

    pq = ParsedQuery(free_text="rust", must_skills=("payments",))
    cands = engine.search(pq, k=5).unwrap()
    ids = [c.entity_id for c in cands]
    assert ids == ["gh:carol"]


def test_unsatisfiable_filter_returns_zero_candidates(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    rust = _onehot(0)
    payments = _onehot(2)

    vecs = VectorIndex(pg_conn, dim=_DIM, kind="graph")
    vecs.add_batch(
        [
            ("gh:alice", rust),
            ("gh:carol", rust + payments),
        ]
    ).unwrap()
    encoder = _FakeEncoder({"rust": rust})
    dense = DenseBranch(encoder=encoder, index=vecs)
    engine = build_engine(pg_conn, text_branch=dense).unwrap()

    pq = ParsedQuery(free_text="rust", must_skills=("haskell",))
    cands = engine.search(pq, k=5).unwrap()
    assert cands == []


def test_ok_is_ok() -> None:
    assert Ok(1).unwrap() == 1


def test_organization_branch_returns_org_candidates(pg_conn: psycopg.Connection) -> None:
    from humetric_core import Organization
    from humetric_store import upsert_organization

    from humetric_retrieval import TypeBranches, build_bm25, build_engine

    _seed(pg_conn)
    for o in (
        Organization(
            id="o:gh:anthropic",
            source="github",
            name="Anthropic",
            org_kind="company",
            headline="AI safety lab",
        ),
        Organization(
            id="o:gh:openai",
            source="github",
            name="OpenAI",
            org_kind="company",
            headline="AI research and deployment",
        ),
    ):
        upsert_organization(pg_conn, o).unwrap()

    persons_bm25 = build_bm25(pg_conn, table="persons").unwrap()
    orgs_bm25 = build_bm25(pg_conn, table="organizations").unwrap()

    engine = build_engine(
        pg_conn,
        persons=TypeBranches(bm25=persons_bm25),
        organizations=TypeBranches(bm25=orgs_bm25),
    ).unwrap()

    pq = ParsedQuery(free_text="ai safety", target_entity_types=("organization",))
    cands = engine.search(pq, k=5).unwrap()
    assert all(c.entity_type == "organization" for c in cands)
    assert any(c.entity_id == "o:gh:anthropic" for c in cands)

    # Multi-type query: persons block first, then orgs block.
    pq_both = ParsedQuery(
        free_text="rust ai",
        target_entity_types=("person", "organization"),
    )
    both = engine.search(pq_both, k=5).unwrap()
    types_in_order = [c.entity_type for c in both]
    if "person" in types_in_order and "organization" in types_in_order:
        first_org = types_in_order.index("organization")
        last_person = len(types_in_order) - 1 - list(reversed(types_in_order)).index("person")
        assert last_person < first_org, "person block should precede org block"
