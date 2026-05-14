from __future__ import annotations

from typing import Any

import numpy as np
from humetric_core import Ok, ParsedQuery, Person, Skill
from humetric_embed import TextEncoder
from humetric_store import VectorIndex, open_db, upsert_person

from humetric_retrieval import DenseBranch, build_engine


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


def _seed():
    conn = open_db(":memory:").unwrap()
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
    return conn


def test_end_to_end_bm25_plus_dense() -> None:
    conn = _seed()
    # Synthetic 3-d space: rust direction, react direction, payments direction.
    rust = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    react = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    payments = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vecs = VectorIndex(conn, dim=3, kind="text")
    vecs.add_batch(
        [
            ("gh:alice", rust),
            ("gh:bob", react),
            ("gh:carol", rust + payments),
        ]
    ).unwrap()

    encoder = _FakeEncoder({"rust engineer": rust, "react dev": react})
    dense = DenseBranch(encoder=encoder, index=vecs)

    engine = build_engine(conn, text_branch=dense).unwrap()
    cands = engine.search(ParsedQuery(free_text="rust engineer"), k=3).unwrap()
    ids = [c.person_id for c in cands]
    assert ids[0] == "gh:alice"
    assert set(ids[:2]) == {"gh:alice", "gh:carol"}


def test_filter_intersects_with_dense_branch() -> None:
    conn = _seed()
    rust = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    payments = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vecs = VectorIndex(conn, dim=3, kind="text")
    vecs.add_batch(
        [
            ("gh:alice", rust),
            ("gh:carol", rust + payments),
        ]
    ).unwrap()
    encoder = _FakeEncoder({"rust": rust})
    dense = DenseBranch(encoder=encoder, index=vecs)
    engine = build_engine(conn, text_branch=dense).unwrap()

    pq = ParsedQuery(free_text="rust", must_skills=("payments",))
    cands = engine.search(pq, k=5).unwrap()
    ids = [c.person_id for c in cands]
    assert ids == ["gh:carol"]


def test_ok_is_ok() -> None:
    # Sanity smoke
    assert Ok(1).unwrap() == 1
