from __future__ import annotations

from humetric_core import ParsedQuery, Person, Skill
from humetric_store import open_db, upsert_person

from humetric_retrieval import CorpusEmpty, build_bm25, candidate_ids


def _seed_conn():
    conn = open_db(":memory:").unwrap()
    people = [
        Person(
            id="gh:alice",
            source="github",
            name="Alice",
            headline="rust engineer building distributed systems",
            location="NYC",
            skills=(Skill.of("rust"), Skill.of("kafka")),
            follower_count=300,
        ),
        Person(
            id="gh:bob",
            source="github",
            name="Bob",
            headline="frontend developer, react fanatic",
            location="SF",
            skills=(Skill.of("react"), Skill.of("typescript")),
            follower_count=80,
        ),
        Person(
            id="gh:carol",
            source="github",
            name="Carol",
            headline="rust + cryptography, payments at fintech",
            location="London",
            skills=(Skill.of("rust"), Skill.of("payments")),
            follower_count=500,
        ),
    ]
    for p in people:
        upsert_person(conn, p).unwrap()
    return conn


def test_bm25_finds_relevant_doc() -> None:
    conn = _seed_conn()
    idx = build_bm25(conn).unwrap()
    results = idx.search("rust distributed", k=3)
    ids = [pid for pid, _ in results]
    assert ids[0] == "gh:alice"
    assert "gh:carol" in ids  # also has rust


def test_bm25_corpus_empty() -> None:
    conn = open_db(":memory:").unwrap()
    r = build_bm25(conn)
    assert r.is_err()
    assert isinstance(r.err(), CorpusEmpty)


def test_filter_by_must_skill() -> None:
    conn = _seed_conn()
    pq = ParsedQuery(free_text="x", must_skills=("rust",))
    ids = candidate_ids(conn, pq).unwrap()
    assert ids == {"gh:alice", "gh:carol"}


def test_filter_by_min_followers_and_location() -> None:
    conn = _seed_conn()
    pq = ParsedQuery(free_text="x", min_followers=200, location="London")
    ids = candidate_ids(conn, pq).unwrap()
    assert ids == {"gh:carol"}


def test_filter_no_constraints_returns_all() -> None:
    conn = _seed_conn()
    pq = ParsedQuery(free_text="anything")
    ids = candidate_ids(conn, pq).unwrap()
    assert ids == {"gh:alice", "gh:bob", "gh:carol"}
