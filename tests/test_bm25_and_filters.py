from __future__ import annotations

from pathlib import Path

import psycopg
from humetric_core import ParsedQuery, Person, Skill
from humetric_store import upsert_person

from humetric_retrieval import (
    Bm25IndexFailed,
    CorpusEmpty,
    build_bm25,
    candidate_ids,
    open_bm25,
)


def _seed(conn: psycopg.Connection) -> None:
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


def test_bm25_finds_relevant_doc(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    idx = build_bm25(pg_conn).unwrap()
    results = idx.search("rust distributed", k=3)
    ids = [pid for pid, _ in results]
    assert ids[0] == "gh:alice"
    assert "gh:carol" in ids


def test_bm25_corpus_empty(pg_conn: psycopg.Connection) -> None:
    r = build_bm25(pg_conn)
    assert r.is_err()
    assert isinstance(r.err(), CorpusEmpty)


def test_filter_by_must_skill(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    pq = ParsedQuery(free_text="x", must_skills=("rust",))
    ids = candidate_ids(pg_conn, pq).unwrap()
    assert ids == {"gh:alice", "gh:carol"}


def test_filter_by_min_followers_and_location(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    pq = ParsedQuery(free_text="x", min_followers=200, location="London")
    ids = candidate_ids(pg_conn, pq).unwrap()
    assert ids == {"gh:carol"}


def test_filter_no_constraints_returns_none(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    pq = ParsedQuery(free_text="anything")
    assert candidate_ids(pg_conn, pq).unwrap() is None


def test_filter_with_unsatisfiable_skill_returns_empty_set(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    pq = ParsedQuery(free_text="x", must_skills=("rust", "haskell"))
    ids = candidate_ids(pg_conn, pq).unwrap()
    assert ids == set()


def test_filter_normalizes_spaced_skill_names(pg_conn: psycopg.Connection) -> None:
    upsert_person(
        pg_conn,
        Person(
            id="gh:dave",
            source="github",
            name="Dave",
            skills=(Skill.of("rust"), Skill.of("distributed-systems")),
        ),
    ).unwrap()
    pq = ParsedQuery(free_text="x", must_skills=("rust", "distributed systems"))
    ids = candidate_ids(pg_conn, pq).unwrap()
    assert ids == {"gh:dave"}


def test_bm25_persists_to_disk_and_reopens(tmp_path: Path, pg_conn: psycopg.Connection) -> None:
    idx_path = tmp_path / "bm25.idx"
    _seed(pg_conn)
    build_bm25(pg_conn, idx_path).unwrap()
    pg_conn.close()

    reopened = open_bm25(idx_path).unwrap()
    results = reopened.search("rust distributed", k=3)
    assert results[0][0] == "gh:alice"


def test_open_bm25_missing_path() -> None:
    r = open_bm25("/nonexistent/bm25/path")
    assert r.is_err()
    assert isinstance(r.err(), Bm25IndexFailed)


def test_bm25_lenient_on_punctuation(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn)
    idx = build_bm25(pg_conn).unwrap()
    results = idx.search("rust && distributed:", k=3)
    assert any(pid == "gh:alice" for pid, _ in results)
