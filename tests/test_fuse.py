from __future__ import annotations

from humetric_retrieval import rrf


def test_rrf_combines_two_rankings_with_consensus_first() -> None:
    r1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    r2 = [("b", 0.95), ("a", 0.85), ("d", 0.5)]
    fused = rrf([r1, r2], k=60)
    ids = [pid for pid, _ in fused]
    # a and b appear in both; should rank above c (only r1) and d (only r2).
    assert ids[:2] == ["a", "b"] or ids[:2] == ["b", "a"]
    assert set(ids[:2]) == {"a", "b"}
    assert ids[2:] == ["c", "d"] or ids[2:] == ["d", "c"]


def test_rrf_empty_input_returns_empty() -> None:
    assert rrf([]) == []
    assert rrf([[]]) == []


def test_rrf_single_ranking_preserves_order() -> None:
    r = [("a", 0.0), ("b", 0.0), ("c", 0.0)]
    fused = rrf([r])
    assert [pid for pid, _ in fused] == ["a", "b", "c"]


def test_rrf_top_n_clips() -> None:
    r = [(f"p{i}", 0.0) for i in range(50)]
    fused = rrf([r], top_n=5)
    assert len(fused) == 5
