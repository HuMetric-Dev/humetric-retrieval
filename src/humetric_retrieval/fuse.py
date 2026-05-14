from __future__ import annotations

from collections.abc import Sequence


def rrf(
    rankings: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across ranked lists of (id, branch_score).

    The branch_score is ignored — only the rank within its list matters.
    A higher final score = a better consensus across branches.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    if top_n is not None:
        fused = fused[:top_n]
    return fused
