from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, cast

import psycopg
import tantivy
from humetric_core import Err, Ok, Result
from humetric_store import list_persons

from humetric_retrieval.errors import (
    Bm25IndexFailed,
    CorpusEmpty,
    RetrievalError,
    StoreWrapped,
)

_PID_FIELD = "pid"
_BLOB_FIELD = "blob"


def _build_schema() -> Any:
    b = tantivy.SchemaBuilder()
    b.add_text_field(_PID_FIELD, stored=True)
    b.add_text_field(_BLOB_FIELD, stored=False)
    return b.build()


class Bm25Index:
    """Tantivy-backed BM25 index. Persistent on disk when built/opened with a
    path, in-memory otherwise. `search(query, k)` returns `(person_id, score)`."""

    __slots__ = ("_index",)

    def __init__(self, index: Any) -> None:
        self._index = index

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not query.strip():
            return []
        # Lenient parser tolerates LLM-output free text (`:`, `&&`, stray
        # punctuation) instead of raising on syntax it doesn't understand.
        parsed, _warnings = self._index.parse_query_lenient(query, [_BLOB_FIELD])
        searcher = self._index.searcher()
        try:
            hits = searcher.search(parsed, limit=k).hits
        except ValueError:
            # Empty query after lenient parsing -> no terms to score against.
            return []
        out: list[tuple[str, float]] = []
        for score, addr in hits:
            doc = searcher.doc(addr)
            pid = doc.get_first(_PID_FIELD)
            if isinstance(pid, str):
                out.append((pid, float(score)))
        return out


def _open_or_create(schema: Any, path: Path) -> Any:
    if cast(Any, tantivy.Index).exists(str(path)):
        return cast(Any, tantivy.Index).open(str(path))
    return tantivy.Index(schema, path=str(path))


def build_bm25(
    conn: psycopg.Connection,
    path: str | Path | None = None,
) -> Result[Bm25Index, RetrievalError]:
    """Materialize a BM25 index from every Person in the store.

    `path=None` builds an in-memory index (tests, short-lived processes).
    A directory path persists the index so future processes can `open_bm25`
    without re-tokenizing the entire corpus.
    """
    schema = _build_schema()

    try:
        if path is None:
            index = tantivy.Index(schema)
            target = "<memory>"
        else:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            index = _open_or_create(schema, p)
            target = str(p)
    except (OSError, ValueError) as e:
        return Err(Bm25IndexFailed(path=str(path) if path else "<memory>", reason=str(e)))

    try:
        writer = index.writer(heap_size=64_000_000, num_threads=1)
        # Full rebuild semantics: callers can re-run `build_bm25` to refresh
        # an existing on-disk index without leftover docs from a prior corpus.
        writer.delete_all_documents()
    except ValueError as e:
        return Err(Bm25IndexFailed(path=target, reason=str(e)))

    page = 0
    total = 0
    while True:
        r = list_persons(conn, limit=2000, offset=page * 2000)
        if isinstance(r, Err):
            return Err(StoreWrapped(cause=r.error))
        rows = r.value
        if not rows:
            break
        for person in rows:
            doc = tantivy.Document()
            doc.add_text(_PID_FIELD, person.id)
            doc.add_text(_BLOB_FIELD, person.text_blob())
            writer.add_document(doc)
            total += 1
        page += 1

    if total == 0:
        with contextlib.suppress(ValueError):
            writer.rollback()
        return Err(CorpusEmpty())

    try:
        writer.commit()
        writer.wait_merging_threads()
        index.reload()
    except ValueError as e:
        return Err(Bm25IndexFailed(path=target, reason=str(e)))

    return Ok(Bm25Index(index))


def open_bm25(path: str | Path) -> Result[Bm25Index, RetrievalError]:
    """Open a previously-built BM25 index from disk. Cheap; no DB scan."""
    p = Path(path)
    if not p.is_dir():
        return Err(Bm25IndexFailed(path=str(p), reason="index directory does not exist"))
    try:
        if not cast(Any, tantivy.Index).exists(str(p)):
            return Err(Bm25IndexFailed(path=str(p), reason="index not found"))
        index = cast(Any, tantivy.Index).open(str(p))
        index.reload()
    except (OSError, ValueError) as e:
        return Err(Bm25IndexFailed(path=str(p), reason=str(e)))
    return Ok(Bm25Index(index))
