from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, cast

import psycopg
import tantivy
from humetric_core import Err, Ok, Result
from humetric_store import list_organizations, list_persons

from humetric_retrieval.errors import (
    Bm25IndexFailed,
    CorpusEmpty,
    RetrievalError,
    StoreWrapped,
)

_ID_FIELD = "eid"
_BLOB_FIELD = "blob"


def _build_schema() -> Any:
    b = tantivy.SchemaBuilder()
    b.add_text_field(_ID_FIELD, stored=True)
    b.add_text_field(_BLOB_FIELD, stored=False)
    return b.build()


class Bm25Index:
    """Tantivy-backed BM25 index. Persistent on disk when built/opened with a
    path, in-memory otherwise. `search(query, k)` returns `(entity_id, score)`.

    One index covers one entity table (persons or organizations). The
    SearchEngine fuses per-type results via RRF.
    """

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
            return []
        out: list[tuple[str, float]] = []
        for score, addr in hits:
            doc = searcher.doc(addr)
            eid = doc.get_first(_ID_FIELD)
            if isinstance(eid, str):
                out.append((eid, float(score)))
        return out


def _open_or_create(schema: Any, path: Path) -> Any:
    if cast(Any, tantivy.Index).exists(str(path)):
        return cast(Any, tantivy.Index).open(str(path))
    return tantivy.Index(schema, path=str(path))


def _iter_corpus(
    conn: psycopg.Connection, table: str
) -> Result[list[tuple[str, str]], RetrievalError]:
    """Stream (entity_id, text_blob) pairs from the chosen entity table."""
    out: list[tuple[str, str]] = []
    page = 0
    while True:
        if table == "persons":
            r = list_persons(conn, limit=2000, offset=page * 2000)
            if isinstance(r, Err):
                return Err(StoreWrapped(cause=r.error))
            rows = r.value
            if not rows:
                break
            for p in rows:
                out.append((p.id, p.text_blob()))
        elif table == "organizations":
            ro = list_organizations(conn, limit=2000, offset=page * 2000)
            if isinstance(ro, Err):
                return Err(StoreWrapped(cause=ro.error))
            rows_o = ro.value
            if not rows_o:
                break
            for o in rows_o:
                out.append((o.id, o.text_blob()))
        else:
            return Err(Bm25IndexFailed(path=table, reason=f"unknown entity table {table!r}"))
        page += 1
    return Ok(out)


def build_bm25(
    conn: psycopg.Connection,
    path: str | Path | None = None,
    table: str = "persons",
) -> Result[Bm25Index, RetrievalError]:
    """Materialize a BM25 index over the `table` corpus.

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
        writer.delete_all_documents()
    except ValueError as e:
        return Err(Bm25IndexFailed(path=target, reason=str(e)))

    corpus_r = _iter_corpus(conn, table)
    if isinstance(corpus_r, Err):
        with contextlib.suppress(ValueError):
            writer.rollback()
        return corpus_r

    total = 0
    for eid, blob in corpus_r.value:
        doc = tantivy.Document()
        doc.add_text(_ID_FIELD, eid)
        doc.add_text(_BLOB_FIELD, blob)
        writer.add_document(doc)
        total += 1

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
