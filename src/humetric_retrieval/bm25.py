from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from humetric_core import Err, Ok, Result
from humetric_store import list_persons
from rank_bm25 import BM25Okapi

from humetric_retrieval.errors import CorpusEmpty, RetrievalError, StoreWrapped

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-+.#]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(slots=True)
class Bm25Index:
    person_ids: tuple[str, ...]
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.person_ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        idx_sorted = sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:k]
        return [(self.person_ids[i], float(scores[i])) for i in idx_sorted if scores[i] > 0]


def build_bm25(conn: sqlite3.Connection) -> Result[Bm25Index, RetrievalError]:
    """Materialize a BM25 index from every Person currently in the store.

    Suitable for tens of thousands of docs; rebuild on each app start. For
    v1.5 corpora >100k, persist the tokenized corpus to disk.
    """
    page = 0
    person_ids: list[str] = []
    docs: list[list[str]] = []
    while True:
        r = list_persons(conn, limit=2000, offset=page * 2000)
        if isinstance(r, Err):
            return Err(StoreWrapped(cause=r.error))
        rows = r.value
        if not rows:
            break
        for p in rows:
            person_ids.append(p.id)
            docs.append(_tokenize(p.text_blob()))
        page += 1

    if not docs:
        return Err(CorpusEmpty())

    idx = Bm25Index(person_ids=tuple(person_ids))
    idx._bm25 = BM25Okapi(docs)
    return Ok(idx)
