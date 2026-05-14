from __future__ import annotations

from dataclasses import dataclass

from humetric_core import HumetricError
from humetric_embed import EmbedError
from humetric_store import StoreError


@dataclass(frozen=True, slots=True)
class CorpusEmpty(HumetricError):
    pass


@dataclass(frozen=True, slots=True)
class StoreWrapped(HumetricError):
    cause: StoreError


@dataclass(frozen=True, slots=True)
class EmbedWrapped(HumetricError):
    cause: EmbedError


@dataclass(frozen=True, slots=True)
class FilterFailed(HumetricError):
    detail: str


type RetrievalError = CorpusEmpty | StoreWrapped | EmbedWrapped | FilterFailed
