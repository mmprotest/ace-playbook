"""High-level façade for ACE playbook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import ACEConfig
from .curator import Curator
from .embeddings import BaseEmbeddings, build_embedding_provider
from .retrieval import Retriever
from .schemas import ContextSlice, Delta, MergeReport
from .storage import PlaybookStorage


@dataclass
class Playbook:
    config: ACEConfig
    storage: PlaybookStorage
    embedder: BaseEmbeddings
    retriever: Retriever
    curator: Curator

    @classmethod
    def initialize(cls, config: ACEConfig | None = None) -> "Playbook":
        config = config or ACEConfig.from_env()
        storage = PlaybookStorage(config)
        embedder = build_embedding_provider(config)
        retriever = Retriever(config, storage)
        curator = Curator(config, storage, embedder)
        return cls(config=config, storage=storage, embedder=embedder, retriever=retriever, curator=curator)

    def retrieve(self, query: str) -> ContextSlice:
        return self.retriever.retrieve_for_query(query, self.embedder)

    def update(self, delta: Delta) -> MergeReport:
        return self.curator.merge(delta)

    def stats(self) -> dict:
        bullets = self.storage.list_bullets()
        return {
            "total_bullets": len(bullets),
            "strategies": sum(1 for b in bullets if b.kind == "strategy"),
            "rules": sum(1 for b in bullets if b.kind == "rule"),
        }
