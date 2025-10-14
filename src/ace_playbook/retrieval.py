"""Playbook retrieval logic."""

from __future__ import annotations

import logging
from datetime import datetime
from math import log1p
from typing import Iterable, List, Tuple

from .config import ACEConfig
from .schemas import Bullet, ContextSlice, cosine_similarity
from .storage import PlaybookStorage

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config: ACEConfig, storage: PlaybookStorage):
        self.config = config
        self.storage = storage

    def retrieve(self, query_embedding: List[float]) -> ContextSlice:
        bullets, vectors = self.storage.fetch_embeddings()
        if not len(bullets):
            return ContextSlice(bullets=[])
        scores: List[Tuple[float, Bullet]] = []
        now = datetime.utcnow()
        for vector, bullet in zip(vectors, bullets):
            if len(query_embedding) != len(vector):
                continue
            sim = cosine_similarity(query_embedding, vector)
            helpful_bonus = self.config.retrieval_beta * log1p(bullet.helpful_count)
            harmful_penalty = self.config.retrieval_gamma * log1p(bullet.harmful_count)
            freshness = 0.0
            if bullet.last_used_at:
                delta = now - bullet.last_used_at
                months = max(delta.days / 30.0, 0.0)
                freshness = self.config.retrieval_freshness / (1 + months)
            rank_score = (
                self.config.retrieval_alpha * float(sim)
                + helpful_bonus
                - harmful_penalty
                + freshness
            )
            scores.append((rank_score, bullet))
        scores.sort(key=lambda x: x[0], reverse=True)
        top_bullets = [bullet for _, bullet in scores[: self.config.retrieval_top_k]]
        return ContextSlice(bullets=top_bullets)

    def retrieve_for_query(self, query: str, embedder) -> ContextSlice:
        embedding = embedder.embed_texts([query]).vectors
        if not embedding:
            return ContextSlice(bullets=[])
        return self.retrieve(embedding[0])
