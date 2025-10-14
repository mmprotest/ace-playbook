"""Embedding provider abstractions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from random import Random
from typing import Iterable, List, Optional

from .config import ACEConfig
from .llm_client import SyncChatClient

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


@dataclass
class EmbeddingResult:
    vectors: List[List[float]]
    model: str


class BaseEmbeddings:
    def embed_texts(self, texts: Iterable[str]) -> EmbeddingResult:  # pragma: no cover - abstract
        raise NotImplementedError


class OpenAIEmbeddings(BaseEmbeddings):
    """Embedding provider backed by an OpenAI-compatible endpoint."""

    def __init__(self, config: ACEConfig):
        self._config = config

    def embed_texts(self, texts: Iterable[str]) -> EmbeddingResult:
        payload = list(texts)
        if not payload:
            return EmbeddingResult([], self._config.embedding_model)

        client = SyncChatClient(self._config)
        try:
            vectors = client.embeddings(
                payload,
                model=self._config.embedding_model,
                base_url=self._config.embedding_base_url,
                api_key=self._config.embedding_api_key(),
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(str(exc)) from exc
        return EmbeddingResult([list(map(float, vec)) for vec in vectors], self._config.embedding_model)


class LocalEmbeddings(BaseEmbeddings):
    """Fallback embeddings based on sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            logger.warning("sentence-transformers unavailable, using hashing embeddings")
            self._model = None
            self._dim = 128
            return
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: Iterable[str]) -> EmbeddingResult:
        payload = list(texts)
        if not payload:
            return EmbeddingResult([], "local")
        if self._model is None:
            vectors = [self._hash_vector(text) for text in payload]
            return EmbeddingResult(vectors, "hash")
        vectors = self._model.encode(payload, convert_to_numpy=False)
        return EmbeddingResult([list(map(float, vec)) for vec in vectors], self._model.__class__.__name__)

    def _hash_vector(self, text: str) -> List[float]:
        seed = abs(hash(text)) % (2**32)
        random = Random(seed)
        return [random.uniform(-1.0, 1.0) for _ in range(self._dim)]


def build_embedding_provider(config: ACEConfig) -> BaseEmbeddings:
    """Construct an embedding provider with graceful fallback."""

    try:
        provider: BaseEmbeddings = OpenAIEmbeddings(config)
        # Attempt a quick dry run to ensure the credentials work.
        provider.embed_texts(["ace health check"])
        logger.info("Using OpenAI embeddings with model %s", config.embedding_model)
        return provider
    except EmbeddingError as exc:
        logger.warning("Falling back to local embeddings: %s", exc)
        return LocalEmbeddings(config.local_embedding_model)
