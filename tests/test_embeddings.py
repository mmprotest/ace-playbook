from __future__ import annotations

from ace_playbook.config import ACEConfig
from ace_playbook.embeddings import (
    EmbeddingError,
    LocalEmbeddings,
    OpenAIEmbeddings,
    build_embedding_provider,
)


def test_hash_fallback_deterministic():
    embedder = LocalEmbeddings()
    vec1 = embedder.embed_texts(["hello"]).vectors
    vec2 = embedder.embed_texts(["hello"]).vectors
    assert vec1 == vec2


def test_build_embedding_provider(monkeypatch):
    config = ACEConfig()

    class Dummy(OpenAIEmbeddings):
        def embed_texts(self, texts):
            raise EmbeddingError("fail")

    from ace_playbook import embeddings as emb_module

    monkeypatch.setattr(emb_module, "OpenAIEmbeddings", Dummy)
    provider = build_embedding_provider(config)
    vectors = provider.embed_texts(["test"]).vectors
    assert len(vectors) == 1
