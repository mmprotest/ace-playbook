from __future__ import annotations

from ace_playbook.config import ACEConfig
from ace_playbook.curator import Curator
from ace_playbook.embeddings import BaseEmbeddings, EmbeddingResult
from ace_playbook.schemas import Bullet, DeltaRuntime
from ace_playbook.storage import PlaybookStorage


class StubEmbeddings(BaseEmbeddings):
    def embed_texts(self, texts):
        vectors = [[1.0, 0.0] for _ in texts]
        return EmbeddingResult(vectors=vectors, model="stub")


def test_curator_deduplicates(tmp_path):
    config = ACEConfig(storage_path=tmp_path / "db.sqlite", dedup_cosine_threshold=0.5)
    storage = PlaybookStorage(config)
    embedder = StubEmbeddings()
    curator = Curator(config, storage, embedder)
    bullet1 = Bullet(kind="strategy", title="One", body="Use math", tags=["math"], embedding=[1, 0])
    bullet2 = Bullet(kind="strategy", title="Two", body="Use math", tags=["math"], embedding=[1, 0])
    delta = DeltaRuntime(bullets=[bullet1, bullet2])
    report = curator.merge(delta)
    assert report.added == 1
    assert report.deduplicated >= 1
