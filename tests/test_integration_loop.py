from __future__ import annotations

import json

from ace_playbook.config import ACEConfig
from ace_playbook.curator import Curator
from ace_playbook.embeddings import BaseEmbeddings, EmbeddingResult
from ace_playbook.pipeline_online import Episode, OnlinePipeline
from ace_playbook.playbook import Playbook
from ace_playbook.retrieval import Retriever
from ace_playbook.storage import PlaybookStorage


class StubEmbeddings(BaseEmbeddings):
    def embed_texts(self, texts):
        vectors = [[1.0, 0.0] for _ in texts]
        return EmbeddingResult(vectors=vectors, model="stub")


def _fake_chat(self, messages, **kwargs):
    system = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    if "ACE Generator" in system:
        bullets = payload.get("bullets", [])
        if bullets:
            bullet_id = bullets[0]["id"]
            content = json.dumps(
                {
                    "answer": "correct",
                    "used_bullet_ids": [bullet_id],
                    "misleading_bullet_ids": [],
                    "attribution_notes": {bullet_id: "used for correctness"},
                }
            )
        else:
            content = json.dumps(
                {
                    "answer": "wrong",
                    "used_bullet_ids": [],
                    "misleading_bullet_ids": [],
                    "attribution_notes": {"gap": "missing guidance"},
                }
            )
        return {"choices": [{"message": {"content": content}}], "usage": {}}
    if "ACE Reflector" in system:
        content = json.dumps(
            {
                "bullets": [
                    {
                        "kind": "strategy",
                        "title": "Use the correct answer",
                        "body": "Always reply with the correct answer when asked.",
                        "tags": ["auto"],
                    }
                ],
                "patches": [],
                "traces": [],
            }
        )
        return {"choices": [{"message": {"content": content}}], "usage": {}}
    return {"choices": [{"message": {"content": "{}"}}], "usage": {}}


def test_generator_reflector_curator_loop(monkeypatch, tmp_path):
    config = ACEConfig(storage_path=tmp_path / "ace.sqlite", n_reflect_iterations=2)
    storage = PlaybookStorage(config)
    embedder = StubEmbeddings()
    retriever = Retriever(config, storage)
    curator = Curator(config, storage, embedder)
    playbook = Playbook(config=config, storage=storage, embedder=embedder, retriever=retriever, curator=curator)
    monkeypatch.setattr("ace_playbook.llm_client.SyncChatClient.chat", _fake_chat)
    pipeline = OnlinePipeline(config, playbook)
    traces = list(pipeline.run([Episode(query="Question?", answer="correct")]))
    assert traces[0].success is False
    assert traces[-1].success is True
