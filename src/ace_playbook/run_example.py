"""Run a minimal ACE loop without external dependencies."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ACEConfig
from .curator import Curator
from .embeddings import BaseEmbeddings, EmbeddingResult
from .pipeline_online import Episode, OnlinePipeline
from .playbook import Playbook
from .retrieval import Retriever
from .storage import PlaybookStorage


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
                    "answer": "example answer",
                    "used_bullet_ids": [bullet_id],
                    "misleading_bullet_ids": [],
                    "attribution_notes": {bullet_id: "used for the answer"},
                }
            )
        else:
            content = json.dumps(
                {
                    "answer": "needs guidance",
                    "used_bullet_ids": [],
                    "misleading_bullet_ids": [],
                    "attribution_notes": {"gap": "no guidance yet"},
                }
            )
        return {"choices": [{"message": {"content": content}}], "usage": {}}
    if "ACE Reflector" in system:
        content = json.dumps(
            {
                "bullets": [
                    {
                        "kind": "strategy",
                        "title": "Provide the example answer",
                        "body": "Reply with the example answer in plain language.",
                        "tags": ["example"],
                    }
                ],
                "patches": [],
                "traces": [],
            }
        )
        return {"choices": [{"message": {"content": content}}], "usage": {}}
    return {"choices": [{"message": {"content": "{}"}}], "usage": {}}


def main() -> None:
    config = ACEConfig(storage_path=Path("ace_example.sqlite"), n_reflect_iterations=2)
    storage = PlaybookStorage(config)
    embedder = StubEmbeddings()
    retriever = Retriever(config, storage)
    curator = Curator(config, storage, embedder)
    playbook = Playbook(config=config, storage=storage, embedder=embedder, retriever=retriever, curator=curator)

    from . import llm_client

    llm_client.SyncChatClient.chat = _fake_chat  # type: ignore[assignment]

    pipeline = OnlinePipeline(config, playbook)
    episodes = [Episode(query="What is ACE?", answer="example answer")]
    for trace in pipeline.run(episodes):
        print(f"Trace success={trace.success} response={trace.response}")


if __name__ == "__main__":
    main()
