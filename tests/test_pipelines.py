from __future__ import annotations

from ace_playbook.config import ACEConfig
from ace_playbook.pipeline_offline import OfflinePipeline, Task
from ace_playbook.pipeline_online import Episode, OnlinePipeline
from ace_playbook.playbook import Playbook
from ace_playbook.schemas import Trace


class DummyGenerator:
    def __init__(self, config, storage):
        self.storage = storage

    def run(self, query, context):
        return Trace(
            query=query,
            selected_bullet_ids=[bullet.id for bullet in context.bullets],
            used_bullet_ids=[bullet.id for bullet in context.bullets],
            misleading_bullet_ids=[],
            attribution_notes={},
            prompt="[]",
            response="answer",
            success=False,
        )


class DummyReflector:
    def __init__(self, config):
        pass

    def reflect(self, traces, label=None):
        from ace_playbook.schemas import Bullet, DeltaRuntime

        bullet = Bullet(kind="strategy", title="Always answer", body="Say answer", tags=["test"])
        return DeltaRuntime(bullets=[bullet])


def test_offline_pipeline(monkeypatch, tmp_path):
    config = ACEConfig(storage_path=tmp_path / "offline.sqlite")
    playbook = Playbook.initialize(config)
    monkeypatch.setattr("ace_playbook.pipeline_offline.Generator", DummyGenerator)
    monkeypatch.setattr("ace_playbook.pipeline_offline.Reflector", DummyReflector)
    pipeline = OfflinePipeline(config, playbook)
    tasks = [Task(query="Q1", answer="answer")]
    pipeline.train(tasks, epochs=1)
    assert playbook.stats()["total_bullets"] >= 1


def test_online_pipeline(monkeypatch, tmp_path):
    config = ACEConfig(storage_path=tmp_path / "online.sqlite")
    playbook = Playbook.initialize(config)
    monkeypatch.setattr("ace_playbook.pipeline_online.Generator", DummyGenerator)
    monkeypatch.setattr("ace_playbook.pipeline_online.Reflector", DummyReflector)
    pipeline = OnlinePipeline(config, playbook)
    traces = list(pipeline.run([Episode(query="Q1", answer="answer")]))
    assert traces
    assert playbook.stats()["total_bullets"] >= 1
