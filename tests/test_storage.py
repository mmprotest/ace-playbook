from __future__ import annotations

from datetime import datetime

import pytest

from ace_playbook.config import ACEConfig
from ace_playbook.schemas import Bullet, Trace
from ace_playbook.storage import PlaybookStorage


@pytest.fixture()
def storage(tmp_path):
    config = ACEConfig(storage_path=tmp_path / "test.sqlite")
    return PlaybookStorage(config)


def test_upsert_and_fetch(storage):
    bullet = Bullet(kind="strategy", title="Test", body="Do X", tags=["unit"], embedding=[0.1, 0.2])
    added, updated = storage.upsert_bullets([bullet])
    assert added == 1
    assert updated == 0
    fetched = storage.get_bullet(bullet.id)
    assert fetched is not None
    assert fetched.title == "Test"
    assert fetched.embedding is not None
    assert pytest.approx(fetched.embedding[0], rel=0.1) == 0.1


def test_trace_round_trip(storage):
    trace = Trace(
        query="What?",
        selected_bullet_ids=[],
        prompt="[]",
        response="done",
        success=True,
    )
    storage.record_trace(trace)
    traces = storage.list_traces()
    assert traces
    assert traces[0].response == "done"
