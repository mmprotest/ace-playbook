from __future__ import annotations

from ace_playbook.config import ACEConfig
from ace_playbook.reflector import Reflector
from ace_playbook.schemas import Trace


def test_reflector_uses_attribution_for_updates():
    config = ACEConfig()
    reflector = Reflector(config)
    trace_gap = Trace(
        query="Explain taxes",
        selected_bullet_ids=[],
        used_bullet_ids=[],
        misleading_bullet_ids=[],
        attribution_notes={"gap": "no guidance available"},
        prompt="[]",
        response="",
        success=False,
    )
    trace_misleading = Trace(
        query="Explain loans",
        selected_bullet_ids=["helpful", "bad"],
        used_bullet_ids=["helpful"],
        misleading_bullet_ids=["bad"],
        attribution_notes={"helpful": "used", "bad": "misleading"},
        prompt="[]",
        response="",
        success=False,
    )
    delta = reflector._heuristic_reflect([trace_gap, trace_misleading])
    assert delta.bullets, "Expected new bullets when gaps exist"
    assert any(patch.bullet_id == "bad" for patch in delta.patches)
    assert all(patch.bullet_id != "helpful" for patch in delta.patches)
