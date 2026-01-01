"""Reflector role: translate traces into deltas."""

from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional

from .config import ACEConfig
from .llm_client import SyncChatClient
from .schemas import Bullet, BulletPatch, DeltaRuntime, DeltaSchema, Trace
from .utils import load_prompt_template

logger = logging.getLogger(__name__)


class Reflector:
    def __init__(self, config: ACEConfig):
        self.config = config
        self.client = SyncChatClient(config)
        self.system_prompt = load_prompt_template("reflector_system.txt")

    def reflect(self, traces: Iterable[Trace], label: Optional[str] = None) -> DeltaRuntime:
        traces = list(traces)
        if not traces:
            return DeltaRuntime()
        try:
            return self._reflect_via_llm(traces, label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM reflection failed (%s). Falling back to heuristic reflector.", exc)
            return self._heuristic_reflect(traces)

    def _reflect_via_llm(self, traces: List[Trace], label: Optional[str]) -> DeltaRuntime:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "label": label,
                        "traces": [trace.to_dict() for trace in traces],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.client.chat(messages, max_tokens=1200)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        try:
            schema = DeltaSchema.model_validate_json(content)
        except Exception:  # noqa: BLE001
            logger.error("Reflector response not valid JSON: %s", content)
            return self._heuristic_reflect(traces)
        delta = schema.to_runtime()
        return self._filter_delta(delta, traces)

    def _heuristic_reflect(self, traces: List[Trace]) -> DeltaRuntime:
        bullets: List[Bullet] = []
        patches: List[BulletPatch] = []
        for trace in traces:
            helpful_ids = set(trace.used_bullet_ids)
            misleading_ids = set(trace.misleading_bullet_ids) - helpful_ids
            if not trace.success and not helpful_ids and not misleading_ids:
                bullets.append(
                    Bullet(
                        kind="strategy",
                        title=f"Address gap: {trace.query[:60]}",
                        body=f"Add guidance to answer: {trace.query}.",
                        tags=["auto", "gap"],
                        source_trace_ids=[trace.id],
                    )
                )
            for bullet_id in misleading_ids:
                patches.append(
                    BulletPatch(
                        bullet_id=bullet_id,
                        op="patch",
                        patch_text="Append a clarification to avoid the observed failure.",
                        patch_mode="append",
                    )
                )
        return self._filter_delta(DeltaRuntime(bullets=bullets, patches=patches, traces=traces), traces)

    def _filter_delta(self, delta: DeltaRuntime, traces: List[Trace]) -> DeltaRuntime:
        helpful_ids = {bid for trace in traces for bid in trace.used_bullet_ids}
        misleading_ids = {bid for trace in traces for bid in trace.misleading_bullet_ids}
        gaps_exist = any(
            (not trace.success) and (not trace.used_bullet_ids) and (not trace.misleading_bullet_ids)
            for trace in traces
        )
        filtered_bullets = delta.bullets if gaps_exist else []
        filtered_patches = [
            patch
            for patch in delta.patches
            if patch.bullet_id in misleading_ids and patch.bullet_id not in helpful_ids
        ]
        trace_ids = [trace.id for trace in traces]
        for bullet in filtered_bullets:
            if not bullet.source_trace_ids:
                bullet.source_trace_ids = trace_ids
        return DeltaRuntime(
            bullets=filtered_bullets,
            patches=filtered_patches,
            traces=delta.traces or traces,
        )
