"""Reflector role: translate traces into deltas."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Iterable, List, Optional

from .config import ACEConfig
from .llm_client import SyncChatClient
from .schemas import Bullet, BulletPatch, Delta, Trace
from .utils import load_prompt_template

logger = logging.getLogger(__name__)


class Reflector:
    def __init__(self, config: ACEConfig):
        self.config = config
        self.client = SyncChatClient(config)
        self.system_prompt = load_prompt_template("reflector_system.txt")

    def reflect(self, traces: Iterable[Trace], label: Optional[str] = None) -> Delta:
        traces = list(traces)
        if not traces:
            return Delta()
        try:
            return self._reflect_via_llm(traces, label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM reflection failed (%s). Falling back to heuristic reflector.", exc)
            return self._heuristic_reflect(traces)

    def _reflect_via_llm(self, traces: List[Trace], label: Optional[str]) -> Delta:
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
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Reflector response not valid JSON: %s", content)
            return self._heuristic_reflect(traces)
        bullets = [Bullet.from_dict(item) for item in payload.get("bullets", [])]
        patches = [BulletPatch.from_dict(item) for item in payload.get("patches", [])]
        delta = Delta(
            bullets=bullets,
            patches=patches,
            traces=traces,
        )
        return delta

    def _heuristic_reflect(self, traces: List[Trace]) -> Delta:
        counter = Counter()
        for trace in traces:
            if trace.success:
                counter.update(trace.selected_bullet_ids)
        bullets: List[Bullet] = []
        for bullet_id, count in counter.items():
            bullets.append(
                Bullet(
                    id=bullet_id,
                    kind="strategy",
                    title=f"Strengthen usage of {bullet_id}",
                    body=f"Repeat the approach captured in bullet {bullet_id}.",
                    helpful_count=count,
                    tags=["auto"],
                )
            )
        return Delta(bullets=bullets, traces=traces)
