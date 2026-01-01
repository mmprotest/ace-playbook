"""Generator role: produce trajectories using playbook context."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from .config import ACEConfig
from .llm_client import SyncChatClient
from .schemas import ContextSlice, Trace
from .storage import PlaybookStorage
from .utils import load_prompt_template

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, config: ACEConfig, storage: PlaybookStorage):
        self.config = config
        self.storage = storage
        self.client = SyncChatClient(config)
        self.system_prompt = load_prompt_template("generator_system.txt")

    def run(self, query: str, context: ContextSlice) -> Trace:
        logger.info("Generator handling query: %s", query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "bullets": [bullet.to_dict() for bullet in context.bullets],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.client.chat(messages)
        output = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = self._parse_output(output)
        trace = Trace(
            query=query,
            selected_bullet_ids=[bullet.id for bullet in context.bullets],
            used_bullet_ids=parsed["used_bullet_ids"],
            misleading_bullet_ids=parsed["misleading_bullet_ids"],
            attribution_notes=parsed["attribution_notes"],
            prompt=json.dumps(messages, ensure_ascii=False),
            response=parsed["answer"],
            success=False,
            metadata={
                "model": self.config.model,
                "usage": json.dumps(response.get("usage", {})),
                "raw_response": output,
            },
        )
        self.storage.record_trace(trace)
        return trace

    def _parse_output(self, output: str) -> Dict[str, object]:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            logger.warning("Generator output not JSON, falling back to plain response.")
            return {
                "answer": output,
                "used_bullet_ids": [],
                "misleading_bullet_ids": [],
                "attribution_notes": {},
            }
        answer = payload.get("answer", "")
        return {
            "answer": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False),
            "used_bullet_ids": list(payload.get("used_bullet_ids", []) or []),
            "misleading_bullet_ids": list(payload.get("misleading_bullet_ids", []) or []),
            "attribution_notes": dict(payload.get("attribution_notes", {}) or {}),
        }
