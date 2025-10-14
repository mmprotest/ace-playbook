"""Generator role: produce trajectories using playbook context."""

from __future__ import annotations

import json
import logging
from typing import List

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
        trace = Trace(
            query=query,
            selected_bullet_ids=[bullet.id for bullet in context.bullets],
            prompt=json.dumps(messages, ensure_ascii=False),
            response=output,
            success=self._determine_success(output),
            metadata={
                "model": self.config.model,
                "usage": json.dumps(response.get("usage", {})),
            },
        )
        self.storage.record_trace(trace)
        for bullet_id in trace.selected_bullet_ids:
            self.storage.update_usage(bullet_id, trace.success)
        return trace

    def _determine_success(self, response: str) -> bool:
        lower = response.lower()
        if "error" in lower or "fail" in lower:
            return False
        return True
