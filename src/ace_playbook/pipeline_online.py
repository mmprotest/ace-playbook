"""Online self-improvement pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .config import ACEConfig
from .generator import Generator
from .playbook import Playbook
from .reflector import Reflector
from .schemas import Trace
from .utils import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    query: str
    answer: Optional[str] = None
    metadata: Optional[dict] = None


class OnlinePipeline:
    def __init__(self, config: ACEConfig, playbook: Playbook):
        self.config = config
        self.playbook = playbook
        seed_everything(config.random_seed)
        self.generator = Generator(config, playbook.storage)
        self.reflector = Reflector(config)

    def run(self, episodes: Iterable[Episode]) -> Iterator[Trace]:
        for episode in episodes:
            context = self.playbook.retrieve(episode.query)
            trace = self.generator.run(episode.query, context)
            trace.success = self._evaluate(trace, episode)
            delta = self.reflector.reflect([trace], label=episode.answer)
            self.playbook.update(delta)
            yield trace

    def _evaluate(self, trace: Trace, episode: Episode) -> bool:
        if episode.answer is None:
            return trace.success
        expected = (episode.answer or "").strip().lower()
        actual = (trace.response or "").strip().lower()
        return expected in actual
