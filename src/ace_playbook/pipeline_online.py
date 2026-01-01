"""Online self-improvement pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from .config import ACEConfig
from .evaluation import get_evaluator
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
    evaluator: str = "exact"
    evaluator_params: dict = field(default_factory=dict)


class OnlinePipeline:
    def __init__(self, config: ACEConfig, playbook: Playbook):
        self.config = config
        self.playbook = playbook
        seed_everything(config.random_seed)
        self.generator = Generator(config, playbook.storage)
        self.reflector = Reflector(config)

    def run(self, episodes: Iterable[Episode]) -> Iterator[Trace]:
        for episode in episodes:
            for iteration in range(self.config.n_reflect_iterations):
                context = self.playbook.retrieve(episode.query)
                trace = self.generator.run(episode.query, context)
                trace.success = self._evaluate(trace, episode)
                delta = self.reflector.reflect([trace], label=episode.answer)
                self.playbook.update(delta)
                yield trace

    def _evaluate(self, trace: Trace, episode: Episode) -> bool:
        expected = episode.answer
        if expected is None and episode.metadata:
            expected = episode.metadata.get("expected")
        if expected is None:
            return False
        evaluator = get_evaluator(episode.evaluator, episode.evaluator_params)
        return evaluator.evaluate(expected or "", trace.response or "", episode.evaluator_params)
