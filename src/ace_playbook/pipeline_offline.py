"""Offline grow-and-refine pipeline."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import ACEConfig
from .generator import Generator
from .playbook import Playbook
from .reflector import Reflector
from .schemas import Trace
from .utils import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class Task:
    query: str
    answer: Optional[str] = None
    metadata: Optional[dict] = None


def _progress(items: Sequence[Task], desc: str):
    for item in items:
        yield item


class CSVQAAdapter:
    """Simple adapter that loads question/answer pairs from a CSV file."""

    def __init__(self, path: Path):
        self.path = path

    def iter_tasks(self) -> Iterable[Task]:
        with self.path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield Task(query=row.get("question", ""), answer=row.get("answer"))


@dataclass
class OfflinePipeline:
    config: ACEConfig
    playbook: Playbook

    def __post_init__(self) -> None:
        seed_everything(self.config.random_seed)
        self.generator = Generator(self.config, self.playbook.storage)
        self.reflector = Reflector(self.config)

    def train(self, tasks: Sequence[Task], epochs: int = 1) -> None:
        for epoch in range(epochs):
            logger.info("Offline epoch %d/%d", epoch + 1, epochs)
            traces: List[Trace] = []
            for task in _progress(tasks, desc=f"epoch-{epoch+1}"):
                context = self.playbook.retrieve(task.query)
                trace = self.generator.run(task.query, context)
                trace.metadata["expected_answer"] = task.answer or ""
                trace.success = self._evaluate(trace, task)
                traces.append(trace)
            delta = self.reflector.reflect(traces)
            self.playbook.update(delta)

    def _evaluate(self, trace: Trace, task: Task) -> bool:
        if task.answer is None:
            return trace.success
        answer = (task.answer or "").strip().lower()
        response = (trace.response or "").strip().lower()
        return answer in response


def run_offline(config: ACEConfig, adapter: CSVQAAdapter, epochs: int = 1) -> Playbook:
    playbook = Playbook.initialize(config)
    pipeline = OfflinePipeline(config, playbook)
    tasks = list(adapter.iter_tasks())
    pipeline.train(tasks, epochs=epochs)
    return playbook


def _progress(items: Sequence[Task], desc: str):
    for item in items:
        yield item

