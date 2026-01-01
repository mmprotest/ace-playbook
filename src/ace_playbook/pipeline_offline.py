"""Offline grow-and-refine pipeline."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .config import ACEConfig
from .evaluation import get_evaluator
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
    evaluator: str = "exact"
    evaluator_params: dict = field(default_factory=dict)


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
                evaluator = row.get("evaluator") or "exact"
                evaluator_params: dict = {}
                if row.get("tolerance"):
                    evaluator_params["tolerance"] = row["tolerance"]
                yield Task(
                    query=row.get("question", ""),
                    answer=row.get("answer"),
                    evaluator=evaluator,
                    evaluator_params=evaluator_params,
                )


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
            for task in _progress(tasks, desc=f"epoch-{epoch+1}"):
                for iteration in range(self.config.n_reflect_iterations):
                    context = self.playbook.retrieve(task.query)
                    trace = self.generator.run(task.query, context)
                    trace.metadata["expected_answer"] = task.answer or ""
                    trace.success = self._evaluate(trace, task)
                    delta = self.reflector.reflect([trace], label=task.answer)
                    self.playbook.update(delta)

    def _evaluate(self, trace: Trace, task: Task) -> bool:
        expected = task.answer
        if expected is None and task.metadata:
            expected = task.metadata.get("expected")
        if expected is None:
            return False
        evaluator = get_evaluator(task.evaluator, task.evaluator_params)
        return evaluator.evaluate(expected or "", trace.response or "", task.evaluator_params)


def run_offline(config: ACEConfig, adapter: CSVQAAdapter, epochs: int = 1) -> Playbook:
    playbook = Playbook.initialize(config)
    pipeline = OfflinePipeline(config, playbook)
    tasks = list(adapter.iter_tasks())
    pipeline.train(tasks, epochs=epochs)
    return playbook
