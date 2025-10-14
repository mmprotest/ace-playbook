"""Toy online loop demonstrating ACE self-improvement."""

from __future__ import annotations

import random

from ace_playbook.config import ACEConfig
from ace_playbook.pipeline_online import Episode, OnlinePipeline
from ace_playbook.playbook import Playbook


def synthetic_math_stream(n: int = 5):
    for _ in range(n):
        a, b = random.randint(1, 9), random.randint(1, 9)
        yield Episode(query=f"What is {a} + {b}?", answer=str(a + b))


def main() -> None:
    config = ACEConfig()
    playbook = Playbook.initialize(config)
    pipeline = OnlinePipeline(config, playbook)
    for trace in pipeline.run(synthetic_math_stream(3)):
        print(f"Q: {trace.query}\nA: {trace.response}\nSuccess: {trace.success}\n---")


if __name__ == "__main__":
    main()
