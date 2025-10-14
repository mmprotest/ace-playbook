"""CLI for online ACE rollout."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import typer

from ace_playbook.config import ACEConfig
from ace_playbook.pipeline_online import Episode, OnlinePipeline
from ace_playbook.playbook import Playbook

app = typer.Typer(help="Online ACE operations")


def _load_episodes(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield Episode(query=row.get("question", ""), answer=row.get("answer"))


@app.command()
def rollout(
    data_path: Path = typer.Argument(..., exists=True),
    storage_path: Optional[Path] = typer.Option(None, help="Existing playbook DB"),
    episodes: int = typer.Option(10, help="Number of episodes to run"),
) -> None:
    config = ACEConfig.from_env()
    if storage_path:
        config = config.copy(update={"storage_path": storage_path})
    playbook = Playbook.initialize(config)
    pipeline = OnlinePipeline(config, playbook)
    traces = []
    for idx, episode in enumerate(_load_episodes(data_path)):
        if idx >= episodes:
            break
        trace = next(iter(pipeline.run([episode])))
        traces.append(trace)
    typer.echo(f"Completed {len(traces)} episodes")


if __name__ == "__main__":
    app()
