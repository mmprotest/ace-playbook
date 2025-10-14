"""CLI for offline ACE training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ace_playbook.config import ACEConfig
from ace_playbook.pipeline_offline import CSVQAAdapter, OfflinePipeline
from ace_playbook.playbook import Playbook
from ace_playbook.storage import dump_playbook
from ace_playbook.utils import render_bullets_table

app = typer.Typer(help="Offline ACE operations")


@app.command()
def train(
    data_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    epochs: int = typer.Option(1, help="Number of offline epochs"),
    storage_path: Optional[Path] = typer.Option(None, help="SQLite database path"),
) -> None:
    config = ACEConfig.from_env()
    if storage_path:
        config = config.copy(update={"storage_path": storage_path})
    playbook = Playbook.initialize(config)
    adapter = CSVQAAdapter(data_path)
    pipeline = OfflinePipeline(config, playbook)
    tasks = list(adapter.iter_tasks())
    pipeline.train(tasks, epochs=epochs)
    typer.echo("Training complete.")


@app.command()
def inspect(storage_path: Path = typer.Argument(..., exists=True)) -> None:
    config = ACEConfig.from_env(storage_path=storage_path)
    playbook = Playbook.initialize(config)
    bullets = dump_playbook(playbook.storage)
    render_bullets_table(bullets)


@app.command()
def export(
    storage_path: Path = typer.Argument(..., exists=True),
    output_path: Path = typer.Option(Path("playbook.json")),
) -> None:
    config = ACEConfig.from_env(storage_path=storage_path)
    playbook = Playbook.initialize(config)
    bullets = dump_playbook(playbook.storage)
    output_path.write_text(json.dumps(bullets, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Exported {len(bullets)} bullets to {output_path}")


if __name__ == "__main__":
    app()
