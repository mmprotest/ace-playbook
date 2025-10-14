"""CLI helpers for inspecting the playbook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ace_playbook.config import ACEConfig
from ace_playbook.playbook import Playbook
from ace_playbook.schemas import Delta

app = typer.Typer(help="Playbook utilities")


@app.command()
def retrieve(query: str, storage_path: Optional[Path] = typer.Option(None)) -> None:
    config = ACEConfig.from_env()
    if storage_path:
        config = config.copy(update={"storage_path": storage_path})
    playbook = Playbook.initialize(config)
    context = playbook.retrieve(query)
    for bullet in context.bullets:
        typer.echo(f"[{bullet.kind}] {bullet.title} ({bullet.id})\n  {bullet.body}\n")


@app.command()
def merge(delta_path: Path, storage_path: Optional[Path] = typer.Option(None)) -> None:
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    delta = Delta(**payload)
    config = ACEConfig.from_env()
    if storage_path:
        config = config.copy(update={"storage_path": storage_path})
    playbook = Playbook.initialize(config)
    report = playbook.update(delta)
    typer.echo(f"Merge complete: {report}")


if __name__ == "__main__":
    app()
