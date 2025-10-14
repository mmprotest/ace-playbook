"""Utility helpers for ACE."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Table = None

from .config import ACEConfig

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt_template(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"prompt template {name} not found")
    return path.read_text(encoding="utf-8").strip()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def render_bullets_table(bullets: Iterable[Dict[str, Any]]) -> None:
    if Console is None or Table is None:  # pragma: no cover
        for bullet in bullets:
            print(f"{bullet.get('id','')} | {bullet.get('kind','')} | {bullet.get('title','')}")
        return
    console = Console()
    table = Table(title="ACE Playbook")
    table.add_column("ID")
    table.add_column("Kind")
    table.add_column("Title")
    table.add_column("Helpful")
    table.add_column("Harmful")
    for bullet in bullets:
        table.add_row(
            bullet.get("id", ""),
            bullet.get("kind", ""),
            bullet.get("title", ""),
            str(bullet.get("helpful_count", 0)),
            str(bullet.get("harmful_count", 0)),
        )
    console.print(table)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
