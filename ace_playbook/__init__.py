"""Bootstrap package for src layout."""

from __future__ import annotations

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent.parent / "src" / "ace_playbook"))

from .config import ACEConfig  # noqa: E402
from .playbook import Playbook  # noqa: E402

__all__ = ["ACEConfig", "Playbook"]
