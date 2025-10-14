"""Data models and schemas for ACE."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conlist

from .sanitize import sanitize_text, validate_bullet_body

BulletKind = Literal["strategy", "rule", "pitfall", "template", "tool", "concept"]
EditOp = Literal["inc_helpful", "inc_harmful", "patch"]


def _now() -> datetime:
    return datetime.utcnow()


def _validate_body(body: str) -> str:
    body = sanitize_text(body)
    validate_bullet_body(body)
    return body


@dataclass
class Bullet:
    kind: BulletKind
    title: str
    body: str
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_now)
    last_used_at: Optional[datetime] = None
    helpful_count: int = 0
    harmful_count: int = 0
    score: float = 0.0
    embedding: Optional[List[float]] = None
    source_trace_ids: List[str] = field(default_factory=list)
    version: int = 0
    duplicate_of: Optional[str] = None

    def __post_init__(self) -> None:
        self.body = _validate_body(self.body)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["last_used_at"] = self.last_used_at.isoformat() if self.last_used_at else None
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Bullet":
        payload = dict(data)
        if isinstance(payload.get("created_at"), str):
            payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        if isinstance(payload.get("last_used_at"), str) and payload["last_used_at"]:
            payload["last_used_at"] = datetime.fromisoformat(payload["last_used_at"])
        return cls(**payload)  # type: ignore[arg-type]


@dataclass
class BulletPatch:
    bullet_id: str
    op: EditOp
    patch_text: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BulletPatch":
        return cls(**data)  # type: ignore[arg-type]


@dataclass
class Trace:
    query: str
    selected_bullet_ids: List[str]
    prompt: str
    response: str
    success: bool
    metadata: Dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Trace":
        payload = dict(data)
        if isinstance(payload.get("created_at"), str):
            payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        return cls(**payload)  # type: ignore[arg-type]


class BulletIn(BaseModel):
    kind: Literal["strategy", "rule", "pitfall", "template", "tool", "concept"]
    title: str = Field(..., max_length=160)
    body: str = Field(..., max_length=1200)
    tags: List[str] = []


class DeltaEdit(BaseModel):
    bullet_id: str
    op: EditOp
    patch_text: Optional[str] = None


class Delta(BaseModel):
    trace_ids: conlist(str, min_items=1)
    new_bullets: List["BulletIn"]
    edits: List[DeltaEdit] = []


Delta.model_rebuild()


@dataclass
class ContextSlice:
    bullets: List[Bullet]

    def to_prompt_fragment(self) -> str:
        lines = []
        for bullet in self.bullets:
            lines.append(f"- [{bullet.kind.upper()}] {bullet.title}: {bullet.body}")
        return "\n".join(lines)


@dataclass
class MergeReport:
    added: int
    updated: int
    skipped: int
    deduplicated: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def export_delta_json_schema(path: str) -> Dict[str, object]:
    """Export the strict Delta JSON schema to ``path``."""

    schema = Delta.model_json_schema(mode="validation")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return dot / denom
