"""Data models for ACE without external dependencies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from math import sqrt
from typing import Dict, List, Literal, Optional

BulletKind = Literal["strategy", "rule", "pitfall", "template", "tool", "concept"]


def _now() -> datetime:
    return datetime.utcnow()


def _validate_body(body: str) -> str:
    if len(body) > 1200:
        raise ValueError("bullet body exceeds 1200 characters")
    if "<<<" in body:
        raise ValueError("prompt injection token detected")
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
    op: Literal["inc_helpful", "inc_harmful", "patch"]
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


@dataclass
class Delta:
    bullets: List[Bullet] = field(default_factory=list)
    patches: List[BulletPatch] = field(default_factory=list)
    traces: List[Trace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "bullets": [b.to_dict() for b in self.bullets],
            "patches": [p.to_dict() for p in self.patches],
            "traces": [t.to_dict() for t in self.traces],
        }


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
