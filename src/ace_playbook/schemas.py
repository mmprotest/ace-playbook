"""Data models and schemas for ACE."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .sanitize import sanitize_text, validate_bullet_body

BulletKind = Literal["strategy", "rule", "pitfall", "template", "tool", "concept"]
EditOp = Literal["inc_helpful", "inc_harmful", "patch"]
PatchMode = Literal["append", "replace"]


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
    patch_mode: PatchMode = "append"

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
    used_bullet_ids: List[str] = field(default_factory=list)
    misleading_bullet_ids: List[str] = field(default_factory=list)
    attribution_notes: Dict[str, str] = field(default_factory=dict)
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
        payload.setdefault("used_bullet_ids", [])
        payload.setdefault("misleading_bullet_ids", [])
        payload.setdefault("attribution_notes", {})
        return cls(**payload)  # type: ignore[arg-type]


class BulletSchema(BaseModel):
    kind: Literal["strategy", "rule", "pitfall", "template", "tool", "concept"]
    title: str = Field(..., max_length=160)
    body: str = Field(..., max_length=1200)
    tags: List[str] = Field(default_factory=list)
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    helpful_count: int = 0
    harmful_count: int = 0
    score: float = 0.0
    embedding: Optional[List[float]] = None
    source_trace_ids: List[str] = Field(default_factory=list)
    version: int = 0
    duplicate_of: Optional[str] = None

    def to_runtime(self) -> Bullet:
        return Bullet(
            kind=self.kind,
            title=self.title,
            body=self.body,
            tags=list(self.tags),
            id=self.id or str(uuid.uuid4()),
            created_at=self.created_at or _now(),
            last_used_at=self.last_used_at,
            helpful_count=self.helpful_count,
            harmful_count=self.harmful_count,
            score=self.score,
            embedding=self.embedding,
            source_trace_ids=list(self.source_trace_ids),
            version=self.version,
            duplicate_of=self.duplicate_of,
        )

    @classmethod
    def from_runtime(cls, bullet: Bullet) -> "BulletSchema":
        return cls(
            kind=bullet.kind,
            title=bullet.title,
            body=bullet.body,
            tags=list(bullet.tags),
            id=bullet.id,
            created_at=bullet.created_at,
            last_used_at=bullet.last_used_at,
            helpful_count=bullet.helpful_count,
            harmful_count=bullet.harmful_count,
            score=bullet.score,
            embedding=bullet.embedding,
            source_trace_ids=list(bullet.source_trace_ids),
            version=bullet.version,
            duplicate_of=bullet.duplicate_of,
        )
class BulletPatchSchema(BaseModel):
    bullet_id: str
    op: EditOp
    patch_text: Optional[str] = None
    patch_mode: PatchMode = "append"

    def to_runtime(self) -> BulletPatch:
        return BulletPatch(
            bullet_id=self.bullet_id,
            op=self.op,
            patch_text=self.patch_text,
            patch_mode=self.patch_mode,
        )

    @classmethod
    def from_runtime(cls, patch: BulletPatch) -> "BulletPatchSchema":
        return cls(
            bullet_id=patch.bullet_id,
            op=patch.op,
            patch_text=patch.patch_text,
            patch_mode=patch.patch_mode,
        )


class TraceSchema(BaseModel):
    query: str
    selected_bullet_ids: List[str]
    used_bullet_ids: List[str] = Field(default_factory=list)
    misleading_bullet_ids: List[str] = Field(default_factory=list)
    attribution_notes: Dict[str, str] = Field(default_factory=dict)
    prompt: str
    response: str
    success: bool
    metadata: Dict[str, str] = Field(default_factory=dict)
    id: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_runtime(self) -> Trace:
        return Trace(
            query=self.query,
            selected_bullet_ids=list(self.selected_bullet_ids),
            used_bullet_ids=list(self.used_bullet_ids),
            misleading_bullet_ids=list(self.misleading_bullet_ids),
            attribution_notes=dict(self.attribution_notes),
            prompt=self.prompt,
            response=self.response,
            success=self.success,
            metadata=dict(self.metadata),
            id=self.id or str(uuid.uuid4()),
            created_at=self.created_at or _now(),
        )

    @classmethod
    def from_runtime(cls, trace: Trace) -> "TraceSchema":
        return cls(
            query=trace.query,
            selected_bullet_ids=list(trace.selected_bullet_ids),
            used_bullet_ids=list(trace.used_bullet_ids),
            misleading_bullet_ids=list(trace.misleading_bullet_ids),
            attribution_notes=dict(trace.attribution_notes),
            prompt=trace.prompt,
            response=trace.response,
            success=trace.success,
            metadata=dict(trace.metadata),
            id=trace.id,
            created_at=trace.created_at,
        )


@dataclass
class DeltaRuntime:
    bullets: List[Bullet] = field(default_factory=list)
    patches: List[BulletPatch] = field(default_factory=list)
    traces: List[Trace] = field(default_factory=list)

    def to_schema(self) -> "DeltaSchema":
        return DeltaSchema.from_runtime(self)


class DeltaSchema(BaseModel):
    bullets: List[BulletSchema] = Field(default_factory=list)
    patches: List[BulletPatchSchema] = Field(default_factory=list)
    traces: List[TraceSchema] = Field(default_factory=list)

    def to_runtime(self) -> DeltaRuntime:
        return DeltaRuntime(
            bullets=[item.to_runtime() for item in self.bullets],
            patches=[item.to_runtime() for item in self.patches],
            traces=[item.to_runtime() for item in self.traces],
        )

    @classmethod
    def from_runtime(cls, delta: DeltaRuntime) -> "DeltaSchema":
        return cls(
            bullets=[BulletSchema.from_runtime(item) for item in delta.bullets],
            patches=[BulletPatchSchema.from_runtime(item) for item in delta.patches],
            traces=[TraceSchema.from_runtime(item) for item in delta.traces],
        )


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

    schema = DeltaSchema.model_json_schema(mode="validation")
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
