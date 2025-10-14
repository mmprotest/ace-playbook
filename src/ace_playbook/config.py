"""Minimal configuration system for ACE."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional


@dataclass
class ACEConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 1024
    request_timeout: float = 120.0

    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"

    embedding_model: str = "text-embedding-3-large"
    embedding_base_url: Optional[str] = None
    embedding_api_key_env: Optional[str] = None
    local_embedding_model: str = "all-MiniLM-L6-v2"

    retrieval_top_k: int = 8
    retrieval_alpha: float = 0.7
    retrieval_beta: float = 0.2
    retrieval_gamma: float = 0.2
    retrieval_freshness: float = 0.1

    dedup_cosine_threshold: float = 0.86
    grow_and_refine: Literal["proactive", "lazy"] = "proactive"
    refine_window_size: int = 50

    storage_path: Path = Path("ace_playbook.sqlite")
    trace_dir: Path = Path("traces")

    n_reflect_iterations: int = 1
    random_seed: int = 42

    ENV_PREFIX: ClassVar[str] = "ACE_"

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_env(cls, **overrides: Any) -> "ACEConfig":
        values: Dict[str, Any] = {}
        for field_info in fields(cls):
            if not field_info.init:
                continue
            env_name = f"{cls.ENV_PREFIX}{field_info.name.upper()}"
            if env_name in os.environ:
                values[field_info.name] = cls._cast_value(field_info.type, os.environ[env_name])
        values.update(overrides)
        return cls(**values)

    def copy(self, update: Optional[Dict[str, Any]] = None) -> "ACEConfig":
        update = update or {}
        data = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        data.update(update)
        return ACEConfig(**data)

    def _validate(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

    @staticmethod
    def _cast_value(field_type: Any, raw: str) -> Any:
        origin = getattr(field_type, "__origin__", None)
        if origin is Literal:
            return raw
        if field_type in (int, float, str):
            return field_type(raw)
        if field_type is Path:
            return Path(raw)
        return raw

    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)

    def embedding_api_key(self) -> Optional[str]:
        env = self.embedding_api_key_env or self.api_key_env
        return os.getenv(env)

    def resolve_trace_dir(self) -> Path:
        path = Path(self.trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
