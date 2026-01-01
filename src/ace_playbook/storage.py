"""SQLite-backed storage layer without external dependencies."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .config import ACEConfig
from .schemas import Bullet, Trace


class PlaybookStorage:
    def __init__(self, config: ACEConfig):
        self.config = config
        self._path = config.storage_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bullets (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    helpful_count INTEGER DEFAULT 0,
                    harmful_count INTEGER DEFAULT 0,
                    score REAL DEFAULT 0,
                    embedding TEXT,
                    source_trace_ids TEXT NOT NULL,
                    version INTEGER DEFAULT 0,
                    duplicate_of TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    selected_bullet_ids TEXT NOT NULL,
                    used_bullet_ids TEXT NOT NULL,
                    misleading_bullet_ids TEXT NOT NULL,
                    attribution_notes TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bullet_usage (
                    bullet_id TEXT PRIMARY KEY,
                    total_uses INTEGER DEFAULT 0,
                    last_used_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bullets_kind ON bullets(kind)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bullets_tags ON bullets(tags)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bullets_score_last_used ON bullets(score, last_used_at)"
            )
            self._run_migrations(conn)
            conn.commit()

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        info = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(bullets)").fetchall()
        }
        if "version" not in info:
            conn.execute(
                "ALTER TABLE bullets ADD COLUMN version INTEGER DEFAULT 0"
            )
        if "duplicate_of" not in info:
            conn.execute(
                "ALTER TABLE bullets ADD COLUMN duplicate_of TEXT"
            )
        trace_info = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(traces)").fetchall()
        }
        if "used_bullet_ids" not in trace_info:
            conn.execute(
                "ALTER TABLE traces ADD COLUMN used_bullet_ids TEXT NOT NULL DEFAULT '[]'"
            )
        if "misleading_bullet_ids" not in trace_info:
            conn.execute(
                "ALTER TABLE traces ADD COLUMN misleading_bullet_ids TEXT NOT NULL DEFAULT '[]'"
            )
        if "attribution_notes" not in trace_info:
            conn.execute(
                "ALTER TABLE traces ADD COLUMN attribution_notes TEXT NOT NULL DEFAULT '{}'"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def upsert_bullets(self, bullets: Iterable[Bullet]) -> Tuple[int, int]:
        added = 0
        updated = 0
        with self._connect() as conn:
            for bullet in bullets:
                payload = self._bullet_to_row(bullet)
                exists = conn.execute(
                    "SELECT 1 FROM bullets WHERE id=?",
                    (payload["id"],),
                ).fetchone()
                conn.execute(
                    f"REPLACE INTO bullets ({', '.join(payload.keys())}) VALUES ({', '.join(['?']*len(payload))})",
                    list(payload.values()),
                )
                if exists:
                    updated += 1
                else:
                    added += 1
            conn.commit()
        return added, updated

    def get_bullet(self, bullet_id: str) -> Optional[Bullet]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM bullets WHERE id=?", (bullet_id,)).fetchone()
            if not row:
                return None
            return self._row_to_bullet(row)

    def list_bullets(self) -> List[Bullet]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM bullets").fetchall()
            return [self._row_to_bullet(row) for row in rows]

    def record_trace(self, trace: Trace) -> None:
        payload = self._trace_to_row(trace)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (id, query, selected_bullet_ids, used_bullet_ids, misleading_bullet_ids, attribution_notes, prompt, response, success, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    payload["id"],
                    payload["query"],
                    payload["selected_bullet_ids"],
                    payload["used_bullet_ids"],
                    payload["misleading_bullet_ids"],
                    payload["attribution_notes"],
                    payload["prompt"],
                    payload["response"],
                    payload["success"],
                    payload["metadata"],
                    payload["created_at"],
                ),
            )
            conn.commit()

    def list_traces(self, limit: int = 100) -> List[Trace]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY datetime(created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_trace(row) for row in rows]

    def update_usage(self, bullet_id: str, success: bool) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT total_uses FROM bullet_usage WHERE bullet_id=?",
                (bullet_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO bullet_usage (bullet_id, total_uses, last_used_at) VALUES (?, ?, ?)",
                    (bullet_id, 1, now),
                )
            else:
                conn.execute(
                    "UPDATE bullet_usage SET total_uses=total_uses+1, last_used_at=? WHERE bullet_id=?",
                    (now, bullet_id),
                )
            if success:
                conn.execute(
                    "UPDATE bullets SET helpful_count=helpful_count+1, last_used_at=? WHERE id=?",
                    (now, bullet_id),
                )
            else:
                conn.execute(
                    "UPDATE bullets SET harmful_count=harmful_count+1, last_used_at=? WHERE id=?",
                    (now, bullet_id),
                )
            conn.commit()

    def fetch_embeddings(self) -> Tuple[List[Bullet], List[List[float]]]:
        bullets = self.list_bullets()
        vectors: List[List[float]] = []
        valid: List[Bullet] = []
        for bullet in bullets:
            if not bullet.embedding:
                continue
            vector = [float(x) for x in bullet.embedding]
            if not vector:
                continue
            vectors.append(vector)
            valid.append(bullet)
        return valid, vectors

    def prune_to_ids(self, keep_ids: List[str]) -> None:
        with self._connect() as conn:
            if keep_ids:
                placeholders = ",".join(["?"] * len(keep_ids))
                conn.execute(
                    f"DELETE FROM bullets WHERE id NOT IN ({placeholders})",
                    keep_ids,
                )
            else:
                conn.execute("DELETE FROM bullets")
            conn.commit()

    def _bullet_to_row(self, bullet: Bullet) -> Dict[str, object]:
        return {
            "id": bullet.id,
            "kind": bullet.kind,
            "title": bullet.title,
            "body": bullet.body,
            "tags": json.dumps(bullet.tags),
            "created_at": bullet.created_at.isoformat(),
            "last_used_at": bullet.last_used_at.isoformat() if bullet.last_used_at else None,
            "helpful_count": bullet.helpful_count,
            "harmful_count": bullet.harmful_count,
            "score": bullet.score,
            "embedding": json.dumps(bullet.embedding) if bullet.embedding else None,
            "source_trace_ids": json.dumps(bullet.source_trace_ids),
            "version": bullet.version,
            "duplicate_of": bullet.duplicate_of,
        }

    def _row_to_bullet(self, row: sqlite3.Row) -> Bullet:
        data = dict(row)
        data["tags"] = json.loads(data["tags"])
        data["source_trace_ids"] = json.loads(data["source_trace_ids"])
        if data.get("embedding"):
            data["embedding"] = json.loads(data["embedding"])
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_used_at"):
            data["last_used_at"] = datetime.fromisoformat(data["last_used_at"])
        if data.get("duplicate_of") is None:
            data["duplicate_of"] = None
        return Bullet.from_dict(data)

    def _trace_to_row(self, trace: Trace) -> Dict[str, object]:
        return {
            "id": trace.id,
            "query": trace.query,
            "selected_bullet_ids": json.dumps(trace.selected_bullet_ids),
            "used_bullet_ids": json.dumps(trace.used_bullet_ids),
            "misleading_bullet_ids": json.dumps(trace.misleading_bullet_ids),
            "attribution_notes": json.dumps(trace.attribution_notes),
            "prompt": trace.prompt,
            "response": trace.response,
            "success": 1 if trace.success else 0,
            "metadata": json.dumps(trace.metadata),
            "created_at": trace.created_at.isoformat(),
        }

    def _row_to_trace(self, row: sqlite3.Row) -> Trace:
        data = dict(row)
        data["selected_bullet_ids"] = json.loads(data["selected_bullet_ids"])
        data["used_bullet_ids"] = json.loads(data.get("used_bullet_ids", "[]"))
        data["misleading_bullet_ids"] = json.loads(data.get("misleading_bullet_ids", "[]"))
        data["attribution_notes"] = json.loads(data.get("attribution_notes", "{}"))
        data["metadata"] = json.loads(data["metadata"])
        data["success"] = bool(data["success"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return Trace.from_dict(data)


def dump_playbook(storage: PlaybookStorage) -> List[Dict[str, object]]:
    return [bullet.to_dict() for bullet in storage.list_bullets()]
