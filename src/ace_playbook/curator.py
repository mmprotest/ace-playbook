"""Curator merges deltas into storage."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .config import ACEConfig
from .embeddings import BaseEmbeddings
from .schemas import Bullet, BulletPatch, DeltaRuntime, MergeReport, cosine_similarity
from .storage import PlaybookStorage

logger = logging.getLogger(__name__)


@dataclass
class Curator:
    config: ACEConfig
    storage: PlaybookStorage
    embedder: BaseEmbeddings

    def merge(self, delta: DeltaRuntime) -> MergeReport:
        logger.info("Merging delta with %d bullets and %d patches", len(delta.bullets), len(delta.patches))
        valid_bullets = self._validate_bullets(delta.bullets)
        vectors = self.embedder.embed_texts([b.body for b in valid_bullets]).vectors
        for bullet, vector in zip(valid_bullets, vectors):
            bullet.embedding = vector
        deduplicated = self._deduplicate(valid_bullets)
        added, updated = self.storage.upsert_bullets(deduplicated)
        updated += self._apply_patches(delta.patches)
        if delta.traces:
            for trace in delta.traces:
                self.storage.record_trace(trace)
        skipped = len(delta.bullets) - len(valid_bullets)
        if self.config.grow_and_refine == "proactive":
            self._prune_if_needed()
        return MergeReport(added=added, updated=updated, skipped=skipped, deduplicated=len(delta.bullets) - len(deduplicated))

    def _validate_bullets(self, bullets: Iterable[Bullet]) -> List[Bullet]:
        seen_titles = set()
        valid: List[Bullet] = []
        for bullet in bullets:
            try:
                if isinstance(bullet, Bullet):
                    bullet = Bullet.from_dict(bullet.to_dict())
                else:
                    bullet = Bullet.from_dict(dict(bullet))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid bullet: %s", exc)
                continue
            key = (bullet.kind, bullet.title)
            if key in seen_titles:
                logger.debug("Duplicate bullet title %s skipped", bullet.title)
                continue
            seen_titles.add(key)
            valid.append(bullet)
        return valid

    def _deduplicate(self, bullets: List[Bullet]) -> List[Bullet]:
        if not bullets:
            return []
        keep: List[Bullet] = []
        existing_bullets, existing_vectors = self.storage.fetch_embeddings()
        for bullet in bullets:
            duplicate_found = False
            if bullet.embedding:
                for stored_bullet, stored_vector in zip(existing_bullets, existing_vectors):
                    if cosine_similarity(stored_vector, bullet.embedding) >= self.config.dedup_cosine_threshold:
                        duplicate_found = True
                        logger.info(
                            "Rejected bullet %s as duplicate of %s (cosine >= %.2f)",
                            bullet.title,
                            stored_bullet.id,
                            self.config.dedup_cosine_threshold,
                        )
                        break
            if not duplicate_found:
                for existing in keep:
                    if not existing.embedding or not bullet.embedding:
                        continue
                    if cosine_similarity(existing.embedding, bullet.embedding) >= self.config.dedup_cosine_threshold:
                        duplicate_found = True
                        existing.helpful_count += bullet.helpful_count
                        existing.tags = sorted(set(existing.tags) | set(bullet.tags))
                        existing.source_trace_ids = list(set(existing.source_trace_ids + bullet.source_trace_ids))
                        logger.info(
                            "Merged duplicate incoming bullet %s into %s",
                            bullet.title,
                            existing.id,
                        )
                        break
            if not duplicate_found:
                keep.append(bullet)
        return keep

    def _apply_patches(self, patches: Iterable[BulletPatch]) -> int:
        updated_bullets: List[Bullet] = []
        for patch in patches:
            bullet = self.storage.get_bullet(patch.bullet_id)
            if bullet is None:
                continue
            if patch.op == "inc_helpful":
                bullet.helpful_count += 1
            elif patch.op == "inc_harmful":
                bullet.harmful_count += 1
            elif patch.op == "patch" and patch.patch_text:
                if patch.patch_mode == "append":
                    bullet.body = f"{bullet.body}\n{patch.patch_text}".strip()
                else:
                    bullet.body = patch.patch_text
            updated_bullets.append(bullet)
        if updated_bullets:
            self.storage.upsert_bullets(updated_bullets)
        return len(updated_bullets)

    def _prune_if_needed(self) -> None:
        bullets = self.storage.list_bullets()
        if len(bullets) <= self.config.refine_window_size:
            return
        bullets.sort(key=lambda b: (b.helpful_count - b.harmful_count, b.created_at), reverse=True)
        to_keep = bullets[: self.config.refine_window_size]
        keep_ids = {b.id for b in to_keep}
        self.storage.prune_to_ids(list(keep_ids))
