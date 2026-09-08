"""Embed episodes stored before the adapter started embedding them.

Semantic recall over episodes only sees an episode that carries
``content_embedding``. Episodes written by an earlier release have none, so
they stay reachable by keywords alone until this one-off pass fills them in::

    uv run python -m scripts.backfill_episode_embeddings

Safe to repeat: it only touches episodes whose embedding is still missing.
"""

import asyncio
import logging
import sys

from src.config import Settings
from src.infrastructure.knowledge.factory import build_graphiti

BATCH = 25
PENDING = (
    "MATCH (episode:Episodic) WHERE episode.content_embedding IS NULL "
    "RETURN episode.uuid AS uuid, episode.content AS content LIMIT $limit"
)
STORE = (
    "MATCH (episode:Episodic {uuid: $uuid}) SET episode.content_embedding = $embedding"
)


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    logging.basicConfig(level=logging.WARNING)
    settings = Settings()  # type: ignore[call-arg]
    if not settings.knowledge_memory_enabled:
        print("Knowledge memory is disabled; nothing to backfill.")
        return 1
    graphiti = build_graphiti(settings)
    embedded = 0
    try:
        while True:
            rows, _, _ = await graphiti.driver.execute_query(PENDING, limit=BATCH)
            if not rows:
                break
            contents = [row["content"] or "" for row in rows]
            embeddings = await graphiti.embedder.create_batch(contents)
            for row, embedding in zip(rows, embeddings, strict=True):
                await graphiti.driver.execute_query(
                    STORE, uuid=row["uuid"], embedding=embedding
                )
                embedded += 1
            print(f"embedded {embedded}")
    finally:
        await graphiti.close()  # type: ignore[no-untyped-call]
    print(f"done: {embedded} episode(s) embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
