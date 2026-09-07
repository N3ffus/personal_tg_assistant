"""Seed a few episodes, search them back and point at the Neo4j Browser.

Run with ``make knowledge-demo`` after ``docker compose up -d neo4j``.
"""

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta

from src.application.services.knowledge import KnowledgeService
from src.config import Settings
from src.domain.knowledge.models import KnowledgeSourceType, namespace_for
from src.infrastructure.knowledge.factory import create_knowledge_service

DEMO_USER_ID = 999_000_001
DEMO_EPISODES = [
    ("Андрей изучает Kafka.", 30),
    ("Андрей работает с Python.", 20),
    ("Андрей хочет изучать AI agents.", 5),
]
DEMO_QUERIES = ["Что Андрей изучает?", "С какими технологиями работает Андрей?"]


async def seed(knowledge: KnowledgeService) -> None:
    now = datetime.now(UTC)
    for index, (content, days_ago) in enumerate(DEMO_EPISODES, start=1):
        reference_time = now - timedelta(days=days_ago)
        print(f"→ remember [{reference_time:%Y-%m-%d}] {content}")
        await knowledge.remember(
            user_id=DEMO_USER_ID,
            content=content,
            source_id=f"demo-{index}",
            source_type=KnowledgeSourceType.NOTE,
            reference_time=reference_time,
        )


async def recall(knowledge: KnowledgeService) -> None:
    for query in DEMO_QUERIES:
        print(f"\n? search: {query}")
        for fact in await knowledge.search(user_id=DEMO_USER_ID, query=query):
            validity = f" valid_from={fact.valid_from} valid_until={fact.valid_until}"
            print(f"  • {fact.fact}{validity} source={fact.source}")


async def main() -> int:
    # Windows consoles default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    logging.basicConfig(level=logging.INFO)
    settings = Settings()  # type: ignore[call-arg]
    knowledge = await create_knowledge_service(settings)
    if knowledge is None:
        print(
            "Knowledge memory is unavailable. Set KNOWLEDGE_MEMORY_ENABLED=true and "
            "start Neo4j with: docker compose up -d neo4j"
        )
        return 1
    try:
        await seed(knowledge)
        await recall(knowledge)
    finally:
        await knowledge.close()
    print(
        "\nOpen http://localhost:7474/browser and run:\n"
        "  MATCH p=(a)-[r]->(b)\n"
        f"  WHERE a.group_id = '{namespace_for(DEMO_USER_ID)}'\n"
        "  RETURN p LIMIT 100;"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
