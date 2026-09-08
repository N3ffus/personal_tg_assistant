"""Repair dates and embeddings of a row-per-episode watched-film CSV import.

Run inside the app environment with CSV path and the import's episode prefix.
Validates every title before writing; preserves a JSON snapshot for recovery.
"""

import argparse
import asyncio
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from src.config import Settings
from src.domain.knowledge.models import namespace_for
from src.infrastructure.knowledge.factory import build_graphiti


async def repair(csv_path: Path, prefix: str, backup: Path) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        films = [
            row
            for row in csv.DictReader(stream)
            if row["Статус"].strip() == "Посмотрел" and row["Название"].strip()
        ]
    settings = Settings()  # type: ignore[call-arg]
    graph = build_graphiti(settings)
    try:
        rows, _, _ = await graph.driver.execute_query(
            "MATCH (e:Episodic {group_id:$ns}) WHERE e.name STARTS WITH $prefix "
            "RETURN e.name AS name, properties(e) AS props ORDER BY e.name",
            ns=namespace_for(settings.telegram_allowed_user_id),
            prefix=prefix,
            routing_="r",
        )
        if len(rows) != len(films):
            raise ValueError("CSV and stored episode counts differ")
        updates = []
        for index, (row, film) in enumerate(zip(rows, films, strict=True), 1):
            content = row["props"]["content"]
            title = film["Название"].strip()
            if row["name"] != f"{prefix}{index:03d}" or not content.startswith(
                f"Просмотренный фильм: {title};"
            ):
                raise ValueError(f"Title mismatch at row {index}")
            raw_date = film["Последний просмотр"].strip()
            when = (
                datetime.strptime(raw_date, "%B %d, %Y").replace(tzinfo=UTC)
                if raw_date
                else None
            )
            content = content.split("; Дата последнего просмотра:")[0]
            content += "; Дата последнего просмотра: " + (
                when.strftime("%d.%m.%Y") if when else "не указана"
            )
            updates.append(
                {"uuid": row["props"]["uuid"], "content": content, "watched_at": when}
            )
        # Exclusive creation prevents overwriting the only pre-repair snapshot.
        with backup.open("x", encoding="utf-8") as stream:
            os.chmod(backup, 0o600)
            json.dump(
                [dict(row) for row in rows], stream, ensure_ascii=False, default=str
            )
        for start in range(0, len(updates), 25):
            batch = updates[start : start + 25]
            vectors = await graph.embedder.create_batch(
                [item["content"] for item in batch]
            )
            for item, vector in zip(batch, vectors, strict=True):
                item["embedding"] = vector
            await graph.driver.execute_query(
                "UNWIND $updates AS item MATCH (e:Episodic {uuid:item.uuid, group_id:$ns}) "
                "SET e.content=item.content, e.watched_at=item.watched_at, "
                "e.content_embedding=item.embedding",
                updates=batch,
                ns=namespace_for(settings.telegram_allowed_user_id),
            )
            print(
                f"repaired {min(start + 25, len(updates))}/{len(updates)}", flush=True
            )
    finally:
        await graph.close()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    asyncio.run(repair(args.csv, args.prefix, args.backup))
