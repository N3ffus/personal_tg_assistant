import aiosqlite

# Polling handles up to 32 updates at once while the business worker writes in
# parallel; without a busy timeout a brief overlap fails as "database is locked".
BUSY_TIMEOUT_SECONDS = 15


def connect(database_path: str) -> aiosqlite.Connection:
    return aiosqlite.connect(database_path, timeout=BUSY_TIMEOUT_SECONDS)


async def use_write_ahead_log(database: aiosqlite.Connection) -> None:
    """Let readers proceed while a writer holds the database.

    The journal mode is stored in the database file, so setting it once at
    startup covers every later connection.
    """
    await database.execute("PRAGMA journal_mode=WAL")
