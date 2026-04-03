from __future__ import annotations

import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

CURSOR_SECRET_NUM_BYTES = 32
SYSTEM_METADATA_TABLE = "system_metadata"


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS system_metadata (
        key TEXT PRIMARY KEY,
        value_text TEXT,
        value_blob BLOB,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        CHECK (
            (value_text IS NOT NULL AND value_blob IS NULL)
            OR (value_text IS NULL AND value_blob IS NOT NULL)
        )
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value_text TEXT,
        value_integer INTEGER,
        updated_at INTEGER NOT NULL,
        CHECK (
            (value_text IS NOT NULL AND value_integer IS NULL)
            OR (value_text IS NULL AND value_integer IS NOT NULL)
        )
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
        etag INTEGER NOT NULL DEFAULT 0,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tasks_status_updated_at
    ON tasks(status, updated_at DESC, id)
    """,
)


@dataclass(frozen=True, slots=True)
class CursorSecrets:
    persistent_instance_id: str
    current_secret: bytes
    previous_secret: bytes | None


def connect_bootstrap_database(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000;")
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_persistence(database_path: str | Path) -> CursorSecrets:
    with connect_bootstrap_database(database_path) as connection:
        connection.execute("PRAGMA journal_mode = WAL;")
        connection.execute("PRAGMA foreign_keys = ON;")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        secrets_state = _ensure_security_foundation(connection)
        _ensure_system_seed_data(connection)
        connection.commit()
        return secrets_state


def load_cursor_secrets(database_path: str | Path) -> CursorSecrets:
    with connect_bootstrap_database(database_path) as connection:
        return _read_cursor_secrets(connection)


def rotate_cursor_secret(database_path: str | Path) -> CursorSecrets:
    with connect_bootstrap_database(database_path) as connection:
        state = _read_cursor_secrets(connection)
        now = _unix_timestamp()
        new_secret = secrets.token_bytes(CURSOR_SECRET_NUM_BYTES)
        _upsert_metadata(connection, "cursor_secret_previous", value_blob=state.current_secret, now=now)
        _upsert_metadata(connection, "cursor_secret_current", value_blob=new_secret, now=now)
        connection.commit()
        return CursorSecrets(
            persistent_instance_id=state.persistent_instance_id,
            current_secret=new_secret,
            previous_secret=state.current_secret,
        )


def _ensure_security_foundation(connection: sqlite3.Connection) -> CursorSecrets:
    now = _unix_timestamp()
    instance_id = _get_metadata_text(connection, "persistent_instance_id")
    if instance_id is None:
        instance_id = str(uuid.uuid4())
        _upsert_metadata(connection, "persistent_instance_id", value_text=instance_id, now=now)

    current_secret = _get_metadata_blob(connection, "cursor_secret_current")
    if current_secret is None:
        current_secret = secrets.token_bytes(CURSOR_SECRET_NUM_BYTES)
        _upsert_metadata(connection, "cursor_secret_current", value_blob=current_secret, now=now)

    previous_secret = _get_metadata_blob(connection, "cursor_secret_previous")
    return CursorSecrets(
        persistent_instance_id=instance_id,
        current_secret=current_secret,
        previous_secret=previous_secret,
    )


def _ensure_system_seed_data(connection: sqlite3.Connection) -> None:
    now = _unix_timestamp()
    connection.execute(
        """
        INSERT INTO system_state (key, value_text, value_integer, updated_at)
        VALUES ('DEPLOY_LOCK', 'IDLE', NULL, ?)
        ON CONFLICT(key) DO NOTHING
        """,
        (now,),
    )
    connection.execute(
        """
        INSERT INTO system_state (key, value_text, value_integer, updated_at)
        VALUES ('DEPLOY_LOCK_INSTANCE_ID', '', NULL, ?)
        ON CONFLICT(key) DO NOTHING
        """,
        (now,),
    )
    connection.execute(
        """
        INSERT INTO tasks (id, title, status, etag, payload_json, created_at, updated_at)
        VALUES ('task-demo', 'Demo Deployment Task', 'pending', 1, '{}', ?, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (now, now),
    )


def _read_cursor_secrets(connection: sqlite3.Connection) -> CursorSecrets:
    instance_id = _get_metadata_text(connection, "persistent_instance_id")
    current_secret = _get_metadata_blob(connection, "cursor_secret_current")
    previous_secret = _get_metadata_blob(connection, "cursor_secret_previous")
    if instance_id is None or current_secret is None:
        raise RuntimeError("Persistence foundation has not been initialized.")
    return CursorSecrets(
        persistent_instance_id=instance_id,
        current_secret=current_secret,
        previous_secret=previous_secret,
    )


def _get_metadata_text(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        f"SELECT value_text FROM {SYSTEM_METADATA_TABLE} WHERE key = ?",
        (key,),
    ).fetchone()
    return None if row is None else row["value_text"]


def _get_metadata_blob(connection: sqlite3.Connection, key: str) -> bytes | None:
    row = connection.execute(
        f"SELECT value_blob FROM {SYSTEM_METADATA_TABLE} WHERE key = ?",
        (key,),
    ).fetchone()
    return None if row is None else row["value_blob"]


def _upsert_metadata(
    connection: sqlite3.Connection,
    key: str,
    *,
    value_text: str | None = None,
    value_blob: bytes | None = None,
    now: int | None = None,
) -> None:
    timestamp = _unix_timestamp() if now is None else now
    connection.execute(
        """
        INSERT INTO system_metadata (key, value_text, value_blob, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value_text = excluded.value_text,
            value_blob = excluded.value_blob,
            updated_at = excluded.updated_at
        """,
        (key, value_text, value_blob, timestamp, timestamp),
    )


def _unix_timestamp() -> int:
    return int(time.time())
