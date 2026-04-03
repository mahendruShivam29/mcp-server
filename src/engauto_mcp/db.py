from __future__ import annotations

import contextvars
import json
import time
import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .compat import aiosqlite, require_dependency
from .models import TaskRecord
from .persistence import initialize_persistence

StagedState = dict[str, tuple[str | None, int | None]]
_staged_live_state: contextvars.ContextVar[StagedState | None] = contextvars.ContextVar(
    "staged_live_state",
    default=None,
)


class DatabaseManager:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.connection: Any | None = None
        self.live_state: dict[str, tuple[str | None, int | None]] = {}
        self._task_staging: dict[int, StagedState] = {}

    async def open(self) -> None:
        initialize_persistence(self.database_path)
        aiosqlite_module = require_dependency("aiosqlite", aiosqlite)
        self.connection = await aiosqlite_module.connect(self.database_path)
        self.connection.row_factory = aiosqlite_module.Row
        await self._execute("PRAGMA journal_mode = WAL;")
        await self._execute("PRAGMA foreign_keys = ON;")
        await self._execute("PRAGMA busy_timeout = 5000;")
        await self._load_live_state()

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
            self.connection = None

    async def transaction(
        self,
        callback: Callable[["DatabaseManager", StagedState], Awaitable[Any]],
    ) -> Any:
        self._ensure_connection()
        staged: StagedState = {}
        token = _staged_live_state.set(staged)
        task = asyncio.current_task()
        task_id = id(task) if task is not None else None
        if task_id is not None:
            self._task_staging[task_id] = staged
        try:
            await self._execute("BEGIN IMMEDIATE;")
            result = await callback(self, staged)
            await self._commit()
            self._apply_staged_state(staged)
            return result
        except Exception:
            await self._rollback()
            raise
        finally:
            if task_id is not None:
                self._task_staging.pop(task_id, None)
            _staged_live_state.reset(token)

    async def fetch_task(self, task_id: str) -> TaskRecord | None:
        row = await self._fetchone(
            "SELECT id, title, status, etag, payload_json, created_at, updated_at FROM tasks WHERE id = ?",
            (task_id,),
        )
        return None if row is None else TaskRecord.model_validate(dict(row))

    async def list_tasks(
        self,
        *,
        status: str,
        limit: int,
        offset: int,
    ) -> list[TaskRecord]:
        rows = await self._fetchall(
            """
            SELECT id, title, status, etag, payload_json, created_at, updated_at
            FROM tasks
            WHERE status = ?
            ORDER BY updated_at DESC, id
            LIMIT ? OFFSET ?
            """,
            (status, limit, offset),
        )
        return [TaskRecord.model_validate(dict(row)) for row in rows]

    async def update_task_document(
        self,
        task: TaskRecord,
        document: dict[str, Any],
        *,
        staged: StagedState | None = None,
    ) -> TaskRecord:
        now = _unix_timestamp()
        next_etag = task.etag + 1
        payload_json = json.dumps(document, sort_keys=True)
        await self._execute(
            """
            UPDATE tasks
            SET title = ?, status = ?, etag = ?, payload_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                document.get("title", task.title),
                document.get("status", task.status),
                next_etag,
                payload_json,
                now,
                task.id,
            ),
        )
        return TaskRecord.model_validate(
            {
                "id": task.id,
                "title": document.get("title", task.title),
                "status": document.get("status", task.status),
                "etag": next_etag,
                "payload_json": payload_json,
                "created_at": task.created_at,
                "updated_at": now,
            }
        )

    async def set_system_state(
        self,
        key: str,
        *,
        value_text: str | None = None,
        value_integer: int | None = None,
        staged: StagedState | None = None,
    ) -> None:
        now = _unix_timestamp()
        await self._execute(
            """
            INSERT INTO system_state (key, value_text, value_integer, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_text = excluded.value_text,
                value_integer = excluded.value_integer,
                updated_at = excluded.updated_at
            """,
            (key, value_text, value_integer, now),
        )
        self.stage_live_state(key, value_text=value_text, value_integer=value_integer, staged=staged)
        if staged is None and self._active_staging() is None:
            await self._commit()

    async def get_system_state(self, key: str) -> tuple[str | None, int | None] | None:
        row = await self._fetchone(
            "SELECT value_text, value_integer FROM system_state WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        return row["value_text"], row["value_integer"]

    async def get_system_state_record(self, key: str) -> dict[str, Any] | None:
        row = await self._fetchone(
            "SELECT key, value_text, value_integer, updated_at FROM system_state WHERE key = ?",
            (key,),
        )
        return None if row is None else dict(row)

    async def get_metadata_text(self, key: str) -> str | None:
        row = await self._fetchone(
            "SELECT value_text FROM system_metadata WHERE key = ?",
            (key,),
        )
        return None if row is None else row["value_text"]

    async def write_read_heartbeat(self, *, staged: StagedState | None = None) -> int:
        now = _unix_timestamp()
        await self.set_system_state("HEARTBEAT", value_integer=now, staged=staged)
        row = await self._fetchone(
            "SELECT value_integer FROM system_state WHERE key = 'HEARTBEAT'"
        )
        return int(row["value_integer"]) if row else now

    async def sqlite_version(self) -> str:
        row = await self._fetchone("SELECT sqlite_version() AS version")
        return str(row["version"])

    async def checkpoint_wal(self) -> None:
        await self._execute("PRAGMA wal_checkpoint(PASSIVE);")

    async def maintenance_checkpoint(self) -> bool:
        row = await self._fetchone("PRAGMA wal_checkpoint(RESTART);")
        if row is None:
            return False
        return int(row[0]) == 0

    def stage_live_state(
        self,
        key: str,
        *,
        value_text: str | None = None,
        value_integer: int | None = None,
        staged: StagedState | None = None,
    ) -> None:
        active = staged if staged is not None else self._active_staging()
        if active is not None:
            active[key] = (value_text, value_integer)
        else:
            self.live_state[key] = (value_text, value_integer)

    async def _load_live_state(self) -> None:
        rows = await self._fetchall("SELECT key, value_text, value_integer FROM system_state")
        self.live_state = {
            row["key"]: (row["value_text"], row["value_integer"])
            for row in rows
        }

    def _apply_staged_state(self, staged: StagedState) -> None:
        if not staged:
            return
        self.live_state.update(staged)

    def _active_staging(self) -> StagedState | None:
        task = asyncio.current_task()
        if task is not None:
            staged = self._task_staging.get(id(task))
            if staged is not None:
                return staged
        return _staged_live_state.get()

    async def _execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        self._ensure_connection()
        staged = self._active_staging()
        cursor = await self.connection.execute(sql, parameters)
        if staged is not None:
            _staged_live_state.set(staged)
        return cursor

    async def _fetchone(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        cursor = await self._execute(sql, parameters)
        return await cursor.fetchone()

    async def _fetchall(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[Any]:
        cursor = await self._execute(sql, parameters)
        return await cursor.fetchall()

    async def _commit(self) -> None:
        self._ensure_connection()
        await self.connection.commit()

    async def _rollback(self) -> None:
        self._ensure_connection()
        await self.connection.rollback()

    def _ensure_connection(self) -> None:
        if self.connection is None:
            raise RuntimeError("DatabaseManager is not open.")


def _unix_timestamp() -> int:
    return int(time.time())
