from __future__ import annotations

import asyncio
from contextvars import ContextVar, copy_context
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .compat import aiosqlite, require_dependency
from .models import TaskRecord
from .persistence import initialize_persistence

StagedState = dict[str, tuple[str | None, int | None]]
_staged_context: ContextVar[dict[str, Any] | None] = ContextVar("staged_context", default=None)
CommitCallback = Callable[[], Awaitable[None]]


class DeferredOnCommitCallback:
    def __init__(self, callback: CommitCallback) -> None:
        self._callback = callback

    async def __call__(self) -> None:
        await self._callback()


class DatabaseManager:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.connection: Any | None = None
        self.live_state: dict[str, tuple[str | None, int | None]] = {}

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
        callback: Callable[
            ["DatabaseManager", list[CommitCallback | DeferredOnCommitCallback]],
            Awaitable[Any],
        ],
    ) -> Any:
        self._ensure_connection()
        staged: StagedState = {}
        token = _staged_context.set(staged)
        on_commit: list[CommitCallback | DeferredOnCommitCallback] = []
        try:
            await self._execute("BEGIN IMMEDIATE;")
            result = await self._run_in_context(callback, self, on_commit)
            await self._commit()
            committed_staged = self._current_staged_state() or staged
            self._apply_staged_state(committed_staged)
            immediate_callbacks = [
                item for item in on_commit if not isinstance(item, DeferredOnCommitCallback)
            ]
            deferred_callbacks = [
                item for item in on_commit if isinstance(item, DeferredOnCommitCallback)
            ]
            for item in immediate_callbacks:
                await item()
            for item in deferred_callbacks:
                await item()
            return result
        except Exception:
            await self._rollback()
            raise
        finally:
            _staged_context.reset(token)

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
        last_updated_at: int | None = None,
        last_id: str | None = None,
    ) -> list[TaskRecord]:
        if last_updated_at is None or last_id is None:
            rows = await self._fetchall(
                """
                SELECT id, title, status, etag, payload_json, created_at, updated_at
                FROM tasks
                WHERE status = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            rows = await self._fetchall(
                """
                SELECT id, title, status, etag, payload_json, created_at, updated_at
                FROM tasks
                WHERE status = ?
                  AND (updated_at, id) < (?, ?)
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (status, last_updated_at, last_id, limit),
            )
        return [TaskRecord.model_validate(dict(row)) for row in rows]

    async def update_task_document(
        self,
        task: TaskRecord,
        document: dict[str, Any],
    ) -> TaskRecord:
        now = _unix_timestamp()
        next_etag = task.etag + 1
        payload_json = json.dumps(document, sort_keys=True)
        next_status = document.get("status", task.status)
        next_title = document.get("title", task.title)
        await self._execute(
            """
            UPDATE tasks
            SET title = ?, status = ?, etag = etag + 1, payload_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                next_title,
                next_status,
                payload_json,
                now,
                task.id,
            ),
        )
        return TaskRecord.model_validate(
            {
                "id": task.id,
                "title": next_title,
                "status": next_status,
                "etag": next_etag,
                "payload_json": payload_json,
                "created_at": task.created_at,
                "updated_at": now,
            }
        )

    async def create_task(
        self,
        *,
        task_id: str,
        title: str,
        status: str = "pending",
        payload: dict[str, Any] | None = None,
    ) -> TaskRecord:
        now = _unix_timestamp()
        payload_json = json.dumps(payload or {}, sort_keys=True)
        try:
            await self._execute(
                """
                INSERT INTO tasks (id, title, status, etag, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                """,
                (task_id, title, status, payload_json, now, now),
            )
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError(f"Task '{task_id}' already exists.") from exc
            raise
        return TaskRecord.model_validate(
            {
                "id": task_id,
                "title": title,
                "status": status,
                "etag": 1,
                "payload_json": payload_json,
                "created_at": now,
                "updated_at": now,
            }
        )

    async def set_system_state(
        self,
        key: str,
        *,
        value_text: str | None = None,
        value_integer: int | None = None,
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
        self.stage_live_state(key, value_text=value_text, value_integer=value_integer)
        if self._current_staged_state() is None:
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

    async def write_read_heartbeat(self) -> int:
        now = _unix_timestamp()
        await self.set_system_state("HEARTBEAT", value_integer=now)
        row = await self._fetchone("SELECT value_integer FROM system_state WHERE key = 'HEARTBEAT'")
        return int(row["value_integer"]) if row else now

    async def sqlite_version(self) -> str:
        row = await self._fetchone("SELECT sqlite_version() AS version")
        return str(row["version"])

    async def integrity_check(self) -> str:
        row = await self._fetchone("PRAGMA integrity_check;")
        return str(row[0]) if row is not None else "unknown"

    async def foreign_key_check(self) -> list[dict[str, Any]]:
        rows = await self._fetchall("PRAGMA foreign_key_check;")
        return [dict(row) for row in rows]

    async def checkpoint_wal(self) -> None:
        await self._execute("PRAGMA wal_checkpoint(PASSIVE);")

    async def maintenance_checkpoint(self) -> bool:
        row = await self._fetchone("PRAGMA wal_checkpoint(RESTART);")
        if row is None:
            return False
        return int(row[0]) == 0

    async def wal_autocheckpoint_pages(self) -> int:
        row = await self._fetchone("PRAGMA wal_autocheckpoint;")
        return int(row[0]) if row is not None else 0

    async def page_count(self) -> int:
        row = await self._fetchone("PRAGMA page_count;")
        return int(row[0]) if row is not None else 0

    async def page_size(self) -> int:
        row = await self._fetchone("PRAGMA page_size;")
        return int(row[0]) if row is not None else 0

    def stage_live_state(
        self,
        key: str,
        *,
        value_text: str | None = None,
        value_integer: int | None = None,
    ) -> None:
        staged = self._current_staged_state()
        if staged is None:
            self.live_state[key] = (value_text, value_integer)
            return
        staged[key] = (value_text, value_integer)

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

    def _current_staged_state(self) -> StagedState | None:
        staged = _staged_context.get()
        if staged is None:
            return None
        return staged

    async def _execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> Any:
        self._ensure_connection()
        return await self._run_in_context(self.connection.execute, sql, parameters)

    async def _fetchone(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> Any:
        cursor = await self._execute(sql, parameters)
        return await self._run_in_context(cursor.fetchone)

    async def _fetchall(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[Any]:
        cursor = await self._execute(sql, parameters)
        return await self._run_in_context(cursor.fetchall)

    async def _commit(self) -> None:
        self._ensure_connection()
        await self._run_in_context(self.connection.commit)

    async def _rollback(self) -> None:
        self._ensure_connection()
        await self._run_in_context(self.connection.rollback)

    async def _run_in_context(self, callback: Callable[..., Any], *args: Any) -> Any:
        context = copy_context()
        result = context.run(callback, *args)
        if inspect.isawaitable(result):
            task = context.run(asyncio.create_task, result)
            return await task
        return result

    def _ensure_connection(self) -> None:
        if self.connection is None:
            raise RuntimeError("DatabaseManager is not open.")


def _unix_timestamp() -> int:
    return int(time.time())
