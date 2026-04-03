from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from .db import DatabaseManager
from .logging_utils import configure_logging
from .subscriptions import SubscriptionManager

LogEmitter = Callable[[str, str], Awaitable[None]]


class BackgroundDeploymentEngine:
    def __init__(
        self,
        db: DatabaseManager,
        subscriptions: SubscriptionManager,
        instance_id: str,
        log_emitter: LogEmitter,
    ) -> None:
        self._db = db
        self._subscriptions = subscriptions
        self._instance_id = instance_id
        self._log_emitter = log_emitter
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._logger = configure_logging()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        stored_instance_state = await self._db.get_system_state("DEPLOY_LOCK_INSTANCE_ID")
        stored_instance_id = stored_instance_state[0] if stored_instance_state else None
        if stored_instance_id == self._instance_id:
            await self._db.transaction(self._reset_lock_state)

    async def enqueue(self, task_id: str, environment: dict[str, Any]) -> None:
        await self._db.transaction(
            lambda db: db.set_system_state("DEPLOY_LOCK_INSTANCE_ID", value_text=self._instance_id)
        )
        await self._queue.put((task_id, environment))

    async def maintenance_checkpoint(self) -> None:
        await self._log_emitter("WARN", "WAL checkpoint will start in 1 second.")
        await asyncio.sleep(1)
        await self._db.maintenance_checkpoint()

    async def _worker(self) -> None:
        while True:
            task_id, environment = await self._queue.get()
            try:
                await asyncio.sleep(0.25)
                await self._complete_task(task_id, environment)
            except Exception as exc:
                self._logger.exception("Background deployment failed: %s", exc)
            finally:
                self._queue.task_done()

    async def _complete_task(self, task_id: str, environment: dict[str, Any]) -> None:
        new_status = "failed" if environment.get("target") == "fail" else "completed"

        async def callback(db: DatabaseManager) -> None:
            task = await db.fetch_task(task_id)
            if task is None:
                return
            document = json.loads(task.payload_json)
            document["deployment_result"] = {
                "environment": environment,
                "outcome": new_status,
            }
            document["status"] = new_status
            await db.update_task_document(task, document)
            await self._reset_lock_state(db)

        await self._db.transaction(callback)
        self._subscriptions.emit_resource_updated("tasks://running")
        self._subscriptions.emit_resource_updated(f"tasks://{new_status}")

    async def _reset_lock_state(self, db: DatabaseManager) -> None:
        await db.set_system_state("DEPLOY_LOCK", value_text="IDLE")
        await db.set_system_state("DEPLOY_LOCK_INSTANCE_ID", value_text="")
