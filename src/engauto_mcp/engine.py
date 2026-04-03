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
        self._metric_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._metric_task: asyncio.Task[None] | None = None
        self._logger = configure_logging()

    async def start(self) -> None:
        await self._recover_interrupted_work()
        if self._task is None:
            self._task = asyncio.create_task(self._worker())
        if self._metric_task is None:
            self._metric_task = asyncio.create_task(self._flush_metrics_worker())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._metric_task is not None:
            self._metric_task.cancel()
            await asyncio.gather(self._metric_task, return_exceptions=True)
            self._metric_task = None
        stored_instance_state = await self._db.get_system_state("DEPLOY_LOCK_INSTANCE_ID")
        stored_instance_id = stored_instance_state[0] if stored_instance_state else None
        if stored_instance_id == self._instance_id:
            await self._db.transaction(self._reset_lock_state)

    async def enqueue(self, task_id: str, environment: dict[str, Any]) -> None:
        await self._db.transaction(
            lambda db, staged, on_commit: db.set_system_state(
                "DEPLOY_LOCK_INSTANCE_ID",
                value_text=self._instance_id,
                staged=staged,
            )
        )
        await self._queue.put((task_id, environment))

    async def maintenance_checkpoint(self) -> None:
        await self._log_emitter("WARN", "WAL checkpoint will start in 1 second.")
        await asyncio.sleep(1)
        if not await self._db.maintenance_checkpoint():
            await self._log_emitter("WARN", "WAL checkpoint skipped due to active readers.")

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

        async def callback(
            db: DatabaseManager,
            staged: dict[str, tuple[str | None, int | None]],
            on_commit: list[Callable[[], Awaitable[None]]],
        ) -> None:
            task = await db.fetch_task(task_id)
            if task is None:
                return
            document = json.loads(task.payload_json)
            document["deployment_result"] = {
                "environment": environment,
                "outcome": new_status,
            }
            document["status"] = new_status
            await db.update_task_document(task, document, staged=staged)
            await self._reset_lock_state(db, staged, on_commit)
            self._subscriptions.emit_resource_updated("tasks://running", on_commit=on_commit)
            self._subscriptions.emit_resource_updated(f"tasks://{new_status}", on_commit=on_commit)

        await self._db.transaction(callback)

    async def _reset_lock_state(
        self,
        db: DatabaseManager,
        staged: dict[str, tuple[str | None, int | None]],
        on_commit: list[Callable[[], Awaitable[None]]],
    ) -> None:
        await db.set_system_state("DEPLOY_LOCK", value_text="IDLE", staged=staged)
        await db.set_system_state("DEPLOY_LOCK_INSTANCE_ID", value_text="", staged=staged)

    def queue_metric(self, key: str, amount: int = 1) -> None:
        self._metric_queue.put_nowait((key, amount))

    async def _flush_metrics_worker(self) -> None:
        while True:
            metric_key, amount = await self._metric_queue.get()
            batch: dict[str, int] = {metric_key: amount}
            try:
                await asyncio.sleep(0.05)
                while True:
                    next_key, next_amount = self._metric_queue.get_nowait()
                    batch[next_key] = batch.get(next_key, 0) + next_amount
                    self._metric_queue.task_done()
            except asyncio.QueueEmpty:
                pass

            try:
                await self._flush_metric_batch(batch)
            finally:
                self._metric_queue.task_done()

    async def _flush_metric_batch(self, batch: dict[str, int]) -> None:
        async def callback(
            db: DatabaseManager,
            staged: dict[str, tuple[str | None, int | None]],
            on_commit: list[Callable[[], Awaitable[None]]],
        ) -> None:
            for key, amount in batch.items():
                current = await db.get_system_state(key)
                current_value = int(current[1] or 0) if current else 0
                await db.set_system_state(key, value_integer=current_value + amount, staged=staged)

        await self._db.transaction(callback)

    async def _recover_interrupted_work(self) -> None:
        lock_state = await self._db.get_system_state("DEPLOY_LOCK")
        lock_owner = await self._db.get_system_state("DEPLOY_LOCK_INSTANCE_ID")
        if (lock_state[0] if lock_state else None) != "RUNNING":
            return
        if (lock_owner[0] if lock_owner else None) != self._instance_id:
            return

        async def callback(
            db: DatabaseManager,
            staged: dict[str, tuple[str | None, int | None]],
            on_commit: list[Callable[[], Awaitable[None]]],
        ) -> None:
            running_tasks = await db.list_tasks(status="running", limit=1000)
            for task in running_tasks:
                document = json.loads(task.payload_json)
                document["status"] = "failed"
                document["recovery_reason"] = "Recovered interrupted deployment during startup."
                await db.update_task_document(task, document, staged=staged)
            await self._reset_lock_state(db, staged, on_commit)
            if running_tasks:
                self._subscriptions.emit_resource_updated("tasks://running", on_commit=on_commit)
                self._subscriptions.emit_resource_updated("tasks://failed", on_commit=on_commit)

        await self._db.transaction(callback)
