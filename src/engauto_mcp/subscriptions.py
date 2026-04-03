from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

NotificationEmitter = Callable[[str, dict[str, Any], set[str]], Awaitable[None]]
OnCommitCallback = Callable[[], Awaitable[None]]


class SubscriptionManager:
    def __init__(self, emitter: NotificationEmitter) -> None:
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        self._emitter = emitter
        self._debounce_tasks: dict[str, asyncio.Task[None]] = {}

    def subscribe(self, client_id: str, uri: str) -> None:
        self._subscriptions[client_id].add(uri)

    def unsubscribe(self, client_id: str, uri: str) -> None:
        if client_id in self._subscriptions:
            self._subscriptions[client_id].discard(uri)

    def interested_clients(self, uri: str) -> set[str]:
        return {
            client_id
            for client_id, uris in self._subscriptions.items()
            if uri in uris
        }

    def active_subscriptions_count(self) -> int:
        return sum(len(uris) for uris in self._subscriptions.values())

    def emit_resource_updated(
        self,
        uri: str,
        *,
        on_commit: list[OnCommitCallback] | None = None,
    ) -> None:
        if on_commit is not None:
            on_commit.append(lambda: self._schedule_emit(uri))
            return
        self._schedule_emit_now(uri)

    async def _schedule_emit(self, uri: str) -> None:
        self._schedule_emit_now(uri)

    def _schedule_emit_now(self, uri: str) -> None:
        existing = self._debounce_tasks.get(uri)
        if existing is not None and not existing.done():
            existing.cancel()
        self._debounce_tasks[uri] = asyncio.create_task(self._debounced_emit(uri))

    async def close(self) -> None:
        for task in self._debounce_tasks.values():
            task.cancel()
        if self._debounce_tasks:
            await asyncio.gather(*self._debounce_tasks.values(), return_exceptions=True)

    async def _debounced_emit(self, uri: str) -> None:
        try:
            await asyncio.sleep(0.1)
            clients = self.interested_clients(uri)
            if clients:
                await self._emitter("notifications/resources/updated", {"uri": uri}, clients)
        except asyncio.CancelledError:
            raise
