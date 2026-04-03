from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from .cursor import HmacCursorCodec
from .errors import CursorValidationError
from .db import DatabaseManager
from .models import ResourcePage

TASKS_TEMPLATE = "tasks://{status}{?cursor}"


class TaskResourceService:
    def __init__(self, db: DatabaseManager, cursor_codec: HmacCursorCodec) -> None:
        self._db = db
        self._cursor_codec = cursor_codec

    async def list_resources(self) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        for status in ("pending", "running", "completed", "failed"):
            page = await self.read_resource(f"tasks://{status}")
            resources.append(
                {
                    "uri": page.uri,
                    "name": f"Tasks ({status})",
                    "mimeType": "application/json",
                }
            )
        return resources

    def list_templates(self) -> list[dict[str, Any]]:
        return [
            {
                "uriTemplate": TASKS_TEMPLATE,
                "name": "Tasks by status",
                "mimeType": "application/json",
            }
        ]

    async def read_resource(self, uri: str, *, page_size: int = 20) -> ResourcePage:
        parsed = urlparse(uri)
        status = parsed.netloc or parsed.path.lstrip("/")
        query = parse_qs(parsed.query)
        cursor = query.get("cursor", [None])[0]
        offset = 0
        migration_hint = False
        if cursor:
            await self._record_cursor_metric(total=1, failures=0)
            try:
                decoded = self._cursor_codec.decode(cursor)
            except CursorValidationError:
                await self._record_cursor_metric(total=0, failures=1)
                raise
            offset = decoded.offset
            migration_hint = decoded.migration_hint

        tasks = await self._db.list_tasks(status=status, limit=page_size, offset=offset)
        items = [
            {
                "id": task.id,
                "title": task.title,
                "status": task.status,
                "etag": task.etag,
                "payload": json.loads(task.payload_json),
                "updated_at": task.updated_at,
            }
            for task in tasks
        ]
        next_cursor = None
        if len(items) == page_size:
            next_cursor = self._cursor_codec.encode(offset + page_size)
        normalized_uri = f"tasks://{status}"
        if cursor:
            normalized_uri = f"{normalized_uri}?{urlencode({'cursor': cursor})}"
        return ResourcePage(
            uri=normalized_uri,
            items=items,
            next_cursor=next_cursor,
            migration_hint=migration_hint,
        )

    def refresh_cursor_codec(self, cursor_codec: HmacCursorCodec) -> None:
        self._cursor_codec = cursor_codec

    async def _record_cursor_metric(self, *, total: int, failures: int) -> None:
        if total:
            total_state = await self._db.get_system_state("cursor_decode_total")
            current_total = int(total_state[1] or 0) if total_state else 0
            await self._db.set_system_state("cursor_decode_total", value_integer=current_total + total)
        if failures:
            failure_state = await self._db.get_system_state("cursor_decode_failures")
            current_failures = int(failure_state[1] or 0) if failure_state else 0
            await self._db.set_system_state("cursor_decode_failures", value_integer=current_failures + failures)
