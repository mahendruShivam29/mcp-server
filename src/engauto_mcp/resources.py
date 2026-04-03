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
    def __init__(
        self,
        db: DatabaseManager,
        cursor_codec: HmacCursorCodec,
        metric_sink: Any | None = None,
    ) -> None:
        self._db = db
        self._cursor_codec = cursor_codec
        self._metric_sink = metric_sink

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
            self._record_cursor_metric(total=1, failures=0)
            try:
                decoded = self._cursor_codec.decode(cursor)
            except CursorValidationError:
                self._record_cursor_metric(total=0, failures=1)
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

    def _record_cursor_metric(self, *, total: int, failures: int) -> None:
        if self._metric_sink is None:
            return
        if total:
            self._metric_sink("cursor_decode_total", total)
        if failures:
            self._metric_sink("cursor_decode_failures", failures)
