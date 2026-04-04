from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount

from .compat import DEPENDENCY_AVAILABILITY
from .config import DEFAULT_DB_PATH
from .cursor import HmacCursorCodec
from .db import DatabaseManager
from .engine import BackgroundDeploymentEngine
from .errors import CursorValidationError, JsonRpcError
from .logging_utils import configure_logging
from .models import CreateTaskRequest, SamplingResponse, TriggerDeploymentRequest, UpdateTaskRequest
from .persistence import load_cursor_secrets, rotate_cursor_secret
from .rate_limiter import RateLimitTier, TieredRateLimiter
from .resources import TaskResourceService
from .sampling import SamplingGuard
from .subscriptions import SubscriptionManager
from .tools import ToolService


class EngineeringAutomationServer:
    def __init__(self, database_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.database_path = Path(database_path)
        self.logger = configure_logging()
        self.db = DatabaseManager(self.database_path)
        self.rate_limiter = TieredRateLimiter(self.db)
        self.sdk = Server("engauto-mcp")
        self.sse_transport = SseServerTransport("/messages")
        self.client_capabilities: dict[str, object] = {}
        self._subscriptions = SubscriptionManager(self._emit_to_clients)
        self._sampling_guard = SamplingGuard(self._sample_client)
        self._resources: TaskResourceService | None = None
        self._tools: ToolService | None = None
        self._engine: BackgroundDeploymentEngine | None = None
        self._cursor_monitor_task: asyncio.Task[None] | None = None
        self._instance_id = ""
        self._sessions: dict[str, Any] = {}
        self._register_handlers()
        self.app = Starlette(
            lifespan=self._lifespan,
            routes=[
                Mount("/sse", app=self._handle_sse),
                Mount("/sse/messages", app=self._handle_messages),
            ],
        )

    async def start(self) -> None:
        await self.db.open()
        secrets = load_cursor_secrets(self.database_path)
        self._instance_id = secrets.persistent_instance_id
        self.logger = configure_logging(
            sensitive_values=[
                secrets.persistent_instance_id,
                secrets.current_secret,
                secrets.previous_secret or b"",
            ]
        )
        self._engine = BackgroundDeploymentEngine(
            self.db,
            self._subscriptions,
            instance_id=secrets.persistent_instance_id,
            log_emitter=self._emit_log_message,
        )
        self._resources = TaskResourceService(
            self.db,
            HmacCursorCodec(secrets),
            metric_sink=self._engine.queue_metric,
        )
        await self._engine.start()
        self._tools = ToolService(
            self.db,
            self.rate_limiter,
            self._subscriptions,
            self._sampling_guard,
            self._engine.enqueue,
            DEPENDENCY_AVAILABILITY,
            self._emit_log_message,
        )
        self._cursor_monitor_task = asyncio.create_task(self._monitor_cursor_failures())

    async def stop(self) -> None:
        if self._cursor_monitor_task is not None:
            self._cursor_monitor_task.cancel()
            await asyncio.gather(self._cursor_monitor_task, return_exceptions=True)
            self._cursor_monitor_task = None
        if self._engine is not None:
            await self._engine.stop()
        await self._subscriptions.close()
        self._sessions.clear()
        await self.db.close()

    @asynccontextmanager
    async def _lifespan(self, app: Starlette):
        await self.start()
        try:
            yield
        finally:
            await self.stop()

    async def _handle_sse(self, scope, receive, send) -> None:
        async with self.sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
            await self.sdk.run(
                read_stream,
                write_stream,
                self.sdk.create_initialization_options(
                    notification_options=NotificationOptions(
                        resources_changed=True,
                        tools_changed=True,
                    ),
                ),
            )

    async def _handle_messages(self, scope, receive, send) -> None:
        await self.sse_transport.handle_post_message(scope, receive, send)

    def _register_handlers(self) -> None:
        @self.sdk.list_tools()
        async def list_tools() -> list[types.Tool]:
            assert self._tools is not None
            self._register_current_session()
            return [
                types.Tool(
                    name=tool["name"],
                    description=tool.get("description"),
                    inputSchema=tool.get("inputSchema") or {"type": "object", "properties": {}},
                )
                for tool in self._tools.list_tools()
            ]

        @self.sdk.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            assert self._tools is not None
            self._register_current_session()
            client_id = self._current_client_id()
            try:
                if name == "create_task":
                    request = CreateTaskRequest.model_validate({"client_id": client_id, **arguments})
                    return await self._tools.create_task(request)
                if name == "update_task":
                    request = UpdateTaskRequest.model_validate({"client_id": client_id, **arguments})
                    return await self._tools.update_task(request)
                if name == "trigger_deployment":
                    request = TriggerDeploymentRequest.model_validate({"client_id": client_id, **arguments})
                    return await self._tools.trigger_deployment(request)
                if name == "get_engine_health":
                    return (await self._tools.get_engine_health()).model_dump()
                if name == "maintenance/checkpoint":
                    assert self._engine is not None
                    await self._engine.maintenance_checkpoint()
                    return {"ok": True}
                raise JsonRpcError(-32601, f"Tool '{name}' was not found.")
            except JsonRpcError as error:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps({"error": error.to_payload()}, indent=2),
                        )
                    ],
                    structuredContent={"error": error.to_payload()},
                    isError=True,
                )
            except Exception as exc:
                payload = {"code": -32603, "message": str(exc)}
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps({"error": payload}, indent=2),
                        )
                    ],
                    structuredContent={"error": payload},
                    isError=True,
                )

        @self.sdk.list_resources()
        async def list_resources() -> list[types.Resource]:
            assert self._resources is not None
            client_id = self._current_client_id()
            await self.rate_limiter.consume(client_id, RateLimitTier.RESOURCE)
            self._register_current_session()
            return [
                types.Resource(
                    uri=resource["uri"],
                    name=resource["name"],
                    mimeType=resource.get("mimeType"),
                )
                for resource in await self._resources.list_resources()
            ]

        @self.sdk.list_resource_templates()
        async def list_resource_templates() -> list[types.ResourceTemplate]:
            assert self._resources is not None
            client_id = self._current_client_id()
            await self.rate_limiter.consume(client_id, RateLimitTier.RESOURCE)
            self._register_current_session()
            return [
                types.ResourceTemplate(
                    uriTemplate=template["uriTemplate"],
                    name=template["name"],
                    mimeType=template.get("mimeType"),
                )
                for template in self._resources.list_templates()
            ]

        @self.sdk.read_resource()
        async def read_resource(uri: str) -> list[ReadResourceContents]:
            assert self._resources is not None
            client_id = self._current_client_id()
            await self.rate_limiter.consume(client_id, RateLimitTier.RESOURCE)
            self._register_current_session()
            try:
                page = await self._resources.read_resource(uri)
            except CursorValidationError:
                await self.rate_limiter.record_hmac_failure(client_id)
                raise
            return [
                ReadResourceContents(
                    content=json.dumps(page.model_dump(), indent=2),
                    mime_type="application/json",
                )
            ]

        @self.sdk.subscribe_resource()
        async def subscribe_resource(uri: str) -> None:
            self._subscriptions.subscribe(self._current_client_id(), uri)
            self._register_current_session()

        @self.sdk.unsubscribe_resource()
        async def unsubscribe_resource(uri: str) -> None:
            self._subscriptions.unsubscribe(self._current_client_id(), uri)

    def _register_current_session(self) -> None:
        context = self.sdk.request_context
        self._sessions[self._current_client_id()] = context.session
        client_params = getattr(context.session, "client_params", None)
        if client_params is not None and getattr(client_params, "capabilities", None) is not None:
            try:
                self.client_capabilities = client_params.capabilities.model_dump(exclude_none=True)
            except AttributeError:
                self.client_capabilities = dict(client_params.capabilities)

    def _current_client_id(self) -> str:
        return f"session:{id(self.sdk.request_context.session)}"

    async def _sample_client(self, request) -> SamplingResponse:
        session = self.sdk.request_context.session
        if not session.check_client_capability(types.ClientCapabilities(sampling=types.SamplingCapability())):
            raise JsonRpcError(-32004, "Client does not advertise sampling capability.")
        result = await session.create_message(
            messages=[
                types.SamplingMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=json.dumps(request.model_dump(), indent=2),
                    ),
                )
            ],
            max_tokens=512,
            system_prompt=request.prompt_instructions or "Evaluate the deployment request and respond safely.",
        )
        content = getattr(result, "content", types.TextContent(type="text", text=""))
        text = content.text if hasattr(content, "text") else ""
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {}
        approved = bool(payload.get("approved", True))
        patch = payload.get("patch")
        message = payload.get("message")
        return SamplingResponse(approved=approved, patch=patch, message=message)

    async def _emit_to_clients(
        self,
        method: str,
        payload: dict[str, object],
        client_ids: set[str],
    ) -> None:
        uri = payload.get("uri")
        if method != "notifications/resources/updated" or not isinstance(uri, str):
            return
        for client_id in client_ids:
            session = self._sessions.get(client_id)
            if session is None:
                continue
            try:
                await session.send_resource_updated(uri)
            except Exception:
                self._sessions.pop(client_id, None)

    async def _emit_log_message(self, level: str, message: str) -> None:
        level_map = {
            "DEBUG": "debug",
            "INFO": "info",
            "WARN": "warning",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
        }
        mapped_level = level_map.get(level.upper(), "info")
        stale_sessions: list[str] = []
        for client_id, session in self._sessions.items():
            try:
                await session.send_log_message(mapped_level, message, logger="engauto_mcp")
            except Exception:
                stale_sessions.append(client_id)
        for client_id in stale_sessions:
            self._sessions.pop(client_id, None)

    async def _monitor_cursor_failures(self) -> None:
        while True:
            await asyncio.sleep(5)
            total_state = await self.db.get_system_state("cursor_decode_total")
            failure_state = await self.db.get_system_state("cursor_decode_failures")
            total = int(total_state[1] or 0) if total_state else 0
            failures = int(failure_state[1] or 0) if failure_state else 0
            if total == 0:
                continue
            if failures / total > 0.05:
                secrets = rotate_cursor_secret(self.database_path)
                self.logger = configure_logging(
                    sensitive_values=[
                        secrets.persistent_instance_id,
                        secrets.current_secret,
                        secrets.previous_secret or b"",
                    ]
                )
                if self._resources is not None:
                    self._resources.refresh_cursor_codec(HmacCursorCodec(secrets))
                await self.db.set_system_state("cursor_decode_total", value_integer=0)
                await self.db.set_system_state("cursor_decode_failures", value_integer=0)
                await self._emit_log_message("WARN", "Cursor secret rotated after elevated HMAC failures.")


server = EngineeringAutomationServer()
app = server.app
