from __future__ import annotations

from pathlib import Path

from .compat import DEPENDENCY_AVAILABILITY
from .config import DEFAULT_DB_PATH
from .cursor import HmacCursorCodec
from .db import DatabaseManager
from .engine import BackgroundDeploymentEngine
from .errors import JsonRpcError
from .jsonrpc import JsonRpcPeer, StdIOTransport
from .logging_utils import configure_logging
from .models import SamplingRequest, SamplingResponse, TriggerDeploymentRequest
from .persistence import load_cursor_secrets
from .rate_limiter import RateLimitTier, TieredRateLimiter
from .resources import TaskResourceService
from .sampling import SamplingGuard
from .subscriptions import SubscriptionManager
from .tools import ToolService


class EngineeringAutomationServer:
    def __init__(self, database_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.database_path = Path(database_path)
        self.logger = configure_logging()
        self.transport = StdIOTransport()
        self.db = DatabaseManager(self.database_path)
        self.rate_limiter = TieredRateLimiter()
        self.peer: JsonRpcPeer | None = None
        self.client_id = "stdio-client"
        self._subscriptions = SubscriptionManager(self._emit_to_clients)
        self._sampling_guard = SamplingGuard(self._sample_client)
        self._resources: TaskResourceService | None = None
        self._tools: ToolService | None = None
        self._engine: BackgroundDeploymentEngine | None = None

    async def start(self) -> None:
        await self.db.open()
        secrets = load_cursor_secrets(self.database_path)
        self._resources = TaskResourceService(self.db, HmacCursorCodec(secrets))
        self._engine = BackgroundDeploymentEngine(
            self.db,
            self._subscriptions,
            instance_id=secrets.persistent_instance_id,
            log_emitter=self._emit_log_message,
        )
        await self._engine.start()
        self._tools = ToolService(
            self.db,
            self.rate_limiter,
            self._subscriptions,
            self._sampling_guard,
            self._engine.enqueue,
            DEPENDENCY_AVAILABILITY,
        )
        self.peer = JsonRpcPeer(
            self.transport.read_message,
            self.transport.write_message,
            self._handle_request,
            self._handle_notification,
        )

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()
        await self._subscriptions.close()
        await self.db.close()

    async def serve(self) -> None:
        await self.start()
        try:
            assert self.peer is not None
            await self.peer.serve_forever()
        finally:
            await self.stop()

    async def _handle_request(self, message: dict[str, object]) -> dict[str, object] | None:
        method = str(message["method"])
        params = dict(message.get("params", {}))

        if method == "initialize":
            return {
                "serverInfo": {"name": "engauto-mcp", "version": "0.1.0"},
                "capabilities": {
                    "resources": {"subscribe": True},
                    "tools": {},
                    "sampling": {},
                },
            }
        if method == "resources/list":
            self.rate_limiter.consume(self.client_id, RateLimitTier.RESOURCE)
            assert self._resources is not None
            return {"resources": await self._resources.list_resources()}
        if method == "resources/templates/list":
            self.rate_limiter.consume(self.client_id, RateLimitTier.RESOURCE)
            assert self._resources is not None
            return {"resourceTemplates": self._resources.list_templates()}
        if method == "resources/read":
            self.rate_limiter.consume(self.client_id, RateLimitTier.RESOURCE)
            assert self._resources is not None
            page = await self._resources.read_resource(str(params["uri"]))
            return page.model_dump()
        if method == "resources/subscribe":
            uri = str(params["uri"])
            self._subscriptions.subscribe(self.client_id, uri)
            return {"subscribed": True, "uri": uri}
        if method == "resources/unsubscribe":
            uri = str(params["uri"])
            self._subscriptions.unsubscribe(self.client_id, uri)
            return {"subscribed": False, "uri": uri}
        if method == "tools/list":
            assert self._tools is not None
            return {"tools": self._tools.list_tools()}
        if method == "tools/call":
            assert self._tools is not None
            name = str(params["name"])
            arguments = dict(params.get("arguments", {}))
            if name == "trigger_deployment":
                request = TriggerDeploymentRequest.model_validate(
                    {"client_id": self.client_id, **arguments}
                )
                return await self._tools.trigger_deployment(request)
            if name == "get_engine_health":
                return (await self._tools.get_engine_health()).model_dump()
            if name == "maintenance/checkpoint":
                assert self._engine is not None
                await self._engine.maintenance_checkpoint()
                return {"ok": True}
        raise JsonRpcError(-32601, f"Method '{method}' was not found.")

    async def _handle_notification(self, message: dict[str, object]) -> None:
        if message.get("method") == "notifications/initialized":
            self.logger.info("Client initialized")

    async def _sample_client(self, request: SamplingRequest) -> SamplingResponse:
        if self.peer is None:
            raise JsonRpcError(-32020, "Peer is not ready for sampling.")
        result = await self.peer.send_request("sampling/createMessage", request.model_dump())
        return SamplingResponse.model_validate(result)

    async def _emit_to_clients(
        self,
        method: str,
        payload: dict[str, object],
        client_ids: set[str],
    ) -> None:
        if self.peer is None or self.client_id not in client_ids:
            return
        await self.peer.send_notification(method, payload)

    async def _emit_log_message(self, level: str, message: str) -> None:
        if self.peer is not None:
            await self.peer.send_notification(
                "notifications/logging/message",
                {"level": level, "data": message},
            )
