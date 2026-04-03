from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from .errors import JsonRpcError
from .jsonrpc import JsonRpcPeer, StdIOTransport, encode_message, read_message_async
from .models import SamplingRequest, SamplingResponse

SamplingHandler = Callable[[SamplingRequest], Awaitable[SamplingResponse]]


class SubprocessTransport:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._stdout_lock = asyncio.Lock()

    async def read_message(self) -> dict[str, Any] | None:
        if self._process.stdout is None:
            return None
        return await read_message_async(self._process.stdout)

    async def write_message(self, payload: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Client subprocess stdin is not available.")
        frame = encode_message(payload)
        async with self._stdout_lock:
            self._process.stdin.write(frame)
            await self._process.stdin.drain()


class ClientStdIOTransport:
    def __init__(self) -> None:
        self._transport = StdIOTransport()

    async def read_message(self) -> dict[str, Any] | None:
        return await self._transport.read_message()

    async def write_message(self, payload: dict[str, Any]) -> None:
        await self._transport.write_message(payload)


class MCPClient:
    def __init__(
        self,
        process: asyncio.subprocess.Process | None = None,
        sampling_handler: SamplingHandler | None = None,
        transport: Any | None = None,
    ) -> None:
        if transport is not None:
            self._transport = transport
        elif process is not None:
            self._transport = SubprocessTransport(process)
        else:
            raise ValueError("Either a subprocess or transport must be provided.")
        self._sampling_handler = sampling_handler or self._default_sampling_handler
        self._peer = JsonRpcPeer(
            self._transport.read_message,
            self._transport.write_message,
            self._handle_request,
            self._handle_notification,
        )
        self._reader_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._reader_task = asyncio.create_task(self._peer.serve_forever())
        await self._peer.send_request("initialize", {})
        await self._peer.send_notification("notifications/initialized", {})

    async def stop(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)

    async def list_resources(self) -> dict[str, Any]:
        return await self._peer.send_request("resources/list", {})

    async def list_templates(self) -> dict[str, Any]:
        return await self._peer.send_request("resources/templates/list", {})

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return await self._peer.send_request("resources/read", {"uri": uri})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._peer.send_request("tools/call", {"name": name, "arguments": arguments})

    async def call_tool_with_policy(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        mutable_arguments = json.loads(json.dumps(arguments))
        while True:
            try:
                return await self.call_tool(name, mutable_arguments)
            except JsonRpcError as error:
                if error.code == -32002:
                    retry_after = float((error.data or {}).get("retry_after", 0))
                    await asyncio.sleep(retry_after)
                    continue
                if error.code == -32003 and await self._repair_patch_from_resource(
                    mutable_arguments,
                    error.data,
                ):
                    continue
                raise

    async def _handle_request(self, message: dict[str, Any]) -> dict[str, Any]:
        if message["method"] == "sampling/createMessage":
            request = SamplingRequest.model_validate(message.get("params", {}))
            response = await self._sampling_handler(request)
            return response.model_dump()
        raise JsonRpcError(-32601, f"Method '{message['method']}' is not supported by the client.")

    async def _handle_notification(self, message: dict[str, Any]) -> None:
        if message.get("method") == "notifications/logging/message":
            params = message.get("params", {})
            print(json.dumps({"log": params}, ensure_ascii=True), file=sys.stderr)

    async def _default_sampling_handler(self, request: SamplingRequest) -> SamplingResponse:
        patch = None
        if request.original_error:
            patch = [{"op": "test", "path": "/status", "value": request.current_status}]
            actual = request.original_error.get("data", {}).get("actual", {})
            if "etag" in actual:
                patch.append({"op": "test", "path": "/etag", "value": actual["etag"]})
        return SamplingResponse(approved=True, patch=patch, message="Auto-approved remediation")

    async def _repair_patch_from_resource(
        self,
        arguments: dict[str, Any],
        error_data: dict[str, Any] | None,
    ) -> bool:
        if not error_data:
            return False
        resource_uri = error_data.get("resource_uri")
        patch = arguments.get("patch")
        if not resource_uri or not isinstance(patch, list):
            return False

        resource = await self.read_resource(resource_uri)
        items = resource.get("items", [])
        task_id = arguments.get("task_id")
        actual = None
        for item in items:
            if item.get("id") == task_id:
                actual = item
                break
        if actual is None:
            actual = error_data.get("actual", {})

        repaired = False
        for op in patch:
            if op.get("op") != "test":
                continue
            if op.get("path") == "/status" and "status" in actual:
                op["value"] = actual["status"]
                repaired = True
            if op.get("path") == "/etag" and "etag" in actual:
                op["value"] = actual["etag"]
                repaired = True
        if repaired:
            arguments["_last_reread_resource_uri"] = resource_uri
        return repaired

    async def run_stdio_forever(self) -> None:
        self._reader_task = asyncio.create_task(self._peer.serve_forever())
        await self._reader_task


async def _spawn_server(command: list[str]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
    )


async def _run_cli(args: argparse.Namespace) -> int:
    if args.command == "stdio-proxy":
        client = MCPClient(transport=ClientStdIOTransport())
        await client.run_stdio_forever()
        return 0

    process = await _spawn_server(args.server_command)
    client = MCPClient(process)
    await client.start()
    try:
        if args.command == "resources-list":
            print(json.dumps(await client.list_resources(), indent=2))
        elif args.command == "templates-list":
            print(json.dumps(await client.list_templates(), indent=2))
        elif args.command == "resource-read":
            print(json.dumps(await client.read_resource(args.uri), indent=2))
        elif args.command == "tool-call":
            arguments = json.loads(args.arguments) if args.arguments else {}
            print(json.dumps(await client.call_tool_with_policy(args.name, arguments), indent=2))
        return 0
    finally:
        await client.stop()
        process.terminate()
        await process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MCP stdio client for EngAuto-MCP")
    parser.add_argument(
        "--server-command",
        nargs="+",
        default=[sys.executable, "-m", "engauto_mcp"],
        help="Server command to execute",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("resources-list")
    subparsers.add_parser("templates-list")
    resource_read = subparsers.add_parser("resource-read")
    resource_read.add_argument("uri")
    tool_call = subparsers.add_parser("tool-call")
    tool_call.add_argument("name")
    tool_call.add_argument("--arguments", default="{}")
    subparsers.add_parser("stdio-proxy")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
