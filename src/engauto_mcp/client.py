from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from openai import OpenAI

from .errors import JsonRpcError
from .jsonrpc import JsonRpcPeer, StdIOTransport, encode_message, read_message_async
from .models import SamplingRequest, SamplingResponse
from .server import EngineeringAutomationServer

SamplingHandler = Callable[[SamplingRequest], Awaitable[SamplingResponse]]
TASK_STATUSES = ("pending", "running", "completed", "failed")
TERMINAL_TASK_STATUSES = {"completed", "failed"}


class _FrameWriter:
    def __init__(self, write_fn: Callable[[bytes], Awaitable[None]]) -> None:
        self._write_fn = write_fn
        self._stdout_lock = asyncio.Lock()

    async def write_message(self, payload: dict[str, Any]) -> None:
        frame = encode_message(payload)
        async with self._stdout_lock:
            await self._write_fn(frame)


class SubprocessTransport:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._writer = _FrameWriter(self._write_frame)

    async def read_message(self) -> dict[str, Any] | None:
        if self._process.stdout is None:
            return None
        return await read_message_async(self._process.stdout)

    async def write_message(self, payload: dict[str, Any]) -> None:
        await self._writer.write_message(payload)

    async def _write_frame(self, frame: bytes) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Client subprocess stdin is not available.")
        self._process.stdin.write(frame)
        await self._process.stdin.drain()

    async def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            wait_closed = getattr(self._process.stdin, "wait_closed", None)
            if callable(wait_closed):
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                    await wait_closed()


class ClientStdIOTransport:
    def __init__(self) -> None:
        self._transport = StdIOTransport()
        self._writer = _FrameWriter(self._write_frame)

    async def read_message(self) -> dict[str, Any] | None:
        return await self._transport.read_message()

    async def write_message(self, payload: dict[str, Any]) -> None:
        await self._writer.write_message(payload)

    async def _write_frame(self, frame: bytes) -> None:
        await self._transport.write_message_bytes(frame)

    async def close(self) -> None:
        return None


class _QueueTransport:
    def __init__(self, reader: asyncio.Queue[dict[str, Any] | None], writer: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._reader = reader
        self._writer = writer

    async def read_message(self) -> dict[str, Any] | None:
        return await self._reader.get()

    async def write_message(self, payload: dict[str, Any]) -> None:
        await self._writer.put(payload)

    async def close(self) -> None:
        return None


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
        await self._peer.send_request(
            "initialize",
            {"capabilities": {"sampling": {}}},
        )
        await self._peer.send_notification("notifications/initialized", {})

    async def stop(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None
        close_transport = getattr(self._transport, "close", None)
        if callable(close_transport):
            await close_transport()

    async def list_resources(self) -> dict[str, Any]:
        return await self._peer.send_request("resources/list", {})

    async def list_templates(self) -> dict[str, Any]:
        return await self._peer.send_request("resources/templates/list", {})

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return await self._peer.send_request("resources/read", {"uri": uri})

    async def list_tools(self) -> dict[str, Any]:
        return await self._peer.send_request("tools/list", {})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._peer.send_request("tools/call", {"name": name, "arguments": arguments})

    async def call_tool_with_policy(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        mutable_arguments = json.loads(json.dumps(arguments))
        while True:
            try:
                result = await self.call_tool(name, mutable_arguments)
                if name == "trigger_deployment":
                    return await self._await_deployment_terminal_state(mutable_arguments, result)
                return result
            except JsonRpcError as error:
                if error.code == -32002:
                    retry_after = float((error.data or {}).get("retry_after", 0))
                    await asyncio.sleep(retry_after)
                    continue
                # The deployment tools rely on JSON Patch "test" operations for /status and /etag
                # as an optimistic concurrency guard. If the server returns -32003, the task was
                # changed after the caller last read it. Re-reading the resource and updating those
                # test values prevents a stale patch from silently applying to a newer task version.
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

    async def _await_deployment_terminal_state(
        self,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return result

        deadline = asyncio.get_running_loop().time() + 5.0
        await asyncio.sleep(0.35)
        delay = 0.1
        latest = result
        while asyncio.get_running_loop().time() < deadline:
            task = await _find_task_item(self, task_id)
            if task is not None:
                latest = {
                    "task_id": task["id"],
                    "status": task["status"],
                    "etag": task["etag"],
                }
                if task["status"] in TERMINAL_TASK_STATUSES:
                    return latest
            await asyncio.sleep(delay)
        return latest

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


async def _start_in_process_server() -> tuple[MCPClient, EngineeringAutomationServer, asyncio.Task[None]]:
    client_to_server: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    server_to_client: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    server = EngineeringAutomationServer()
    server.transport = _QueueTransport(client_to_server, server_to_client)
    await server.start()
    assert server.peer is not None
    server_task = asyncio.create_task(server.peer.serve_forever())
    client = MCPClient(transport=_QueueTransport(server_to_client, client_to_server))
    await client.start()
    return client, server, server_task


async def _shutdown_process(process: asyncio.subprocess.Process) -> None:
    transport = getattr(process, "_transport", None)
    try:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await asyncio.wait_for(process.wait(), timeout=5.0)
    finally:
        if transport is not None:
            with contextlib.suppress(RuntimeError, OSError):
                transport.close()
        await asyncio.sleep(0)


async def _run_shell(client: MCPClient) -> int:
    print("Connected. Commands: resources, tools, read <uri>, call <tool> <json>, exit", file=sys.stderr)
    while True:
        raw_command = await asyncio.to_thread(input, "engauto> ")
        command = raw_command.strip()
        if not command:
            continue
        lowered = command.lower()
        if lowered in {"exit", "quit"}:
            return 0
        if lowered == "resources":
            print(json.dumps(await client.list_resources(), indent=2))
            continue
        if lowered == "tools":
            print(json.dumps(await client.list_tools(), indent=2))
            continue
        if lowered.startswith("read "):
            uri = command[5:].strip()
            if not uri:
                print("Usage: read <uri>", file=sys.stderr)
                continue
            print(json.dumps(await client.read_resource(uri), indent=2))
            continue
        if lowered.startswith("call "):
            remainder = command[5:].strip()
            if not remainder:
                print("Usage: call <tool> <json-arguments>", file=sys.stderr)
                continue
            name, _, json_payload = remainder.partition(" ")
            arguments = {}
            if json_payload.strip():
                try:
                    arguments = json.loads(json_payload)
                except json.JSONDecodeError as exc:
                    print(f"Invalid JSON arguments: {exc}", file=sys.stderr)
                    continue
            print(json.dumps(await client.call_tool_with_policy(name, arguments), indent=2))
            continue
        print("Unknown command. Use: resources, tools, read <uri>, call <tool> <json>, exit", file=sys.stderr)


def _load_tool_arguments(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "arguments_file", None):
        raw_arguments = Path(args.arguments_file).read_text(encoding="utf-8")
    else:
        raw_arguments = getattr(args, "arguments", "{}")
    return json.loads(raw_arguments) if raw_arguments else {}


async def _read_resource_with_policy(client: MCPClient, uri: str) -> dict[str, Any]:
    while True:
        try:
            return await client.read_resource(uri)
        except JsonRpcError as error:
            if error.code != -32002:
                if error.code == -32603 and "cannot start a transaction within a transaction" in error.message:
                    await asyncio.sleep(0.1)
                    continue
                raise
            retry_after = float((error.data or {}).get("retry_after", 0))
            await asyncio.sleep(retry_after)


async def _find_task_item(client: MCPClient, task_id: str) -> dict[str, Any] | None:
    for status in TASK_STATUSES:
        uri = f"tasks://{status}"
        while uri:
            page = await _read_resource_with_policy(client, uri)
            for item in page.get("items", []):
                if item.get("id") == task_id:
                    return item
            uri = None
            next_cursor = page.get("next_cursor")
            if next_cursor:
                uri = f"tasks://{status}?cursor={next_cursor}"
    return None


def _merge_deployment_patch(
    patch: list[dict[str, Any]] | None,
    *,
    status: str,
    etag: int,
) -> list[dict[str, Any]]:
    merged_patch = json.loads(json.dumps(patch or []))
    status_set = False
    etag_set = False
    for op in merged_patch:
        if op.get("op") != "test":
            continue
        if op.get("path") == "/status":
            op["value"] = status
            status_set = True
        elif op.get("path") == "/etag":
            op["value"] = etag
            etag_set = True
    if not status_set:
        merged_patch.append({"op": "test", "path": "/status", "value": status})
    if not etag_set:
        merged_patch.append({"op": "test", "path": "/etag", "value": etag})
    return merged_patch


async def _finalize_planned_tool_call(
    client: MCPClient,
    tool_call: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    if tool_call.get("name") != "trigger_deployment":
        return tool_call

    arguments = json.loads(json.dumps(tool_call.get("arguments", {})))
    task_id = arguments.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return tool_call

    task = await _find_task_item(client, task_id)
    if task is None:
        return tool_call

    arguments.setdefault("reason", query)
    environment = arguments.get("environment")
    if not isinstance(environment, dict):
        arguments["environment"] = {}
    arguments["patch"] = _merge_deployment_patch(
        arguments.get("patch"),
        status=str(task["status"]),
        etag=int(task["etag"]),
    )
    return {"name": "trigger_deployment", "arguments": arguments}


async def _run_cli(args: argparse.Namespace) -> int:
    if args.command == "stdio-proxy":
        client = MCPClient(transport=ClientStdIOTransport())
        await client.run_stdio_forever()
        return 0

    process: asyncio.subprocess.Process | None = None
    server: EngineeringAutomationServer | None = None
    server_task: asyncio.Task[None] | None = None
    try:
        process = await _spawn_server(args.server_command)
        client = MCPClient(process)
        await client.start()
    except (PermissionError, NotImplementedError, OSError, RuntimeError):
        client, server, server_task = await _start_in_process_server()
    try:
        if args.command == "resources-list":
            print(json.dumps(await client.list_resources(), indent=2))
        elif args.command == "templates-list":
            print(json.dumps(await client.list_templates(), indent=2))
        elif args.command == "resource-read":
            print(json.dumps(await client.read_resource(args.uri), indent=2))
        elif args.command == "tool-call":
            arguments = _load_tool_arguments(args)
            print(json.dumps(await client.call_tool_with_policy(args.name, arguments), indent=2))
        elif args.command == "ask":
            resource_uri = _infer_task_resource_uri(args.query)
            if resource_uri is not None:
                print(json.dumps(await client.read_resource(resource_uri), indent=2))
            else:
                direct_response = await _maybe_handle_direct_query(client, args.query)
                if direct_response is not None:
                    print(json.dumps(direct_response, indent=2))
                else:
                    tool_call = await _plan_tool_call_with_openai(client, args.query)
                    tool_call = await _finalize_planned_tool_call(client, tool_call, args.query)
                    print(
                        json.dumps(
                            await client.call_tool_with_policy(tool_call["name"], tool_call["arguments"]),
                            indent=2,
                        )
                    )
        elif args.command == "shell":
            return await _run_shell(client)
        return 0
    finally:
        await client.stop()
        if server_task is not None:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        if server is not None:
            await server.stop()
        if process is not None:
            if process.stdin is not None:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                    process.stdin.close()
                    wait_closed = getattr(process.stdin, "wait_closed", None)
                    if callable(wait_closed):
                        await wait_closed()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
            transport = getattr(process, "_transport", None)
            if transport is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    transport.close()
            await asyncio.sleep(0)


async def _plan_tool_call_with_openai(
    client: MCPClient,
    query: str,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. The ask command requires OpenAI access.", file=sys.stderr)
        raise SystemExit(2)

    openai_client = OpenAI(api_key=api_key)
    tools_response = await client.list_tools()
    openai_tools = _build_openai_tools(tools_response.get("tools", []))
    if not openai_tools:
        print("The server did not return any callable tool schemas.", file=sys.stderr)
        raise SystemExit(2)

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a planner for a minimal MCP client. "
                            "Choose exactly one tool call that best satisfies the user request. "
                            "Use the provided tool schemas exactly as given. "
                            "Do not invent argument names. "
                            "If required arguments are missing from the user request, make the best safe guess only when the schema allows it."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                tools=openai_tools,
                tool_choice="required",
            ),
            timeout=20.0,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            raise ValueError("OpenAI returned no tool call.")
        selected_call = tool_calls[0]
        name = selected_call.function.name
        arguments = json.loads(selected_call.function.arguments or "{}")
    except asyncio.TimeoutError as exc:
        print("OpenAI request timed out after 20 seconds.", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"OpenAI failed to produce a valid tool call: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if not isinstance(name, str) or not isinstance(arguments, dict):
        print("OpenAI did not return a valid schema-constrained tool call.", file=sys.stderr)
        raise SystemExit(2)
    return {"name": name, "arguments": arguments}


def _infer_task_resource_uri(query: str) -> str | None:
    lowered = query.lower()
    if "task" not in lowered:
        return None
    for status in ("pending", "running", "completed", "failed"):
        if status in lowered:
            return f"tasks://{status}"
    return None


async def _maybe_handle_direct_query(client: MCPClient, query: str) -> dict[str, Any] | None:
    lowered = " ".join(query.lower().split())

    if _mentions_any(lowered, ("tool list", "tools list", "list tools", "show tools", "show tool list")):
        return await client.list_tools()
    if _mentions_any(
        lowered,
        (
            "resource list",
            "resources list",
            "list resources",
            "show resources",
            "show resource list",
        ),
    ):
        return await client.list_resources()
    if _mentions_any(
        lowered,
        (
            "template list",
            "templates list",
            "list templates",
            "show templates",
            "show template list",
        ),
    ):
        return await client.list_templates()

    return None


def _mentions_any(query: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in query for phrase in phrases)

def _build_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_definitions: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        input_schema = tool.get("inputSchema") or tool.get("input_schema") or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        if not isinstance(input_schema, dict):
            continue
        tool_definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": input_schema,
                },
            }
        )
    return tool_definitions


def _default_server_command() -> list[str]:
    if os.name == "nt":
        root = Path(__file__).resolve().parents[2]
        return [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "scripts" / "run_server.ps1"),
        ]
    return [sys.executable, "-m", "engauto_mcp"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MCP stdio client for EngAuto-MCP")
    parser.add_argument(
        "--server-command",
        nargs="+",
        default=_default_server_command(),
        help="Server command to execute",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("resources-list")
    subparsers.add_parser("templates-list")
    resource_read = subparsers.add_parser("resource-read")
    resource_read.add_argument("uri")
    tool_call = subparsers.add_parser("tool-call")
    tool_call.add_argument("name")
    tool_call_arguments = tool_call.add_mutually_exclusive_group()
    tool_call_arguments.add_argument("--arguments", default="{}")
    tool_call_arguments.add_argument("--arguments-file")
    ask = subparsers.add_parser("ask")
    ask.add_argument("query")
    subparsers.add_parser("shell")
    subparsers.add_parser("stdio-proxy")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_cli(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Client failed: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

