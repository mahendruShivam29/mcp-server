from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
import sys
from typing import Any

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

from .errors import JsonRpcError
from .models import SamplingRequest, SamplingResponse

TASK_STATUSES = ("pending", "running", "completed", "failed")
TERMINAL_TASK_STATUSES = {"completed", "failed"}


class MCPClient:
    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def list_resources(self) -> dict[str, Any]:
        result = await self._session.list_resources()
        return {
            "resources": [
                resource.model_dump(mode="json", exclude_none=True) for resource in result.resources
            ]
        }

    async def list_templates(self) -> dict[str, Any]:
        result = await self._session.list_resource_templates()
        return {
            "resourceTemplates": [
                template.model_dump(mode="json", exclude_none=True)
                for template in result.resourceTemplates
            ]
        }

    async def read_resource(self, uri: str) -> dict[str, Any]:
        result = await self._session.read_resource(uri)
        contents = result.contents
        if not contents:
            return {"uri": uri, "contents": []}
        first = contents[0]
        text = getattr(first, "text", "")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {
                    "uri": uri,
                    "contents": [
                        content.model_dump(mode="json", exclude_none=True) for content in contents
                    ],
                }
        return {
            "uri": uri,
            "contents": [content.model_dump(mode="json", exclude_none=True) for content in contents],
        }

    async def list_tools(self) -> dict[str, Any]:
        result = await self._session.list_tools()
        return {"tools": [tool.model_dump(mode="json", exclude_none=True) for tool in result.tools]}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._session.call_tool(name, arguments)
        if result.isError:
            raise _tool_error_to_jsonrpc(result)
        if result.structuredContent is not None:
            return dict(result.structuredContent)
        return _content_blocks_to_payload(result.content)

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
                # Deployment uses JSON Patch test operations on /status and /etag as an
                # optimistic concurrency check. If the task changed after the caller last
                # read it, the server rejects the request with a TOCTOU conflict. Re-reading
                # the resource and updating those test values ensures the retry applies only
                # to the latest task version instead of silently overwriting stale state.
                if error.code == -32003 and await self._repair_patch_from_resource(
                    mutable_arguments,
                    error.data,
                ):
                    continue
                raise

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


def _content_blocks_to_payload(content_blocks: list[Any]) -> dict[str, Any]:
    if len(content_blocks) == 1 and getattr(content_blocks[0], "type", None) == "text":
        text = getattr(content_blocks[0], "text", "")
        if isinstance(text, str):
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return {"text": text}
    return {"content": [block.model_dump(mode="json", exclude_none=True) for block in content_blocks]}


def _tool_error_to_jsonrpc(result: types.CallToolResult) -> JsonRpcError:
    if isinstance(result.structuredContent, dict):
        error_payload = result.structuredContent.get("error")
        if isinstance(error_payload, dict):
            return JsonRpcError(
                int(error_payload.get("code", -32603)),
                str(error_payload.get("message", "Tool call failed.")),
                error_payload.get("data"),
            )

    payload = _content_blocks_to_payload(result.content)
    if "error" in payload and isinstance(payload["error"], dict):
        error_payload = payload["error"]
        return JsonRpcError(
            int(error_payload.get("code", -32603)),
            str(error_payload.get("message", "Tool call failed.")),
            error_payload.get("data"),
        )
    return JsonRpcError(-32603, json.dumps(payload))


async def _logging_callback(params: types.LoggingMessageNotificationParams) -> None:
    print(
        json.dumps(
            {
                "log": {
                    "level": params.level,
                    "logger": params.logger,
                    "data": params.data,
                }
            },
            ensure_ascii=True,
        ),
        file=sys.stderr,
    )


async def _sampling_callback(
    context: Any,
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    request = _parse_sampling_request(params)
    approved = await _confirm_sampling_request(request)
    if approved:
        response = _build_sampling_response(request)
    else:
        response = SamplingResponse(approved=False, patch=None, message="Deployment was denied by the terminal operator.")
    return types.CreateMessageResult(
        role="assistant",
        model="engauto-mcp-client",
        content=types.TextContent(type="text", text=json.dumps(response.model_dump())),
        stopReason="endTurn",
    )


def _parse_sampling_request(params: types.CreateMessageRequestParams) -> SamplingRequest:
    for message in params.messages:
        content = getattr(message, "content", None)
        if getattr(content, "type", None) == "text":
            text = getattr(content, "text", "")
            if isinstance(text, str):
                try:
                    return SamplingRequest.model_validate(json.loads(text))
                except Exception:
                    continue
    return SamplingRequest(
        reason="Sampling request",
        task_id="unknown",
        current_status="unknown",
        environment={},
        diff={},
    )


def _build_sampling_response(request: SamplingRequest) -> SamplingResponse:
    patch = None
    if request.original_error:
        patch = [{"op": "test", "path": "/status", "value": request.current_status}]
        actual = request.original_error.get("data", {}).get("actual", {})
        if "etag" in actual:
            patch.append({"op": "test", "path": "/etag", "value": actual["etag"]})
    return SamplingResponse(approved=True, patch=patch, message="Auto-approved remediation")


async def _confirm_sampling_request(request: SamplingRequest) -> bool:
    if not sys.stdin.isatty():
        print(
            "Sampling approval requires an interactive terminal. Denying deployment by default.",
            file=sys.stderr,
        )
        return False

    prompt = _format_sampling_prompt(request)
    response = await asyncio.to_thread(input, prompt)
    return response.strip().lower() in {"y", "yes"}


def _format_sampling_prompt(request: SamplingRequest) -> str:
    lines = [
        "",
        "Deployment approval required.",
        f"Task: {request.task_id}",
        f"Reason: {request.reason}",
        f"Current status: {request.current_status}",
        f"Environment: {json.dumps(request.environment, ensure_ascii=True, sort_keys=True)}",
    ]
    if request.diff:
        lines.append(f"Diff: {json.dumps(request.diff, ensure_ascii=True, sort_keys=True)}")
    if request.original_error:
        lines.append(
            f"Conflict: {json.dumps(request.original_error.get('data', {}), ensure_ascii=True, sort_keys=True)}"
        )
    lines.append("Approve deployment? [y/N]: ")
    return "\n".join(lines)


@contextlib.asynccontextmanager
async def _get_mcp_client(
    server_url: str,
    *,
    sampling_callback: Any | None = None,
) -> AsyncIterator[MCPClient]:
    async with sse_client(server_url) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            sampling_callback=sampling_callback or _sampling_callback,
            sampling_capabilities=types.SamplingCapability(),
            logging_callback=_logging_callback,
            client_info=types.Implementation(name="engauto-mcp-client", version="0.1.0"),
        ) as session:
            await session.initialize()
            yield MCPClient(session)


async def _run_client_operation(
    server_url: str,
    operation: Any,
    *,
    sampling_callback: Any | None = None,
) -> Any:
    pending_error: Exception | None = None
    pending_result: Any = None
    async with _get_mcp_client(server_url, sampling_callback=sampling_callback) as client:
        try:
            pending_result = await operation(client)
        except Exception as exc:
            pending_error = exc
    if pending_error is not None:
        raise pending_error
    return pending_result


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
    if args.command == "resources-list":
        result = await _run_client_operation(args.server_url, lambda client: client.list_resources())
        print(json.dumps(result, indent=2))
    elif args.command == "tools-list":
        result = await _run_client_operation(args.server_url, lambda client: client.list_tools())
        print(json.dumps(result, indent=2))
    elif args.command == "templates-list":
        result = await _run_client_operation(args.server_url, lambda client: client.list_templates())
        print(json.dumps(result, indent=2))
    elif args.command == "resource-read":
        result = await _run_client_operation(args.server_url, lambda client: client.read_resource(args.uri))
        print(json.dumps(result, indent=2))
    elif args.command == "tool-call":
        arguments = _load_tool_arguments(args)
        result = await _run_client_operation(
            args.server_url,
            lambda client: client.call_tool_with_policy(args.name, arguments),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "ask":
        async def operation(client: MCPClient) -> dict[str, Any]:
            planned_action = await _plan_tool_call_with_openai(client, args.query)
            return await _execute_planned_action(client, planned_action, args.query)

        result = await _run_client_operation(args.server_url, operation)
        print(json.dumps(result, indent=2))
    elif args.command == "shell":
        return await _run_client_operation(args.server_url, _run_shell)
    return 0


async def _plan_tool_call_with_openai(
    client: MCPClient,
    query: str,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. The ask command requires OpenAI access.")

    openai_client = OpenAI(api_key=api_key)
    tools_response = await client.list_tools()
    resources_response = await client.list_resources()
    openai_tools = _build_openai_tools(
        tools_response.get("tools", []),
        resources_response.get("resources", []),
    )
    if not openai_tools:
        raise RuntimeError("The server did not return any callable tool schemas.")

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
                            "Choose exactly one function call that best satisfies the user request. "
                            "The available functions include both server tools and client-side resource actions. "
                            "Use the provided schemas exactly as given. "
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
    except Exception as exc:
        if isinstance(exc, asyncio.TimeoutError):
            raise RuntimeError("OpenAI request timed out after 20 seconds.") from exc
        raise RuntimeError(f"OpenAI failed to produce a valid tool call: {exc}") from exc

    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise RuntimeError("OpenAI did not return a valid schema-constrained tool call.")
    return {"name": name, "arguments": arguments}


def _build_openai_tools(
    tools: list[dict[str, Any]],
    resources: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
    resource_uris = [
        resource.get("uri")
        for resource in (resources or [])
        if isinstance(resource, dict) and isinstance(resource.get("uri"), str)
    ]
    if resource_uris:
        tool_definitions.append(
            {
                "type": "function",
                "function": {
                    "name": "read_resource",
                    "description": (
                        "Read one server resource by URI. Use this for queries about current task state, "
                        "such as pending, running, completed, or failed tasks."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "enum": resource_uris,
                                "description": "The exact resource URI to read.",
                            }
                        },
                        "required": ["uri"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    tool_definitions.extend(
        [
            {
                "type": "function",
                "function": {
                    "name": "list_resources",
                    "description": "List currently available server resources.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_resource_templates",
                    "description": "List currently available server resource templates.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_tools",
                    "description": "List currently available server tools and their schemas.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]
    )
    return tool_definitions


async def _execute_planned_action(
    client: MCPClient,
    planned_action: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    name = planned_action.get("name")
    arguments = planned_action.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise RuntimeError("OpenAI did not return a valid schema-constrained tool call.")

    if name == "read_resource":
        uri = arguments.get("uri")
        if not isinstance(uri, str) or not uri:
            raise RuntimeError("OpenAI selected read_resource without a valid uri.")
        return await client.read_resource(uri)
    if name == "list_resources":
        return await client.list_resources()
    if name == "list_resource_templates":
        return await client.list_templates()
    if name == "list_tools":
        return await client.list_tools()

    tool_call = await _finalize_planned_tool_call(client, planned_action, query)
    return await client.call_tool_with_policy(tool_call["name"], tool_call["arguments"])


def _default_server_url() -> str:
    return os.environ.get("ENGAUTO_MCP_SERVER_URL", "http://localhost:8000/sse/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MCP SSE client for EngAuto-MCP")
    parser.add_argument(
        "--server-url",
        default=_default_server_url(),
        help="SSE server URL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("resources-list")
    subparsers.add_parser("tools-list")
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
