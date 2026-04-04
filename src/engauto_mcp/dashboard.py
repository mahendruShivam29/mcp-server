from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
import shlex
import traceback
from typing import Any

import streamlit as st

from engauto_mcp.client import MCPClient, _shutdown_process, _spawn_server, _start_in_process_server
from engauto_mcp.errors import JsonRpcError

STATUS_ORDER = ("pending", "running", "completed", "failed")
DEPLOY_TARGETS = ("dev", "staging", "prod")


async def _with_client_session(
    operation: Callable[[MCPClient], Awaitable[Any]],
    server_command: list[str],
) -> Any:
    client: MCPClient | None = None
    process: asyncio.subprocess.Process | None = None
    server: Any | None = None
    server_task: asyncio.Task[None] | None = None
    try:
        try:
            process = await _spawn_server(server_command)
            client = MCPClient(process)
            await client.start()
        except (PermissionError, NotImplementedError, OSError, RuntimeError):
            client, server, server_task = await _start_in_process_server()
        return await operation(client)
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                await client.stop()
        if server_task is not None:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        if server is not None:
            await server.stop()
        if process is not None:
            await _shutdown_process(process)


async def _fetch_task_snapshot(server_command: list[str]) -> dict[str, list[dict[str, Any]]]:
    async def operation(client: MCPClient) -> dict[str, list[dict[str, Any]]]:
        snapshot: dict[str, list[dict[str, Any]]] = {}
        for status in STATUS_ORDER:
            snapshot[status] = await _read_all_tasks(client, status)
        return snapshot

    return await _with_client_session(operation, server_command)


async def _read_all_tasks(client: MCPClient, status: str) -> list[dict[str, Any]]:
    uri = f"tasks://{status}"
    items: list[dict[str, Any]] = []
    while uri:
        page = await _read_resource_with_policy(client, uri)
        items.extend(page.get("items", []))
        next_cursor = page.get("next_cursor")
        uri = f"tasks://{status}?cursor={next_cursor}" if next_cursor else ""
    return items


async def _deploy_task(
    server_command: list[str],
    task: dict[str, Any],
    environment_target: str,
) -> dict[str, Any]:
    async def operation(client: MCPClient) -> dict[str, Any]:
        arguments = {
            "task_id": task["id"],
            "reason": f"Deploy {task['title']} from the dashboard to {environment_target}.",
            "environment": {"target": environment_target},
            "patch": [
                {"op": "test", "path": "/status", "value": task["status"]},
                {"op": "test", "path": "/etag", "value": task["etag"]},
            ],
        }
        return await client.call_tool_with_policy("trigger_deployment", arguments)

    return await _with_client_session(operation, server_command)


async def _read_resource_with_policy(client: MCPClient, uri: str) -> dict[str, Any]:
    while True:
        try:
            return await client.read_resource(uri)
        except JsonRpcError as exc:
            if exc.code != -32002:
                raise
            retry_after = float((exc.data or {}).get("retry_after", 0.5))
            await asyncio.sleep(retry_after)


def _run_async(coro: Awaitable[Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()


def _default_server_command() -> list[str]:
    return [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str((st.session_state.get("_dashboard_root") / "scripts" / "run_server.ps1").resolve()),
    ]


def render_dashboard() -> None:
    st.set_page_config(page_title="EngAuto MCP Dashboard", page_icon=":gear:", layout="wide")
    st.session_state.setdefault("_dashboard_root", Path(__file__).resolve().parents[2])
    st.title("EngAuto MCP Dashboard")
    st.caption("Visual client for task state and deployment triggers.")

    with st.sidebar:
        st.header("Controls")
        selected_status = st.selectbox("Task Status", [status.title() for status in STATUS_ORDER], index=0)
        deploy_target = st.selectbox("Deploy Target", list(DEPLOY_TARGETS), index=0)
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_seconds = st.slider("Refresh Interval", min_value=1, max_value=15, value=3)
        server_command_input = st.text_input(
            "Server Command",
            value=" ".join(_default_server_command()),
            help="Command used when a subprocess transport is available. The dashboard falls back to an in-process server if Windows blocks subprocess pipes.",
        )
        manual_refresh = st.button("Refresh Now", use_container_width=True)

    server_command = (
        shlex.split(server_command_input, posix=False)
        if server_command_input.strip()
        else _default_server_command()
    )
    if auto_refresh:
        st.markdown(
            f"<meta http-equiv='refresh' content='{refresh_seconds}'>",
            unsafe_allow_html=True,
        )
    if manual_refresh:
        st.rerun()

    if "dashboard_message" in st.session_state:
        message = st.session_state.pop("dashboard_message")
        st.success(message)
    if "dashboard_error" in st.session_state:
        error_message = st.session_state.pop("dashboard_error")
        st.error(error_message)

    try:
        snapshot = _run_async(_fetch_task_snapshot(server_command))
    except Exception as exc:
        st.error(f"Failed to load task snapshot: {exc!r}")
        st.code(traceback.format_exc())
        return

    metrics = st.columns(len(STATUS_ORDER))
    for column, status in zip(metrics, STATUS_ORDER, strict=False):
        column.metric(status.title(), len(snapshot.get(status, [])))

    active_status = selected_status.lower()
    tasks = snapshot.get(active_status, [])
    st.subheader(f"{selected_status} Tasks")
    if not tasks:
        st.info(f"No {active_status} tasks are currently available.")
        return

    display_rows = [
        {
            "id": task["id"],
            "title": task["title"],
            "status": task["status"],
            "etag": task["etag"],
            "updated_at": task["updated_at"],
        }
        for task in tasks
    ]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    if active_status != "pending":
        return

    st.subheader("Deploy Pending Tasks")
    for task in tasks:
        row = st.columns([3, 2, 2, 1])
        row[0].markdown(f"**{task['title']}**")
        row[1].code(task["id"])
        row[2].write(f"etag: {task['etag']}")
        if row[3].button("Deploy", key=f"deploy:{task['id']}", use_container_width=True):
            try:
                result = _run_async(_deploy_task(server_command, task, deploy_target))
            except JsonRpcError as exc:
                st.session_state["dashboard_error"] = f"Deployment failed: {exc.message}"
            except Exception as exc:
                st.session_state["dashboard_error"] = f"Deployment failed: {exc!r}"
            else:
                st.session_state["dashboard_message"] = (
                    f"Triggered deployment for {task['title']} with status {result['status']}."
                )
            st.rerun()


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
