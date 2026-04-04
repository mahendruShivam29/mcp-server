from __future__ import annotations

import asyncio
import contextlib
import traceback
from collections.abc import Awaitable
from typing import Any

import streamlit as st

from engauto_mcp.client import MCPClient, _get_mcp_client
from engauto_mcp.errors import JsonRpcError

STATUS_ORDER = ("pending", "running", "completed", "failed")
DEPLOY_TARGETS = ("dev", "staging", "prod")


async def _fetch_task_snapshot(server_url: str) -> dict[str, list[dict[str, Any]]]:
    async with _get_mcp_client(server_url) as client:
        snapshot: dict[str, list[dict[str, Any]]] = {}
        for status in STATUS_ORDER:
            snapshot[status] = await _read_all_tasks(client, status)
        return snapshot


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
    server_url: str,
    task: dict[str, Any],
    environment_target: str,
) -> dict[str, Any]:
    async with _get_mcp_client(server_url) as client:
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


def _default_server_url() -> str:
    return "http://localhost:8000/sse"


def render_dashboard() -> None:
    st.set_page_config(page_title="EngAuto MCP Dashboard", page_icon=":gear:", layout="wide")
    st.title("EngAuto MCP Dashboard")
    st.caption("Visual client for task state and deployment triggers.")

    with st.sidebar:
        st.header("Controls")
        selected_status = st.selectbox("Task Status", [status.title() for status in STATUS_ORDER], index=0)
        deploy_target = st.selectbox("Deploy Target", list(DEPLOY_TARGETS), index=0)
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_seconds = st.slider("Refresh Interval", min_value=1, max_value=15, value=3)
        server_url = st.text_input(
            "Server URL",
            value=st.session_state.get("dashboard_server_url", _default_server_url()),
            help="SSE endpoint for the long-lived EngAuto MCP server.",
        )
        st.session_state["dashboard_server_url"] = server_url
        manual_refresh = st.button("Refresh Now", use_container_width=True)

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
        snapshot = _run_async(_fetch_task_snapshot(server_url))
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
                result = _run_async(_deploy_task(server_url, task, deploy_target))
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
