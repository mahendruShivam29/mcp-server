from __future__ import annotations

import asyncio
import contextlib
import json
import traceback
from collections.abc import Awaitable
from typing import Any

import streamlit as st
from mcp import types

from engauto_mcp.client import (
    MCPClient,
    _build_sampling_response,
    _execute_planned_action,
    _get_mcp_client,
    _parse_sampling_request,
    _plan_tool_call_with_openai,
)
from engauto_mcp.errors import JsonRpcError

STATUS_ORDER = ("pending", "running", "completed", "failed")
DEPLOY_TARGETS = ("dev", "staging", "prod")


async def _dashboard_sampling_callback(
    context: Any,
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    request = _parse_sampling_request(params)
    response = _build_sampling_response(request)
    return types.CreateMessageResult(
        role="assistant",
        model="engauto-mcp-dashboard",
        content=types.TextContent(type="text", text=json.dumps(response.model_dump())),
        stopReason="endTurn",
    )


async def _fetch_task_snapshot(server_url: str) -> dict[str, list[dict[str, Any]]]:
    async with _get_mcp_client(server_url, sampling_callback=_dashboard_sampling_callback) as client:
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
    pending_error: Exception | None = None
    pending_result: dict[str, Any] | None = None
    async with _get_mcp_client(server_url, sampling_callback=_dashboard_sampling_callback) as client:
        try:
            arguments = {
                "task_id": task["id"],
                "reason": f"Deploy {task['title']} from the dashboard to {environment_target}.",
                "environment": {"target": environment_target},
                "patch": [
                    {"op": "test", "path": "/status", "value": task["status"]},
                    {"op": "test", "path": "/etag", "value": task["etag"]},
                ],
            }
            pending_result = await client.call_tool_with_policy("trigger_deployment", arguments)
        except Exception as exc:
            pending_error = exc
    if pending_error is not None:
        raise pending_error
    assert pending_result is not None
    return pending_result


async def _create_task(
    server_url: str,
    task_id: str,
    title: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    pending_error: Exception | None = None
    pending_result: dict[str, Any] | None = None
    async with _get_mcp_client(server_url, sampling_callback=_dashboard_sampling_callback) as client:
        try:
            pending_result = await client.call_tool_with_policy(
                "create_task",
                {
                    "task_id": task_id,
                    "title": title,
                    "payload": payload,
                },
            )
        except Exception as exc:
            pending_error = exc
    if pending_error is not None:
        raise pending_error
    assert pending_result is not None
    return pending_result


async def _update_task(
    server_url: str,
    task_id: str,
    title: str | None,
    status: str | None,
    payload_updates: dict[str, Any],
    expected_status: str | None,
    expected_etag: int | None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "task_id": task_id,
        "payload_updates": payload_updates,
    }
    if title:
        arguments["title"] = title
    if status:
        arguments["status"] = status
    if expected_status:
        arguments["expected_status"] = expected_status
    if expected_etag is not None:
        arguments["expected_etag"] = expected_etag

    pending_error: Exception | None = None
    pending_result: dict[str, Any] | None = None
    async with _get_mcp_client(server_url, sampling_callback=_dashboard_sampling_callback) as client:
        try:
            pending_result = await client.call_tool_with_policy("update_task", arguments)
        except Exception as exc:
            pending_error = exc
    if pending_error is not None:
        raise pending_error
    assert pending_result is not None
    return pending_result


async def _run_nlp_action(server_url: str, query: str) -> dict[str, Any]:
    pending_error: Exception | None = None
    pending_result: dict[str, Any] | None = None
    async with _get_mcp_client(server_url, sampling_callback=_dashboard_sampling_callback) as client:
        try:
            tool_call = await _plan_tool_call_with_openai(client, query)
            result = await _execute_planned_action(client, tool_call, query)
            pending_result = {
                "query": query,
                "tool_call": tool_call,
                "result": result,
            }
        except Exception as exc:
            pending_error = exc
    if pending_error is not None:
        raise pending_error
    assert pending_result is not None
    return pending_result


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
    return "http://localhost:8000/sse/"


def _parse_json_object(raw_text: str, *, field_name: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return payload


def _flatten_tasks(snapshot: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for status in STATUS_ORDER:
        tasks.extend(snapshot.get(status, []))
    return tasks


def _describe_exception(exc: BaseException) -> str:
    if isinstance(exc, ExceptionGroup) and exc.exceptions:
        return _describe_exception(exc.exceptions[0])
    message = str(exc)
    return message or repr(exc)


def render_dashboard() -> None:
    st.set_page_config(page_title="EngAuto MCP Dashboard", page_icon=":gear:", layout="wide")
    st.title("EngAuto MCP Dashboard")
    st.caption("Visual client for task state, task edits, NLP actions, and deployment triggers.")

    with st.sidebar:
        st.header("Controls")
        selected_status = st.selectbox("Task Status", [status.title() for status in STATUS_ORDER], index=0)
        deploy_target = st.selectbox("Deploy Target", list(DEPLOY_TARGETS), index=0)
        auto_refresh = st.checkbox("Auto Refresh", value=False, disabled=True)
        refresh_seconds = st.slider("Refresh Interval", min_value=1, max_value=15, value=3, disabled=True)
        server_url = st.text_input(
            "Server URL",
            value=st.session_state.get("dashboard_server_url", _default_server_url()),
            help="SSE endpoint for the long-lived EngAuto MCP server.",
        )
        st.session_state["dashboard_server_url"] = server_url
        manual_refresh = st.button("Refresh Now", use_container_width=True)
        st.caption("Auto refresh is disabled to avoid interrupting forms and text input. Use Refresh Now after changes.")
    if manual_refresh:
        st.rerun()

    if "dashboard_message" in st.session_state:
        st.success(st.session_state.pop("dashboard_message"))
    if "dashboard_error" in st.session_state:
        st.error(st.session_state.pop("dashboard_error"))

    try:
        snapshot = _run_async(_fetch_task_snapshot(server_url))
    except Exception as exc:
        st.error(f"Failed to load task snapshot: {exc!r}")
        st.code(traceback.format_exc())
        return

    all_tasks = _flatten_tasks(snapshot)
    task_lookup = {task["id"]: task for task in all_tasks}

    metrics = st.columns(len(STATUS_ORDER))
    for column, status in zip(metrics, STATUS_ORDER, strict=False):
        column.metric(status.title(), len(snapshot.get(status, [])))

    actions_left, actions_right = st.columns(2)

    with actions_left:
        with st.expander("Add Task", expanded=True):
            with st.form("create-task-form", clear_on_submit=True):
                create_task_id = st.text_input("Task ID", placeholder="demo-task-1")
                create_title = st.text_input("Title", placeholder="Demo task")
                create_payload = st.text_area("Payload JSON", value="{}", height=120)
                create_submitted = st.form_submit_button("Create Task", use_container_width=True)
            if create_submitted:
                try:
                    payload = _parse_json_object(create_payload, field_name="Payload JSON")
                    result = _run_async(_create_task(server_url, create_task_id.strip(), create_title.strip(), payload))
                except Exception as exc:
                    st.session_state["dashboard_error"] = f"Create task failed: {exc}"
                else:
                    st.session_state["dashboard_message"] = (
                        f"Created task {result['task_id']} with status {result['status']}."
                    )
                st.rerun()

    with actions_right:
        with st.expander("OpenAI NLP Action", expanded=True):
            with st.form("nlp-action-form", clear_on_submit=False):
                query = st.text_area(
                    "Instruction",
                    placeholder="Create a task with id demo-ai-1 and title Demo AI Task",
                    height=120,
                )
                nlp_submitted = st.form_submit_button("Run NLP Action", use_container_width=True)
            if nlp_submitted:
                try:
                    result = _run_async(_run_nlp_action(server_url, query.strip()))
                except Exception as exc:
                    st.session_state["dashboard_error"] = f"NLP action failed: {_describe_exception(exc)}"
                else:
                    tool_name = result.get("tool_call", {}).get("name") or result.get("mode", "action")
                    st.session_state["dashboard_message"] = f"NLP executed {tool_name} successfully."
                    st.session_state["dashboard_last_nlp"] = result
                st.rerun()

    with st.expander("Update Task", expanded=False):
        if not all_tasks:
            st.info("No tasks available to update.")
        else:
            selected_task_id = st.selectbox("Task", list(task_lookup), format_func=lambda task_id: f"{task_id} ({task_lookup[task_id]['status']})")
            selected_task = task_lookup[selected_task_id]
            payload_placeholder = json.dumps(selected_task.get("payload", {}), indent=2)
            with st.form("update-task-form", clear_on_submit=False):
                update_title = st.text_input("New Title", value=selected_task["title"])
                update_status = st.selectbox(
                    "New Status",
                    ["", *STATUS_ORDER],
                    index=0,
                    help="Leave blank to keep the current status.",
                )
                update_payload = st.text_area(
                    "Payload Updates JSON",
                    value="{}",
                    height=120,
                    help=f"Current payload:\n{payload_placeholder}",
                )
                expected_status = st.selectbox(
                    "Expected Status",
                    ["", *STATUS_ORDER],
                    index=(STATUS_ORDER.index(selected_task["status"]) + 1),
                    help="Used for optimistic concurrency checks. Leave as current status unless you intentionally want a stale-write failure.",
                )
                expected_etag = st.number_input(
                    "Expected ETag",
                    min_value=0,
                    value=int(selected_task["etag"]),
                    step=1,
                )
                update_submitted = st.form_submit_button("Update Task", use_container_width=True)
            if update_submitted:
                try:
                    payload_updates = _parse_json_object(update_payload, field_name="Payload Updates JSON")
                    result = _run_async(
                        _update_task(
                            server_url,
                            selected_task_id,
                            update_title.strip(),
                            update_status or None,
                            payload_updates,
                            expected_status or None,
                            int(expected_etag),
                        )
                    )
                except Exception as exc:
                    st.session_state["dashboard_error"] = f"Update task failed: {exc}"
                else:
                    st.session_state["dashboard_message"] = (
                        f"Updated task {result['task_id']} to status {result['status']}."
                    )
                st.rerun()

    last_nlp = st.session_state.get("dashboard_last_nlp")
    if last_nlp:
        with st.expander("Last NLP Result", expanded=False):
            st.json(last_nlp)

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
    st.caption("Dashboard deploys auto-approve the MCP sampling gate because the button click is treated as explicit operator approval.")
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
