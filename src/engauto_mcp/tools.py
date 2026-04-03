from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .compat import jsonpatch, require_dependency
from .config import LLM_INSTRUCTIONS
from .db import DatabaseManager
from .errors import JsonRpcError, TOCTOUConflictError
from .models import EngineHealth, SamplingRequest, TriggerDeploymentRequest
from .rate_limiter import RateLimitTier, TieredRateLimiter
from .sampling import SamplingGuard
from .subscriptions import SubscriptionManager


class ToolService:
    def __init__(
        self,
        db: DatabaseManager,
        rate_limiter: TieredRateLimiter,
        subscriptions: SubscriptionManager,
        sampling_guard: SamplingGuard,
        enqueue_deployment: Any,
        dependency_availability: dict[str, bool],
        log_emitter: Any,
    ) -> None:
        self._db = db
        self._rate_limiter = rate_limiter
        self._subscriptions = subscriptions
        self._sampling_guard = sampling_guard
        self._enqueue_deployment = enqueue_deployment
        self._dependency_availability = dependency_availability
        self._log_emitter = log_emitter

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "trigger_deployment",
                "description": "Trigger a mock deployment for a task with TOCTOU-safe JSON Patch validation.",
            },
            {
                "name": "get_engine_health",
                "description": "Run a deep heartbeat health check.",
            },
        ]

    async def trigger_deployment(self, request: TriggerDeploymentRequest) -> dict[str, Any]:
        await self._rate_limiter.consume(request.client_id, RateLimitTier.TOOL)
        patch_ops = list(request.patch)
        current_task = await self._db.fetch_task(request.task_id)
        if current_task is None:
            raise JsonRpcError(-32010, f"Task '{request.task_id}' was not found.")
        current_document = {
            "id": current_task.id,
            "title": current_task.title,
            "status": current_task.status,
            "etag": current_task.etag,
            **json.loads(current_task.payload_json),
        }
        current_environment = current_document.get("deployment_environment", {})

        approval = await self._sampling_guard.request_preflight(
            SamplingRequest(
                reason=request.reason,
                task_id=request.task_id,
                current_status=current_task.status,
                environment=request.environment,
                diff={
                    "before": current_environment,
                    "after": request.environment,
                },
                prompt_instructions=LLM_INSTRUCTIONS,
            ),
            JsonRpcError(-32004, "Sampling approval is required before deployment."),
        )
        if not approval.approved:
            raise JsonRpcError(-32004, "Deployment was not approved by the sampling client.")
        if approval.patch:
            patch_ops = approval.patch

        for _ in range(2):
            try:
                result = await self._db.transaction(
                    lambda db, on_commit: self._apply_deployment_patch(
                        db,
                        on_commit,
                        request.task_id,
                        patch_ops,
                        request.environment,
                    )
                )
                await self._enqueue_deployment(result["task_id"], request.environment)
                return result
            except TOCTOUConflictError as error:
                remediation = await self._sampling_guard.request_remediation(
                    SamplingRequest(
                        reason="TOCTOU remediation required",
                        task_id=request.task_id,
                        current_status=error.data["actual"]["status"],
                        environment=request.environment,
                        diff={"expected": error.data["expected"], "actual": error.data["actual"]},
                        original_error=error.to_payload(),
                        prompt_instructions=LLM_INSTRUCTIONS,
                    ),
                    error,
                )
                if not remediation.approved:
                    raise error
                if remediation.patch:
                    patch_ops = remediation.patch
        raise JsonRpcError(-32005, "Deployment remediation attempts exhausted.")

    async def get_engine_health(self) -> EngineHealth:
        heartbeat = await self._db.transaction(lambda db, on_commit: db.write_read_heartbeat())
        wal_path = self._db.database_path.with_name(self._db.database_path.name + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        page_count = await self._db.page_count()
        page_size = await self._db.page_size()
        wal_autocheckpoint_pages = await self._db.wal_autocheckpoint_pages()
        main_db_size_bytes = page_count * page_size
        wal_to_main_size_ratio = (
            wal_size / main_db_size_bytes if main_db_size_bytes > 0 else 0.0
        )
        persistent_instance_id = await self._db.get_metadata_text("persistent_instance_id")
        warnings: list[str] = []
        if wal_size > 100 * 1024 * 1024:
            warnings.append("WAL file exceeds 100MB.")
        if wal_to_main_size_ratio > 2.0:
            warnings.append(
                "WAL is more than 2x the main database size; trigger maintenance/checkpoint."
            )
        return EngineHealth(
            sqlite_version=await self._db.sqlite_version(),
            persistent_instance_id=persistent_instance_id or "",
            integrity_check_result=await self._db.integrity_check(),
            foreign_key_check_results=await self._db.foreign_key_check(),
            wal_autocheckpoint_pages=wal_autocheckpoint_pages,
            page_count=page_count,
            main_db_size_bytes=main_db_size_bytes,
            wal_file_size_bytes=wal_size,
            wal_to_main_size_ratio=wal_to_main_size_ratio,
            active_subscriptions_count=self._subscriptions.active_subscriptions_count(),
            heartbeat_timestamp=heartbeat,
            dependency_availability=self._dependency_availability,
            warning=" ".join(warnings) if warnings else None,
        )

    async def _apply_deployment_patch(
        self,
        db: DatabaseManager,
        on_commit: list[Callable[[], Awaitable[None]]],
        task_id: str,
        patch_ops: list[dict[str, Any]],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        task = await db.fetch_task(task_id)
        if task is None:
            raise JsonRpcError(-32010, f"Task '{task_id}' was not found.")

        document = {
            "id": task.id,
            "title": task.title,
            "status": task.status,
            "etag": task.etag,
            **json.loads(task.payload_json),
        }
        await self._claim_or_validate_lock(db)
        self._validate_test_operations(task_id, document, patch_ops)

        patch_module = require_dependency("jsonpatch", jsonpatch)
        patch = patch_module.JsonPatch([op for op in patch_ops if op.get("op") != "test"])
        updated_document = patch.apply(document, in_place=False)
        updated_document["status"] = "running"
        updated_document["deployment_environment"] = environment
        updated_task = await db.update_task_document(task, updated_document)
        await db.set_system_state("DEPLOY_LOCK", value_text="RUNNING")
        self._subscriptions.emit_resource_updated(f"tasks://{task.status}", on_commit=on_commit)
        self._subscriptions.emit_resource_updated(f"tasks://{updated_task.status}", on_commit=on_commit)
        return {
            "task_id": updated_task.id,
            "status": updated_task.status,
            "etag": updated_task.etag,
        }

    @staticmethod
    def _validate_test_operations(
        task_id: str,
        current_document: dict[str, Any],
        patch_ops: list[dict[str, Any]],
    ) -> None:
        test_values: dict[str, Any] = {}
        required_paths = {"/status", "/etag"}
        payload_field_paths: dict[str, str] = {}

        for op in patch_ops:
            operation = op.get("op")
            path = op.get("path")
            if operation == "test":
                if isinstance(path, str):
                    test_values[path] = op.get("value")
                continue
            if not isinstance(path, str):
                raise JsonRpcError(-32012, "Patch operation is missing a valid path.", {"task_id": task_id})

            top_level_field = ToolService._top_level_field_for_path(path)
            if top_level_field not in {"id", "title", "status", "etag"}:
                required_paths.add(f"/{top_level_field}")
                payload_field_paths[f"/{top_level_field}"] = top_level_field

            if operation == "move":
                from_path = op.get("from")
                if not isinstance(from_path, str):
                    raise JsonRpcError(
                        -32012,
                        "Move operation is missing a valid from path.",
                        {"task_id": task_id},
                    )
                from_field = ToolService._top_level_field_for_path(from_path)
                if from_field not in {"id", "title", "status", "etag"}:
                    required_paths.add(f"/{from_field}")
                    payload_field_paths[f"/{from_field}"] = from_field

        missing_paths = sorted(path for path in required_paths if path not in test_values)
        if missing_paths:
            raise JsonRpcError(
                -32012,
                "Patch must include test operations for /status, /etag, and each modified top-level payload field.",
                {"task_id": task_id, "missing_tests": missing_paths},
            )

        expected = {"status": test_values["/status"], "etag": test_values["/etag"]}
        actual = {"status": current_document.get("status"), "etag": current_document.get("etag")}
        for path, field_name in payload_field_paths.items():
            expected[field_name] = test_values[path]
            actual[field_name] = current_document.get(field_name)

        if any(expected[key] != actual[key] for key in expected):
            raise TOCTOUConflictError(task_id, expected, actual, f"tasks://{actual['status']}")

    @staticmethod
    def _top_level_field_for_path(path: str) -> str:
        if not path.startswith("/") or path == "/":
            raise JsonRpcError(-32012, "Patch path must target a top-level document field.", {"path": path})
        segment = path[1:].split("/", 1)[0]
        return segment.replace("~1", "/").replace("~0", "~")

    async def _claim_or_validate_lock(self, db: DatabaseManager) -> None:
        lock_record = await db.get_system_state_record("DEPLOY_LOCK")
        if lock_record is None:
            return
        if lock_record["value_text"] != "RUNNING":
            return
        age_seconds = int(time.time()) - int(lock_record["updated_at"])
        if age_seconds > 30 * 60:
            await self._log_emitter("WARN", "Stale Lock Reclaimed")
            return
        raise JsonRpcError(
            -32013,
            "Deployment lock is already RUNNING.",
            {"updated_at": lock_record["updated_at"]},
        )
