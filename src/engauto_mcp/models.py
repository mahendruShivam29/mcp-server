from __future__ import annotations

from typing import Any, Literal

from .compat import BaseModel, ConfigDict, Field


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    status: Literal["pending", "running", "completed", "failed"]
    etag: int
    payload_json: str
    created_at: int
    updated_at: int


class ResourcePage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    items: list[dict[str, Any]]
    next_cursor: str | None = None
    migration_hint: bool = False


class SamplingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    task_id: str
    current_status: str
    environment: dict[str, Any]
    diff: dict[str, Any]
    original_error: dict[str, Any] | None = None
    prompt_instructions: str | None = None


class SamplingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    approved: bool = False
    patch: list[dict[str, Any]] | None = None
    message: str | None = None


class TriggerDeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    task_id: str
    patch: list[dict[str, Any]] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    reason: str


class EngineHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sqlite_version: str
    persistent_instance_id: str
    wal_file_size_bytes: int
    active_subscriptions_count: int
    heartbeat_timestamp: int
    dependency_availability: dict[str, bool]
    warning: str | None = None
