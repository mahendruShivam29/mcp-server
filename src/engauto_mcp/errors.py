from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class JsonRpcError(Exception):
    code: int
    message: str
    data: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            payload["data"] = self.data
        return payload


class CursorValidationError(JsonRpcError):
    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(-32001, message, data)


class TOCTOUConflictError(JsonRpcError):
    def __init__(self, task_id: str, expected: dict[str, Any], actual: dict[str, Any]) -> None:
        super().__init__(
            -32003,
            "Task state changed during deployment trigger.",
            {
                "task_id": task_id,
                "expected": expected,
                "actual": actual,
            },
        )
