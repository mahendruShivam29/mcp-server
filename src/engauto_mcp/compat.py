from __future__ import annotations

from typing import Any

try:
    import aiosqlite as _aiosqlite
except ImportError:  # pragma: no cover - environment fallback
    _aiosqlite = None

try:
    import jsonpatch as _jsonpatch
except ImportError:  # pragma: no cover - environment fallback
    _jsonpatch = None

try:
    import mcp as _mcp
except ImportError:  # pragma: no cover - environment fallback
    _mcp = None

try:
    from pydantic import BaseModel, ConfigDict, Field, SecretStr
except ImportError:  # pragma: no cover - environment fallback
    class SecretStr:
        def __init__(self, value: str) -> None:
            self._value = value

        def get_secret_value(self) -> str:
            return self._value

        def __repr__(self) -> str:
            return "SecretStr('**********')"

        def __str__(self) -> str:
            return "**********"

    def Field(default: Any = None, **kwargs: Any) -> Any:
        if "default_factory" in kwargs:
            return kwargs["default_factory"]()
        return default

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    class BaseModel:
        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> "BaseModel":
            return cls(**data)


aiosqlite = _aiosqlite
jsonpatch = _jsonpatch
mcp = _mcp
DEPENDENCY_AVAILABILITY = {
    "aiosqlite": aiosqlite is not None,
    "jsonpatch": jsonpatch is not None,
    "mcp": mcp is not None,
    "pydantic": BaseModel.__module__ != __name__,
}


def require_dependency(name: str, module: Any) -> Any:
    if module is None:
        raise RuntimeError(
            f"Missing optional dependency '{name}'. Install project dependencies first."
        )
    return module
