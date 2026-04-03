from __future__ import annotations

from pathlib import Path

PACKAGE_NAME = "engauto_mcp"
DEFAULT_DB_PATH = Path("data") / "engauto_mcp.sqlite3"
LLM_INSTRUCTIONS = (
    "Include 'test' operations for 'status' and 'etag' in all JSON-Patches. "
    "If you receive error -32003, re-read the resource URI. "
    "If you receive -32002, respect the 'retry_after' duration."
)
