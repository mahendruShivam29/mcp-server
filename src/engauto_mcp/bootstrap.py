from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_DB_PATH
from .persistence import CursorSecrets, initialize_persistence


def bootstrap(database_path: str | Path = DEFAULT_DB_PATH) -> CursorSecrets:
    return initialize_persistence(database_path)
