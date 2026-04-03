from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

from .compat import SecretStr

LOGGER_NAME = "engauto_mcp"


class RedactingJsonFormatter(logging.Formatter):
    _deep_scrub_patterns = (
        re.compile(r'("persistent_instance_id"\s*:\s*")([^"]+)(")', re.IGNORECASE),
        re.compile(r'((?:cursor_secret|CURSOR_SECRET)[A-Za-z0-9_]*["\'=: ]+)([A-Za-z0-9_\-+/=]+)'),
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact(record.msg),
        }
        if record.args:
            payload["args"] = self._redact(record.args)
        formatted = json.dumps(payload, default=str)
        return self._deep_scrub(formatted)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, SecretStr):
            return "**********"
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._redact(item) for item in value]
        if hasattr(value, "get_secret_value") and value.__class__.__name__ == "SecretStr":
            return "**********"
        return value

    def _deep_scrub(self, formatted: str) -> str:
        scrubbed = formatted
        scrubbed = self._deep_scrub_patterns[0].sub(r'\1**********\3', scrubbed)
        scrubbed = self._deep_scrub_patterns[1].sub(r'\1**********', scrubbed)
        return scrubbed


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingJsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
