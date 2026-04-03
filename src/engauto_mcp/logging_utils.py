from __future__ import annotations

import json
import logging
import re
import sys
from base64 import urlsafe_b64encode
from typing import Any

from .compat import SecretStr

LOGGER_NAME = "engauto_mcp"


class RedactingJsonFormatter(logging.Formatter):
    _deep_scrub_patterns = (
        re.compile(r'("persistent_instance_id"\s*:\s*")([^"]+)(")', re.IGNORECASE),
        re.compile(r'((?:cursor_secret|CURSOR_SECRET)[A-Za-z0-9_]*["\'=: ]+)([A-Za-z0-9_\-+/=]+)'),
    )

    def __init__(self, sensitive_literals: list[str] | None = None) -> None:
        super().__init__()
        self._sensitive_literals: set[str] = set()
        if sensitive_literals:
            self.set_sensitive_literals(sensitive_literals)

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

    def set_sensitive_literals(self, values: list[str]) -> None:
        self._sensitive_literals = {value for value in values if value}

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
        for literal in sorted(self._sensitive_literals, key=len, reverse=True):
            scrubbed = scrubbed.replace(literal, "[REDACTED]")
        return scrubbed


def configure_logging(
    level: int = logging.INFO,
    *,
    sensitive_values: list[str | bytes] | None = None,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    sensitive_literals = _normalize_sensitive_values(sensitive_values or [])
    if logger.handlers:
        _update_handler_redaction(logger, sensitive_literals)
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingJsonFormatter(sensitive_literals))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _update_handler_redaction(logger: logging.Logger, sensitive_literals: list[str]) -> None:
    for handler in logger.handlers:
        formatter = handler.formatter
        if isinstance(formatter, RedactingJsonFormatter):
            formatter.set_sensitive_literals(sensitive_literals)


def _normalize_sensitive_values(values: list[str | bytes]) -> list[str]:
    normalized: set[str] = set()
    for value in values:
        if isinstance(value, bytes):
            normalized.add(repr(value))
            normalized.add(value.hex())
            normalized.add(urlsafe_b64encode(value).decode("ascii").rstrip("="))
            continue
        normalized.add(str(value))
    return [value for value in normalized if value]
