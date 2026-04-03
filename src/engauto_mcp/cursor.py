from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from dataclasses import dataclass

from .errors import CursorValidationError
from .persistence import CursorSecrets

HMAC_LENGTH = 8


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(encoded + padding)
    except (binascii.Error, ValueError) as exc:
        raise CursorValidationError("Cursor payload is malformed.") from exc


@dataclass(frozen=True, slots=True)
class DecodedCursor:
    updated_at: int
    task_id: str
    migration_hint: bool = False


class HmacCursorCodec:
    def __init__(self, secrets: CursorSecrets) -> None:
        self._secrets = secrets

    def encode(self, updated_at: int, task_id: str, timestamp: int | None = None) -> str:
        issued_at = int(time.time()) if timestamp is None else timestamp
        body = (
            f"{updated_at}|{task_id}|{issued_at}|{self._secrets.persistent_instance_id}"
        ).encode("utf-8")
        mac = self._sign(body, self._secrets.current_secret)
        return _b64url_encode(body + mac)

    def decode(self, cursor: str) -> DecodedCursor:
        raw = _b64url_decode(cursor)
        if len(raw) <= HMAC_LENGTH:
            raise CursorValidationError("Cursor payload is too short.")
        body, signature = raw[:-HMAC_LENGTH], raw[-HMAC_LENGTH:]
        used_previous_secret = False
        if hmac.compare_digest(signature, self._sign(body, self._secrets.current_secret)):
            pass
        elif self._secrets.previous_secret and hmac.compare_digest(
            signature, self._sign(body, self._secrets.previous_secret)
        ):
            used_previous_secret = True
        else:
            raise CursorValidationError("Cursor signature is invalid.")

        try:
            updated_at_raw, task_id, timestamp_raw, instance_id = body.decode("utf-8").split("|", 3)
        except ValueError as exc:
            raise CursorValidationError("Cursor payload is malformed.") from exc

        if instance_id != self._secrets.persistent_instance_id:
            raise CursorValidationError(
                "Cursor was issued by a different server instance.",
                {"reason": "persistent_instance_id_mismatch"},
            )

        return DecodedCursor(
            updated_at=int(updated_at_raw),
            task_id=task_id,
            migration_hint=used_previous_secret,
        )

    @staticmethod
    def _sign(payload: bytes, secret: bytes) -> bytes:
        digest = hmac.new(secret, payload, hashlib.sha256).digest()
        return digest[:HMAC_LENGTH]
