from __future__ import annotations

import asyncio
from collections import deque
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from .errors import JsonRpcError, map_exception_to_jsonrpc


def encode_message(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def read_message_sync(stream: Any) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers["content-length"])
    body = stream.read(length)
    return json.loads(body.decode("utf-8"))


async def read_message_async(reader: Any) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers["content-length"])
    body = await reader.readexactly(length)
    return json.loads(body.decode("utf-8"))


class JsonRpcPeer:
    def __init__(
        self,
        read_message: Callable[[], Awaitable[dict[str, Any] | None]],
        write_message: Callable[[dict[str, Any]], Awaitable[None]],
        request_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]],
        notification_handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._read_message = read_message
        self._write_message = write_message
        self._request_handler = request_handler
        self._notification_handler = notification_handler
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._expired_ids: deque[int | str] = deque(maxlen=1000)
        self._expired_id_lookup: set[int | str] = set()

    async def serve_forever(self) -> None:
        while True:
            message = await self._read_message()
            if message is None:
                return
            if "id" in message and ("result" in message or "error" in message):
                response_id = message["id"]
                if response_id in self._expired_id_lookup:
                    self._expired_id_lookup.discard(response_id)
                    continue
                future = self._pending.pop(response_id, None)
                if future and not future.done():
                    future.set_result(message)
                continue
            if "id" in message:
                asyncio.create_task(self._handle_request(message))
            else:
                await self._notification_handler(message)

    async def send_request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float = 30.0,
        request_id: int | str | None = None,
    ) -> dict[str, Any]:
        if request_id is None:
            request_id = self._next_id
            self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            if len(self._expired_ids) == self._expired_ids.maxlen:
                expired = self._expired_ids.popleft()
                self._expired_id_lookup.discard(expired)
            self._expired_ids.append(request_id)
            self._expired_id_lookup.add(request_id)
            raise JsonRpcError(
                -32022,
                f"Request timed out waiting for '{method}'.",
                {"method": method, "timeout": timeout},
            ) from exc
        if "error" in response:
            error = response["error"]
            raise JsonRpcError(error["code"], error["message"], error.get("data"))
        return response["result"]

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    async def _handle_request(self, message: dict[str, Any]) -> None:
        try:
            result = await self._request_handler(message)
            await self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": result if result is not None else {},
                }
            )
        except JsonRpcError as error:
            await self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": error.to_payload(),
                }
            )
        except Exception as exc:
            error = map_exception_to_jsonrpc(exc)
            await self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": error.to_payload(),
                }
            )


class StdIOTransport:
    def __init__(self) -> None:
        self._stdout_lock = asyncio.Lock()
        self._stdout_buffer = sys.stdout.buffer

    async def read_message(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(read_message_sync, sys.stdin.buffer)

    async def write_message(self, payload: dict[str, Any]) -> None:
        frame = encode_message(payload)
        await self.write_message_bytes(frame)

    async def write_message_bytes(self, frame: bytes) -> None:
        async with self._stdout_lock:
            await asyncio.to_thread(self._write_sync, self._stdout_buffer, frame)

    @staticmethod
    def _write_sync(buffer: Any, frame: bytes) -> None:
        buffer.write(frame)
        buffer.flush()
