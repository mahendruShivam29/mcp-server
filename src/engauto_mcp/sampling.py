from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable

from .errors import JsonRpcError
from .models import SamplingRequest, SamplingResponse

RemediationSampler = Callable[[SamplingRequest], Awaitable[SamplingResponse]]
_remediation_attempts: contextvars.ContextVar[int] = contextvars.ContextVar(
    "remediation_attempts",
    default=0,
)


class SamplingGuard:
    def __init__(self, sampler: RemediationSampler | None, *, max_remediation_attempts: int = 2) -> None:
        self._sampler = sampler
        self._max_remediation_attempts = max_remediation_attempts

    async def request_preflight(
        self,
        request: SamplingRequest,
        original_error: JsonRpcError,
    ) -> SamplingResponse:
        if self._sampler is None:
            raise original_error
        try:
            return await self._sampler(request)
        except JsonRpcError:
            raise original_error

    async def request_remediation(
        self,
        request: SamplingRequest,
        original_error: JsonRpcError,
    ) -> SamplingResponse:
        attempts = _remediation_attempts.get()
        if attempts >= self._max_remediation_attempts:
            raise original_error
        if self._sampler is None:
            raise original_error
        token = _remediation_attempts.set(attempts + 1)
        try:
            try:
                return await self._sampler(request)
            except JsonRpcError:
                raise original_error
        finally:
            _remediation_attempts.reset(token)
