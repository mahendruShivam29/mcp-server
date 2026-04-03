from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from enum import Enum

from .db import DatabaseManager


class RateLimitTier(str, Enum):
    RESOURCE = "resource"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class TokenBucketConfig:
    capacity: int
    refill_window_seconds: float
    base_retry_after_seconds: float
    max_retry_after_seconds: float = 60.0

    @property
    def refill_rate(self) -> float:
        return self.capacity / self.refill_window_seconds


@dataclass(slots=True)
class TokenBucketState:
    tokens: float
    last_refill_epoch: float
    denial_count: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "tokens": self.tokens,
                "last_refill_epoch": self.last_refill_epoch,
                "denial_count": self.denial_count,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str, *, default_tokens: float, now: float) -> "TokenBucketState":
        data = json.loads(raw)
        return cls(
            tokens=float(data.get("tokens", default_tokens)),
            last_refill_epoch=float(data.get("last_refill_epoch", now)),
            denial_count=int(data.get("denial_count", 0)),
        )


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    tokens_remaining: float
    retry_after_seconds: float | None = None
    denial_count: int = 0


@dataclass(frozen=True, slots=True)
class RateLimitExceeded(Exception):
    tier: RateLimitTier
    retry_after_seconds: float
    denial_count: int

    def as_payload(self) -> dict[str, float | int | str]:
        return {
            "code": -32002,
            "tier": self.tier.value,
            "retry_after": self.retry_after_seconds,
            "denial_count": self.denial_count,
        }


class TieredRateLimiter:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self.resource_config = TokenBucketConfig(
            capacity=20,
            refill_window_seconds=10.0,
            base_retry_after_seconds=0.5,
        )
        self.tool_config = TokenBucketConfig(
            capacity=5,
            refill_window_seconds=10.0,
            base_retry_after_seconds=1.0,
        )
        self._lock = asyncio.Lock()

    async def consume(
        self,
        client_id: str,
        tier: RateLimitTier,
        *,
        cost: int = 1,
        now: float | None = None,
    ) -> RateLimitDecision:
        async with self._lock:
            decision = await self._consume_locked(client_id, tier, cost=cost, now=now)
        if not decision.allowed and decision.retry_after_seconds is not None:
            raise RateLimitExceeded(
                tier=tier,
                retry_after_seconds=decision.retry_after_seconds,
                denial_count=decision.denial_count,
            )
        return decision

    async def inspect(
        self,
        client_id: str,
        tier: RateLimitTier,
        *,
        now: float | None = None,
    ) -> RateLimitDecision:
        async with self._lock:
            current_time = time.time() if now is None else now
            state = await self._read_state(client_id, tier, current_time)
            config = self._config_for(tier)
            self._refill(state, config, current_time)
            return RateLimitDecision(
                allowed=True,
                tokens_remaining=state.tokens,
                denial_count=state.denial_count,
            )

    async def _consume_locked(
        self,
        client_id: str,
        tier: RateLimitTier,
        *,
        cost: int,
        now: float | None,
    ) -> RateLimitDecision:
        current_time = time.time() if now is None else now
        config = self._config_for(tier)
        state = await self._read_state(client_id, tier, current_time)
        self._refill(state, config, current_time)

        if state.tokens >= cost:
            state.tokens -= cost
            state.denial_count = 0
            await self._write_state(client_id, tier, state)
            return RateLimitDecision(
                allowed=True,
                tokens_remaining=state.tokens,
                denial_count=0,
            )

        state.denial_count += 1
        shortfall = cost - state.tokens
        refill_wait = shortfall / config.refill_rate
        retry_after = max(
            refill_wait,
            min(
                config.base_retry_after_seconds * (2 ** (state.denial_count - 1)),
                config.max_retry_after_seconds,
            ),
        )
        await self._write_state(client_id, tier, state)
        return RateLimitDecision(
            allowed=False,
            tokens_remaining=max(state.tokens, 0.0),
            retry_after_seconds=math.ceil(retry_after * 1000) / 1000,
            denial_count=state.denial_count,
        )

    async def _read_state(
        self,
        client_id: str,
        tier: RateLimitTier,
        now: float,
    ) -> TokenBucketState:
        key = self._state_key(client_id, tier)
        record = await self._db.get_system_state_record(key)
        config = self._config_for(tier)
        if record is None or record["value_text"] is None:
            return TokenBucketState(tokens=float(config.capacity), last_refill_epoch=now)
        return TokenBucketState.from_json(record["value_text"], default_tokens=float(config.capacity), now=now)

    async def _write_state(
        self,
        client_id: str,
        tier: RateLimitTier,
        state: TokenBucketState,
    ) -> None:
        await self._db.set_system_state(
            self._state_key(client_id, tier),
            value_text=state.to_json(),
        )

    @staticmethod
    def _refill(bucket: TokenBucketState, config: TokenBucketConfig, now: float) -> None:
        elapsed = max(0.0, now - bucket.last_refill_epoch)
        bucket.last_refill_epoch = now
        bucket.tokens = min(config.capacity, bucket.tokens + elapsed * config.refill_rate)

    @staticmethod
    def _state_key(client_id: str, tier: RateLimitTier) -> str:
        return f"rate_limit:{tier.value}:{client_id}"

    def _config_for(self, tier: RateLimitTier) -> TokenBucketConfig:
        if tier is RateLimitTier.RESOURCE:
            return self.resource_config
        return self.tool_config
