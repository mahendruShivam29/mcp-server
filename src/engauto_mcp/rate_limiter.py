from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


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
    last_refill_monotonic: float
    denial_count: int = 0


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


@dataclass(slots=True)
class TieredRateLimiter:
    resource_config: TokenBucketConfig = field(
        default_factory=lambda: TokenBucketConfig(
            capacity=20,
            refill_window_seconds=10.0,
            base_retry_after_seconds=0.5,
        )
    )
    tool_config: TokenBucketConfig = field(
        default_factory=lambda: TokenBucketConfig(
            capacity=5,
            refill_window_seconds=10.0,
            base_retry_after_seconds=1.0,
        )
    )
    _lock: Lock = field(init=False, repr=False)
    _buckets: dict[tuple[str, RateLimitTier], TokenBucketState] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._buckets: dict[tuple[str, RateLimitTier], TokenBucketState] = {}

    def consume(
        self,
        client_id: str,
        tier: RateLimitTier,
        *,
        cost: int = 1,
        now: float | None = None,
    ) -> RateLimitDecision:
        with self._lock:
            decision = self._consume_locked(client_id, tier, cost=cost, now=now)
            if not decision.allowed and decision.retry_after_seconds is not None:
                raise RateLimitExceeded(
                    tier=tier,
                    retry_after_seconds=decision.retry_after_seconds,
                    denial_count=decision.denial_count,
                )
            return decision

    def inspect(
        self,
        client_id: str,
        tier: RateLimitTier,
        *,
        now: float | None = None,
    ) -> RateLimitDecision:
        with self._lock:
            current_time = time.monotonic() if now is None else now
            bucket = self._get_bucket(client_id, tier, current_time)
            config = self._config_for(tier)
            self._refill(bucket, config, current_time)
            return RateLimitDecision(
                allowed=True,
                tokens_remaining=bucket.tokens,
                denial_count=bucket.denial_count,
            )

    def _consume_locked(
        self,
        client_id: str,
        tier: RateLimitTier,
        *,
        cost: int,
        now: float | None,
    ) -> RateLimitDecision:
        current_time = time.monotonic() if now is None else now
        bucket = self._get_bucket(client_id, tier, current_time)
        config = self._config_for(tier)
        self._refill(bucket, config, current_time)

        if bucket.tokens >= cost:
            bucket.tokens -= cost
            bucket.denial_count = 0
            return RateLimitDecision(
                allowed=True,
                tokens_remaining=bucket.tokens,
                denial_count=0,
            )

        bucket.denial_count += 1
        shortfall = cost - bucket.tokens
        refill_wait = shortfall / config.refill_rate
        retry_after = max(
            refill_wait,
            min(
                config.base_retry_after_seconds * (2 ** (bucket.denial_count - 1)),
                config.max_retry_after_seconds,
            ),
        )
        return RateLimitDecision(
            allowed=False,
            tokens_remaining=max(bucket.tokens, 0.0),
            retry_after_seconds=math.ceil(retry_after * 1000) / 1000,
            denial_count=bucket.denial_count,
        )

    def _get_bucket(
        self,
        client_id: str,
        tier: RateLimitTier,
        now: float,
    ) -> TokenBucketState:
        bucket_key = (client_id, tier)
        bucket = self._buckets.get(bucket_key)
        if bucket is None:
            config = self._config_for(tier)
            bucket = TokenBucketState(
                tokens=float(config.capacity),
                last_refill_monotonic=now,
            )
            self._buckets[bucket_key] = bucket
        return bucket

    @staticmethod
    def _refill(bucket: TokenBucketState, config: TokenBucketConfig, now: float) -> None:
        elapsed = max(0.0, now - bucket.last_refill_monotonic)
        bucket.last_refill_monotonic = now
        bucket.tokens = min(config.capacity, bucket.tokens + elapsed * config.refill_rate)

    def _config_for(self, tier: RateLimitTier) -> TokenBucketConfig:
        if tier is RateLimitTier.RESOURCE:
            return self.resource_config
        return self.tool_config
