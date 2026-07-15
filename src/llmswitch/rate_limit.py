import time
import threading
from typing import Dict, Optional
from llmswitch.config import Endpoint, RateLimit


def estimate_tokens(messages: list) -> int:
    """A quick estimate of tokens in messages to avoid importing tiktoken."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4 + 4  # ~4 chars per token + overhead
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    total += len(text) // 4 + 4
    return total


class TokenBucket:
    """Thread-safe Token Bucket for rate limit tracking."""

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, amount: float) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def can_consume(self, amount: float) -> bool:
        with self._lock:
            self._refill()
            return self.tokens >= amount

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)


class EndpointRateLimiter:
    """Manages Token Buckets for all limits defined on an endpoint."""

    def __init__(self, limits: RateLimit):
        self.limits = limits
        self.req_minute_bucket = (
            TokenBucket(limits.rpm, limits.rpm / 60.0)
            if limits.rpm and limits.rpm > 0
            else None
        )
        self.req_day_bucket = (
            TokenBucket(limits.rpd, limits.rpd / 86400.0)
            if limits.rpd and limits.rpd > 0
            else None
        )
        self.tok_minute_bucket = (
            TokenBucket(limits.tpm, limits.tpm / 60.0)
            if limits.tpm and limits.tpm > 0
            else None
        )
        self.tok_day_bucket = (
            TokenBucket(limits.tpd, limits.tpd / 86400.0)
            if limits.tpd and limits.tpd > 0
            else None
        )

    def can_accept(self, estimated_tokens: int) -> bool:
        if self.req_minute_bucket and not self.req_minute_bucket.can_consume(1):
            return False
        if self.req_day_bucket and not self.req_day_bucket.can_consume(1):
            return False
        if self.tok_minute_bucket and not self.tok_minute_bucket.can_consume(
            estimated_tokens
        ):
            return False
        if self.tok_day_bucket and not self.tok_day_bucket.can_consume(
            estimated_tokens
        ):
            return False
        return True

    def consume(self, estimated_tokens: int):
        if self.req_minute_bucket:
            self.req_minute_bucket.consume(1)
        if self.req_day_bucket:
            self.req_day_bucket.consume(1)
        if self.tok_minute_bucket:
            self.tok_minute_bucket.consume(estimated_tokens)
        if self.tok_day_bucket:
            self.tok_day_bucket.consume(estimated_tokens)

    def adjust_tokens(self, estimated_tokens: int, actual_tokens: int):
        """Refunds unused token capacity or consumes additional tokens if estimate was exceeded."""
        diff = estimated_tokens - actual_tokens
        if self.tok_minute_bucket:
            with self.tok_minute_bucket._lock:
                self.tok_minute_bucket.tokens = min(
                    self.tok_minute_bucket.capacity,
                    max(0.0, self.tok_minute_bucket.tokens + diff),
                )
        if self.tok_day_bucket:
            with self.tok_day_bucket._lock:
                self.tok_day_bucket.tokens = min(
                    self.tok_day_bucket.capacity,
                    max(0.0, self.tok_day_bucket.tokens + diff),
                )


class RateLimitManager:
    """Registry to manage and retrieve stateless endpoint limiters."""

    def __init__(self):
        self._limiters: Dict[str, EndpointRateLimiter] = {}
        self._lock = threading.Lock()

    def get_limiter(self, endpoint: Endpoint) -> EndpointRateLimiter:
        key = f"{endpoint.provider}:{endpoint.base_url}:{endpoint.model}"
        with self._lock:
            if key not in self._limiters:
                self._limiters[key] = EndpointRateLimiter(endpoint.limits)
            return self._limiters[key]
