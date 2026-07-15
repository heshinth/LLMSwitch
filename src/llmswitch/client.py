import time
import threading
from typing import List, Union, Dict, Any, Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI, OpenAIError
from loguru import logger
from llmswitch.config import LLMSwitchConfig, Endpoint, RateLimit, test_logging  # noqa: F401


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


class ErrorClassifier:
    """Helper to classify API errors and parse rate limit metadata."""

    @staticmethod
    def is_rate_limit(error: Exception) -> bool:
        if not isinstance(error, OpenAIError):
            return False
        return "rate_limit" in str(error).lower() or (
            hasattr(error, "status_code") and error.status_code == 429
        )

    @staticmethod
    def get_retry_after(error: Exception) -> float:
        """Extracts the retry-after duration (in seconds) from error response headers if available."""
        if not hasattr(error, "response") or error.response is None:
            return 60.0  # default fallback

        headers = getattr(error.response, "headers", {})

        # Standard HTTP Header
        if "retry-after" in headers:
            try:
                return float(headers["retry-after"])
            except ValueError:
                pass

        # OpenAI specific reset headers (e.g. '12ms', '6s', '2m')
        for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            if key in headers:
                val = str(headers[key]).strip().lower()
                try:
                    if val.endswith("ms"):
                        return float(val[:-2]) / 1000.0
                    elif val.endswith("s"):
                        return float(val[:-1])
                    elif val.endswith("m"):
                        return float(val[:-1]) * 60.0
                    elif val.endswith("h"):
                        return float(val[:-1]) * 3600.0
                    else:
                        return float(val)
                except ValueError:
                    pass

        return 60.0  # default fallback


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


class CircuitBreaker:
    """Tracks and manages provider cooldowns."""

    def __init__(self, cooldowns: Dict[str, float]):
        self._cooldowns = cooldowns

    def is_cooling_down(self, provider: str) -> bool:
        if provider in self._cooldowns:
            if time.time() < self._cooldowns[provider]:
                return True
            del self._cooldowns[provider]
        return False

    def trigger_cooldown(self, provider: str, duration: float = 60.0):
        self._cooldowns[provider] = time.time() + duration


class Router:
    """Selects and filters active endpoints for a given model alias."""

    def __init__(
        self, routing_map: Dict[str, List[Endpoint]], circuit_breaker: CircuitBreaker
    ):
        self._routing_map = routing_map
        self._circuit_breaker = circuit_breaker

    def get_active_endpoints(self, model: str) -> List[Endpoint]:
        candidates = self._routing_map.get(model)
        if not candidates:
            raise ValueError(f"Model alias '{model}' is not registered.")
        return candidates


class ClientRegistry:
    """Caches OpenAI and AsyncOpenAI clients for connection reuse."""

    def __init__(self):
        self._sync_clients: Dict[str, OpenAI] = {}
        self._async_clients: Dict[str, AsyncOpenAI] = {}

    def get_sync_client(self, endpoint: Endpoint) -> OpenAI:
        key = f"{endpoint.base_url}:{endpoint.api_key}"
        if key not in self._sync_clients:
            self._sync_clients[key] = OpenAI(
                base_url=endpoint.base_url, api_key=endpoint.api_key
            )
        return self._sync_clients[key]

    def get_async_client(self, endpoint: Endpoint) -> AsyncOpenAI:
        key = f"{endpoint.base_url}:{endpoint.api_key}"
        if key not in self._async_clients:
            self._async_clients[key] = AsyncOpenAI(
                base_url=endpoint.base_url, api_key=endpoint.api_key
            )
        return self._async_clients[key]


class Completions:
    """Handles synchronous LLM completions, fallbacks, and client-side rate limits."""

    def __init__(
        self,
        router: Router,
        client_registry: ClientRegistry,
        circuit_breaker: CircuitBreaker,
        rate_limit_manager: RateLimitManager,
    ):
        self._router = router
        self._client_registry = client_registry
        self._circuit_breaker = circuit_breaker
        self._rate_limit_manager = rate_limit_manager

    def create(self, model: str, messages: list, **kwargs) -> Any:
        endpoints = self._router.get_active_endpoints(model)
        max_tokens = (
            kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 1000
        )
        req_tokens = estimate_tokens(messages) + max_tokens

        for idx, endpoint in enumerate(endpoints):
            provider = endpoint.provider
            if self._circuit_breaker.is_cooling_down(provider):
                continue

            limiter = self._rate_limit_manager.get_limiter(endpoint)
            if not limiter.can_accept(req_tokens):
                logger.warning(
                    f"Endpoint for {provider} ({endpoint.model}) skipped: local rate limit exceeded."
                )
                continue

            limiter.consume(req_tokens)

            try:
                client = self._client_registry.get_sync_client(endpoint)
                logger.info(f"Routing '{model}' to {provider} ({endpoint.model})...")

                if kwargs.get("stream"):
                    return self._stream_wrapper(
                        idx, endpoints, model, messages, req_tokens, **kwargs
                    )

                response = client.chat.completions.create(
                    model=endpoint.model, messages=messages, **kwargs
                )

                if hasattr(response, "usage") and response.usage is not None:
                    actual_tokens = (
                        response.usage.prompt_tokens + response.usage.completion_tokens
                    )
                    limiter.adjust_tokens(req_tokens, actual_tokens)

                return response
            except OpenAIError as e:
                limiter.adjust_tokens(req_tokens, 0)  # Refund full allocation on error

                if ErrorClassifier.is_rate_limit(e):
                    cooldown = ErrorClassifier.get_retry_after(e)
                    logger.warning(
                        f"Rate limit hit on {provider}. Activating circuit breaker for {cooldown}s."
                    )
                    self._circuit_breaker.trigger_cooldown(provider, duration=cooldown)
                else:
                    logger.error(f"Error from {provider}: {e}")
                continue

        raise RuntimeError(f"All providers under alias '{model}' are exhausted.")

    def _stream_wrapper(
        self,
        start_idx: int,
        endpoints: List[Endpoint],
        model: str,
        messages: list,
        req_tokens: int,
        **kwargs,
    ) -> Generator[Any, None, None]:
        yielded_any = False
        remaining_endpoints = endpoints[start_idx:]

        for endpoint in remaining_endpoints:
            provider = endpoint.provider
            if self._circuit_breaker.is_cooling_down(provider):
                continue

            limiter = self._rate_limit_manager.get_limiter(endpoint)
            if not limiter.can_accept(req_tokens):
                logger.warning(
                    f"Endpoint for {provider} ({endpoint.model}) skipped: local rate limit exceeded."
                )
                continue

            limiter.consume(req_tokens)

            try:
                client = self._client_registry.get_sync_client(endpoint)
                logger.info(
                    f"Routing '{model}' to {provider} ({endpoint.model}) [Stream]..."
                )
                stream = client.chat.completions.create(
                    model=endpoint.model, messages=messages, **kwargs
                )
                generated_text = []
                for chunk in stream:
                    yielded_any = True
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            generated_text.append(delta.content)
                    yield chunk

                actual_prompt_tokens = estimate_tokens(messages)
                actual_completion_tokens = len("".join(generated_text)) // 4
                limiter.adjust_tokens(
                    req_tokens, actual_prompt_tokens + actual_completion_tokens
                )
                return
            except OpenAIError as e:
                limiter.adjust_tokens(req_tokens, 0)

                if yielded_any:
                    logger.error(f"Error from {provider} mid-stream: {e}")
                    raise

                if ErrorClassifier.is_rate_limit(e):
                    cooldown = ErrorClassifier.get_retry_after(e)
                    logger.warning(
                        f"Rate limit hit on {provider}. Activating circuit breaker for {cooldown}s."
                    )
                    self._circuit_breaker.trigger_cooldown(provider, duration=cooldown)
                else:
                    logger.error(f"Error from {provider} during stream init: {e}")
                continue

        raise RuntimeError(f"All providers under alias '{model}' are exhausted.")


class Chat:
    """Synchronous Chat namespace."""

    def __init__(
        self,
        router: Router,
        client_registry: ClientRegistry,
        circuit_breaker: CircuitBreaker,
        rate_limit_manager: RateLimitManager,
    ):
        self.completions = Completions(
            router, client_registry, circuit_breaker, rate_limit_manager
        )


class Client:
    """Synchronous LLMSwitch Client."""

    def __init__(self, configs: Union[LLMSwitchConfig, List[LLMSwitchConfig]]):
        self.configs = configs if isinstance(configs, list) else [configs]
        self._routing_map: Dict[str, List[Endpoint]] = {
            c.alias: c.endpoints for c in self.configs
        }
        self._cooldowns: Dict[str, float] = {}

        self._circuit_breaker = CircuitBreaker(self._cooldowns)
        self._router = Router(self._routing_map, self._circuit_breaker)
        self._client_registry = ClientRegistry()
        self._rate_limit_manager = RateLimitManager()

        self.chat = Chat(
            self._router,
            self._client_registry,
            self._circuit_breaker,
            self._rate_limit_manager,
        )


class AsyncCompletions:
    """Handles asynchronous LLM completions, fallbacks, and client-side rate limits."""

    def __init__(
        self,
        router: Router,
        client_registry: ClientRegistry,
        circuit_breaker: CircuitBreaker,
        rate_limit_manager: RateLimitManager,
    ):
        self._router = router
        self._client_registry = client_registry
        self._circuit_breaker = circuit_breaker
        self._rate_limit_manager = rate_limit_manager

    async def create(self, model: str, messages: list, **kwargs) -> Any:
        endpoints = self._router.get_active_endpoints(model)
        max_tokens = (
            kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 1000
        )
        req_tokens = estimate_tokens(messages) + max_tokens

        for idx, endpoint in enumerate(endpoints):
            provider = endpoint.provider
            if self._circuit_breaker.is_cooling_down(provider):
                continue

            limiter = self._rate_limit_manager.get_limiter(endpoint)
            if not limiter.can_accept(req_tokens):
                logger.warning(
                    f"Endpoint for {provider} ({endpoint.model}) skipped: local rate limit exceeded."
                )
                continue

            limiter.consume(req_tokens)

            try:
                client = self._client_registry.get_async_client(endpoint)
                logger.info(f"Routing '{model}' to {provider} ({endpoint.model})...")

                if kwargs.get("stream"):
                    return self._stream_wrapper(
                        idx, endpoints, model, messages, req_tokens, **kwargs
                    )

                response = await client.chat.completions.create(
                    model=endpoint.model, messages=messages, **kwargs
                )

                if hasattr(response, "usage") and response.usage is not None:
                    actual_tokens = (
                        response.usage.prompt_tokens + response.usage.completion_tokens
                    )
                    limiter.adjust_tokens(req_tokens, actual_tokens)

                return response
            except OpenAIError as e:
                limiter.adjust_tokens(req_tokens, 0)

                if ErrorClassifier.is_rate_limit(e):
                    cooldown = ErrorClassifier.get_retry_after(e)
                    logger.warning(
                        f"Rate limit hit on {provider}. Activating circuit breaker for {cooldown}s."
                    )
                    self._circuit_breaker.trigger_cooldown(provider, duration=cooldown)
                else:
                    logger.error(f"Error from {provider}: {e}")
                continue

        raise RuntimeError(f"All providers under alias '{model}' are exhausted.")

    async def _stream_wrapper(
        self,
        start_idx: int,
        endpoints: List[Endpoint],
        model: str,
        messages: list,
        req_tokens: int,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        yielded_any = False
        remaining_endpoints = endpoints[start_idx:]

        for endpoint in remaining_endpoints:
            provider = endpoint.provider
            if self._circuit_breaker.is_cooling_down(provider):
                continue

            limiter = self._rate_limit_manager.get_limiter(endpoint)
            if not limiter.can_accept(req_tokens):
                logger.warning(
                    f"Endpoint for {provider} ({endpoint.model}) skipped: local rate limit exceeded."
                )
                continue

            limiter.consume(req_tokens)

            try:
                client = self._client_registry.get_async_client(endpoint)
                logger.info(
                    f"Routing '{model}' to {provider} ({endpoint.model}) [Async Stream]..."
                )
                stream = await client.chat.completions.create(
                    model=endpoint.model, messages=messages, **kwargs
                )
                generated_text = []
                async for chunk in stream:
                    yielded_any = True
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            generated_text.append(delta.content)
                    yield chunk

                actual_prompt_tokens = estimate_tokens(messages)
                actual_completion_tokens = len("".join(generated_text)) // 4
                limiter.adjust_tokens(
                    req_tokens, actual_prompt_tokens + actual_completion_tokens
                )
                return
            except OpenAIError as e:
                limiter.adjust_tokens(req_tokens, 0)

                if yielded_any:
                    logger.error(f"Error from {provider} mid-stream: {e}")
                    raise

                if ErrorClassifier.is_rate_limit(e):
                    cooldown = ErrorClassifier.get_retry_after(e)
                    logger.warning(
                        f"Rate limit hit on {provider}. Activating circuit breaker for {cooldown}s."
                    )
                    self._circuit_breaker.trigger_cooldown(provider, duration=cooldown)
                else:
                    logger.error(f"Error from {provider} during stream init: {e}")
                continue

        raise RuntimeError(f"All providers under alias '{model}' are exhausted.")


class AsyncChat:
    """Asynchronous Chat namespace."""

    def __init__(
        self,
        router: Router,
        client_registry: ClientRegistry,
        circuit_breaker: CircuitBreaker,
        rate_limit_manager: RateLimitManager,
    ):
        self.completions = AsyncCompletions(
            router, client_registry, circuit_breaker, rate_limit_manager
        )


class AsyncClient:
    """Asynchronous LLMSwitch Client."""

    def __init__(self, configs: Union[LLMSwitchConfig, List[LLMSwitchConfig]]):
        self.configs = configs if isinstance(configs, list) else [configs]
        self._routing_map: Dict[str, List[Endpoint]] = {
            c.alias: c.endpoints for c in self.configs
        }
        self._cooldowns: Dict[str, float] = {}

        self._circuit_breaker = CircuitBreaker(self._cooldowns)
        self._router = Router(self._routing_map, self._circuit_breaker)
        self._client_registry = ClientRegistry()
        self._rate_limit_manager = RateLimitManager()

        self.chat = AsyncChat(
            self._router,
            self._client_registry,
            self._circuit_breaker,
            self._rate_limit_manager,
        )
