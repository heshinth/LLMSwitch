import time
from typing import List, Union, Dict, Any, Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI, OpenAIError
from loguru import logger

from llmswitch.config import LLMSwitchConfig, Endpoint, test_logging
from llmswitch.errors import ErrorClassifier
from llmswitch.rate_limit import RateLimitManager, estimate_tokens


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
