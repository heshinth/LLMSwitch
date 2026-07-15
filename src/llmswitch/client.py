import time
from typing import List, Union, Dict, Any
from openai import OpenAI, OpenAIError
from loguru import logger
from llmswitch.config import LLMSwitchConfig, Endpoint, test_logging

class Completions:
    def __init__(self, routing_map: Dict[str, List[Endpoint]], cooldowns: Dict[str, float]):
        self._routing_map = routing_map
        self._cooldowns = cooldowns

    def _is_cooling_down(self, provider: str) -> bool:
        if provider in self._cooldowns:
            if time.time() < self._cooldowns[provider]:
                return True
            del self._cooldowns[provider]
        return False

    def _trigger_cooldown(self, provider: str, duration: float = 60.0):
        self._cooldowns[provider] = time.time() + duration

    def create(self, model: str, messages: list, **kwargs) -> Any:
        candidates = self._routing_map.get(model)
        if not candidates:
            raise ValueError(f"Model alias '{model}' is not registered.")

        for endpoint in candidates:
            provider = endpoint.provider
            if self._is_cooling_down(provider):
                continue

            try:
                client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)
                logger.info(f"Routing '{model}' to {provider} ({endpoint.model})...")
                return client.chat.completions.create(
                    model=endpoint.model,
                    messages=messages,
                    **kwargs
                )
            except OpenAIError as e:
                is_429 = "rate_limit" in str(e).lower() or (hasattr(e, "status_code") and e.status_code == 429)
                if is_429:
                    logger.warning(f"Rate limit hit on {provider}. Activating circuit breaker.")
                    self._trigger_cooldown(provider, duration=60.0)
                else:
                    logger.error(f"Error from {provider}: {e}")
                continue

        raise RuntimeError(f"All providers under alias '{model}' are exhausted.")

class Chat:
    def __init__(self, routing_map: Dict[str, List[Endpoint]], cooldowns: Dict[str, float]):
        self.completions = Completions(routing_map, cooldowns)

class SwitchClient:
    def __init__(self, configs: Union[LLMSwitchConfig, List[LLMSwitchConfig]]):
        self.configs = configs if isinstance(configs, list) else [configs]
        self._routing_map: Dict[str, List[Endpoint]] = {c.alias: c.endpoints for c in self.configs}
        self._cooldowns: Dict[str, float] = {}
        self.chat = Chat(self._routing_map, self._cooldowns)
