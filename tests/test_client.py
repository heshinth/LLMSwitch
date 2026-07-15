import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from openai import OpenAIError
from llmswitch.config import RateLimit
from llmswitch import Client, AsyncClient, LLMSwitchConfig, Endpoint


@pytest.fixture
def sample_config():
    return LLMSwitchConfig(
        alias="gpt-4o",
        endpoints=[
            Endpoint(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="sk-openai",
                model="gpt-4o",
            ),
            Endpoint(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-openrouter",
                model="openai/gpt-4o",
            ),
        ],
    )


# --- Sync Client Tests ---


@patch("llmswitch.client.OpenAI")
def test_sync_routing_success(mock_openai, sample_config):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = "response_ok"

    client = Client(sample_config)
    res = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )

    assert res == "response_ok"
    mock_openai.assert_called_once_with(
        base_url="https://api.openai.com/v1", api_key="sk-openai"
    )


@patch("llmswitch.client.OpenAI")
def test_sync_fallback_on_normal_error(mock_openai, sample_config):
    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create.side_effect = OpenAIError("Auth failed")

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create.return_value = "response_fallback"

    mock_openai.side_effect = [mock_client_fail, mock_client_success]

    client = Client(sample_config)
    res = client.chat.completions.create(model="gpt-4o", messages=[])

    assert res == "response_fallback"
    assert "openai" not in client._cooldowns


@patch("llmswitch.client.OpenAI")
def test_sync_circuit_breaker_on_429(mock_openai, sample_config):
    err = OpenAIError("Rate limit exceeded")
    err.status_code = 429

    mock_client_429 = MagicMock()
    mock_client_429.chat.completions.create.side_effect = err

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create.return_value = "response_ok"

    mock_openai.side_effect = [mock_client_429, mock_client_success]

    client = Client(sample_config)
    res1 = client.chat.completions.create(model="gpt-4o", messages=[])
    assert res1 == "response_ok"
    assert "openai" in client._cooldowns

    # 2nd call: First provider (openai) should be skipped preemptively
    mock_openai.reset_mock()
    mock_client_success.chat.completions.create.reset_mock()

    res2 = client.chat.completions.create(model="gpt-4o", messages=[])
    assert res2 == "response_ok"
    # Verify no new OpenAI clients were instantiated (proving cache hit)
    mock_openai.assert_not_called()
    # Verify we still called the completions API of the cached openrouter client
    mock_client_success.chat.completions.create.assert_called_once_with(
        model="openai/gpt-4o",
        messages=[],
    )


@patch("llmswitch.client.OpenAI")
def test_sync_streaming_init_failure_fallback(mock_openai, sample_config):
    err = OpenAIError("Rate limit exceeded")
    err.status_code = 429

    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create.side_effect = err

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create.return_value = ["chunk1", "chunk2"]

    mock_openai.side_effect = [mock_client_fail, mock_client_success]

    client = Client(sample_config)
    stream = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    chunks = list(stream)

    assert chunks == ["chunk1", "chunk2"]
    assert "openai" in client._cooldowns


# --- Async Client Tests ---


@pytest.mark.asyncio
@patch("llmswitch.client.AsyncOpenAI")
async def test_async_routing_success(mock_async_openai, sample_config):
    mock_client = MagicMock()
    mock_async_openai.return_value = mock_client
    mock_client.chat.completions.create = AsyncMock(return_value="async_ok")

    client = AsyncClient(sample_config)
    res = await client.chat.completions.create(model="gpt-4o", messages=[])

    assert res == "async_ok"


@pytest.mark.asyncio
@patch("llmswitch.client.AsyncOpenAI")
async def test_async_streaming_init_fallback(mock_async_openai, sample_config):
    err = OpenAIError("Rate limit")
    err.status_code = 429

    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create = AsyncMock(side_effect=err)

    async def mock_async_generator():
        yield "chunk_a"
        yield "chunk_b"

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create = AsyncMock(
        return_value=mock_async_generator()
    )

    mock_async_openai.side_effect = [mock_client_fail, mock_client_success]

    client = AsyncClient(sample_config)
    stream = await client.chat.completions.create(
        model="gpt-4o", messages=[], stream=True
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == ["chunk_a", "chunk_b"]
    assert "openai" in client._cooldowns


# --- Client-Side Rate Limit & Retry-After Tests ---


@patch("llmswitch.client.OpenAI")
def test_local_rate_limit_preemptive_skip(mock_openai):
    # Setup config where first endpoint has a strict limit (rpm=1)
    config = LLMSwitchConfig(
        alias="gpt-4o",
        endpoints=[
            Endpoint(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="sk-openai",
                model="gpt-4o",
                limits=RateLimit(rpm=1),
            ),
            Endpoint(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-openrouter",
                model="openai/gpt-4o",
            ),
        ],
    )

    # Success client responses
    mock_client_openai = MagicMock()
    mock_client_openai.chat.completions.create.return_value = MagicMock(
        usage=MagicMock(prompt_tokens=10, completion_tokens=10)
    )

    mock_client_openrouter = MagicMock()
    mock_client_openrouter.chat.completions.create.return_value = MagicMock(
        usage=MagicMock(prompt_tokens=10, completion_tokens=10)
    )

    mock_openai.side_effect = [mock_client_openai, mock_client_openrouter]

    client = Client(config)

    # 1. First call should route to openai (and consume the 1 request token bucket capacity)
    res1 = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello"}]
    )
    assert res1 is not None
    mock_client_openai.chat.completions.create.assert_called_once()

    # Reset mock call counts
    mock_client_openai.chat.completions.create.reset_mock()
    mock_client_openrouter.chat.completions.create.reset_mock()

    # 2. Second call immediately should skip openai due to local rate limit (rpm=1 exceeded)
    # and route directly to openrouter.
    res2 = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello"}]
    )
    assert res2 is not None
    mock_client_openai.chat.completions.create.assert_not_called()
    mock_client_openrouter.chat.completions.create.assert_called_once()


@patch("llmswitch.client.OpenAI")
def test_dynamic_retry_after_header_parsing(mock_openai, sample_config):
    # Mock response with headers
    mock_response = MagicMock()
    mock_response.headers = {"retry-after": "5.5", "x-ratelimit-reset-requests": "12s"}

    err = OpenAIError("Rate limited")
    err.response = mock_response
    err.status_code = 429

    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create.side_effect = err

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create.return_value = "ok"

    mock_openai.side_effect = [mock_client_fail, mock_client_success]

    client = Client(sample_config)

    start_time = time.time()
    client.chat.completions.create(model="gpt-4o", messages=[])

    assert "openai" in client._cooldowns
    # The cooldown expiration time should be start_time + 5.5 seconds (retry-after header takes priority)
    cooldown_expiry = client._cooldowns["openai"]
    assert cooldown_expiry >= start_time + 5.5
    assert cooldown_expiry <= start_time + 6.5


@patch("llmswitch.client.OpenAI")
def test_dynamic_retry_after_fallback_openai_headers(mock_openai, sample_config):
    # Mock response with OpenAI x-ratelimit headers only
    mock_response = MagicMock()
    mock_response.headers = {"x-ratelimit-reset-requests": "12s"}

    err = OpenAIError("Rate limited")
    err.response = mock_response
    err.status_code = 429

    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create.side_effect = err

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create.return_value = "ok"

    mock_openai.side_effect = [mock_client_fail, mock_client_success]

    client = Client(sample_config)

    start_time = time.time()
    client.chat.completions.create(model="gpt-4o", messages=[])

    assert "openai" in client._cooldowns
    cooldown_expiry = client._cooldowns["openai"]
    assert cooldown_expiry >= start_time + 12.0
    assert cooldown_expiry <= start_time + 13.0


@patch("llmswitch.client.OpenAI")
def test_routing_strategy_round_robin(mock_openai):
    config = LLMSwitchConfig(
        alias="gpt-4o",
        strategy="round_robin",
        endpoints=[
            Endpoint(
                provider="openai",
                base_url="https://api.a.com",
                api_key="sk-a",
                model="m",
            ),
            Endpoint(
                provider="openrouter",
                base_url="https://api.b.com",
                api_key="sk-b",
                model="m",
            ),
            Endpoint(
                provider="anthropic",
                base_url="https://api.c.com",
                api_key="sk-c",
                model="m",
            ),
        ],
    )

    mock_a = MagicMock()
    mock_a.chat.completions.create.return_value = "res_a"
    mock_b = MagicMock()
    mock_b.chat.completions.create.return_value = "res_b"
    mock_c = MagicMock()
    mock_c.chat.completions.create.return_value = "res_c"

    # We map mock_openai instantiations to their respective mocks
    def get_mock_client(base_url, api_key):
        if "api.a.com" in base_url:
            return mock_a
        elif "api.b.com" in base_url:
            return mock_b
        elif "api.c.com" in base_url:
            return mock_c
        return MagicMock()

    mock_openai.side_effect = get_mock_client

    client = Client(config)

    # 1st call: Should route to A first (index 0)
    assert client.chat.completions.create(model="gpt-4o", messages=[]) == "res_a"
    mock_a.chat.completions.create.assert_called_once()
    mock_b.chat.completions.create.assert_not_called()
    mock_c.chat.completions.create.assert_not_called()

    mock_a.chat.completions.create.reset_mock()

    # 2nd call: Should route to B first (index 1)
    assert client.chat.completions.create(model="gpt-4o", messages=[]) == "res_b"
    mock_a.chat.completions.create.assert_not_called()
    mock_b.chat.completions.create.assert_called_once()
    mock_c.chat.completions.create.assert_not_called()

    mock_b.chat.completions.create.reset_mock()

    # 3rd call: Should route to C first (index 2)
    assert client.chat.completions.create(model="gpt-4o", messages=[]) == "res_c"
    mock_a.chat.completions.create.assert_not_called()
    mock_b.chat.completions.create.assert_not_called()
    mock_c.chat.completions.create.assert_called_once()

    mock_c.chat.completions.create.reset_mock()

    # 4th call: Should cycle back to A first (index 0)
    assert client.chat.completions.create(model="gpt-4o", messages=[]) == "res_a"
    mock_a.chat.completions.create.assert_called_once()


@patch("llmswitch.client.OpenAI")
def test_routing_strategy_random(mock_openai):
    config = LLMSwitchConfig(
        alias="gpt-4o",
        strategy="random",
        endpoints=[
            Endpoint(
                provider="openai",
                base_url="https://api.a.com",
                api_key="sk-a",
                model="m",
            ),
            Endpoint(
                provider="openrouter",
                base_url="https://api.b.com",
                api_key="sk-b",
                model="m",
            ),
            Endpoint(
                provider="anthropic",
                base_url="https://api.c.com",
                api_key="sk-c",
                model="m",
            ),
        ],
    )

    mock_a = MagicMock()
    mock_a.chat.completions.create.return_value = "res"
    mock_b = MagicMock()
    mock_b.chat.completions.create.return_value = "res"
    mock_c = MagicMock()
    mock_c.chat.completions.create.return_value = "res"

    def get_mock_client(base_url, api_key):
        if "api.a.com" in base_url:
            return mock_a
        elif "api.b.com" in base_url:
            return mock_b
        elif "api.c.com" in base_url:
            return mock_c
        return MagicMock()

    mock_openai.side_effect = get_mock_client

    client = Client(config)

    first_called_endpoints = set()

    for _ in range(30):
        mock_a.chat.completions.create.reset_mock()
        mock_b.chat.completions.create.reset_mock()
        mock_c.chat.completions.create.reset_mock()

        client.chat.completions.create(model="gpt-4o", messages=[])

        if mock_a.chat.completions.create.called:
            first_called_endpoints.add("a")
        elif mock_b.chat.completions.create.called:
            first_called_endpoints.add("b")
        elif mock_c.chat.completions.create.called:
            first_called_endpoints.add("c")

    # Over 30 runs, random routing should hit more than one starting endpoint (probabilistically guaranteed)
    assert len(first_called_endpoints) > 1
