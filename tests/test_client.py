import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from openai import OpenAIError
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
                model="gpt-4o"
            ),
            Endpoint(
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-openrouter",
                model="openai/gpt-4o"
            )
        ]
    )

# --- Sync Client Tests ---

@patch("llmswitch.client.OpenAI")
def test_sync_routing_success(mock_openai, sample_config):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = "response_ok"

    client = Client(sample_config)
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}]
    )

    assert res == "response_ok"
    mock_openai.assert_called_once_with(
        base_url="https://api.openai.com/v1",
        api_key="sk-openai"
    )

@patch("llmswitch.client.OpenAI")
def test_sync_fallback_on_normal_error(mock_openai, sample_config):
    mock_client_fail = MagicMock()
    mock_client_fail.chat.completions.create.side_effect = OpenAIError("Auth failed")

    mock_client_success = MagicMock()
    mock_client_success.chat.completions.create.return_value = "response_fallback"

    mock_openai.side_effect = [mock_client_fail, mock_client_success]

    client = Client(sample_config)
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[]
    )

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
    mock_client_success.chat.completions.create = AsyncMock(return_value=mock_async_generator())

    mock_async_openai.side_effect = [mock_client_fail, mock_client_success]

    client = AsyncClient(sample_config)
    stream = await client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == ["chunk_a", "chunk_b"]
    assert "openai" in client._cooldowns
