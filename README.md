# LLMSwitch

A lightweight, production-ready routing and fallback client for LLM providers. It acts as a proxy for the official OpenAI Python SDK, implementing priority-based fallbacks, circuit breakers (with smart `Retry-After` header parsing), and client-side token/request rate limits.

---

## Features

- **Priority-Based Fallback Routing**: Define a model alias (e.g. `gpt-4o`) mapping to a list of provider endpoints (e.g., OpenAI, OpenRouter, Anthropic). If one fails, it automatically routes to the next candidate.
- **Async & Sync Support**: Native implementation for both `Client` and `AsyncClient` mirroring the standard OpenAI interface.
- **Circuit Breaker on 429**: Temporarily cools down rate-limited providers.
- **Smart Retry-After Parsing**: Extracts precise cooldown times from HTTP headers (standard `Retry-After` and custom OpenAI reset headers).
- **Client-Side Rate Limiting**: Token-bucket rate limiter that enforces configured requests-per-minute (RPM) and tokens-per-minute (TPM) limits locally.
- **Streaming Fallback**: Gracefully falls back to secondary endpoints if initial streaming connection fails.
- **Client Pooling**: Reuses underlying `OpenAI` and `AsyncOpenAI` instances to optimize connection pooling.

---

## Installation

Add it to your Python project using `pip`:
```bash
pip install llmswitch-client
```

Or using `uv`:
```bash
uv add llmswitch-client
```

---

## Quickstart

### 1. Define Configuration

Configure virtual model aliases and register their respective target endpoints:

```python
from llmswitch import LLMSwitchConfig, Endpoint

# Configure gpt-4o fallback path
gpt4o_config = LLMSwitchConfig(
    alias="gpt-4o",
    endpoints=[
        Endpoint(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="your-openai-api-key",
            model="gpt-4o"
        ),
        Endpoint(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="your-openrouter-api-key",
            model="openai/gpt-4o"
        )
    ]
)
```

### 2. Synchronous Client

Use the client exactly like the official OpenAI Python SDK:

```python
from llmswitch import Client

client = Client(configs=gpt4o_config)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a joke!"}]
)

print(response.choices[0].message.content)
```

### 3. Asynchronous Client

Use the `AsyncClient` for asynchronous python runtimes (FastAPI, Quart, asyncio, etc.):

```python
import asyncio
from llmswitch import AsyncClient

async def main():
    client = AsyncClient(configs=gpt4o_config)
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story!"}]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## Advanced Usage

### Local Preemptive Rate Limiting (TPM / RPM)

Avoid sending calls that you know will fail. You can enforce token-bucket rate limits client-side by setting limits on endpoints. If an endpoint is rate-limited locally, the client will bypass it:

```python
from llmswitch import Endpoint, RateLimit

endpoint_with_limits = Endpoint(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4o",
    limits=RateLimit(
        rpm=10,       # 10 requests per minute
        tpm=40000     # 40k tokens per minute
    )
)
```

### Routing Strategies

You can configure different strategies to distribute load across your endpoints:

- `"priority"` (default): Tries endpoints strictly in the order they are listed in configuration.
- `"round_robin"`: Cycles the starting endpoint for each request, distributing traffic evenly.
- `"random"`: Randomly shuffles the endpoints list on every request.

```python
round_robin_config = LLMSwitchConfig(
    alias="gpt-4o",
    strategy="round_robin",  # Can also be "random" or "priority"
    endpoints=[...]
)
```

### Streaming with Fallback

Streaming works out-of-the-box. If the initial stream setup fails, `LLMSwitch` falls back to the next healthy provider:

```python
# Sync Streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a long essay."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

```python
# Async Streaming
stream = await async_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a long essay."}],
    stream=True
)

async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Enabling Logs

By default, logs are disabled to keep your logs clean. You can enable them for debugging:

```python
from llmswitch import enable_logging

# Enable and print clean colorized logs to standard output
enable_logging(level="INFO")
```

---

## License

This project is licensed under the Apache-2.0 License.
