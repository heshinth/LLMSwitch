from loguru import logger
from typing import List, Optional
from pydantic import BaseModel, Field


class RateLimit(BaseModel):
    rpm: Optional[int] = Field(None, description="Requests per minute")
    rpd: Optional[int] = Field(None, description="Requests per day")
    tpm: Optional[int] = Field(None, description="Tokens per minute")
    tpd: Optional[int] = Field(None, description="Tokens per day")


class Endpoint(BaseModel):
    provider: str = Field(..., description="The provider name for this endpoint")
    base_url: str = Field(..., description="The base URL for this endpoint")
    api_key: str = Field(..., description="The API key for this endpoint")
    model: str = Field(..., description="The model name for this endpoint")
    limits: RateLimit = Field(default_factory=RateLimit)


class LLMSwitchConfig(BaseModel):
    alias: str = Field(..., description="The virtual model name used in your app")
    endpoints: List[Endpoint] = Field(
        ..., description="List of fallback targets for this alias"
    )


def test_logging():
    """Verify that logging is working correctly inside the library."""
    logger.debug("llmswitch debug message")
    logger.info("llmswitch info message")
    logger.warning("llmswitch warning message")
