import sys
from loguru import logger

# Disable logging for the llmswitch library by default to avoid polluting host applications.
logger.disable("llmswitch")


def enable_logging(level: str = "INFO") -> int:
    """
    Enable logging for the llmswitch library namespace and configure a custom
    stdout sink with clean formatting for llmswitch log messages.
    """
    logger.enable("llmswitch")

    handler_id = logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level=level,
        filter=lambda record: record["name"].startswith("llmswitch"),
    )
    return handler_id
