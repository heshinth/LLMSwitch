from openai import OpenAIError


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
