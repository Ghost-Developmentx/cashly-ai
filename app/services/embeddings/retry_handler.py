"""
Handles retry logic for async operations.
"""

import asyncio
import logging
from typing import Callable, Optional, TypeVar, Awaitable
import openai
from app.core.config import Settings


logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncRetryHandler:
    """
    AsyncRetryHandler is a handler to execute asynchronous functions with retry logic
    using exponential backoff. This class is designed to handle rate limit errors and
    API errors gracefully, ensuring minimal disruption to operations by automatically
    retrying the operation within configured constraints.

    It helps manage scenarios where transient errors occur, such as rate limits or
    temporary server unavailability, allowing the caller to continue without needing
    to manually implement retry logic.

    Attributes
    ----------
    max_retries : int
        The maximum number of retry attempts for the operation.
    base_delay : float
        The initial delay between retries, in seconds.
    max_delay : float
        The maximum delay between retries, in seconds.
    exponential_base : float
        The base value used for calculating exponential backoff delay.
    """

    def __init__(self, config: Settings):
        self.max_retries = config.max_retries
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.exponential_base = 2

    async def execute_with_retry(
        self, func: Callable[..., Awaitable[T]], *args, **kwargs
    ) -> Optional[T]:
        """
        Execute an async function with exponential backoff retry.
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)

            except openai.RateLimitError as e:
                last_exception = e
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Rate limit exceeded after {self.max_retries} attempts"
                    )
                    break

                wait_time = self._calculate_delay(attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)

            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                last_exception = e
                logger.error(f"OpenAI API error: {type(e).__name__}: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = self._calculate_delay(attempt)
                    logger.info(f"Retrying after {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retries failed. Last error: {e}")

            except openai.AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                break  # Don't retry auth errors

            except Exception as e:
                logger.error(f"Unexpected error in retry handler: {e}")
                last_exception = e
                break

        return None

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)

        # Add jitter
        import random

        jitter = random.uniform(0, 0.1 * delay)

        return delay + jitter
