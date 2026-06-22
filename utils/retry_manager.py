"""
Retry Manager - Intelligent retry logic with exponential backoff

This module provides retry capabilities for browser operations and API calls,
with exponential backoff and configurable retry policies.

Usage:
    from utils.retry_manager import retry, RetryManager

    # Using decorator
    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def my_function():
        # Function that might fail transiently
        pass

    # Using context manager
    retry_mgr = RetryManager()
    result = retry_mgr.retry_with_backoff(my_function, arg1, arg2)
"""

import time
import logging
from functools import wraps
from typing import Callable, List, Optional, Any, Tuple, Type
from utils.timeout_manager import timeout_mgr

logger = logging.getLogger(__name__)


class RetryManager:
    """
    Manages retry logic with exponential backoff

    Supports configurable retry attempts, backoff delays, and error filtering.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize RetryManager

        Args:
            config: Retry configuration dict. If None, loads from timeout_mgr.
        """
        if config is None:
            config = timeout_mgr.get_retry_config()

        self.max_attempts = config.get("max_attempts", 3)
        self.initial_delay = config.get("initial_delay", 1)
        self.backoff_multiplier = config.get("backoff_multiplier", 2)
        self.max_delay = config.get("max_delay", 30)
        self.retryable_errors = config.get("retryable_errors", [])

    def is_retryable_error(self, exception: Exception) -> bool:
        """
        Check if an exception is retryable

        Args:
            exception: The exception to check

        Returns:
            True if the exception should trigger a retry
        """
        if not self.retryable_errors:
            # If no specific errors configured, retry all exceptions
            return True

        exception_name = type(exception).__name__

        return any(
            error_type in exception_name or error_type in str(exception)
            for error_type in self.retryable_errors
        )

    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        attempts = 0
        delay = self.initial_delay

        while attempts < self.max_attempts:
            try:
                result = func(*args, **kwargs)
                if attempts > 0:
                    logger.info(
                        f"Success after {attempts} retries: {func.__name__}"
                    )
                return result

            except Exception as e:
                attempts += 1

                # Check if error is retryable
                if not self.is_retryable_error(e):
                    logger.error(
                        f"Non-retryable error in {func.__name__}: {e}"
                    )
                    raise

                if attempts >= self.max_attempts:
                    logger.error(
                        f"Failed after {self.max_attempts} attempts: "
                        f"{func.__name__}: {e}"
                    )
                    raise

                # Calculate next backoff delay
                delay = min(
                    delay * self.backoff_multiplier,
                    self.max_delay
                )

                logger.warning(
                    f"Retry {attempts}/{self.max_attempts} for {func.__name__} "
                    f"after {delay}s: {e}"
                )
                time.sleep(delay)

        raise Exception(f"Failed after {self.max_attempts} attempts")


# Global singleton instance
_retry_mgr = RetryManager()


def retry(
    max_attempts: int = 3,
    backoff_seconds: Optional[List[float]] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Decorator for adding retry logic to functions

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_seconds: List of delay seconds between retries (default: [1, 5, 30])
        retryable_exceptions: Tuple of exception types to retry. If None, retry all.

    Returns:
        Decorated function

    Example:
        @retry(max_attempts=3, backoff_seconds=[1, 2, 5])
        def api_call():
            # Make API call that might fail
            pass

        @retry(retryable_exceptions=(TimeoutError, ConnectionError))
        def browser_operation():
            # Browser operation that might timeout
            pass
    """
    if backoff_seconds is None:
        backoff_seconds = [1, 5, 30]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt, delay in enumerate(backoff_seconds[:max_attempts], 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            f"Success after {attempt-1} retries: {func.__name__}"
                        )
                    return result

                except Exception as e:
                    # Check if this exception type should be retried
                    if retryable_exceptions and not isinstance(e, retryable_exceptions):
                        logger.error(
                            f"Non-retryable exception in {func.__name__}: {e}"
                        )
                        raise

                    if attempt == max_attempts:
                        logger.error(
                            f"Failed after {max_attempts} attempts: "
                            f"{func.__name__}: {e}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    backoff_seconds: Optional[List[float]] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Decorator for adding retry logic to async functions

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_seconds: List of delay seconds between retries (default: [1, 5, 30])
        retryable_exceptions: Tuple of exception types to retry. If None, retry all.

    Returns:
        Decorated async function

    Example:
        @retry_async(max_attempts=3)
        async def async_api_call():
            # Async API call that might fail
            pass
    """
    import asyncio

    if backoff_seconds is None:
        backoff_seconds = [1, 5, 30]

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt, delay in enumerate(backoff_seconds[:max_attempts], 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            f"Async success after {attempt-1} retries: {func.__name__}"
                        )
                    return result

                except Exception as e:
                    if retryable_exceptions and not isinstance(e, retryable_exceptions):
                        logger.error(
                            f"Non-retryable exception in async {func.__name__}: {e}"
                        )
                        raise

                    if attempt == max_attempts:
                        logger.error(
                            f"Async failed after {max_attempts} attempts: "
                            f"{func.__name__}: {e}"
                        )
                        raise

                    logger.warning(
                        f"Async attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

        return wrapper
    return decorator


# Convenience retry functions with preset configurations
def retry_api_call(func: Callable) -> Callable:
    """Decorator for API calls with preset retry configuration"""
    return retry(
        max_attempts=3,
        backoff_seconds=[1, 5, 30],
        retryable_exceptions=(ConnectionError, TimeoutError)
    )(func)


def retry_browser_operation(func: Callable) -> Callable:
    """Decorator for browser operations with preset retry configuration"""
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
        retryable = (PlaywrightTimeoutError, PlaywrightError, TimeoutError)
    except ImportError:
        retryable = (TimeoutError,)

    return retry(
        max_attempts=3,
        backoff_seconds=[2, 5, 10],
        retryable_exceptions=retryable
    )(func)


def retry_network_operation(func: Callable) -> Callable:
    """Decorator for network operations with preset retry configuration"""
    return retry(
        max_attempts=5,
        backoff_seconds=[1, 2, 5, 10, 30],
        retryable_exceptions=(ConnectionError, TimeoutError, OSError)
    )(func)


if __name__ == "__main__":
    # Test the retry manager
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=== Retry Manager Test ===\n")

    # Test 1: Successful retry after failures
    attempt_count = [0]

    @retry(max_attempts=3, backoff_seconds=[0.1, 0.2, 0.5])
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError(f"Failed attempt {attempt_count[0]}")
        return "Success!"

    try:
        result = flaky_function()
        print(f"Test 1 PASSED: {result} (after {attempt_count[0]} attempts)")
    except Exception as e:
        print(f"Test 1 FAILED: {e}")

    # Test 2: Maximum attempts exhausted
    @retry(max_attempts=2, backoff_seconds=[0.1, 0.2])
    def always_fails():
        raise TimeoutError("Always fails")

    try:
        always_fails()
        print("Test 2 FAILED: Should have raised exception")
    except TimeoutError:
        print("Test 2 PASSED: Correctly raised exception after max attempts")

    # Test 3: Nonretryable exception
    @retry(max_attempts=3, retryable_exceptions=(ConnectionError,))
    def raises_nonretryable():
        raise ValueError("Non-retryable error")

    try:
        raises_nonretryable()
        print("Test 3 FAILED: Should have raised exception")
    except ValueError:
        print("Test 3 PASSED: Correctly raised non-retryable exception immediately")

    print("\n=== Test Complete ===")
