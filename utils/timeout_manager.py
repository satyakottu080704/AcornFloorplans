"""
Timeout Manager - Centralized timeout configuration management

This module provides a single source of truth for all timeout values used throughout
the application, replacing hardcoded values with configurable settings.

Usage:
    from utils.timeout_manager import timeout_mgr

    # Get timeout values
    api_timeout = timeout_mgr.get_timeout("api", "default_timeout")

    # Get wait times
    time.sleep(timeout_mgr.get_wait_time("after_login"))
"""

import json
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class TimeoutManager:
    """
    Manages timeout and wait time configurations with fallback to defaults

    Configuration is loaded from config/timeouts_config.json with graceful
    fallback to hardcoded defaults if config file is missing or invalid.
    """

    # Default fallback values (used if config file not found)
    DEFAULT_CONFIG = {
        "api": {
            "default_timeout": 30,
            "long_operation_timeout": 60,
            "batch_operation_timeout": 120
        },
        "browser": {
            "page_load": 60000,
            "element_visibility": 30000,
            "navigation": 45000
        },
        "wait_times": {
            "after_login": 3,
            "after_navigation": 5,
            "before_save": 2,
            "button_click": 1,
            "form_fill": 0.5,
            "modal_open": 2
        },
        "ai": {
            "image_analysis": 60,
            "spell_check": 30,
            "confidence_scoring": 120
        },
        "retry": {
            "max_attempts": 3,
            "initial_delay": 1,
            "backoff_multiplier": 2,
            "max_delay": 30
        }
    }

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize TimeoutManager

        Args:
            config_path: Path to timeout configuration JSON file.
                        Defaults to "config/timeouts_config.json"
        """
        if config_path is None:
            # Try to find config relative to project root
            config_path = Path(__file__).parent.parent / "config" / "timeouts_config.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load configuration from JSON file with fallback to defaults

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            logger.warning(
                f"Timeout config not found at {self.config_path}. "
                f"Using default values."
            )
            return self.DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded timeout configuration from {self.config_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(
                f"Error parsing timeout config at {self.config_path}: {e}. "
                f"Using default values."
            )
            return self.DEFAULT_CONFIG.copy()
        except Exception as e:
            logger.error(
                f"Error loading timeout config: {e}. Using default values."
            )
            return self.DEFAULT_CONFIG.copy()

    def reload(self) -> bool:
        """
        Reload configuration from file

        Returns:
            True if reload successful, False otherwise
        """
        try:
            self.config = self._load_config()
            logger.info("Timeout configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading timeout config: {e}")
            return False

    def get_timeout(
        self,
        category: str,
        operation: str,
        default: Optional[float] = None
    ) -> float:
        """
        Get timeout value for a specific category and operation

        Args:
            category: Timeout category (e.g., "api", "browser", "ai")
            operation: Specific operation (e.g., "default_timeout", "page_load")
            default: Fallback value if not found in config

        Returns:
            Timeout value in seconds (for API/AI) or milliseconds (for browser)

        Examples:
            >>> timeout_mgr.get_timeout("api", "default_timeout")
            30
            >>> timeout_mgr.get_timeout("browser", "page_load")
            60000
            >>> timeout_mgr.get_timeout("ai", "image_analysis", default=60)
            60
        """
        category_config = self.config.get(category, {})

        # Try to get the value, fall back to default
        value = category_config.get(operation)

        if value is None:
            if default is not None:
                return default
            # Try to get from DEFAULT_CONFIG
            value = self.DEFAULT_CONFIG.get(category, {}).get(operation)

        if value is None:
            logger.warning(
                f"Timeout not found: {category}.{operation}. "
                f"Using fallback: {default if default is not None else 30}"
            )
            return default if default is not None else 30

        return value

    def get_wait_time(self, operation: str, default: float = 1.0) -> float:
        """
        Get wait time for a specific operation

        This is a convenience method for getting "wait_times" category values.

        Args:
            operation: Operation name (e.g., "after_login", "before_save")
            default: Fallback value if not found

        Returns:
            Wait time in seconds

        Examples:
            >>> timeout_mgr.get_wait_time("after_login")
            3
            >>> timeout_mgr.get_wait_time("custom_wait", default=2.5)
            2.5
        """
        return self.get_timeout("wait_times", operation, default)

    def get_retry_config(self) -> dict:
        """
        Get retry configuration

        Returns:
            Dictionary with retry settings (max_attempts, initial_delay, etc.)

        Example:
            >>> config = timeout_mgr.get_retry_config()
            >>> print(config['max_attempts'])
            3
        """
        return self.config.get("retry", self.DEFAULT_CONFIG["retry"].copy())

    def get_all_timeouts(self, category: str) -> dict:
        """
        Get all timeouts for a specific category

        Args:
            category: Category name (e.g., "api", "browser", "ai")

        Returns:
            Dictionary of all timeouts in that category

        Example:
            >>> api_timeouts = timeout_mgr.get_all_timeouts("api")
            >>> print(api_timeouts)
            {"default_timeout": 30, "long_operation_timeout": 60, ...}
        """
        return self.config.get(category, {})

    def set_timeout(
        self,
        category: str,
        operation: str,
        value: float,
        persist: bool = False
    ) -> bool:
        """
        Set a timeout value at runtime

        Args:
            category: Timeout category
            operation: Operation name
            value: New timeout value
            persist: If True, save to config file (default: False)

        Returns:
            True if successful, False otherwise

        Example:
            >>> timeout_mgr.set_timeout("api", "default_timeout", 45)
            True
        """
        if category not in self.config:
            self.config[category] = {}

        self.config[category][operation] = value
        logger.info(f"Set timeout: {category}.{operation} = {value}")

        if persist:
            return self._save_config()

        return True

    def _save_config(self) -> bool:
        """
        Save current configuration to file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved timeout configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving timeout config: {e}")
            return False


# Global singleton instance for convenience
timeout_mgr = TimeoutManager()


# Convenience functions for common operations
def get_api_timeout(operation: str = "default_timeout", default: float = 30) -> float:
    """Get API timeout value"""
    return timeout_mgr.get_timeout("api", operation, default)


def get_browser_timeout(operation: str = "page_load", default: float = 60000) -> float:
    """Get browser timeout value (in milliseconds)"""
    return timeout_mgr.get_timeout("browser", operation, default)


def get_wait_time(operation: str, default: float = 1.0) -> float:
    """Get wait time value (in seconds)"""
    return timeout_mgr.get_wait_time(operation, default)


def get_ai_timeout(operation: str = "image_analysis", default: float = 60) -> float:
    """Get AI operation timeout value"""
    return timeout_mgr.get_timeout("ai", operation, default)


if __name__ == "__main__":
    # Test the timeout manager
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=== Timeout Manager Test ===\n")

    # Test getting timeouts
    print(f"API default timeout: {get_api_timeout()}")
    print(f"Browser page_load timeout: {get_browser_timeout()}")
    print(f"Wait time after_login: {get_wait_time('after_login')}")
    print(f"AI image_analysis timeout: {get_ai_timeout()}")

    # Test retry config
    retry_config = timeout_mgr.get_retry_config()
    print(f"\nRetry configuration: {retry_config}")

    # Test getting all timeouts for a category
    all_api_timeouts = timeout_mgr.get_all_timeouts("api")
    print(f"\nAll API timeouts: {all_api_timeouts}")

    # Test fallback for missing value
    custom_timeout = timeout_mgr.get_timeout("custom_category", "custom_operation", default=10)
    print(f"\nCustom timeout (fallback): {custom_timeout}")

    print("\n=== Test Complete ===")
