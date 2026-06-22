"""
Configuration Manager - Hot-reload configuration management with file watching

This module provides centralized configuration management with automatic
hot-reload when configuration files change.

Usage:
    from utils.config_manager import config_manager

    # Get configuration values
    timeout = config_manager.get("timeouts_config", "api.default_timeout")

    # Reload configuration
    config_manager.reload("timeouts_config")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import threading

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None
    print("watchdog not available - hot-reload disabled")


if WATCHDOG_AVAILABLE:
    class ConfigFileHandler(FileSystemEventHandler):
        """Handles configuration file change events"""

        def __init__(self, config_manager):
            self.config_manager = config_manager
            super().__init__()

        def on_modified(self, event):
            """Called when a file is modified"""
            if event.is_directory:
                return

            if event.src_path.endswith('.json'):
                file_path = Path(event.src_path)
                config_name = file_path.stem

                logger.info(f"Config file changed: {config_name}, reloading...")
                self.config_manager._load_config_file(file_path)
else:
    class ConfigFileHandler:
        """Dummy handler when watchdog not available"""
        def __init__(self, config_manager):
            pass


class ConfigManager:
    """
    Centralized configuration manager with hot-reload support

    Singleton instance that loads all JSON configuration files from the config
    directory and watches for changes to automatically reload them.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager"""
        if not hasattr(self, 'initialized'):
            self.configs = {}
            self.config_dir = self._find_config_dir()
            self.observers = {}
            self.initialized = True
            self._load_all_configs()
            if WATCHDOG_AVAILABLE:
                self._start_watchers()

    def _find_config_dir(self) -> Path:
        """Find config directory relative to project root"""
        # Try to find config directory
        current = Path(__file__).parent.parent
        config_dir = current / "config"

        if not config_dir.exists():
            logger.warning(f"Config directory not found at {config_dir}")
            config_dir.mkdir(parents=True, exist_ok=True)

        return config_dir

    def _load_all_configs(self):
        """Load all JSON configuration files"""
        if not self.config_dir.exists():
            logger.warning(f"Config directory does not exist: {self.config_dir}")
            return

        for config_file in self.config_dir.glob("**/*.json"):
            self._load_config_file(config_file)

    def _load_config_file(self, file_path: Path):
        """
        Load a single configuration file

        Args:
            file_path: Path to configuration file
        """
        try:
            with open(file_path, 'r') as f:
                config_name = file_path.stem
                self.configs[config_name] = json.load(f)
                logger.debug(f"Loaded config: {config_name} from {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading config {file_path}: {e}")

    def get(
        self,
        config_name: str,
        key: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Get configuration value

        Args:
            config_name: Name of configuration file (without .json)
            key: Dot-separated key path (e.g., "api.default_timeout")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config_manager.get("timeouts_config")
            {...}  # entire config
            >>> config_manager.get("timeouts_config", "api.default_timeout")
            30
            >>> config_manager.get("timeouts_config", "api.custom", default=60)
            60
        """
        config = self.configs.get(config_name, {})

        if key is None:
            return config if config else default

        # Support nested keys like "browser.page_load"
        keys = key.split('.')
        value = config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(
        self,
        config_name: str,
        key: str,
        value: Any,
        persist: bool = False
    ) -> bool:
        """
        Set configuration value at runtime

        Args:
            config_name: Name of configuration file
            key: Dot-separated key path
            value: Value to set
            persist: If True, save to file (default: False)

        Returns:
            True if successful, False otherwise

        Example:
            >>> config_manager.set("timeouts_config", "api.custom_timeout", 45)
            True
        """
        if config_name not in self.configs:
            self.configs[config_name] = {}

        # Navigate to nested location
        keys = key.split('.')
        config = self.configs[config_name]

        # Create nested structure if needed
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value
        logger.info(f"Set config: {config_name}.{key} = {value}")

        if persist:
            return self._save_config(config_name)

        return True

    def reload(self, config_name: Optional[str] = None) -> bool:
        """
        Reload configuration from file

        Args:
            config_name: Specific config to reload, or None for all

        Returns:
            True if successful, False otherwise
        """
        try:
            if config_name:
                config_file = self.config_dir / f"{config_name}.json"
                if config_file.exists():
                    self._load_config_file(config_file)
                    logger.info(f"Reloaded config: {config_name}")
                else:
                    logger.warning(f"Config file not found: {config_file}")
                    return False
            else:
                self._load_all_configs()
                logger.info("Reloaded all configurations")

            return True
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return False

    def _save_config(self, config_name: str) -> bool:
        """
        Save configuration to file

        Args:
            config_name: Name of configuration to save

        Returns:
            True if successful, False otherwise
        """
        try:
            config_file = self.config_dir / f"{config_name}.json"
            with open(config_file, 'w') as f:
                json.dump(self.configs[config_name], f, indent=2)
            logger.info(f"Saved config to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config {config_name}: {e}")
            return False

    def _start_watchers(self):
        """Start file watchers for hot-reload"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available - file watching disabled")
            return

        try:
            event_handler = ConfigFileHandler(self)
            observer = Observer()
            observer.schedule(
                event_handler,
                str(self.config_dir),
                recursive=True
            )
            observer.start()
            self.observers['main'] = observer
            logger.info(f"Started config file watcher on {self.config_dir}")
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")

    def stop_watchers(self):
        """Stop all file watchers"""
        for name, observer in self.observers.items():
            observer.stop()
            observer.join()
            logger.info(f"Stopped config file watcher: {name}")
        self.observers.clear()

    def list_configs(self) -> list:
        """Get list of all loaded configuration names"""
        return list(self.configs.keys())

    def has_config(self, config_name: str) -> bool:
        """Check if configuration exists"""
        return config_name in self.configs

    def get_all(self, config_name: str) -> Dict:
        """Get entire configuration dictionary"""
        return self.configs.get(config_name, {}).copy()


# Global singleton instance
config_manager = ConfigManager()


# Convenience functions
def get_config(config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(config_name, key, default)


def reload_config(config_name: Optional[str] = None) -> bool:
    """Reload configuration"""
    return config_manager.reload(config_name)


def set_config(config_name: str, key: str, value: Any, persist: bool = False) -> bool:
    """Set configuration value"""
    return config_manager.set(config_name, key, value, persist)


if __name__ == "__main__":
    # Test the configuration manager
    import sys
    import time
    logging.basicConfig(level=logging.INFO)

    print("=== Configuration Manager Test ===\n")

    # Test loading configs
    print("Loaded configurations:")
    for config_name in config_manager.list_configs():
        print(f"  - {config_name}")

    # Test getting values
    print("\nTest getting values:")
    timeout = config_manager.get("timeouts_config", "api.default_timeout", default=30)
    print(f"  API timeout: {timeout}")

    selectors = config_manager.get("selectors_config", "login.username")
    print(f"  Login username selectors: {selectors}")

    # Test setting values
    print("\nTest setting runtime value:")
    config_manager.set("timeouts_config", "api.custom_timeout", 45)
    custom = config_manager.get("timeouts_config", "api.custom_timeout")
    print(f"  Custom timeout: {custom}")

    # Test reload
    print("\nTest reload:")
    success = config_manager.reload("timeouts_config")
    print(f"  Reload successful: {success}")

    # Test hot-reload (if watchdog available)
    if WATCHDOG_AVAILABLE:
        print("\nHot-reload is enabled")
        print("  Modify a config file to see automatic reload")
        print("  (Watching for 5 seconds...)")
        time.sleep(5)
    else:
        print("\nHot-reload is disabled (watchdog not installed)")

    print("\n=== Test Complete ===")
