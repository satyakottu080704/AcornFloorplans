"""
Configuration loader for Acorn Survey Validation System.
Uses config_manager as the single source of truth when possible.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .config_manager import config_manager

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

_config_override: Optional[Dict[str, Any]] = None
_config_path: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file or config_manager."""
    global _config_override, _config_path
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            _config_override = json.load(f)
        _config_path = config_path
        return _config_override
    _config_override = None
    _config_path = None
    return config_manager.get("validation_config", default={})


def get_config() -> Dict[str, Any]:
    """Get the full configuration (override if loaded, else config_manager)."""
    if _config_override is not None:
        return _config_override
    return config_manager.get("validation_config", default={})


def reload_config() -> bool:
    """Force reload configuration from file (clears override)."""
    global _config_override, _config_path
    if _config_path:
        # Reload explicit override file
        if not os.path.exists(_config_path):
            raise FileNotFoundError(f"Config file not found: {_config_path}")
        with open(_config_path, "r", encoding="utf-8") as f:
            _config_override = json.load(f)
        return True
    _config_override = None
    return config_manager.reload("validation_config")


def get_api_config() -> Dict[str, Any]:
    """Get API configuration - Environment variables take priority over config file."""
    api = get_config().get("api", {})
    return {
        "base_url": os.environ.get("ALPHA_TRACKER_BASE_URL") or api.get("base_url") or "https://manager.alphatracker.co.uk/api",
        "api_key": os.environ.get("ALPHA_TRACKER_API_KEY") or api.get("api_key") or "",
        "client_id": os.environ.get("ALPHA_TRACKER_CLIENT_ID") or api.get("client_id") or "",
        "timeout": api.get("timeout", 30),
        "max_retries": api.get("max_retries", 3)
    }


def _resolve_env(value: str, env_key: str, default: str = "") -> str:
    """Resolve a config value that may be a ${VAR} placeholder.

    Priority: env var > config value (if not a placeholder) > default.
    """
    env_val = os.environ.get(env_key, "")
    if env_val:
        return env_val
    # If the config value is a ${...} placeholder, treat it as unset
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return default
    return value or default


def get_browser_config() -> Dict[str, Any]:
    """Get browser automation configuration - Environment variables take priority."""
    browser = get_config().get("browser", {})
    return {
        "web_url": _resolve_env(browser.get("web_url", ""), "ALPHA_TRACKER_WEB_URL", "https://acorn-as.co.uk:82"),
        "web_username": _resolve_env(browser.get("web_username", ""), "ALPHA_TRACKER_WEB_USERNAME"),
        "web_password": _resolve_env(browser.get("web_password", ""), "ALPHA_TRACKER_WEB_PASSWORD"),
        "headless": browser.get("headless", False),
        "incognito": browser.get("incognito", False),
        "download_dir": browser.get("download_dir", "./downloads"),
        "enabled": browser.get("enabled", True)
    }


def get_check_config(check_name: str) -> Dict[str, Any]:
    """Get configuration for a specific check (000, 999, samples, etc.)."""
    return get_config().get("checks", {}).get(check_name, {})


def get_client_rules() -> Dict[str, Any]:
    """Get client-specific rules."""
    return get_config().get("checks", {}).get("client", {}).get("rules", {})


def get_notification_config() -> Dict[str, Any]:
    """Get notification configuration."""
    return get_config().get("notifications", {})


def get_project_filter() -> str:
    """Get default project filter."""
    return get_config().get("project_filters", {}).get("default_filter", "")


def is_check_enabled(check_name: str) -> bool:
    """Check if a specific check is enabled."""
    return get_check_config(check_name).get("enabled", True)


def get_keywords(check_name: str, keyword_type: str) -> List[str]:
    """Get keywords for a check (e.g., get_keywords('000', 'building_type_keywords'))."""
    return get_check_config(check_name).get(keyword_type, [])
