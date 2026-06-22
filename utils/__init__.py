"""
Utils package - Configuration, AI features, and helper functions for plan generation.
"""

# Lazy imports only - modules are imported on demand by plans/ code
# This avoids circular imports and missing-module errors at startup.

__all__ = [
    "config",
    "config_manager",
    "retry_manager",
    "timeout_manager",
    "room_detection",
    "visio",
]

from . import config_automation as config
import sys
sys.modules['utils.config'] = config


