"""
Alpha Tracker API functions.
Uses the local api/alphatracker_api implementation.
"""

import os
from typing import Dict, Any, Optional, List

# Import from local implementation
from .alphatracker_api import (
    AlphaTrackerAPI,
    Config,
    APIResponse,
    # Date utilities
    get_uk_bank_holidays,
    is_uk_bank_holiday,
    calculate_working_days,
    count_working_days_between,
    format_date,
    convert_to_iso_date,
    parse_date,
    parse_time_range,
    # Address utilities
    extract_site_name_and_address,
    parse_uk_postcode,
)

# Create a singleton API instance
_api_instance: Optional[AlphaTrackerAPI] = None


def _get_api() -> AlphaTrackerAPI:
    """Get or create API singleton instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = AlphaTrackerAPI.from_env()
    return _api_instance


# =============================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility with existing code)
# =============================================================================

def get_project(project_number: str) -> Optional[Dict[str, Any]]:
    """Get project details."""
    return _get_api().get_project(project_number)


def get_survey_items(project_number: str) -> List[Dict[str, Any]]:
    """Get all survey items for a project."""
    return _get_api().get_survey_items(project_number)


def get_files(project_number: str) -> List[Dict[str, Any]]:
    """Get all files for a project."""
    return _get_api().get_project_files(project_number)


def download_file(project_number: str, filename: str, save_path: Optional[str] = None) -> Optional[str]:
    """
    Download a file from project folder.

    Args:
        project_number: Project number (e.g., N-98813)
        filename: Name of file to download
        save_path: Optional path to save file (otherwise saves to temp directory)

    Returns:
        Path to downloaded file, or None if failed
    """
    import tempfile

    content = _get_api().download_file(project_number, filename)
    if not content:
        return None

    # Determine save location
    if save_path:
        file_path = save_path
    else:
        # Save to temp directory
        temp_dir = tempfile.gettempdir()
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        file_path = os.path.join(temp_dir, safe_filename)

    # Write file
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to save file: {e}")
        return None


def upload_file(project_number: str, file_path: str, file_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a file to project folder.

    Args:
        project_number: Project number (e.g., N-98813)
        file_path: Local file path to upload
        file_type: Optional file type label

    Returns:
        Dict with success status, error message, and data
    """
    response = _get_api().upload_file(project_number, file_path, file_type=file_type)
    return {"success": response.success, "error": response.error, "data": response.data}


def get_milestones(project_number: str) -> List[Dict[str, Any]]:
    """Get milestones for a project."""
    return _get_api().get_milestones(project_number) or []


def get_buildings(project_number: str) -> List[Dict[str, Any]]:
    """Get buildings for a project."""
    return _get_api().get_buildings(project_number)


def get_site(site_id: int) -> Optional[Dict[str, Any]]:
    """Get site details by site ID."""
    return _get_api().get_site(site_id)


def get_survey_item(survey_item_id: int) -> Optional[Dict[str, Any]]:
    """Get a single survey item by ID."""
    return _get_api().get_survey_item(survey_item_id)


def get_samples(project_number: str) -> List[Dict[str, Any]]:
    """Get samples for a project."""
    return _get_api().get_samples(project_number)


def get_appointments(project_number: str) -> List[Dict[str, Any]]:
    """Get appointments for a project."""
    return _get_api().get_appointments(project_number)


def get_staff() -> List[Dict[str, Any]]:
    """Get all staff members."""
    return _get_api().get_staff()


def find_staff_id(name: str) -> Optional[int]:
    """Find staff ID by name."""
    return _get_api().find_staff_id(name)


def get_staff_email(name: str) -> Optional[str]:
    """Get staff email by name."""
    return _get_api().get_staff_email(name)


def get_staff_by_id(staff_id: int) -> Optional[Dict[str, Any]]:
    """Get staff member by ID. Returns dict with staffId, name, emailAddress, etc."""
    return _get_api().get_staff_by_id(staff_id)


def update_project(project_number: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Update project fields in Alpha Tracker."""
    response = _get_api().update_project(project_number, data)
    return {"success": response.success, "error": response.error, "data": response.data}


def update_survey_item(project_number: str, survey_item_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a survey item in Alpha Tracker.

    Args:
        project_number: Project number (e.g., "T-78473")
        survey_item_id: Survey item ID to update
        data: Dict of fields to update (e.g., {"notes": "new notes"})

    Returns:
        Dict with success status, error message, and data
    """
    api = _get_api()
    response = api.update_survey_item(project_number, survey_item_id, data)
    return {"success": response.success, "error": response.error, "data": response.data}


def search_projects(filter_query: str, limit: int = 1000, page: int = 1,
                   due_date_to: str = None) -> List[Dict[str, Any]]:
    """Search projects with filter."""
    response = _get_api().search_projects(filter=filter_query, limit=limit, page=page)
    if response.success and response.data:
        return response.data.get("data", []) if isinstance(response.data, dict) else response.data
    return []


def search_all_projects(filter_query: str, max_pages: int = 50) -> List[Dict[str, Any]]:
    """Search and retrieve ALL matching projects (handles pagination)."""
    return _get_api().search_all_projects(filter=filter_query, max_pages=max_pages)


def create_project(project_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new project."""
    response = _get_api().create_project(project_data)
    return {"success": response.success, "error": response.error, "data": response.data}


def create_appointment(project_number: str, staff_id: int, start_time: str,
                       end_time: str, **kwargs) -> Dict[str, Any]:
    """Create a diary appointment."""
    response = _get_api().create_appointment(
        project_number=project_number,
        staff_id=staff_id,
        start_time=start_time,
        end_time=end_time,
        **kwargs
    )
    return {"success": response.success, "error": response.error, "data": response.data}


def test_connection() -> tuple:
    """Test API connection."""
    return _get_api().test_connection()


# =============================================================================
# EXPOSE API CONFIG (for backward compatibility)
# =============================================================================

# Get config from the API instance
_config = Config.from_env()
BASE_URL = _config.base_url
API_KEY = _config.api_key
CLIENT_ID = _config.client_id
HEADERS = _config.headers


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core API class
    "AlphaTrackerAPI",
    "Config",
    "APIResponse",
    # Convenience functions
    "get_project",
    "get_survey_items",
    "get_files",
    "download_file",
    "upload_file",
    "get_milestones",
    "get_buildings",
    "get_site",
    "get_survey_item",
    "get_samples",
    "get_appointments",
    "get_staff",
    "find_staff_id",
    "get_staff_email",
    "update_project",
    "update_survey_item",
    "search_projects",
    "search_all_projects",
    "create_project",
    "create_appointment",
    "test_connection",
    # Date utilities
    "get_uk_bank_holidays",
    "is_uk_bank_holiday",
    "calculate_working_days",
    "count_working_days_between",
    "format_date",
    "convert_to_iso_date",
    "parse_date",
    "parse_time_range",
    # Address utilities
    "extract_site_name_and_address",
    "parse_uk_postcode",
    # Config values (backward compatibility)
    "BASE_URL",
    "API_KEY",
    "CLIENT_ID",
    "HEADERS",
]
