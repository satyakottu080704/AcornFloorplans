"""
API package - Alpha Tracker API functions.
Uses the local api/alphatracker_api implementation.
"""

from .alpha_tracker import (
    # Core API class
    AlphaTrackerAPI,
    Config,
    APIResponse,
    # Convenience functions
    get_project,
    get_survey_items,
    get_files,
    upload_file,
    download_file,
    get_milestones,
    get_buildings,
    get_site,
    get_survey_item,
    get_samples,
    get_appointments,
    get_staff,
    find_staff_id,
    get_staff_email,
    get_staff_by_id,
    update_project,
    update_survey_item,
    search_projects,
    search_all_projects,
    create_project,
    create_appointment,
    test_connection,
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
    # Config values (backward compatibility)
    BASE_URL,
    API_KEY,
    CLIENT_ID,
    HEADERS,
)

__all__ = [
    # Core API class
    "AlphaTrackerAPI",
    "Config",
    "APIResponse",
    # Convenience functions
    "get_project",
    "get_survey_items",
    "get_files",
    "upload_file",
    "download_file",
    "get_milestones",
    "get_buildings",
    "get_site",
    "get_survey_item",
    "get_samples",
    "get_appointments",
    "get_staff",
    "find_staff_id",
    "get_staff_email",
    "get_staff_by_id",
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
