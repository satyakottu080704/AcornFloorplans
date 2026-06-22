#!/usr/bin/env python3
"""
AlphaTracker API - Complete Single-File Module
===============================================

A comprehensive, self-contained module for the AlphaTracker REST API.
All features in one file for easy reuse across different projects.

FEATURES:
---------
- Full API coverage (all endpoints from AlphaTracker API v1.0.0)
- Configuration management (env vars, .env files, dict)
- Connection pooling for performance
- UK date utilities (bank holidays, working days)
- UK address parsing
- Staff lookup with fuzzy matching
- Automatic pagination handling

API ENDPOINTS COVERED:
----------------------
✓ Projects     - CRUD, Search, Pagination
✓ Sites        - CRUD, Search, Find by reference
✓ Appointments - CRUD (diary entries)
✓ Milestones   - Get project milestones, due dates
✓ Staff        - List, lookup by name/email
✓ Samples      - Get/Create samples
✓ Survey Items - Get survey data
✓ Buildings    - Project buildings
✓ Site Buildings - Site-level buildings
✓ Files        - List, download, bulk certificates
✓ Reports      - List, run reports
✓ Repair Instructions - Search, get details
✓ Reinspection - Send to handset
✓ Project Types - List available types

USAGE:
------
    from alphatracker_api import AlphaTrackerAPI, Config
    
    # Load config from environment/.env
    config = Config.from_env()
    api = AlphaTrackerAPI(config)
    
    # Or quick setup
    api = AlphaTrackerAPI.from_env()
    
    # Get project
    project = api.get_project("N-12345")
    
    # Search projects
    projects = api.search_all_projects("status = 'Scheduled' AND clientId = 'GUHG'")
    
    # Create project
    result = api.create_project({
        "clientId": "GUHG",
        "siteId": 12345,
        "status": "Scheduled",
    })
    
    # Create appointment
    api.create_appointment("N-12345", staff_id=123, 
                          start_time="2025-01-20T09:00:00",
                          end_time="2025-01-20T12:00:00")

ENVIRONMENT VARIABLES:
----------------------
    ALPHA_TRACKER_BASE_URL=https://manager.alphatracker.co.uk/api
    ALPHA_TRACKER_API_KEY=your-api-key
    ALPHA_TRACKER_CLIENT_ID=your-client-id

Author: Acorn Analytical Services
Version: 2.0.0
"""

import os
import re
import sys
import math
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure src/ is on sys.path so utils.* imports resolve correctly
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Import retry logic for API call resilience
from utils.retry_manager import retry

__version__ = "2.0.0"
__all__ = [
    "AlphaTrackerAPI",
    "Config", 
    "APIResponse",
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
]

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _load_env_file(env_paths: List[Path] = None) -> bool:
    """
    Load .env file into environment variables.
    
    Args:
        env_paths: List of paths to check for .env file
        
    Returns:
        True if .env file was loaded
    """
    if env_paths is None:
        env_paths = [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
            Path.home() / ".acorn" / ".env",
        ]
    
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            os.environ.setdefault(key, value)
                return True
            except Exception:
                pass
    return False


@dataclass
class Config:
    """
    AlphaTracker API Configuration.
    
    Attributes:
        base_url: API base URL
        api_key: API key for authentication
        client_id: Client ID for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """
    base_url: str = "https://manager.alphatracker.co.uk/api"
    api_key: str = ""
    client_id: str = ""
    timeout: int = 30
    max_retries: int = 2
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "x-api-key": self.api_key,
            "client-id": self.client_id,
            "accept": "application/json",
            "Content-Type": "application/json",
        }
    
    def validate(self) -> bool:
        """Check if required credentials are set."""
        return bool(self.api_key and self.client_id)
    
    @classmethod
    def from_env(cls, load_dotenv: bool = True) -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            load_dotenv: If True, attempt to load .env file first
            
        Returns:
            Config instance
        """
        if load_dotenv:
            _load_env_file()
        
        return cls(
            base_url=os.environ.get("ALPHA_TRACKER_BASE_URL", "https://manager.alphatracker.co.uk/api"),
            api_key=os.environ.get("ALPHA_TRACKER_API_KEY", ""),
            client_id=os.environ.get("ALPHA_TRACKER_CLIENT_ID", ""),
            timeout=int(os.environ.get("ALPHA_TRACKER_TIMEOUT", "30")),
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary."""
        return cls(
            base_url=data.get("base_url", data.get("ALPHA_TRACKER_BASE_URL", "https://manager.alphatracker.co.uk/api")),
            api_key=data.get("api_key", data.get("ALPHA_TRACKER_API_KEY", "")),
            client_id=data.get("client_id", data.get("ALPHA_TRACKER_CLIENT_ID", "")),
            timeout=data.get("timeout", 30),
        )


@dataclass
class APIResponse:
    """Standardized API response wrapper."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    
    def __bool__(self):
        return self.success


# =============================================================================
# DATE UTILITIES - UK Bank Holidays & Working Days
# =============================================================================

def get_uk_bank_holidays(year: int) -> Set[datetime]:
    """
    Get UK (England & Wales) bank holidays for a given year.
    
    Includes:
        - New Year's Day (+ substitute if weekend)
        - Good Friday & Easter Monday
        - Early May Bank Holiday (first Monday)
        - Spring Bank Holiday (last Monday of May)
        - Summer Bank Holiday (last Monday of August)
        - Christmas Day & Boxing Day (+ substitutes)
    
    Args:
        year: Year to get holidays for
        
    Returns:
        Set of datetime objects representing bank holidays
    """
    holidays = set()
    
    # New Year's Day
    new_year = datetime(year, 1, 1)
    holidays.add(new_year)
    if new_year.weekday() == 5:  # Saturday -> Monday
        holidays.add(datetime(year, 1, 3))
    elif new_year.weekday() == 6:  # Sunday -> Monday
        holidays.add(datetime(year, 1, 2))
    
    # Christmas Day & Boxing Day
    christmas = datetime(year, 12, 25)
    boxing = datetime(year, 12, 26)
    holidays.add(christmas)
    holidays.add(boxing)
    
    if christmas.weekday() == 5:  # Saturday
        holidays.add(datetime(year, 12, 27))
        holidays.add(datetime(year, 12, 28))
    elif christmas.weekday() == 6:  # Sunday
        holidays.add(datetime(year, 12, 27))
        holidays.add(datetime(year, 12, 28))
    elif boxing.weekday() == 6:  # Boxing Day on Sunday
        holidays.add(datetime(year, 12, 28))
    
    # Easter (Anonymous Gregorian algorithm)
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter_sunday = datetime(year, month, day)
    
    holidays.add(easter_sunday - timedelta(days=2))  # Good Friday
    holidays.add(easter_sunday + timedelta(days=1))  # Easter Monday
    
    # Early May Bank Holiday (first Monday of May)
    may_first = datetime(year, 5, 1)
    days_until_monday = (7 - may_first.weekday()) % 7
    if may_first.weekday() == 0:
        days_until_monday = 0
    holidays.add(may_first + timedelta(days=days_until_monday))
    
    # Spring Bank Holiday (last Monday of May)
    may_last = datetime(year, 5, 31)
    holidays.add(may_last - timedelta(days=may_last.weekday()))
    
    # Summer Bank Holiday (last Monday of August)
    aug_last = datetime(year, 8, 31)
    holidays.add(aug_last - timedelta(days=aug_last.weekday()))
    
    return holidays


def is_uk_bank_holiday(check_date: Union[datetime, str]) -> bool:
    """Check if a date is a UK bank holiday."""
    if isinstance(check_date, str):
        check_date = parse_date(check_date)
        if not check_date:
            return False
    
    holidays = get_uk_bank_holidays(check_date.year)
    holidays.update(get_uk_bank_holidays(check_date.year + 1))
    
    check_date_only = datetime(check_date.year, check_date.month, check_date.day)
    return check_date_only in holidays


def parse_date(date_value: Union[str, datetime, float, int, None]) -> Optional[datetime]:
    """
    Parse date from various formats to datetime.
    
    Supports: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, datetime, Excel serial numbers
    """
    if not date_value:
        return None
    
    if isinstance(date_value, datetime):
        return date_value
    
    date_str = str(date_value).strip()
    
    # Try common formats (UK format first)
    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d %b %Y', '%d %B %Y']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try Excel serial number
    try:
        serial = float(date_str)
        if 1 < serial < 100000:
            return datetime(1899, 12, 30) + timedelta(days=serial)
    except (ValueError, TypeError):
        pass
    
    return None


def format_date(date_value: Union[str, datetime, float, int, None], 
                output_format: str = "%d/%m/%Y") -> str:
    """Format any date value to specified format (default: DD/MM/YYYY)."""
    if not date_value:
        return ""
    
    parsed = parse_date(date_value)
    if parsed:
        return parsed.strftime(output_format)
    
    date_str = str(date_value)
    if "/" in date_str or "-" in date_str:
        return date_str
    
    return ""


def convert_to_iso_date(date_value: Union[str, datetime, float, int, None]) -> str:
    """Convert any date value to ISO format (YYYY-MM-DD)."""
    return format_date(date_value, "%Y-%m-%d")


def calculate_working_days(start_date: Union[str, datetime, float, int], 
                           days: int = 10) -> str:
    """
    Calculate date that is N working days from start date.
    Excludes weekends and UK bank holidays.
    
    Args:
        start_date: Start date in any format
        days: Number of working days to add (default 10)
        
    Returns:
        Result date in DD/MM/YYYY format
    """
    if not start_date:
        return ""
    
    try:
        if isinstance(start_date, datetime):
            current_date = start_date
        else:
            current_date = parse_date(start_date)
            if not current_date:
                return str(start_date)
        
        bank_holidays = get_uk_bank_holidays(current_date.year)
        bank_holidays.update(get_uk_bank_holidays(current_date.year + 1))
        
        days_added = 0
        while days_added < days:
            current_date += timedelta(days=1)
            is_weekend = current_date.weekday() >= 5
            is_holiday = datetime(current_date.year, current_date.month, current_date.day) in bank_holidays
            
            if not is_weekend and not is_holiday:
                days_added += 1
        
        return current_date.strftime('%d/%m/%Y')
        
    except Exception as e:
        logger.warning(f"Error calculating working days: {e}")
        return str(start_date) if start_date else ""


def count_working_days_between(start_date: Union[str, datetime], 
                                end_date: Union[str, datetime]) -> int:
    """Count working days between two dates (exclusive of start, inclusive of end)."""
    if not start_date or not end_date:
        return 0
    
    try:
        start = parse_date(start_date) if isinstance(start_date, str) else start_date
        end = parse_date(end_date) if isinstance(end_date, str) else end_date
        
        if not start or not end:
            return 0
        
        bank_holidays = get_uk_bank_holidays(start.year)
        bank_holidays.update(get_uk_bank_holidays(end.year))
        
        working_days = 0
        current = start
        
        while current < end:
            current += timedelta(days=1)
            is_weekend = current.weekday() >= 5
            is_holiday = datetime(current.year, current.month, current.day) in bank_holidays
            
            if not is_weekend and not is_holiday:
                working_days += 1
        
        return working_days
        
    except Exception:
        return 0


def parse_time_range(time_str: str) -> Tuple[int, int]:
    """
    Parse time of attendance string to start and end hours.
    
    Examples: "9-12" -> (9, 12), "AM" -> (9, 12), "PM" -> (13, 17)
    """
    if not time_str:
        return (9, 17)
    
    time_str = time_str.lower().strip()
    
    if time_str == "am":
        return (9, 12)
    if time_str == "pm":
        return (13, 17)
    if time_str in ["anytime", "any time", "flexible", "tbc", "all day"]:
        return (9, 17)
    
    match = re.search(r'(\d{1,2})(?::\d{2})?\s*[-–]\s*(\d{1,2})(?::\d{2})?', time_str)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if start < 7:
            start += 12
        if end < 7:
            end += 12
        return (start, end)
    
    return (9, 17)


# =============================================================================
# ADDRESS UTILITIES - UK Address Parsing
# =============================================================================

UK_STREET_TYPES = {
    'road', 'street', 'avenue', 'lane', 'drive', 'way', 'close', 'court',
    'place', 'crescent', 'gardens', 'terrace', 'grove', 'park', 'rise',
    'walk', 'square', 'row', 'hill', 'green', 'view', 'meadow', 'mews',
    'baulk', 'yard', 'passage', 'alley', 'end', 'gate', 'path', 'circus',
}

UK_TOWNS = {
    'bedford', 'luton', 'biggleswade', 'letchworth', 'hitchin', 'stevenage',
    'london', 'birmingham', 'manchester', 'leeds', 'sheffield', 'liverpool',
    'bristol', 'nottingham', 'leicester', 'coventry', 'cambridge', 'oxford',
    'reading', 'northampton', 'peterborough', 'milton keynes', 'slough',
}


def extract_site_name_and_address(full_address: str, 
                                   postcode: str = "") -> Tuple[str, str]:
    """
    Extract site name (street address) and location (town/city) from full address.
    
    Args:
        full_address: Complete address string
        postcode: Postcode to remove from address (optional)
        
    Returns:
        Tuple of (site_name, location)
    """
    if not full_address:
        return ("", "")
    
    full_address = full_address.strip()
    
    # Remove postcode if provided
    if postcode:
        postcode_clean = postcode.strip()
        full_address = full_address.replace(postcode_clean, "").strip()
        full_address = full_address.replace(postcode_clean.replace(" ", ""), "").strip()
    
    full_address = full_address.rstrip(", ")
    
    # Split by comma or newline
    for delimiter in [',', '\n', '\r']:
        if delimiter in full_address:
            parts = [p.strip() for p in full_address.split(delimiter) if p.strip()]
            if len(parts) >= 2:
                return (parts[0], ", ".join(parts[1:]).strip(", "))
            elif len(parts) == 1:
                full_address = parts[0]
                break
    
    words = full_address.split()
    
    # Detect ALL-CAPS town name at end
    if len(words) >= 2:
        last_word = words[-1]
        if last_word.isupper() and len(last_word) >= 4:
            return (" ".join(words[:-1]), last_word.title())
    
    # Match known town names
    if len(words) >= 2:
        for word_count in [3, 2, 1]:
            if len(words) >= word_count + 1:
                potential_town = " ".join(words[-word_count:]).lower()
                if potential_town in UK_TOWNS:
                    return (" ".join(words[:-word_count]), " ".join(words[-word_count:]).title())
    
    # Split after street type
    for i, word in enumerate(words):
        if word.lower().rstrip('.,;') in UK_STREET_TYPES:
            if i + 1 < len(words):
                return (" ".join(words[:i+1]), " ".join(words[i+1:]))
            break
    
    return (full_address, "")


def parse_uk_postcode(postcode: str) -> Optional[str]:
    """Validate and format UK postcode."""
    if not postcode:
        return None
    
    clean = postcode.strip().upper().replace(" ", "")
    pattern = r'^([A-Z]{1,2}[0-9][0-9A-Z]?)([0-9][A-Z]{2})$'
    match = re.match(pattern, clean)
    
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


# =============================================================================
# ALPHATRACKER API CLIENT
# =============================================================================

class AlphaTrackerAPI:
    """
    AlphaTracker REST API Client.
    
    Complete coverage of all API endpoints with:
    - Connection pooling for performance
    - Automatic retry logic
    - Consistent error handling
    - Response caching where appropriate
    - Pagination handling
    
    Usage:
        api = AlphaTrackerAPI.from_env()
        project = api.get_project("N-12345")
    """
    
    def __init__(self, config: Union[Config, Dict[str, Any]] = None):
        """
        Initialize API client.
        
        Args:
            config: Config instance, dict with credentials, or None to load from env
        """
        if config is None:
            config = Config.from_env()
        elif isinstance(config, dict):
            config = Config.from_dict(config)
        
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.timeout
        
        # Session for connection pooling
        self._session: Optional[requests.Session] = None
        
        # Caches
        self._staff_cache: Optional[List[Dict]] = None
        self._project_types_cache: Optional[List[Dict]] = None
    
    @classmethod
    def from_env(cls) -> "AlphaTrackerAPI":
        """Create API client from environment variables."""
        return cls(Config.from_env())
    
    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session with connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.config.headers)

            # Honor AT_SSL_VERIFY=false (self-hosted AlphaTracker with a
            # self-signed / untrusted cert). Default keeps verification on.
            if os.environ.get("AT_SSL_VERIFY", "true").strip().lower() in ("false", "0", "no"):
                self._session.verify = False
                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except Exception:
                    pass

            adapter = requests.adapters.HTTPAdapter(
                pool_connections=50,
                pool_maxsize=50,
                max_retries=requests.adapters.Retry(
                    total=self.config.max_retries,
                    backoff_factor=0.5,
                    status_forcelist=[500, 502, 503, 504],
                )
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
        
        return self._session
    
    def _request(self, method: str, endpoint: str, 
                 params: Dict = None, json: Dict = None,
                 retries: int = None) -> APIResponse:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"
        retries = retries if retries is not None else self.config.max_retries
        
        for attempt in range(retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    timeout=self.timeout,
                )
                
                if response.status_code in (200, 201):
                    return APIResponse(
                        success=True,
                        data=response.json() if response.content else {"success": True},
                        status_code=response.status_code,
                    )
                elif response.status_code == 404:
                    return APIResponse(success=False, error="Not found", status_code=404)
                elif response.status_code == 401:
                    return APIResponse(success=False, error="Unauthorized - check API credentials", status_code=401)
                elif response.status_code == 400:
                    return APIResponse(success=False, error=response.text[:200], status_code=400)
                else:
                    if attempt < retries:
                        continue
                    return APIResponse(
                        success=False,
                        error=f"HTTP {response.status_code}: {response.text[:100]}",
                        status_code=response.status_code,
                    )
                    
            except requests.exceptions.Timeout:
                if attempt < retries:
                    continue
                return APIResponse(success=False, error="Request timeout")
            except requests.exceptions.ConnectionError as e:
                if attempt < retries:
                    continue
                return APIResponse(success=False, error=f"Connection error: {e}")
            except Exception as e:
                return APIResponse(success=False, error=str(e))
        
        return APIResponse(success=False, error="Max retries exceeded")
    
    # =========================================================================
    # CONNECTION TEST
    # =========================================================================
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test API connection. Returns (success, message)."""
        response = self._request("GET", "/projects/N-99000")
        
        if response.success:
            return (True, "Connection OK")
        elif response.status_code == 404:
            return (True, "Connection OK (test project not found)")
        elif response.status_code == 401:
            return (False, "Authentication failed - check API key")
        elif response.status_code == 403:
            return (False, "Access denied - check client ID")
        else:
            return (False, response.error or "Unknown error")
    
    # =========================================================================
    # PROJECTS - Full CRUD + Search
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_project(self, project_number: str) -> Optional[Dict[str, Any]]:
        """
        Get project details.
        
        Returns project with: projectNumber, clientId, clientName, clientOrderNumber,
        clientProjectRef, siteId, siteName, siteAddress, sitePostcode, siteReference,
        siteContact, landlord, reportRecipients, projectManager, projectType, status,
        statusNotes, allocationStatus, dates (opened, quote, order, report, invoice, closed),
        projectNotes, coordinates, timestamp, showOnWeb
        """
        response = self._request("GET", f"/projects/{project_number}")
        return response.data if response.success else None

    @retry(max_attempts=5, backoff_seconds=[1, 2, 5, 10, 30])
    def create_project(self, project_data: Dict[str, Any]) -> APIResponse:
        """
        Create a new project.
        
        Args:
            project_data: Dict with fields:
                - projectLetter (default "N" for surveys)
                - clientId (required)
                - siteId (required)
                - clientOrderNumber
                - reportRecipientName1-5, reportRecipientEmailAddress1-5
                - invoiceRecipientName
                - projectManagerId
                - projectTypeId
                - estimatedTotalProjectValue
                - status, statusNotes, allocationStatus
                - projectOpened, quoteProduced, orderReceived, reportProduced,
                  projectInvoiced, projectClosed (ISO dates)
                - projectNotes
                - showOnWeb
                
        Returns:
            APIResponse with projectNumber in data
        """
        if "projectLetter" not in project_data:
            project_data["projectLetter"] = "N"

        return self._request("POST", "/projects", json=project_data)

    @retry(max_attempts=5, backoff_seconds=[1, 2, 5, 10, 30])
    def update_project(self, project_number: str, updates: Dict[str, Any]) -> APIResponse:
        """
        Update project fields via PATCH.
        
        Updatable fields: clientId, clientOrderNumber, siteId, reportRecipients,
        invoiceRecipientName, projectManagerId, projectTypeId, estimatedTotalProjectValue,
        status, statusNotes, allocationStatus, dates, projectNotes, showOnWeb
        """
        return self._request("PATCH", f"/projects/{project_number}", json=updates)

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def delete_project(self, project_number: str) -> APIResponse:
        """Delete a project."""
        return self._request("DELETE", f"/projects/{project_number}")

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def search_projects(self, filter: str = None, limit: int = 100,
                        paginate: bool = True, page: int = 1) -> APIResponse:
        """
        Search projects using SQL-like filter.
        
        Filter operators: =, >, <, >=, <=, BETWEEN, LIKE, IN
        Combine with: AND, OR, NOT
        
        Examples:
            "status = 'Scheduled'"
            "projectNumber LIKE 'N-%'"
            "clientId = 'GUHG' AND status = 'Scheduled'"
            "orderReceived BETWEEN '2025-01-01' AND '2025-01-31'"
        """
        params = {
            "filter": filter,
            "limit": limit,
            "paginate": str(paginate).lower(),
            "pageNumber": page,
        }
        return self._request("GET", "/projects/search", params=params)

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def search_all_projects(self, filter: str = None, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Search and retrieve ALL matching projects (handles pagination automatically)."""
        all_projects = []
        page = 1
        
        while page <= max_pages:
            response = self.search_projects(filter=filter, limit=100, page=page)
            
            if not response.success:
                break
            
            data = response.data
            if isinstance(data, dict):
                projects = data.get("data", [])
                pagination = data.get("pagination", {})
            else:
                projects = data if isinstance(data, list) else []
                pagination = {}
            
            all_projects.extend(projects)
            
            total_pages = pagination.get("totalPages", 1)
            if page >= total_pages:
                break
            
            page += 1
        
        return all_projects
    
    # =========================================================================
    # SITES - Full CRUD + Search
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_site(self, site_id: int) -> Optional[Dict[str, Any]]:
        """
        Get site details.
        
        Returns: siteId, clientId, clientName, siteName, siteAddress, sitePostcode,
        siteReference, siteContact, lastInspectionDate, nextInspectionDate,
        inspectionFrequency, status, category, longitude, latitude
        """
        response = self._request("GET", f"/sites/{site_id}")
        return response.data if response.success else None

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def create_site(self, site_data: Dict[str, Any]) -> APIResponse:
        """
        Create a new site.
        
        Args:
            site_data: Dict with fields:
                - clientId (required)
                - siteName (required)
                - siteAddress (required)
                - sitePostcode (required)
                - siteReference (UPRN)
                - siteContactName, siteContactTelephone, siteContactEmail
                - landlord
                
        Returns:
            APIResponse with siteId in data
        """
        return self._request("POST", "/sites", json=site_data)

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def update_site(self, site_id: int, updates: Dict[str, Any]) -> APIResponse:
        """Update site fields."""
        return self._request("PATCH", f"/sites/{site_id}", json=updates)

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def delete_site(self, site_id: int) -> APIResponse:
        """Delete a site."""
        return self._request("DELETE", f"/sites/{site_id}")

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def search_sites(self, filter: str = None, limit: int = 100, page: int = 1) -> APIResponse:
        """Search sites using SQL-like filter."""
        params = {"filter": filter, "limit": limit, "paginate": "true", "pageNumber": page}
        return self._request("GET", "/sites/search", params=params)
    
    def find_site_by_reference(self, client_id: str, site_reference: str) -> Optional[Dict[str, Any]]:
        """Find a site by client ID and site reference (UPRN)."""
        filter_str = f"clientId = '{client_id}' AND siteReference = '{site_reference}'"
        response = self.search_sites(filter=filter_str, limit=1)
        
        if response.success and response.data:
            sites = response.data.get("data", [])
            return sites[0] if sites else None
        return None
    
    # =========================================================================
    # APPOINTMENTS - Diary Entries
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_appointments(self, project_number: str) -> List[Dict[str, Any]]:
        """
        Get diary appointments for a project.
        
        Returns list with: appointmentId, staffId, staffName, staffRole,
        startTime, endTime, location, confirmed, cannotMove, tentative,
        writeUp, incomplete, incompleteDate, incompleteReason, notes
        """
        response = self._request("GET", f"/appointments/{project_number}")
        if response.success:
            data = response.data
            return data if isinstance(data, list) else [data] if data else []
        return []

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def create_appointment(self, project_number: str,
                           staff_id: int,
                           start_time: str,
                           end_time: str,
                           location: str = "",
                           staff_role: str = "Surveyor",
                           notes: str = "",
                           confirmed: bool = False,
                           cannot_move: bool = False,
                           tentative: bool = False,
                           write_up: bool = False) -> APIResponse:
        """
        Create a diary appointment.
        
        Args:
            project_number: Project to attach appointment to
            staff_id: Staff member ID
            start_time: ISO datetime (YYYY-MM-DDTHH:MM:SS)
            end_time: ISO datetime
            location: Location text
            staff_role: Role (e.g., "Surveyor")
            notes: Appointment notes
            confirmed, cannot_move, tentative, write_up: Flags
            
        Returns:
            APIResponse with appointmentId in data
        """
        payload = {
            "staffId": staff_id,
            "staffRole": staff_role,
            "startTime": start_time,
            "endTime": end_time,
            "location": location,
            "confirmed": confirmed,
            "cannotMove": cannot_move,
            "tentative": tentative,
            "writeUp": write_up,
            "notes": notes,
        }
        return self._request("POST", f"/appointments/{project_number}", json=payload)
    
    def create_appointment_from_order(self, project_number: str,
                                       order_data: Dict[str, Any]) -> APIResponse:
        """
        Create appointment from order data (convenience method).
        
        Args:
            project_number: Project number
            order_data: Dict with surveyor, dateBookedIn, timeOfAttendance, address, scope
        """
        surveyor = order_data.get("surveyor", "")
        date_booked = order_data.get("dateBookedIn", "")
        time_attendance = order_data.get("timeOfAttendance", "")
        
        if not surveyor or not date_booked:
            return APIResponse(success=False, error="Missing surveyor or date")
        
        staff_id = self.find_staff_id(surveyor)
        if not staff_id:
            return APIResponse(success=False, error=f"Staff not found: {surveyor}")
        
        appointment_date = convert_to_iso_date(date_booked)
        if not appointment_date:
            return APIResponse(success=False, error=f"Invalid date: {date_booked}")
        
        start_hour, end_hour = parse_time_range(time_attendance)
        
        return self.create_appointment(
            project_number=project_number,
            staff_id=staff_id,
            start_time=f"{appointment_date}T{start_hour:02d}:00:00",
            end_time=f"{appointment_date}T{end_hour:02d}:00:00",
            location=order_data.get("address", ""),
            notes=f"Survey: {order_data.get('scope', '')} | Time: {time_attendance}",
        )

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def update_appointment(self, appointment_id: int, updates: Dict[str, Any]) -> APIResponse:
        """
        Update an appointment.
        
        Updatable: projectNumber, staffId, staffRole, startTime, endTime,
        location, confirmed, cannotMove, tentative, writeUp, notes
        """
        return self._request("PATCH", f"/appointments/appointment/{appointment_id}", json=updates)

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def delete_appointment(self, appointment_id: int) -> APIResponse:
        """Delete an appointment."""
        return self._request("DELETE", f"/appointments/appointment/{appointment_id}")
    
    # =========================================================================
    # MILESTONES
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_milestones(self, project_number: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get milestones for a project.

        Returns list with: milestone, position, targetDate, assignedToId,
        completionDate, completedById, notes
        """
        response = self._request("GET", f"/milestones/{project_number}")
        if response.success:
            return response.data if isinstance(response.data, list) else []
        return None
    
    def get_project_due_date(self, project_number: str) -> Optional[str]:
        """Get the due date from project milestones (looks for '** Project Due Date **')."""
        milestones = self.get_milestones(project_number)
        if not milestones:
            return None
        
        for ms in milestones:
            if "project due date" in ms.get("milestone", "").lower():
                return ms.get("targetDate")
        return None
    
    # =========================================================================
    # STAFF
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_staff(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of all staff members.

        Returns list with: staffId, name, emailAddress, mobileNumber, unitId
        """
        if use_cache and self._staff_cache is not None:
            return self._staff_cache
        
        response = self._request("GET", "/staff")
        if response.success:
            self._staff_cache = response.data if isinstance(response.data, list) else []
            return self._staff_cache
        return []
    
    def find_staff_id(self, name: str) -> Optional[int]:
        """Find staff ID by name (supports fuzzy matching)."""
        if not name:
            return None
        
        staff_list = self.get_staff()
        name_lower = name.lower().strip()
        
        # Exact match
        for staff in staff_list:
            if staff.get("name", "").lower().strip() == name_lower:
                return staff["staffId"]
        
        # Partial match
        for staff in staff_list:
            staff_name = staff.get("name", "").lower().strip()
            if name_lower in staff_name or staff_name in name_lower:
                return staff["staffId"]
        
        # Word match
        name_parts = name_lower.split()
        for staff in staff_list:
            staff_parts = staff.get("name", "").lower().split()
            for part in name_parts:
                if len(part) > 2 and part in staff_parts:
                    return staff["staffId"]
        
        return None
    
    def get_staff_email(self, name: str) -> Optional[str]:
        """Get staff email address by name."""
        if not name:
            return None
        
        staff_list = self.get_staff()
        name_lower = name.lower().strip()
        
        for staff in staff_list:
            if name_lower in staff.get("name", "").lower():
                return staff.get("emailAddress")
        return None
    
    def get_staff_by_id(self, staff_id: int) -> Optional[Dict[str, Any]]:
        """Get staff member by ID."""
        for staff in self.get_staff():
            if staff.get("staffId") == staff_id:
                return staff
        return None
    
    # =========================================================================
    # PROJECT TYPES
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_project_types(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of project types.

        Returns list with: projectTypeId, name, description, category,
        dateActive, dateInactive
        """
        if use_cache and self._project_types_cache is not None:
            return self._project_types_cache

        response = self._request("GET", "/projecttypes")
        if response.success:
            self._project_types_cache = response.data if isinstance(response.data, list) else []
            return self._project_types_cache
        return []
    
    def find_project_type_id(self, name: str) -> Optional[int]:
        """Find project type ID by name."""
        if not name:
            return None
        
        types = self.get_project_types()
        name_lower = name.lower().strip()
        
        for pt in types:
            if pt.get("name", "").lower().strip() == name_lower:
                return pt["projectTypeId"]
            if name_lower in pt.get("name", "").lower():
                return pt["projectTypeId"]
        return None
    
    # =========================================================================
    # SAMPLES
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_samples(self, project_number: str) -> List[Dict[str, Any]]:
        """
        Get samples for a project.

        Returns list with: sampleId, locationDescription, item, material, identification
        """
        response = self._request("GET", f"/samples/{project_number}")
        return response.data if response.success and isinstance(response.data, list) else []
    
    def create_samples(self, project_number: str, samples: List[Dict[str, Any]]) -> APIResponse:
        """
        Create samples for a project.
        
        Args:
            project_number: Project to add samples to
            samples: List of dicts with: sampleId, locationDescription, item, material
        """
        return self._request("POST", f"/samples/{project_number}", json=samples)
    
    # =========================================================================
    # SURVEY ITEMS
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_survey_items(self, project_number: str) -> List[Dict[str, Any]]:
        """
        Get survey items for a project.

        Returns list with: date, floor, location, locationDescription, item,
        materialCode, materialDescription, noAccess, approach, sampleNumber,
        sampleNotes, extent, UoM, scores (productType, condition, surfaceTreatment,
        asbestosType, material, priority, total), identification, recommendedAction,
        recommendationComments, photoFilename, closeUpPhotoFilename
        """
        response = self._request("GET", f"/surveyitems/{project_number}")
        return response.data if response.success and isinstance(response.data, list) else []

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_survey_item(self, survey_item_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific survey item by ID."""
        response = self._request("GET", f"/surveyitems/item/{survey_item_id}")
        return response.data if response.success else None

    @retry(max_attempts=5, backoff_seconds=[1, 2, 5, 10, 30])
    def update_survey_item(self, project_number: str, survey_item_id: int,
                           updates: Dict[str, Any]) -> APIResponse:
        """
        Update a survey item's fields via PATCH.

        Args:
            project_number: Project number (e.g., "T-78473")
            survey_item_id: Survey item ID to update
            updates: Dict of fields to update (e.g., {"notes": "new notes"})

        Returns:
            APIResponse with success status

        Example:
            api.update_survey_item("T-78473", 1796876, {"notes": "Updated notes text"})
        """
        # The API requires the project number in the URL, not the item ID
        # The item ID is passed in the body to identify which item to update
        payload = {"id": survey_item_id, **updates}
        return self._request("PATCH", f"/surveyitems/{project_number}", json=payload)

    # =========================================================================
    # BUILDINGS (Project-level)
    # =========================================================================

    @retry(max_attempts=3, backoff_seconds=[1, 5, 30])
    def get_buildings(self, project_number: str) -> List[Dict[str, Any]]:
        """
        Get buildings for a project.

        Returns list with: buildingId, buildingRef, buildingName
        """
        response = self._request("GET", f"/buildings/{project_number}")
        data = response.data if response.success else None
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    
    def create_building(self, project_number: str,
                        building_ref: str,
                        building_name: str) -> APIResponse:
        """Create a building under a project."""
        return self._request("POST", f"/buildings/{project_number}", json={
            "buildingRef": building_ref,
            "buildingName": building_name,
        })
    
    # =========================================================================
    # SITE BUILDINGS (Site-level)
    # =========================================================================
    
    def get_site_buildings(self, site_id: int) -> List[Dict[str, Any]]:
        """
        Get buildings for a site.
        
        Returns list with: siteBuildingId, buildingRef, buildingName,
        buildingAddress, buildingPostcode
        """
        response = self._request("GET", f"/sitebuildings/{site_id}")
        data = response.data if response.success else None
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    
    def create_site_building(self, site_id: int,
                             building_ref: str,
                             building_name: str,
                             building_address: str = "",
                             building_postcode: str = "") -> APIResponse:
        """Create a building under a site."""
        return self._request("POST", f"/sitebuildings/{site_id}", json={
            "buildingRef": building_ref,
            "buildingName": building_name,
            "buildingAddress": building_address,
            "buildingPostcode": building_postcode,
        })
    
    # =========================================================================
    # FILES
    # =========================================================================
    
    def get_project_files(self, project_number: str) -> List[Dict[str, Any]]:
        """
        Get list of files in project folder.
        
        Returns list with: filename, bytes, modified, created
        """
        response = self._request("GET", f"/files/projects/{project_number}")
        return response.data if response.success and isinstance(response.data, list) else []
    
    def download_file(self, project_number: str, filename: str) -> Optional[bytes]:
        """Download a file from project folder."""
        url = f"{self.base_url}/files/projects/{project_number}/download"
        try:
            response = self.session.get(url, params={"filename": filename}, timeout=60)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.error(f"File download failed: {e}")
        return None

    def upload_file(self, project_number: str, file_path: str, file_type: str = None) -> APIResponse:
        """
        Upload a file to a project's files folder.

        Args:
            project_number: Project number (e.g., N-98813)
            file_path: Local path to file
            file_type: Optional file type label (API-dependent)

        Returns:
            APIResponse with success flag and server response/error
        """
        url = f"{self.base_url}/files/projects/{project_number}/upload"
        try:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f)}
                data = {}
                if file_type:
                    data["fileType"] = file_type
                response = self.session.post(url, files=files, data=data, timeout=60)

            if response.status_code in (200, 201):
                try:
                    payload = response.json()
                except Exception:
                    payload = {"message": response.text}
                return APIResponse(success=True, data=payload, status_code=response.status_code)

            return APIResponse(
                success=False,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def download_bulk_certificate(self, project_number: str) -> Optional[bytes]:
        """Download bulk certificate PDF from project folder."""
        url = f"{self.base_url}/files/projects/{project_number}/download/bulkcertificate"
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            logger.error(f"Bulk certificate download failed: {e}")
        return None
    
    # =========================================================================
    # REPORTS
    # =========================================================================
    
    def list_reports(self, category: str) -> List[Dict[str, Any]]:
        """
        Get list of reports in a category.
        
        Returns list with: reportName, reportDescription, parameter1-9
        """
        response = self._request("GET", f"/reports/{category}")
        return response.data if response.success and isinstance(response.data, list) else []
    
    def run_report(self, category: str, report_name: str,
                   output_format: str = "pdf",
                   parameters: Dict[str, str] = None) -> APIResponse:
        """
        Run a report.
        
        Args:
            category: Report category
            report_name: Name of report
            output_format: Output format (pdf, excel, etc.)
            parameters: Dict with parameter1-9
        """
        payload = {
            "format": output_format,
            "parameters": parameters or {},
        }
        return self._request("POST", f"/reports/{category}/{report_name}", json=payload)
    
    # =========================================================================
    # REPAIR INSTRUCTIONS
    # =========================================================================
    
    def search_repair_instructions(self, filter: str = None, limit: int = 100,
                                    page: int = 1) -> APIResponse:
        """
        Search repair instructions.
        
        Returns list with: repairInstructionId, surveyItemId, originalSurveyItemId,
        issueDate, inspectionDate, targetCompletionDate, issuedBy, issuedById,
        issuedTo, projectNumber, siteId, buildingId, location, roomName, position,
        actionCode, materialType, workToBeCompleted, dateCompleted, comments,
        remediationProjectNumber, remediationAnalyst, remediationAnalystId, certificateIssued
        """
        params = {"filter": filter, "limit": limit, "paginate": "true", "pageNumber": page}
        return self._request("GET", "/repairinstructions/search", params=params)
    
    def get_repair_instruction(self, repair_instruction_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific repair instruction by ID."""
        response = self._request("GET", f"/repairinstructions/repairinstruction/{repair_instruction_id}")
        return response.data if response.success else None
    
    # =========================================================================
    # REINSPECTION
    # =========================================================================
    
    def send_reinspection(self, project_number: str, staff_id: int,
                          items: List[Dict[str, Any]]) -> APIResponse:
        """
        Send re-inspection data to a handset.
        
        Args:
            project_number: Project number
            staff_id: Staff member to send to
            items: List of inspection item dicts with:
                buildingId, floor, location, locationDescription, item, materialCode,
                materialDescription, noAccess, approach, sampleNumber, sampleNotes,
                extent, UoM, scores, identification, recommendedAction, recommendationComments
        """
        return self._request("POST", "/reinspection", json={
            "projectNumber": project_number,
            "staffId": staff_id,
            "data": {"items": items},
        })
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_project_with_details(self, project_number: str) -> Dict[str, Any]:
        """
        Get project with all related data (milestones, appointments, samples).
        
        Convenience method that combines multiple API calls.
        """
        project = self.get_project(project_number)
        if not project:
            return {}
        
        project["milestones"] = self.get_milestones(project_number) or []
        project["appointments"] = self.get_appointments(project_number)
        project["samples"] = self.get_samples(project_number)
        project["buildings"] = self.get_buildings(project_number)
        project["files"] = self.get_project_files(project_number)
        project["due_date"] = self.get_project_due_date(project_number)
        
        return project
    
    def find_projects_due_on(self, target_date: Union[str, datetime] = None,
                              include_overdue: bool = False,
                              max_projects: int = 5000) -> List[Dict[str, Any]]:
        """
        Find projects due on a specific date.
        
        Args:
            target_date: Date to check (default: today)
            include_overdue: Include projects due before target_date
            max_projects: Maximum projects to check
            
        Returns:
            List of matching projects
        """
        if target_date is None:
            target = datetime.now().date()
        elif isinstance(target_date, str):
            parsed = parse_date(target_date)
            target = parsed.date() if parsed else datetime.now().date()
        elif isinstance(target_date, datetime):
            target = target_date.date()
        else:
            target = target_date
        
        # Search for scheduled survey projects
        filter_str = "status = 'Scheduled' AND projectNumber LIKE 'N-%'"
        all_projects = self.search_all_projects(filter_str, max_pages=max_projects // 100)
        
        matching = []
        
        def check_project(project):
            pn = project.get("projectNumber")
            if not pn:
                return None
            
            # Check project type contains "Survey"
            pt = project.get("projectType", "")
            if "survey" not in pt.lower():
                return None
            
            # Get due date from milestones
            due_date_str = self.get_project_due_date(pn)
            if not due_date_str:
                return None
            
            due_date = parse_date(due_date_str)
            if not due_date:
                return None
            
            due = due_date.date()
            
            if include_overdue:
                if due > target:
                    return None
            else:
                if due != target:
                    return None
            
            # Get surveyor from appointments
            appointments = self.get_appointments(pn)
            surveyor = None
            surveyor_email = None
            for appt in appointments:
                if "surveyor" in appt.get("staffRole", "").lower():
                    surveyor = appt.get("staffName")
                    if surveyor:
                        surveyor_email = self.get_staff_email(surveyor)
                    break
            
            return {
                "project_number": pn,
                "client_id": project.get("clientId"),
                "client_name": project.get("clientName"),
                "site_name": project.get("siteName"),
                "site_address": project.get("siteAddress"),
                "site_postcode": project.get("sitePostcode"),
                "project_type": pt,
                "status": project.get("status"),
                "due_date": format_date(due),
                "surveyor": surveyor,
                "surveyor_email": surveyor_email,
            }
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(check_project, p): p for p in all_projects[:max_projects]}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    matching.append(result)
        
        matching.sort(key=lambda x: x.get("project_number", ""))
        return matching
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def clear_cache(self):
        """Clear internal caches."""
        self._staff_cache = None
        self._project_types_cache = None
    
    def close(self):
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# MAIN - CLI Usage
# =============================================================================

def main():
    """Command line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaTracker API Client")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    parser.add_argument("--project", type=str, help="Get project by number")
    parser.add_argument("--search", type=str, help="Search projects with filter")
    parser.add_argument("--due-today", action="store_true", help="Find projects due today")
    parser.add_argument("--staff", action="store_true", help="List all staff")
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_env()
    if not config.validate():
        print("ERROR: API credentials not configured!")
        print("Set ALPHA_TRACKER_API_KEY and ALPHA_TRACKER_CLIENT_ID")
        return
    
    api = AlphaTrackerAPI(config)
    
    if args.test:
        ok, msg = api.test_connection()
        print(f"Connection test: {'OK' if ok else 'FAILED'} - {msg}")
    
    elif args.project:
        project = api.get_project(args.project)
        if project:
            print(json.dumps(project, indent=2, default=str))
        else:
            print(f"Project {args.project} not found")
    
    elif args.search:
        response = api.search_projects(args.search, limit=10)
        if response.success:
            print(json.dumps(response.data, indent=2, default=str))
        else:
            print(f"Search failed: {response.error}")
    
    elif args.due_today:
        projects = api.find_projects_due_on()
        print(f"Found {len(projects)} projects due today:")
        for p in projects:
            print(f"  {p['project_number']}: {p['client_name']} - {p['site_name']}")
    
    elif args.staff:
        staff = api.get_staff()
        print(f"Found {len(staff)} staff members:")
        for s in staff:
            print(f"  {s['staffId']}: {s['name']} - {s.get('emailAddress', 'N/A')}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
