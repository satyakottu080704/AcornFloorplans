"""
Google Gemini Vision AI Module
================================
Provides building age extraction, construction details analysis, and floor plan generation
using Google Gemini 2.5 Flash vision models (upgraded March 2026).

Features:
- Building age detection (Victorian, Edwardian, Post-war, etc.)
- Construction type identification (Masonry, Timber frame, etc.)
- Material detection from photos
- Floor plan analysis and generation
- Sample label reading (S001, S002, etc.)
- Usage tracking (stay under free tier limit)

Model: Gemini 2.5 Flash (PRIMARY vision provider)
Accuracy: ~90%+ (significant upgrade from 2.0 Flash)
Cost: FREE up to 1,000 requests/day
"""

import os
import json
import base64
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

# Gemini API Configuration - Multi-Key Support
# Supports multiple API keys for increased capacity (1,500/day per key)
GEMINI_API_KEYS = []
for i in range(1, 11):  # Support up to 10 keys
    key = os.getenv(f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY", "")
    if key:
        GEMINI_API_KEYS.append(key)

GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_DAILY_LIMIT = int(os.getenv("GEMINI_DAILY_LIMIT", "1500"))
GEMINI_RATE_LIMIT = int(os.getenv("GEMINI_RATE_LIMIT_PER_MINUTE", "15"))

# Gemini API base URL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Usage tracking file
USAGE_FILE = Path(__file__).parent.parent / "logs" / "gemini_usage.json"
USAGE_FILE.parent.mkdir(exist_ok=True)

# Current key index (rotates between keys)
_current_key_index = 0

if GEMINI_API_KEYS:
    logger.info(f"[GEMINI] Loaded {len(GEMINI_API_KEYS)} API key(s) - Capacity: {len(GEMINI_API_KEYS) * GEMINI_DAILY_LIMIT}/day")


# =============================================================================
# USAGE TRACKING
# =============================================================================

def get_usage_today() -> Dict[str, Any]:
    """Get today's Gemini API usage (all keys combined)."""
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        if USAGE_FILE.exists():
            with open(USAGE_FILE, 'r') as f:
                usage = json.load(f)
        else:
            usage = {}
    except:
        usage = {}

    if usage.get("date") != today:
        # New day - reset counters for all keys
        usage = {
            "date": today,
            "total_count": 0,
            "by_key": {i: {"count": 0, "quota_exceeded": False} for i in range(len(GEMINI_API_KEYS))},
            "by_feature": {},
            "last_429_time": None
        }

    return usage


def get_available_key() -> Optional[tuple[int, str]]:
    """
    Get next available API key (round-robin with quota checking).

    Returns:
        Tuple of (key_index, api_key) or None if all keys exhausted
    """
    global _current_key_index

    if not GEMINI_API_KEYS:
        return None

    usage = get_usage_today()
    now = datetime.now().timestamp()

    # Try each key starting from current index
    for _ in range(len(GEMINI_API_KEYS)):
        key_idx = _current_key_index
        key_usage = usage["by_key"].get(str(key_idx), {"count": 0, "quota_exceeded": False})

        # Skip keys with permanent daily quota exceeded
        if key_usage.get("quota_exceeded", False):
            _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
            continue

        # Skip keys still in rate-limit cooldown
        rate_limit_until = key_usage.get("rate_limit_until", 0)
        if rate_limit_until and now < rate_limit_until:
            secs_left = int(rate_limit_until - now)
            logger.debug(f"[GEMINI] Key #{key_idx+1} in rate-limit cooldown ({secs_left}s remaining)")
            _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
            continue

        # Skip keys over daily count limit
        if key_usage.get("count", 0) >= GEMINI_DAILY_LIMIT:
            _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
            continue

        # This key is available
        return (key_idx, GEMINI_API_KEYS[key_idx])

    # All keys exhausted or in cooldown
    logger.error(f"[GEMINI] All {len(GEMINI_API_KEYS)} API keys exhausted or cooling down for today!")
    return None


def increment_usage(key_index: int = 0, feature: str = "general") -> int:
    """
    Increment usage counter for specific key and return current count.

    Args:
        key_index: Index of the API key used
        feature: Feature name for tracking

    Returns:
        Current usage count for this key today
    """
    usage = get_usage_today()

    # Increment total count
    usage["total_count"] = usage.get("total_count", 0) + 1

    # Increment key-specific count
    key_str = str(key_index)
    if key_str not in usage["by_key"]:
        usage["by_key"][key_str] = {"count": 0, "quota_exceeded": False}

    usage["by_key"][key_str]["count"] += 1

    # Increment feature count
    usage["by_feature"][feature] = usage["by_feature"].get(feature, 0) + 1

    # Save
    with open(USAGE_FILE, 'w') as f:
        json.dump(usage, f, indent=2)

    key_count = usage["by_key"][key_str]["count"]

    # Warn if approaching limit for this key
    if key_count == int(GEMINI_DAILY_LIMIT * 0.8):
        logger.warning(f"⚠️ Gemini key #{key_index+1} at 80%: {key_count}/{GEMINI_DAILY_LIMIT}")
    elif key_count == int(GEMINI_DAILY_LIMIT * 0.95):
        logger.warning(f"⚠️ Gemini key #{key_index+1} at 95%: {key_count}/{GEMINI_DAILY_LIMIT}")
    elif key_count >= GEMINI_DAILY_LIMIT:
        logger.error(f"🚫 Gemini key #{key_index+1} daily limit reached: {key_count}/{GEMINI_DAILY_LIMIT}")
        # Rotate to next key automatically
        global _current_key_index
        _current_key_index = (key_index + 1) % len(GEMINI_API_KEYS)
        logger.info(f"[GEMINI] Auto-rotating to key #{_current_key_index+1}")

    return key_count


def mark_quota_exceeded(key_index: int, is_rate_limit: bool = False):
    """
    Mark that a specific key hit a 429 error.

    Args:
        key_index: Index of the key that hit 429
        is_rate_limit: True = per-minute rate limit (temporary cooldown only).
                       False = daily quota exhausted (skip key for rest of day).
    """
    usage = get_usage_today()

    key_str = str(key_index)
    if key_str not in usage["by_key"]:
        usage["by_key"][key_str] = {"count": 0, "quota_exceeded": False}

    usage["last_429_time"] = datetime.now().isoformat()

    if is_rate_limit:
        # Temporary rate limit — apply a 60-second cooldown, do NOT permanently block
        cooldown_until = datetime.now().timestamp() + 60
        usage["by_key"][key_str]["rate_limit_until"] = cooldown_until
        logger.warning(f"[GEMINI] Key #{key_index+1} rate-limited (QPM). Cooldown 60s, rotating to next key.")
    else:
        # Daily quota exhausted — block for rest of day
        usage["by_key"][key_str]["quota_exceeded"] = True
        usage["by_key"][key_str]["count"] = GEMINI_DAILY_LIMIT
        logger.warning(f"[GEMINI] Key #{key_index+1} daily quota exceeded. Blocking for rest of day.")

    with open(USAGE_FILE, 'w') as f:
        json.dump(usage, f, indent=2)

    # Rotate to next key
    global _current_key_index
    _current_key_index = (key_index + 1) % len(GEMINI_API_KEYS)
    logger.info(f"[GEMINI] Rotated to key #{_current_key_index+1}")


def check_can_use_gemini() -> bool:
    """Check if we can make another Gemini API call (any key available)."""
    return get_available_key() is not None


def get_gemini_usage_stats() -> Dict[str, Any]:
    """Get current Gemini usage statistics."""
    usage = get_usage_today()
    return {
        "total_today": usage.get("total_count", 0),
        "limit": len(GEMINI_API_KEYS) * GEMINI_DAILY_LIMIT,
        "keys_count": len(GEMINI_API_KEYS),
        "by_key": usage.get("by_key", {}),
        "by_feature": usage.get("by_feature", {}),
        "date": usage.get("date")
    }


# =============================================================================
# =============================================================================
# HTTP RETRY HELPER
# =============================================================================

def _post_with_retry(
    url: str,
    params: Dict,
    payload: Dict,
    timeout: int = 30,
    max_retries: int = 3,
    backoff: float = 1.0
) -> requests.Response:
    """
    POST to Gemini API with exponential backoff retry on transient errors.

    Retries on:  timeout, connection errors, 5xx server errors
    No retry on: 429 (handled by key rotation), 4xx client errors

    Args:
        url:         Full API endpoint URL
        params:      Query params (API key)
        payload:     JSON body
        timeout:     Per-attempt timeout in seconds
        max_retries: Max retry attempts after initial failure (total = max_retries + 1)
        backoff:     Base backoff in seconds (doubles each retry: 1s, 2s, 4s)

    Returns:
        requests.Response from the last attempt
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, params=params, json=payload, timeout=timeout)

            # Retry on 5xx server errors only
            if response.status_code >= 500 and attempt < max_retries:
                wait = backoff * (2 ** attempt)
                logger.warning(f"[GEMINI] HTTP {response.status_code} on attempt {attempt+1}, "
                               f"retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue

            # For all other responses (200, 4xx, 429) return immediately
            return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < max_retries:
                wait = backoff * (2 ** attempt)
                logger.warning(f"[GEMINI] {type(e).__name__} on attempt {attempt+1}, "
                               f"retrying in {wait:.0f}s... ({e})")
                time.sleep(wait)
            else:
                logger.error(f"[GEMINI] {type(e).__name__} after {max_retries+1} attempts: {e}")

    # All attempts exhausted — raise so caller can return a clean error dict
    raise last_exc or requests.exceptions.ConnectionError("All retry attempts failed")


# =============================================================================
# =============================================================================
# GEMINI API CALLS
# =============================================================================

def call_gemini_vision(
    prompt: str,
    image_data: str,
    model: str = None,
    max_tokens: int = 2048,
    response_mime_type: str = None
) -> Dict[str, Any]:
    """
    Call Gemini vision API with image.

    Args:
        prompt: Text prompt describing what to analyze
        image_data: Base64 encoded image data
        model: Model to use (default: GEMINI_VISION_MODEL)
        max_tokens: Max response tokens

    Returns:
        API response dict with "success", "text", "error"
    """
    # Get available API key (round-robin with quota checking)
    key_result = get_available_key()
    if not key_result:
        usage = get_usage_today()
        total_limit = len(GEMINI_API_KEYS) * GEMINI_DAILY_LIMIT
        return {
            "success": False,
            "error": f"All Gemini API keys exhausted ({usage['total_count']}/{total_limit}). Falls back to Ollama."
        }

    key_index, api_key = key_result
    model = model or GEMINI_VISION_MODEL

    try:
        url = f"{GEMINI_API_BASE}/{model}:generateContent"

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens,
                # Force structured JSON when the caller needs it (e.g. layout
                # extraction). Without this, thinking-capable models wrap/truncate
                # the JSON and json.loads() fails.
                **({"responseMimeType": response_mime_type} if response_mime_type else {})
            }
        }

        response = _post_with_retry(
            url=url,
            params={"key": api_key},
            payload=payload,
            timeout=30
        )

        if response.status_code != 200:
            if response.status_code == 429:
                err_body = response.text.lower()
                is_rate_limit = any(kw in err_body for kw in ["per minute", "rate", "qpm", "retry"])
                mark_quota_exceeded(key_index, is_rate_limit=is_rate_limit)
                kind = "rate-limited" if is_rate_limit else "daily quota exceeded"
                return {
                    "success": False,
                    "error": f"Gemini key #{key_index+1} {kind} (429). Rotating to next key."
                }
            return {
                "success": False,
                "error": f"Gemini API error {response.status_code}: {response.text[:200]}"
            }

        result = response.json()
        candidates = result.get("candidates", [])
        if not candidates:
            return {"success": False, "error": "No response from Gemini"}

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if "text" in part)
        increment_usage(key_index, "vision")

        return {
            "success": True,
            "text": text,
            "model": model,
            "key_used": key_index + 1
        }

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def call_gemini_text(
    prompt: str,
    model: str = None,
    max_tokens: int = 2048,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Call Gemini text-only API.

    Args:
        prompt: Text prompt
        model: Model to use (default: GEMINI_TEXT_MODEL)
        max_tokens: Max response tokens
        temperature: Temperature for generation

    Returns:
        API response dict with "success", "text", "error"
    """
    # Get available API key (round-robin with quota checking)
    key_result = get_available_key()
    if not key_result:
        usage = get_usage_today()
        total_limit = len(GEMINI_API_KEYS) * GEMINI_DAILY_LIMIT
        return {
            "success": False,
            "error": f"All Gemini API keys exhausted ({usage['total_count']}/{total_limit}). Falls back to Ollama."
        }

    key_index, api_key = key_result
    model = model or GEMINI_TEXT_MODEL

    try:
        url = f"{GEMINI_API_BASE}/{model}:generateContent"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        response = _post_with_retry(
            url=url,
            params={"key": api_key},
            payload=payload,
            timeout=30
        )

        if response.status_code != 200:
            if response.status_code == 429:
                # Distinguish rate limit (QPM) from daily quota exhaustion
                err_body = response.text.lower()
                is_rate_limit = any(kw in err_body for kw in ["per minute", "rate", "qpm", "retry"])
                mark_quota_exceeded(key_index, is_rate_limit=is_rate_limit)
                kind = "rate-limited" if is_rate_limit else "daily quota exceeded"
                return {
                    "success": False,
                    "error": f"Gemini key #{key_index+1} {kind} (429). Rotating to next key."
                }

            return {
                "success": False,
                "error": f"Gemini API error {response.status_code}: {response.text[:200]}"
            }

        result = response.json()

        # Extract text from response
        candidates = result.get("candidates", [])
        if not candidates:
            return {
                "success": False,
                "error": "No response from Gemini"
            }

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if "text" in part)

        # Increment usage counter for this key
        increment_usage(key_index, "text")

        return {
            "success": True,
            "text": text,
            "model": model,
            "key_used": key_index + 1  # 1-indexed for logging
        }

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def image_path_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        return None


# =============================================================================
# BUILDING AGE EXTRACTION
# =============================================================================

def extract_building_age_gemini(
    image_path: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Extract building age and construction details from photo using Gemini.

    Args:
        image_path: Path to building photo
        context: Optional context (existing survey data)

    Returns:
        Dict with:
        - building_age: "1900s" | "1960s-1970s" | "Modern (2000+)"
        - construction_era: "Victorian" | "Post-war" | "Modern"
        - building_type: "Masonry" | "Timber frame" | "Steel frame"
        - materials_visible: ["Brick", "Concrete", "Timber"]
        - condition: "Good" | "Fair" | "Poor"
        - confidence: "HIGH" | "MEDIUM" | "LOW"
        - details: Free-form description
        - key_features: List of dating indicators
    """
    result = {
        "success": False,
        "source": "gemini_vision",
        "image_path": image_path
    }

    # Convert image to base64
    image_data = image_path_to_base64(image_path)
    if not image_data:
        result["error"] = "Failed to read image file"
        return result

    # Build prompt
    prompt = """Analyze this UK building photo and determine construction details.

Identify:

1. BUILDING AGE/ERA (UK classification):
   - Pre-1900 (Victorian/Georgian)
   - 1900-1919 (Edwardian)
   - 1920-1945 (Inter-war)
   - 1946-1979 (Post-war)
   - 1980-1999 (Late 20th century)
   - 2000+ (Modern)

2. CONSTRUCTION TYPE:
   - Solid masonry (9" brick, stone)
   - Cavity wall (post-1920s standard)
   - Timber frame (modern, post-1980)
   - Steel frame (commercial, post-1960)
   - Concrete (panel/monolithic, 1950s-1970s)
   - Mixed construction

3. VISIBLE MATERIALS:
   - Wall: Brick | Stone | Render | Cladding | Concrete
   - Roof: Slate | Tile | Felt | Metal
   - Windows: Timber | uPVC | Metal | Glazing type

4. KEY DATING INDICATORS:
   - Brickwork pattern (Flemish=pre-1920, Stretcher=post-1920)
   - Window style (Sash=pre-1960, Casement=post-1960)
   - Roof pitch (Steep=pre-1960, Shallow=modern)
   - Architectural features (Corbels, quoins, lintels)

5. CONDITION:
   - Good: Well-maintained, no visible defects
   - Fair: Some wear, minor repairs needed
   - Poor: Significant deterioration

Respond in this EXACT format:
AGE_ERA: [era from list]
AGE_ESTIMATE: [decade or range]
CONSTRUCTION_TYPE: [type]
WALL_MATERIAL: [material]
ROOF_MATERIAL: [material]
WINDOW_TYPE: [type and glazing]
KEY_FEATURES: [bullet list]
CONDITION: GOOD/FAIR/POOR
CONFIDENCE: HIGH/MEDIUM/LOW
REASONING: [brief explanation]

Only respond with the format above."""

    # Call Gemini
    logger.info(f"[GEMINI] Analyzing building age from: {Path(image_path).name}")
    api_result = call_gemini_vision(prompt, image_data)

    if not api_result["success"]:
        result["error"] = api_result["error"]
        return result

    # Parse response
    text = api_result["text"]
    result["raw_response"] = text
    result["success"] = True
    result["model"] = api_result["model"]

    # Extract fields using simple parsing
    import re

    def extract_field(pattern: str) -> Optional[str]:
        match = re.search(rf"{pattern}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    result["age_era"] = extract_field("AGE_ERA")
    result["age_estimate"] = extract_field("AGE_ESTIMATE")
    result["construction_type"] = extract_field("CONSTRUCTION_TYPE")
    result["wall_material"] = extract_field("WALL_MATERIAL")
    result["roof_material"] = extract_field("ROOF_MATERIAL")
    result["window_type"] = extract_field("WINDOW_TYPE")
    result["condition"] = extract_field("CONDITION")
    result["confidence"] = extract_field("CONFIDENCE") or "MEDIUM"
    result["reasoning"] = extract_field("REASONING")

    # Extract key features (multi-line)
    features_match = re.search(r"KEY_FEATURES:\s*(.+?)(?:\nCONDITION|$)", text, re.IGNORECASE | re.DOTALL)
    if features_match:
        features_text = features_match.group(1).strip()
        result["key_features"] = [f.strip(" -•") for f in features_text.split("\n") if f.strip()]
    else:
        result["key_features"] = []

    # Generate summary for Building Details field
    result["building_details_summary"] = generate_building_details_text(result)

    logger.info(f"[GEMINI] Building age extracted: {result['age_estimate']} ({result['confidence']} confidence)")

    return result


def generate_building_details_text(analysis: Dict[str, Any]) -> str:
    """Generate Building Details text from Gemini analysis."""
    parts = []

    if analysis.get("age_estimate"):
        era = analysis.get("age_era", "")
        parts.append(f"A {analysis['age_estimate']} {era} building.")

    if analysis.get("construction_type"):
        parts.append(f"{analysis['construction_type']} construction.")

    if analysis.get("wall_material"):
        parts.append(f"Walls: {analysis['wall_material']}.")

    if analysis.get("roof_material"):
        parts.append(f"Roof: {analysis['roof_material']}.")

    if analysis.get("window_type"):
        parts.append(f"Windows: {analysis['window_type']}.")

    if analysis.get("condition"):
        parts.append(f"Condition: {analysis['condition']}.")

    return " ".join(parts)


# =============================================================================
# FLOOR PLAN ANALYSIS
# =============================================================================

def analyze_floor_plan_gemini(
    image_path: str,
    client_type: str = "generic"
) -> Dict[str, Any]:
    """
    Analyze floor plan image with Gemini to extract layout and sample labels.

    Args:
        image_path: Path to floor plan image
        client_type: Client type for specific requirements

    Returns:
        Dict with room layout, sample labels, features
    """
    result = {
        "success": False,
        "source": "gemini_vision",
        "image_path": image_path,
        "client_type": client_type
    }

    # Convert image to base64
    image_data = image_path_to_base64(image_path)
    if not image_data:
        result["error"] = "Failed to read image file"
        return result

    # Build prompt based on client type
    if client_type.lower() == "cardtronics":
        prompt = """Analyze this Cardtronics ATM survey floor plan.

Identify:
1. RED areas (ACM - asbestos containing materials)
2. BLUE areas (No Access)
3. GREEN line (Cable route from DB to ATM)
4. Sample labels (S001, S002, etc.)
5. ATM location
6. Distribution Board (DB) location
7. Room names/labels
8. Legend

Respond in this format:
FLOOR_TITLE: [title]
ROOM_COUNT: [number]
ROOMS: [list with positions]
SAMPLE_LABELS: S001, S002, ...
HAS_RED_ACM: YES/NO
HAS_BLUE_NO_ACCESS: YES/NO
HAS_GREEN_CABLE: YES/NO
ATM_LOCATION: [description]
DB_LOCATION: [description]
HAS_LEGEND: YES/NO
QUALITY: GOOD/ACCEPTABLE/POOR

Only respond with the format above."""
    else:
        prompt = """Analyze this asbestos survey floor plan.

Identify:
1. Rooms and their names
2. Sample labels (S001, S002, etc.)
3. Color coding (ACM areas, no access)
4. Layout structure
5. Legend if present

Respond in this format:
FLOOR_TITLE: [title]
ROOM_COUNT: [number]
ROOMS: [list]
SAMPLE_LABELS: S001, S002, ...
HAS_COLOR_CODING: YES/NO
HAS_LEGEND: YES/NO
QUALITY: GOOD/ACCEPTABLE/POOR

Only respond with the format above."""

    # Call Gemini
    logger.info(f"[GEMINI] Analyzing floor plan: {Path(image_path).name}")
    api_result = call_gemini_vision(prompt, image_data, max_tokens=4096)

    if not api_result["success"]:
        result["error"] = api_result["error"]
        return result

    # Parse response
    text = api_result["text"]
    result["raw_response"] = text
    result["success"] = True
    result["model"] = api_result["model"]

    # Extract fields
    import re

    def extract_yes_no(field: str) -> Optional[bool]:
        match = re.search(rf"{field}:\s*(YES|NO)", text, re.IGNORECASE)
        return match.group(1).upper() == "YES" if match else None

    result["floor_title"] = re.search(r"FLOOR_TITLE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    result["floor_title"] = result["floor_title"].group(1).strip() if result["floor_title"] else None

    # Sample labels
    samples_match = re.search(r"SAMPLE_LABELS:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if samples_match:
        labels_text = samples_match.group(1).strip()
        result["sample_labels"] = re.findall(r"S\d{3}", labels_text.upper())
    else:
        result["sample_labels"] = []

    # Cardtronics-specific
    if client_type.lower() == "cardtronics":
        result["has_red_acm"] = extract_yes_no("HAS_RED_ACM")
        result["has_blue_no_access"] = extract_yes_no("HAS_BLUE_NO_ACCESS")
        result["has_green_cable"] = extract_yes_no("HAS_GREEN_CABLE")

    result["has_legend"] = extract_yes_no("HAS_LEGEND")
    result["quality"] = re.search(r"QUALITY:\s*(GOOD|ACCEPTABLE|POOR)", text, re.IGNORECASE)
    result["quality"] = result["quality"].group(1).upper() if result["quality"] else "UNKNOWN"

    logger.info(f"[GEMINI] Floor plan analyzed: {len(result.get('sample_labels', []))} samples, Quality: {result['quality']}")

    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gemini_usage_stats() -> Dict[str, Any]:
    """Get Gemini API usage statistics (all keys combined)."""
    usage = get_usage_today()
    total_limit = len(GEMINI_API_KEYS) * GEMINI_DAILY_LIMIT
    total_count = usage.get("total_count", 0)

    return {
        "date": usage.get("date"),
        "total_today": total_count,
        "limit": total_limit,
        "remaining": total_limit - total_count,
        "percent_used": round((total_count / total_limit) * 100, 1) if total_limit > 0 else 0,
        "keys_count": len(GEMINI_API_KEYS),
        "by_key": usage.get("by_key", {}),
        "by_feature": usage.get("by_feature", {}),
        "can_make_request": check_can_use_gemini()
    }


# =============================================================================
# CONFIDENCE GATE
# =============================================================================

def check_requires_manual_review(ai_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if AI result requires manual review based on confidence level.
    
    Adds requires_review flag and reason when confidence is LOW or uncertain.
    
    Args:
        ai_result: Result from building age extraction or other AI analysis
        
    Returns:
        Updated result dict with:
        - requires_review: bool
        - review_reason: str (if requires_review=True)
        - review_fields: list of fields needing review
    """
    requires_review = False
    review_reason = None
    review_fields = []
    
    confidence = ai_result.get("confidence", "MEDIUM").upper()
    
    # Low confidence always requires review
    if confidence == "LOW":
        requires_review = True
        review_reason = "AI confidence is LOW - manual verification required"
        review_fields.append("all_fields")
    
    # Check for missing critical fields
    critical_fields = ["age_estimate", "construction_type", "wall_material"]
    missing_fields = [f for f in critical_fields if not ai_result.get(f)]
    
    if missing_fields:
        requires_review = True
        review_reason = review_reason or "Missing critical building details"
        review_fields.extend(missing_fields)
    
    # Check for uncertain age estimates
    age_estimate = ai_result.get("age_estimate", "")
    uncertain_keywords = ["unclear", "uncertain", "unknown", "possibly", "maybe", "difficult"]
    if any(kw in age_estimate.lower() for kw in uncertain_keywords):
        requires_review = True
        review_reason = review_reason or "Age estimate is uncertain"
        if "age_estimate" not in review_fields:
            review_fields.append("age_estimate")
    
    # Check for conflicting context
    if ai_result.get("context_conflict"):
        requires_review = True
        review_reason = review_reason or "AI result conflicts with survey context"
        review_fields.append("context_conflict")
    
    # Update result
    ai_result["requires_review"] = requires_review
    if requires_review:
        ai_result["review_reason"] = review_reason
        ai_result["review_fields"] = list(set(review_fields))
        logger.warning(f"[AI] Manual review required: {review_reason}")
    
    return ai_result


def apply_confidence_gate(
    ai_result: Dict[str, Any],
    context: Dict[str, Any] = None,
    auto_downgrade: bool = True
) -> Dict[str, Any]:
    """
    Apply confidence gate to AI extraction result.
    
    If confidence is LOW and auto_downgrade is True:
    - Uses conservative defaults for age ("Built circa mid 1900s")
    - Flags result for manual review
    
    Args:
        ai_result: Result from AI analysis
        context: Survey context for validation
        auto_downgrade: Whether to auto-apply conservative defaults
        
    Returns:
        Updated result with confidence gate applied
    """
    # First check if review is needed
    ai_result = check_requires_manual_review(ai_result)
    
    # Check for context conflicts
    if context:
        survey_notes = context.get("project_notes", "") + context.get("building_details", "")
        ai_age = ai_result.get("age_estimate", "").lower()
        
        # Check for modern AI estimate vs traditional context
        modern_keywords = ["2000", "modern", "contemporary", "recent", "new build"]
        traditional_keywords = ["traditional", "brick", "masonry", "chimney", "period", "old"]
        
        ai_says_modern = any(kw in ai_age for kw in modern_keywords)
        context_says_traditional = any(kw in survey_notes.lower() for kw in traditional_keywords)
        
        if ai_says_modern and context_says_traditional:
            ai_result["context_conflict"] = True
            ai_result["requires_review"] = True
            ai_result["review_reason"] = "AI detected modern building but context suggests traditional construction"
            
            if auto_downgrade:
                # Apply conservative estimate
                ai_result["age_estimate_original"] = ai_result.get("age_estimate")
                ai_result["age_estimate"] = "Built circa mid 1900s"
                ai_result["age_era"] = "Post-war"
                ai_result["auto_downgraded"] = True
                logger.info("[AI] Auto-downgraded age estimate due to context conflict")
    
    return ai_result


# Export
__all__ = [
    "extract_building_age_gemini",
    "analyze_floor_plan_gemini",
    "call_gemini_vision",
    "call_gemini_text",
    "get_gemini_usage_stats",
    "check_can_use_gemini",
    "increment_usage",
    "check_requires_manual_review",
    "apply_confidence_gate",
]
