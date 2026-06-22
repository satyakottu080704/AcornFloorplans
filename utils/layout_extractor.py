"""Gemini floor-plan layout extraction.

Turns a surveyor's sketch image into a structured vector ``layout`` dict that
the native Visio exporters (``automation/container/generate_plan.py``) render
directly — walls, doors, windows, room labels and asbestos sample pins.

All coordinates are returned in a 1000x1000 space (origin top-left, x right,
y down) which the exporters scale to the A3 page. This module reuses the
existing Gemini plumbing in ``utils.gemini_vision`` (API-key rotation, retries
and daily-quota handling), so it honours every ``GEMINI_API_KEY`` /
``GEMINI_API_KEY_2`` ... configured for the project.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

try:
    # Repo layout (top-level ``utils`` package).
    from utils.gemini_vision import call_gemini_vision
except ImportError:
    try:
        # Container layout (files live under ``src/``).
        from src.utils.gemini_vision import call_gemini_vision
    except ImportError:
        # OpenAI extraction must remain available even if the optional Gemini
        # helper is missing from an image.
        call_gemini_vision = None

logger = logging.getLogger(__name__)

# The exporters expect this 1000x1000 coordinate space.
COORD_MAX = 1000.0

_LAYOUT_PROMPT = """You are a CAD draughtsman converting a hand-drawn or scanned floor plan
sketch into structured vector geometry.

Return ONLY a single JSON object (no markdown, no commentary) describing the
plan in a 1000x1000 coordinate grid where (0,0) is the TOP-LEFT corner,
x increases to the RIGHT and y increases DOWNWARD. Scale the drawing so it
fills most of the grid while keeping its real proportions.

Schema:
{
  "walls":   [{"x1":int,"y1":int,"x2":int,"y2":int,"type":"exterior"|"interior"}],
  "doors":   [{"hinge_x":int,"hinge_y":int,"open_x":int,"open_y":int,"closed_x":int,"closed_y":int,"label":str}],
  "windows": [{"x1":int,"y1":int,"x2":int,"y2":int}],
  "rooms":   [{"name":str,"x":int,"y":int}],
  "samples": [{"id":str,"x":int,"y":int}]
}

Rules:
- "walls" are straight line segments. Trace the actual wall lines so rooms read
  as closed shapes. Mark the outer perimeter walls as "exterior", internal
  partitions as "interior".
- "doors": hinge is the pivot point, "open_x/open_y" is the open leaf tip,
  "closed_x/closed_y" is the closed leaf tip, so the swing can be drawn.
- "windows" are the wall openings drawn as short segments along a wall.
- "rooms": (x,y) is the CENTRE of the room where its name label belongs. Use
  the room name written on the sketch; if none, use a sensible type
  (e.g. "Bedroom", "Kitchen", "Bathroom", "Hall", "Landing").
- "samples": small numbered/lettered markers (e.g. "1", "S01") for material
  samples, placed at their drawn position. Omit if none are shown.
- Only output geometry you can actually see. Do NOT invent rooms, walls or
  doors that are not in the sketch. Use [] for any category that is absent.
"""


def _encode_image(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def _mime_type(image_path: str) -> str:
    return {
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(Path(image_path).suffix.lower(), "image/jpeg")


def _configured_keys(prefix: str) -> List[str]:
    """Load PREFIX, PREFIX_2 ... PREFIX_10 without logging secret values."""
    keys = []
    for index in range(1, 11):
        name = prefix if index == 1 else f"{prefix}_{index}"
        value = os.environ.get(name, "").strip()
        if value:
            keys.append(value)
    return keys


def _extract_floor_plan_layout_openai(image_path: str) -> Dict[str, Any]:
    """Extract layout through OpenAI's vision API, rotating configured keys."""
    keys = _configured_keys("OPENAI_API_KEY")
    if not keys:
        raise ValueError("No OpenAI API keys configured")

    image_url = f"data:{_mime_type(image_path)};base64,{_encode_image(image_path)}"
    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o").strip() or "gpt-4o"
    errors = []

    for index, key in enumerate(keys, start=1):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 4096,
                    "response_format": {"type": "json_object"},
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _LAYOUT_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                        ],
                    }],
                },
                timeout=120,
            )
            if response.status_code != 200:
                errors.append(f"key {index}: HTTP {response.status_code}")
                continue
            content = response.json()["choices"][0]["message"]["content"]
            layout = _normalize_layout(_extract_json_object(content))
            if layout["walls"] or layout["rooms"]:
                return layout
            errors.append(f"key {index}: no walls or rooms")
        except Exception as exc:
            errors.append(f"key {index}: {type(exc).__name__}")

    raise ValueError("OpenAI layout extraction failed: " + ", ".join(errors))


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model response, tolerating fences."""
    if not text:
        raise ValueError("Empty response from Gemini")

    # Strip ```json ... ``` / ``` ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in Gemini response: {text[:200]}")

    return json.loads(candidate[start:end + 1])


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(COORD_MAX, value))


# Room name normalization — common abbreviations on Acorn sketches
_ROOM_NAME_MAP = {
    'k': 'Kitchen', 'kit': 'Kitchen', 'kitch': 'Kitchen',
    'lr': 'Living Room', 'lounge': 'Living Room', 'living': 'Living Room',
    'lobby': 'Lobby', 'loby': 'Lobby', 'lob': 'Lobby',
    'br1': 'Bedroom 1', 'br2': 'Bedroom 2', 'br3': 'Bedroom 3',
    'bed1': 'Bedroom 1', 'bed2': 'Bedroom 2', 'bed3': 'Bedroom 3',
    'bsd': 'Bed', 'bsp': 'Bed',
    'bath': 'Bathroom', 'bathrm': 'Bathroom',
    'wc': 'WC', 'toilet': 'WC',
    'corr': 'Corridor', 'hall': 'Hallway', 'landing': 'Landing',
    'cup': 'Cupboard', 'cpd': 'CPD', 'ac': 'Airing Cupboard',
    'gar': 'Garage', 'util': 'Utility Room', 'ut': 'Utility Room',
    'con': 'Conservatory', 'porch': 'Porch', 'ent': 'Entrance',
    'st': 'Store Room', 'store': 'Store Room',
    'off': 'Office', 'rec': 'Reception',
    'din': 'Dining Room', 'dr': 'Dining Room',
    'loft': 'Loft', 'attic': 'Loft',
    'stairs': 'Stairs', 'staircase': 'Stairs',
    'en': 'En-Suite', 'ensuite': 'En-Suite', 'en-suite': 'En-Suite',
    'esu suite': 'En-Suite', 'es suite': 'En-Suite', 'en suite': 'En-Suite',
    'cloak': 'Cloakroom', 'cloakrm': 'Cloakroom',
    'boiler': 'Boiler Room', 'plant': 'Plant Room',
    'wait': 'Waiting Area', 'waiting': 'Waiting Area',
}


def _normalize_room_name(name: str) -> str:
    """Normalize abbreviated room names to full names."""
    if not name:
        return name
    key = name.lower().strip().rstrip('.')
    key = re.sub(r"\s+", " ", key)
    if "candidate" in key:
        return ""
    # Exact match
    if key in _ROOM_NAME_MAP:
        return _ROOM_NAME_MAP[key]
    # Check if already a full name
    full_names = set(_ROOM_NAME_MAP.values())
    for fn in full_names:
        if key == fn.lower() or name.strip() == fn:
            return name.strip()
    # Check partial matches (e.g. "bed 1" → "Bedroom 1")
    prefix_map = dict(_ROOM_NAME_MAP)
    prefix_map['bed'] = 'Bedroom'
    prefix_map['br'] = 'Bedroom'
    for abbr, full in prefix_map.items():
        if key.startswith(abbr + ' ') and len(abbr) >= 2:
            suffix = name[len(abbr):].strip()
            if suffix.lower() in full.lower():
                return full
            return f"{full} {suffix}" if suffix else full
    return name


def _clean_number(val: Any) -> str:
    if not val:
        return ""
    s = str(val).strip().lstrip('#').lstrip('0')
    if not s:
        return ""
    m = re.search(r'\d+', s)
    if m:
        return m.group(0).zfill(3)
    return ""


def _parse_room_name_and_number(raw_name: str) -> Tuple[str, str]:
    """Parse name and number from raw string (e.g. '008 BSP' -> 'Bed', '008')."""
    raw = str(raw_name or "").strip()
    num = ""

    # Check for digits at start: "008 BSP"
    m1 = re.match(r'^(\d{1,4})\s+(.+)$', raw)
    if m1:
        num = m1.group(1)
        raw = m1.group(2).strip()
    else:
        # Check for digits at end: "BSP 008"
        m2 = re.match(r'^(.+?)\s+(\d{1,4})$', raw)
        if m2:
            num = m2.group(2)
            raw = m2.group(1).strip()

    norm_name = _normalize_room_name(raw)
    cleaned_num = _clean_number(num)
    return norm_name, cleaned_num


def _normalize_layout(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the raw model JSON into the exact shape the exporters consume."""
    walls: List[Dict[str, Any]] = []
    for w in raw.get("walls") or []:
        w_type = str(w.get("type", "interior")).strip().lower()
        if w_type not in ("exterior", "interior"):
            w_type = "interior"
        walls.append({
            "x1": _clamp(_num(w.get("x1"))), "y1": _clamp(_num(w.get("y1"))),
            "x2": _clamp(_num(w.get("x2"))), "y2": _clamp(_num(w.get("y2"))),
            "type": w_type,
        })

    doors: List[Dict[str, Any]] = []
    for d in raw.get("doors") or []:
        doors.append({
            "hinge_x": _clamp(_num(d.get("hinge_x"))), "hinge_y": _clamp(_num(d.get("hinge_y"))),
            "open_x": _clamp(_num(d.get("open_x"))), "open_y": _clamp(_num(d.get("open_y"))),
            "closed_x": _clamp(_num(d.get("closed_x"))), "closed_y": _clamp(_num(d.get("closed_y"))),
            "label": str(d.get("label", "")),
        })

    windows: List[Dict[str, Any]] = []
    for win in raw.get("windows") or []:
        windows.append({
            "x1": _clamp(_num(win.get("x1"))), "y1": _clamp(_num(win.get("y1"))),
            "x2": _clamp(_num(win.get("x2"))), "y2": _clamp(_num(win.get("y2"))),
        })

    rooms: List[Dict[str, Any]] = []
    for r in raw.get("rooms") or []:
        raw_name = str(r.get("name", "")).strip() or "Room"
        norm_name, num = _parse_room_name_and_number(raw_name)
        if not norm_name:
            continue
        room_dict = {
            "name": norm_name,
            "x": _clamp(_num(r.get("x"))),
            "y": _clamp(_num(r.get("y"))),
        }
        if num:
            room_dict["room_number"] = num
        rooms.append(room_dict)

    samples: List[Dict[str, Any]] = []
    for s in raw.get("samples") or []:
        samples.append({
            "id": str(s.get("id", "")).strip() or "S001",
            "x": _clamp(_num(s.get("x"))),
            "y": _clamp(_num(s.get("y"))),
        })

    return {
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "rooms": rooms,
        "samples": samples,
    }


def extract_floor_plan_layout(image_path: str) -> Dict[str, Any]:
    """Extract a vector floor-plan layout from a sketch image via GPT-4o-mini.

    Args:
        image_path: Path to the surveyor sketch image (JPG/PNG).

    Returns:
        Layout dict with ``walls``, ``doors``, ``windows``, ``rooms`` and
        ``samples`` in 1000x1000 coordinates.

    Raises:
        ValueError: if the image cannot be read or OpenAI returns no usable
            geometry (the caller decides whether to fall back to a placeholder).
    """
    image_file = Path(image_path)
    if not image_file.is_file():
        raise ValueError(f"Sketch image not found: {image_path}")

    providers = ["openai"]
    errors = []

    for provider in providers:
        try:
            logger.info("Extracting floor-plan layout from %s via %s...", image_file.name, provider)
            if provider == "openai":
                layout = _extract_floor_plan_layout_openai(image_path)
            elif provider == "gemini":
                if call_gemini_vision is None:
                    raise ValueError("Gemini helper module is not installed")
                result = call_gemini_vision(
                    _LAYOUT_PROMPT,
                    _encode_image(image_path),
                    max_tokens=16384,
                    response_mime_type="application/json",
                )
                if not result.get("success"):
                    raise ValueError(result.get("error") or "unknown Gemini failure")
                layout = _normalize_layout(_extract_json_object(result.get("text", "")))
            else:
                errors.append(f"{provider}: unsupported provider")
                continue

            if not (layout["walls"] or layout["rooms"]):
                raise ValueError("provider returned no walls or rooms")

            logger.info(
                "%s layout: %d walls, %d doors, %d windows, %d rooms, %d samples",
                provider,
                len(layout["walls"]), len(layout["doors"]), len(layout["windows"]),
                len(layout["rooms"]), len(layout["samples"]),
            )
            return layout
        except Exception as exc:
            logger.warning("%s layout extraction failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")

    raise ValueError("All configured layout providers failed: " + "; ".join(errors))
