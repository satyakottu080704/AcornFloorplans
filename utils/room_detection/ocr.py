"""OCR integration for reading room labels and sample numbers from sketches.

Priority order:
  1. GPT-4o vision  - best handwriting accuracy, returns structured JSON with locations
  2. RapidOCR       - fast local ONNX, good handwriting, no API needed
  3. Tesseract      - free local OCR when installed on the OS
  4. EasyOCR        - fallback, cross-platform

Cross-platform (Windows + Ubuntu).
"""

import re
import math
import base64
import json
import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .models import DetectedRoom
from .ocr_paddle import run_paddle_ocr


# Module-level OCR reader caches
_rapidocr_engine = None
_easyocr_reader = None


def _get_rapidocr():
    """Lazy-initialize RapidOCR engine."""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _get_easyocr_reader():
    """Lazy-initialize EasyOCR reader (last resort fallback)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


def _configure_tesseract_cmd():
    """Point pytesseract at an explicitly configured Windows binary if needed."""
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if not cmd:
        return
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd
    except Exception:
        pass


SAMPLE_RE = re.compile(r'^[SP]\d{1,3}$')  # S or P prefix required (S01-S999, P01-P999)
REF_SAMPLE_RE = re.compile(r'^REF\s+S?\d{1,3}$', re.IGNORECASE)  # "Ref S001" cross-references
COMMERCIAL_ROOM_RE = re.compile(
    r'^\d{1,3}[A-Z]$|'         # 006A, 99A
    r'^\d{1,2}[A-Z]\d{1,3}$|'  # 2J43
    r'^\d{1,2}\.\d{1,3}$|'     # 2.112
    r'^\d{1,2}[A-Z]\d{1,3}[A-Z]$',  # 2J42A
    re.IGNORECASE
)

# Words/patterns that should NOT be treated as room labels
JUNK_PATTERNS = re.compile(
    r'(controlled\s*document|issue\s*\d|revision\s*\d|acorn|analytical|'
    r'surveyor|date|client|floor\s*:|site|address|page|ref\s*:|0129|'
    r'january|february|march|april|may|june|july|august|september|'
    r'october|november|december|prepared\s*by|limited|services)',
    re.IGNORECASE
)

# Quick junk filter for PaddleOCR/RapidOCR raw text (strips form area noise)
_FORM_JUNK_RE = re.compile(
    r'(controlled\s*document|issue\s*\d|revision|acorn|analytical|'
    r'surveyor|prepared\s*by|limited|services|0129|address|'
    r'january|february|march|april|may|june|july|august|september|'
    r'october|november|december|^\d{2}/\d{2}/\d{4}$|page\s*\d)',
    re.IGNORECASE
)

MARKER_MAP = {
    "ATM": "atm_location", "DB": "db_location",
    "ELECTRICS": "db_location", "ELECTRICAL": "db_location",
    "FLECTRI": "db_location",   # RapidOCR misreads of "ELECTRICAL"
    "DIS BOARD": "db_location", "DIS BOARP": "db_location",
    "DISBOARP": "db_location",  "DISBOARD": "db_location",
    "DISTRIBUTION": "db_location",
    "GAS": "gas_meter", "WATER": "water_stop_tap",
    "STOP TAP": "water_stop_tap",
}
ROOM_WORDS = {
    "KITCHEN", "BEDROOM", "BATHROOM", "HALL", "HALLWAY", "CORRIDOR",
    "LIVING", "LOUNGE", "DINING", "TOILET", "WC", "UTILITY",
    "OFFICE", "STORE", "STORAGE", "CUPBOARD", "CUPROARD", "LANDING", "STAIRS",
    "STAIR", "STAIRCASE", "STAIRWELL", "STEP", "STEPS",
    "LOBBY", "RECEPTION", "SHOP", "SECURE", "STAFF", "GARAGE",
    "LOFT", "ATTIC", "CELLAR", "BASEMENT", "PORCH", "ENTRANCE",
    "BOILER", "PLANT", "SERVER", "COMMS", "ROOM", "FRONT",
    "TILL", "COUNTER", "BACK", "FLOOR", "SHOPFLOON", "FRONTOF",
}


MATERIAL_ABBREVS = {
    "FT": "Floor tiles", "FB": "Fibreboard", "IB": "Insulating board",
    "TC": "Textured coating", "AIB": "Asbestos insulating board",
    "AC": "Asbestos cement", "ACM": "Asbestos-containing material",
    "MASTIC": "Mastic", "VINYL": "Vinyl floor tiles",
    "PUTTY": "Putty", "CEMENT": "Cement", "ROPE": "Rope seal",
    "GASKET": "Gasket", "PAPER": "Paper lining", "FELT": "Bitumen felt",
    "COATING": "Textured coating", "ARTEX": "Artex coating",
}


def _expand_material(abbrev: str) -> str:
    """Expand material abbreviation to full name. E.g. 'FT' -> 'Floor tiles'."""
    if not abbrev:
        return ""
    upper = abbrev.strip().upper()
    if upper in MATERIAL_ABBREVS:
        return MATERIAL_ABBREVS[upper]
    # If already a full word, return as-is
    return abbrev.strip()


def _normalize_ocr_text(text: str) -> str:
    """Normalize OCR text for consensus matching across engines."""
    if not text:
        return ""
    cleaned = text.upper().strip()
    cleaned = cleaned.replace("O", "0")
    cleaned = re.sub(r"[^A-Z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _text_has_consensus(label_text: str, secondary_texts: List[str]) -> bool:
    """Return True when label text is corroborated by a secondary OCR pass."""
    if not label_text or not secondary_texts:
        return False
    norm = _normalize_ocr_text(label_text)
    if not norm:
        return False
    norm_tokens = {t for t in norm.split() if t}
    for st in secondary_texts:
        if norm == st:
            return True
        st_tokens = {t for t in st.split() if t}
        if norm_tokens and st_tokens and (norm_tokens & st_tokens):
            return True
    return False


def _secondary_ocr_items(sketch: np.ndarray, debug_dir: Optional[str], primary_source: str) -> List[Dict]:
    """Run a lightweight secondary OCR pass for label-consensus filtering."""
    secondary: List[Dict] = []
    if primary_source != "paddle":
        paddle_items = run_paddle_ocr(sketch)
        secondary.extend({
            "text": item.get("text", ""),
            "conf": float(item.get("confidence", 0.0)),
        } for item in paddle_items if item.get("text"))
    if not secondary and primary_source != "rapid":
        secondary.extend(_run_rapidocr(sketch, debug_dir))
    if not secondary and primary_source != "tesseract":
        secondary.extend(_run_tesseract(sketch, debug_dir))
    if not secondary and primary_source != "easy":
        secondary.extend(_run_easyocr(sketch, debug_dir))
    return secondary


def run_ocr(
    sketch: np.ndarray,
    form_area: Optional[np.ndarray] = None,
    rooms: Optional[List[DetectedRoom]] = None,
    debug_dir: Optional[str] = None,
    use_paddle: bool = True,
) -> Dict:
    """
    Run OCR on sketch image to find sample labels, room names, and markers.

    Priority: GPT-4o (best handwriting) -> RapidOCR -> EasyOCR

    Returns:
        {
            "samples": [{"id": "S01", "location": (x,y), "material": None}],
            "labels":  [{"text": "Kitchen", "location": (x,y)}],
            "markers": {"atm_location": (x,y), "db_location": (x,y), ...},
            "floor_title": "Ground Floor" or None,
        }
    """
    samples = []
    labels = []
    markers = {}
    caveats = []
    floor_title = None
    local_only = os.environ.get("ACORN_LOCAL_OCR_ONLY", "").strip().lower() in {"1", "true", "yes"}

    # Try PaddleOCR first (free, local, precise bboxes), then GPT-4o as fallback.
    text_items = []
    ocr_source = "none"
    if use_paddle:
        paddle_items = run_paddle_ocr(sketch)
        text_items = [
            {
                "text": item.get("text", ""),
                "upper": str(item.get("text", "")).upper(),
                "center": item.get("location"),
                "conf": float(item.get("confidence", 0.0)),
            }
            for item in paddle_items
            if item.get("location")
        ]
        if text_items:
            ocr_source = "paddle"
    if (not text_items or len(text_items) < 3) and not local_only:
        text_items = _run_gpt4o_ocr(sketch, debug_dir)
        if text_items:
            ocr_source = "gpt4o"
    if not text_items:
        text_items = _run_rapidocr(sketch, debug_dir)
        if text_items:
            ocr_source = "rapid"
    if not text_items:
        text_items = _run_tesseract(sketch, debug_dir)
        if text_items:
            ocr_source = "tesseract"
    if not text_items:
        text_items = _run_easyocr(sketch, debug_dir)
        if text_items:
            ocr_source = "easy"

    secondary_items = _secondary_ocr_items(sketch, debug_dir, ocr_source)
    secondary_texts = [_normalize_ocr_text(i.get("text", "")) for i in secondary_items if i.get("text")]
    secondary_texts = [t for t in secondary_texts if t]

    try:
        print(f"[OCR] Found {len(text_items)} text items: "
              f"{[t['text'] for t in text_items[:20]]}")
    except UnicodeEncodeError:
        # Windows console can't handle some characters (e.g. Chinese from RapidOCR)
        safe_texts = [t['text'].encode('ascii', 'replace').decode() for t in text_items[:20]]
        print(f"[OCR] Found {len(text_items)} text items: {safe_texts}")

    # Classify each text item into samples / markers / room labels / floor titles
    for item in text_items:
        t = item["upper"]
        center = item["center"]
        item_type = item.get("type", "")
        material = item.get("material") or item.get("text_material", "")

        # --- Floor titles: "First Floor", "Ground Floor", "Ground", "Basement" ---
        if item_type == "floor_title" or any(ft in t for ft in ["FIRST FLOOR", "GROUND FLOOR", "GROUND", "BASEMENT", "LOFT SPACE"]):
            if not floor_title:
                floor_title = item["text"].strip()
            continue

        # --- Room numbers: circled digits like 01, 02, 03, 04 or 3-digit codes ---
        # Check BEFORE samples so bare digits with type=room_number aren't misclassified
        if item_type == "room_number":
            labels.append({"text": item["text"], "location": center, "is_room_number": True, "conf": item.get("conf", 1.0)})
            continue

        # Bare 1-2 digit numbers (01, 02, 03, 04) without material are room numbers, not samples
        # Even if GPT-4o says "sample", bare digits inside circled bubbles are room IDs
        cleaned = t.replace("O", "0").replace(",", "")
        # Don't strip dots for commercial room numbers like "2.112"
        cleaned_no_dot = cleaned.replace(".", "")
        is_bare_digit = cleaned_no_dot.isdigit() and len(cleaned_no_dot) <= 2 and not material
        if is_bare_digit and item_type == "sample":
            # Reclassify: bare digits with no material are room numbers
            labels.append({"text": item["text"], "location": center, "is_room_number": True, "conf": item.get("conf", 1.0)})
            continue

        # --- Commercial/alphanumeric room numbers: 006A, 2J43, 2.112 ---
        if (COMMERCIAL_ROOM_RE.match(cleaned) and not material
                and item_type not in ("sample", "ref_sample")):
            labels.append({"text": item["text"], "location": center, "is_room_number": True, "conf": item.get("conf", 1.0)})
            continue

        # --- "Ref S001" cross-references: same material found in another room ---
        ref_match = REF_SAMPLE_RE.match(t)
        if ref_match or item_type == "ref_sample":
            ref_text = t if ref_match else item["text"].upper()
            ref_num = re.sub(r'^REF\s+', '', ref_text, flags=re.IGNORECASE).lstrip("S")
            if ref_num and ref_num.isdigit():
                s = f"S{ref_num.zfill(2)}"
                mat = _expand_material(material) if material else None
                samples.append({"id": s, "location": center, "material": mat,
                                "is_reference": True})
            continue

        # --- Sample numbers: S01, S02, S001, P001, or digit+material like "301 FT" ---
        # Detect ACM positive "+" marker before normalizing
        acm_positive = "+" in cleaned_no_dot
        cleaned_for_parse = cleaned_no_dot.replace("+", "").strip()
        has_s_prefix = cleaned_for_parse.startswith("S") and len(cleaned_for_parse) >= 2 and cleaned_for_parse[1:].isdigit()
        has_p_prefix = cleaned_for_parse.startswith("P") and len(cleaned_for_parse) >= 2 and cleaned_for_parse[1:].isdigit()
        is_digit_with_material = cleaned_for_parse.isdigit() and material
        if has_s_prefix or has_p_prefix or item_type == "sample" or is_digit_with_material:
            prefix = "P" if has_p_prefix else "S"
            num_part = cleaned_for_parse.lstrip("SP")
            if num_part and num_part.isdigit():
                s = f"{prefix}{num_part.zfill(2)}"
                if not any(existing["id"] == s for existing in samples):
                    mat = _expand_material(material) if material else None
                    samples.append({"id": s, "location": center, "material": mat,
                                    "is_reference": False, "acm_positive": acm_positive})
            continue

        # --- Markers: ATM, Dis Board, Electrical, Gas, Water ---
        matched = False
        for kw, field in MARKER_MAP.items():
            if kw in t:
                markers[field] = center
                matched = True
                break
        if matched:
            continue

        # --- Catch "NUMBER MATERIAL" patterns that GPT-4o classified as "other" ---
        # e.g. "301 FT" = sample S301 with material Floor Tiles
        words = t.split()
        if len(words) == 2 and words[0].isdigit() and words[1].upper() in MATERIAL_ABBREVS:
            num_part = words[0]
            s = f"S{num_part.zfill(2)}"
            mat = _expand_material(words[1])
            if not any(existing["id"] == s for existing in samples):
                samples.append({"id": s, "location": center, "material": mat,
                                "is_reference": False, "acm_positive": False})
            continue

        # --- No-access caveat text ---
        caveat_patterns = ["NO ACCESS", "UNABLE TO GAIN ACCESS", "ACCESS NOT GAINED",
                           "COULD NOT ACCESS", "NOT ACCESSIBLE", "NO ACCESS WAS GAINED"]
        if any(cp in t for cp in caveat_patterns):
            caveats.append({"text": item["text"], "location": center})
            continue

        # --- Room labels: filter junk then accept if looks like a room ---
        if JUNK_PATTERNS.search(t):
            continue
        if any(w in t for w in ROOM_WORDS) or (1 <= len(t.split()) <= 4 and len(t) > 2):
            labels.append({"text": item["text"], "location": center, "conf": item.get("conf", 0.0)})

    # OCR consensus filter: remove weak room-name labels not corroborated by a second pass.
    # Room numbers are kept even without consensus.
    if secondary_texts and labels:
        min_conf = float(os.environ.get("OCR_CONSENSUS_MIN_CONF", "0.55"))
        filtered_labels = []
        dropped = 0
        for label in labels:
            if label.get("is_room_number"):
                filtered_labels.append(label)
                continue
            conf = float(label.get("conf", 0.0))
            if conf >= min_conf or _text_has_consensus(label.get("text", ""), secondary_texts):
                filtered_labels.append(label)
            else:
                dropped += 1
        if dropped:
            print(f"[OCR] Consensus filter dropped {dropped} weak labels")
        labels = filtered_labels

    # Floor title from form area
    if form_area is not None:
        floor_title = _extract_floor_title(form_area)

    try:
        print(f"[OCR] Samples: {[s['id'] for s in samples]}, "
              f"Labels: {[l['text'] for l in labels]}, "
              f"Markers: {list(markers.keys())}, "
              f"Floor: {floor_title}")
    except UnicodeEncodeError:
        safe_labels = [l['text'].encode('ascii', 'replace').decode() for l in labels]
        print(f"[OCR] Samples: {[s['id'] for s in samples]}, "
              f"Labels: {safe_labels}, "
              f"Markers: {list(markers.keys())}, "
              f"Floor: {floor_title}")

    return {
        "samples": samples,
        "labels": labels,
        "markers": markers,
        "floor_title": floor_title,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# GPT-4o Vision OCR  (primary - best handwriting accuracy)
# ---------------------------------------------------------------------------

_GPT4O_PROMPT = """You are reading a hand-drawn asbestos survey floor plan sketch.
Extract ALL text visible in the SKETCH AREA ONLY (not the form on the left/top).

Return ONLY valid JSON, no other text:
{
  "items": [
    {
      "text": "exact text as written",
      "type": "sample|ref_sample|room|room_number|marker|floor_title|caveat|other",
      "material": "",
      "x_pct": 45.2,
      "y_pct": 30.1
    }
  ]
}

Rules:
- "room_number": numbers inside CIRCLED bubbles within rooms: 01, 02, 03, 04 etc. These are the surveyor's room identification numbers. They are written in BLACK pen inside small circles. IMPORTANT: these are NOT samples.
  - Also includes COMMERCIAL/alphanumeric codes: 006A, 2J43, 2.112, 2J42A etc.
- "sample": sample references written in RED pen, usually near room edges or outside rooms. Format: "S01 FT", "S02 IB", "301 FT", "P001" etc.
  - Include the material abbreviation in "material" field: "FT" (floor tiles), "FB" (fibreboard), "IB" (insulating board), "Vinyl", "Mastic", "TC" (textured coating), "Cement", "Putty"
  - A number followed by a material abbreviation (e.g. "301 FT") is a sample reference, NOT a dimension
  - P-prefix samples (P001, P002) are preliminary/bulk samples — classify as "sample"
  - If a "+" appears next to the sample number (e.g. "S005 +"), include the "+" in the text — it means ACM positive
  - Example: text="S01", material="FT"  OR  text="301", material="FT"  OR  text="P001", material="TC"
- "ref_sample": cross-reference to an existing sample. Written as "Ref S001" or "REF S01" — means the same material was found in a different room. Example: text="Ref S001", material="TC"
- "room": room names like SHOP FLOOR, CASINO, KITCHEN, WC, STORE, LOBBY, STAFF ROOM, CORRIDOR, OFFICE, STAIRS, FRONT OF SHOP
- "marker": special equipment: ELECTRICAL DIS BOARD, ATM LOCATION, GAS METER, WATER STOP TAP
- "floor_title": floor section labels like "First Floor", "Ground Floor", "Ground", "Basement", "Loft"
- "caveat": no-access notes like "No access at tenants request", "Unable to gain access", "No access due to..."
- "other": everything else (scale, arrows, notes)
- x_pct and y_pct = position as percentage of IMAGE WIDTH and HEIGHT (0-100)
- Read ALL text carefully including RED pen (samples), BLUE pen (notes), and BLACK pen (room labels)
- Do NOT include text from the survey form border/title block
- CRITICAL: Circled numbers (01, 02, 03, 04) are ROOM NUMBERS, not samples. Only classify as "sample" if written in RED pen with material abbreviation nearby."""


def _run_gpt4o_ocr(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """
    Use GPT-4o vision to read all text from the sketch with accurate coordinates.

    Sends the sketch image (not the full photo) to GPT-4o and gets back
    structured JSON with every text item and its approximate position.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-proj-") is False and "sk-" not in api_key:
        print("[OCR-GPT4O] No OpenAI API key configured, skipping")
        return []

    try:
        import httpx

        # Encode sketch as JPEG (resize if huge to save tokens - 1500px is plenty)
        h, w = sketch.shape[:2]
        img = sketch.copy()
        MAX_DIM = 1500
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()

        max_attempts = int(os.environ.get("OPENAI_RETRY_ATTEMPTS", "3"))
        base_sleep = float(os.environ.get("OPENAI_RETRY_BASE_SECONDS", "1.5"))
        model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o").strip() or "gpt-4o"
        print(f"[OCR-GPT4O] Sending sketch to {model} vision...")

        resp = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 42,
                        "max_tokens": 1500,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high"
                                }},
                                {"type": "text", "text": _GPT4O_PROMPT},
                            ]
                        }]
                    },
                    timeout=45.0
                )
                resp.raise_for_status()
                break
            except Exception as e:
                should_retry = attempt < max_attempts
                print(f"[OCR-GPT4O] Attempt {attempt}/{max_attempts} failed: {e}")
                if not should_retry:
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1))
                time.sleep(min(sleep_s, 8.0))

        if resp is None:
            return []
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)
        items = data.get("items", [])

        img_h, img_w = sketch.shape[:2]
        text_items = []
        for it in items:
            text = str(it.get("text", "")).strip()
            if not text:
                continue
            x_pct = float(it.get("x_pct", 50))
            y_pct = float(it.get("y_pct", 50))
            px = int(x_pct / 100 * img_w)
            py = int(y_pct / 100 * img_h)
            item_type = it.get("type", "other")
            material = str(it.get("material", "")).strip()

            # Handle combined text like "S03 FB" — split into sample + material
            if item_type == "sample" and not material:
                parts = text.split(None, 1)
                if len(parts) == 2 and re.match(r'^S?\d{1,3}$', parts[0].upper()):
                    text = parts[0]
                    material = parts[1]

            text_items.append({
                "text": text,
                "upper": text.upper(),
                "center": (px, py),
                "conf": 0.99,
                "type": item_type,
                "material": material,
            })

        try:
            print(f"[OCR-GPT4O] Got {len(text_items)} items: "
                  f"{[t['text'] for t in text_items]}")
        except UnicodeEncodeError:
            print(f"[OCR-GPT4O] Got {len(text_items)} items (some non-ASCII)")
        return text_items

    except ImportError:
        print("[OCR-GPT4O] httpx not installed, skipping")
        return []
    except Exception as e:
        print(f"[OCR-GPT4O] Error ({e}), falling back to local OCR")
        return []


# ---------------------------------------------------------------------------
# GPT-4o Room Layout Extraction
# ---------------------------------------------------------------------------

_GPT4O_LAYOUT_PROMPT = """You are analyzing a hand-drawn asbestos survey floor plan sketch on grid paper.
The surveyor has drawn room outlines in BLACK PEN with room names and circled room numbers (01, 02, etc.) inside each room.

Your task: identify EVERY enclosed area drawn with black pen walls. The rooms must TILE the building — no gaps, no overlaps.

Return ONLY valid JSON (no markdown, no explanation):
{
  "floors": [
    {"title": "Ground Floor", "y1_pct": 0.0, "y2_pct": 100.0}
  ],
  "rooms": [
    {
      "name": "SHOP FLOOR",
      "number": "01",
      "floor": "Ground Floor",
      "has_acm": false,
      "x1_pct": 10.0,
      "y1_pct": 20.0,
      "x2_pct": 60.0,
      "y2_pct": 70.0
    }
  ],
  "samples": [
    {"id": "S301", "material": "FT", "x_pct": 30.0, "y_pct": 70.0, "is_ref": false, "acm_positive": false}
  ]
}

CRITICAL RULES FOR BOUNDING BOXES:
- Only return rooms that have a VISIBLE LABEL written by the surveyor. Do NOT invent rooms to fill space.
- Each room's bbox must TIGHTLY fit the BLACK PEN walls that enclose it.
- x1_pct/y1_pct = top-left corner, x2_pct/y2_pct = bottom-right corner (as % of image, 0-100).
- SHARED WALLS: adjacent rooms MUST use the EXACT SAME coordinate for their shared edge. If Room A's right edge is x=60% and Room B is next to it, Room B's x1_pct must be 60.0. NO GAPS between adjacent rooms.
- COLUMN ALIGNMENT: if Room A is above Room B (stacked vertically), they should share the same left edge (x1_pct) and same right edge (x2_pct) — unless an internal wall subdivides the column.
- The building OUTLINE is NOT a room.
- Include small labelled areas (till, counter, cupboard, WC) but ONLY if the surveyor drew and labelled them.

ROOM IDENTIFICATION:
- "name": read EXACTLY as written (e.g. "SHOP FLOOR", "CUPBOARD", "WC", "TILL")
- "number": the circled number inside the room (e.g. "01", "02", "03"). Empty string if none
- "has_acm": true ONLY if the room has BLUE diagonal hatching/shading
- Multiple rooms can share the same name (e.g. three rooms all called "SHOP FLOOR")

ROOM NUMBERS:
- Can be simple digits (01, 02, 03) or alphanumeric codes (006A, 2J43, 2.112) in commercial buildings
- Always inside CIRCLED bubbles written in BLACK pen

SAMPLES (RED PEN):
- Written in RED pen near room edges: "S01 FT", "301 FT", "S02 IB", "P001 TC"
- Materials: FT=floor tiles, FB=fibreboard, IB=insulating board, TC=textured coating
- P-prefix (P001) = preliminary/bulk sample, include as regular sample
- "Ref S001" = cross-reference (same material in different room), set "is_ref": true
- "S005 +" = ACM positive confirmed, set "acm_positive": true
- Circled BLACK numbers (01, 02, 03) inside rooms are ROOM NUMBERS, NOT samples

SCAN PROCEDURE:
1. Find the building outline (outermost black rectangle) — note its x1,y1,x2,y2
2. Find ALL internal walls that divide the building into rooms
3. For each enclosed area, read the room name and number
4. Assign bounding boxes so rooms TILE the building — shared walls use same coordinates, outer walls use building edge coordinates"""


def extract_room_layout_gpt4o(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """
    Use GPT-4o to identify room boundaries (bounding boxes) in the sketch.

    Returns list of:
        {"name": "Shop Floor", "bbox_pct": (x1, y1, x2, y2) as percentages}
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or "sk-" not in api_key:
        return []

    try:
        import httpx

        h, w = sketch.shape[:2]
        img = sketch.copy()
        # Higher res for better room detection — GPT-4o uses "high" detail mode
        MAX_DIM = 2000
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64 = base64.b64encode(buf.tobytes()).decode()

        model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o").strip() or "gpt-4o"
        print(f"[LAYOUT-GPT4O] Extracting room layout from sketch with {model}...")

        max_attempts = int(os.environ.get("OPENAI_RETRY_ATTEMPTS", "3"))
        base_sleep = float(os.environ.get("OPENAI_RETRY_BASE_SECONDS", "1.5"))
        resp = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 42,
                        "max_tokens": 4000,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high"
                                }},
                                {"type": "text", "text": _GPT4O_LAYOUT_PROMPT},
                            ]
                        }]
                    },
                    timeout=60.0
                )
                resp.raise_for_status()
                break
            except Exception as e:
                should_retry = attempt < max_attempts
                print(f"[LAYOUT-GPT4O] Attempt {attempt}/{max_attempts} failed: {e}")
                if not should_retry:
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1))
                time.sleep(min(sleep_s, 8.0))

        if resp is None:
            return []
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)
        rooms = data.get("rooms", [])

        result = []
        for room in rooms:
            name = str(room.get("name", "")).strip()
            if not name:
                continue
            x1 = float(room.get("x1_pct", 0))
            y1 = float(room.get("y1_pct", 0))
            x2 = float(room.get("x2_pct", 100))
            y2 = float(room.get("y2_pct", 100))
            room_data = {
                "name": name,
                "bbox_pct": (x1, y1, x2, y2),
            }
            # Preserve surveyor's room number if detected
            room_num = str(room.get("number", "")).strip()
            if room_num:
                room_data["number"] = room_num
            # Preserve ACM detection
            if room.get("has_acm"):
                room_data["has_acm"] = True
            # Preserve floor assignment
            floor = str(room.get("floor", "")).strip()
            if floor:
                room_data["floor"] = floor
            result.append(room_data)

        # Also extract floors and samples from GPT-4o layout response
        floors = data.get("floors", [])
        gpt4o_samples = data.get("samples", [])

        if floors:
            print(f"[LAYOUT-GPT4O] Floors: {[f.get('title', '?') for f in floors]}")
        if gpt4o_samples:
            print(f"[LAYOUT-GPT4O] Samples: {[s.get('id', '?') for s in gpt4o_samples]}")

        # Attach extra data as attributes on the result list for caller to use
        result_meta = {
            "floors": floors,
            "samples": gpt4o_samples,
        }
        # Store metadata on a special key that the caller can check
        if result:
            result[0]["_meta"] = result_meta

        print(f"[LAYOUT-GPT4O] Found {len(result)} rooms: {[r['name'] for r in result]}")
        return result

    except Exception as e:
        print(f"[LAYOUT-GPT4O] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# Gemini Room Layout Extraction (native bounding box support)
# ---------------------------------------------------------------------------

_GEMINI_LAYOUT_PROMPT = """You are analyzing a hand-drawn asbestos survey floor plan sketch on grid paper.
The surveyor has drawn room outlines in BLACK PEN with room names and circled room numbers (01, 02, etc.) inside each room.

Your task: identify EVERY enclosed area drawn with black pen walls. The rooms must TILE the building — no gaps, no overlaps.

Return ONLY valid JSON (no markdown, no explanation):
{
  "floors": [
    {"title": "Ground Floor", "y1_pct": 0.0, "y2_pct": 100.0}
  ],
  "rooms": [
    {
      "name": "SHOP FLOOR",
      "number": "01",
      "floor": "Ground Floor",
      "has_acm": false,
      "y_min": 200,
      "x_min": 100,
      "y_max": 700,
      "x_max": 600
    }
  ],
  "samples": [
    {"id": "S301", "material": "FT", "x_pct": 30.0, "y_pct": 70.0, "is_ref": false, "acm_positive": false}
  ]
}

BOUNDING BOX FORMAT:
- Coordinates are NORMALIZED 0-1000 (0=top/left edge of image, 1000=bottom/right edge).
- y_min/x_min = top-left corner of room, y_max/x_max = bottom-right corner.
- Each room's bbox must TIGHTLY fit the BLACK PEN walls that enclose it.

CRITICAL RULES:
- Only return rooms that have a VISIBLE LABEL written by the surveyor. Do NOT invent rooms.
- SHARED WALLS: adjacent rooms MUST use the EXACT SAME coordinate for their shared edge.
- COLUMN ALIGNMENT: if Room A is above Room B (stacked vertically), they should share the same left edge and same right edge — unless an internal wall subdivides the column.
- The building OUTLINE is NOT a room.
- Include small labelled areas (till, cupboard, WC) but ONLY if the surveyor drew and labelled them.

ROOM IDENTIFICATION:
- "name": read EXACTLY as written (e.g. "SHOP FLOOR", "CUPBOARD", "TILL")
- "number": the circled number inside the room (e.g. "01", "02"). Empty string if none visible.
- "floor": which floor this room belongs to (e.g. "Ground Floor", "First Floor", "Loft")
- "has_acm": true ONLY if the room has BLUE diagonal hatching/shading
- Multiple rooms can share the same name (e.g. three rooms all called "SHOP FLOOR")

MULTI-FLOOR PLANS:
- If the sketch shows multiple floors (e.g. Ground Floor and First Floor drawn side by side or top/bottom), list each floor in "floors" with its vertical position range (y1_pct/y2_pct as % of image).
- Assign each room to its correct floor in the "floor" field.

ROOM NUMBERS:
- Can be simple digits (01, 02, 03) or alphanumeric codes (006A, 2J43, 2.112) in commercial buildings
- Always inside CIRCLED bubbles written in BLACK pen

SAMPLES (RED PEN):
- Written in RED pen: "S01 FT", "301 FT", "P001 TC"
- Materials: FT=floor tiles, FB=fibreboard, IB=insulating board, TC=textured coating
- P-prefix (P001) = preliminary/bulk sample, include as regular sample
- "Ref S001" = cross-reference (same material in different room), set "is_ref": true
- "S005 +" = ACM positive confirmed, set "acm_positive": true
- Circled BLACK numbers inside rooms are ROOM NUMBERS, NOT samples
- Sample coordinates: x_pct and y_pct as percentage of image (0-100)"""


def extract_room_layout_gemini(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """
    Use Gemini to identify room boundaries using native bounding box coordinates.

    Gemini returns normalized 0-1000 coordinates which are more precise
    than GPT-4o's percentage-based estimation.

    Returns list of:
        {"name": "Shop Floor", "bbox_pct": (x1, y1, x2, y2) as percentages 0-100}
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []

    try:
        import httpx

        h, w = sketch.shape[:2]
        img = sketch.copy()
        MAX_DIM = 2000
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64 = base64.b64encode(buf.tobytes()).decode()

        print("[LAYOUT-GEMINI] Extracting room layout from sketch...")

        model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        resp = httpx.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                        {"text": _GEMINI_LAYOUT_PROMPT},
                    ]
                }],
                "generationConfig": {
                    "temperature": 0,
                    "topP": 1,
                    "maxOutputTokens": 4000,
                }
            },
            timeout=60.0
        )
        resp.raise_for_status()
        raw_resp = resp.json()

        # Extract text from Gemini response
        raw = raw_resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE).strip()

        data = json.loads(raw)
        rooms = data.get("rooms", [])

        result = []
        for room in rooms:
            name = str(room.get("name", "")).strip()
            if not name:
                continue

            # Gemini returns 0-1000 normalized coords → convert to 0-100 percentage
            y_min = float(room.get("y_min", 0)) / 10.0
            x_min = float(room.get("x_min", 0)) / 10.0
            y_max = float(room.get("y_max", 1000)) / 10.0
            x_max = float(room.get("x_max", 1000)) / 10.0

            room_data = {
                "name": name,
                "bbox_pct": (x_min, y_min, x_max, y_max),  # (x1, y1, x2, y2) as %
            }
            room_num = str(room.get("number", "")).strip()
            if room_num:
                room_data["number"] = room_num
            if room.get("has_acm"):
                room_data["has_acm"] = True
            # Per-room floor assignment (multi-floor support)
            floor = str(room.get("floor", "")).strip()
            if floor:
                room_data["floor"] = floor
            else:
                room_data["floor"] = data.get("floor_title", "Ground Floor")
            result.append(room_data)

        # Extract floors and samples (matching GPT-4o format)
        floors = data.get("floors", [])
        gemini_samples = data.get("samples", [])
        floor_title = data.get("floor_title", "Ground Floor")

        if floors:
            print(f"[LAYOUT-GEMINI] Floors: {[f.get('title', '?') for f in floors]}")
        if gemini_samples:
            print(f"[LAYOUT-GEMINI] Samples: {[s.get('id', '?') for s in gemini_samples]}")

        # Attach metadata on first room (same format as GPT-4o)
        if result:
            result[0]["_meta"] = {
                "floors": floors if floors else [{"title": floor_title, "y1_pct": 0, "y2_pct": 100}],
                "samples": gemini_samples,
            }

        print(f"[LAYOUT-GEMINI] Found {len(result)} rooms: {[r['name'] for r in result]}")
        return result

    except ImportError:
        print("[LAYOUT-GEMINI] httpx not installed")
        return []
    except Exception as e:
        print(f"[LAYOUT-GEMINI] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# PaddleOCR  (PRIMARY - free, local, precise pixel bboxes)
# ---------------------------------------------------------------------------

_paddle_ocr = None


def _get_paddle_ocr():
    """Lazy-init PaddleOCR (caches model weights on first call)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        try:
            _paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        except TypeError:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return _paddle_ocr


def _run_paddleocr(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """Run PaddleOCR (local, free, precise bboxes). Returns list of text items."""
    try:
        ocr = _get_paddle_ocr()
        print("[OCR] Running PaddleOCR on sketch...")

        h, w = sketch.shape[:2]
        scale = 1.0
        ocr_img = sketch.copy()
        MAX_DIM = 2500
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            ocr_img = cv2.resize(ocr_img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

        # CLAHE + denoise for better handwriting recognition
        ocr_img = _preprocess_for_easyocr(ocr_img)

        result = ocr.ocr(ocr_img, cls=True)
        if not result or not result[0]:
            return []

        text_items = []
        for line in result[0]:
            bbox_pts = line[0]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = str(line[1][0]).strip()
            conf = float(line[1][1])
            if conf < 0.3 or not text:
                continue
            # Filter junk from form area
            if _FORM_JUNK_RE.search(text):
                continue
            # Center of bbox, scaled back to original coords
            xs = [p[0] / scale for p in bbox_pts]
            ys = [p[1] / scale for p in bbox_pts]
            tx, ty = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
            text_items.append({
                "text": text,
                "upper": text.upper(),
                "center": (tx, ty),
                "conf": conf,
            })

        try:
            print(f"[OCR-PADDLE] Found {len(text_items)} items: "
                  f"{[t['text'] for t in text_items[:15]]}")
        except UnicodeEncodeError:
            print(f"[OCR-PADDLE] Found {len(text_items)} items (some non-ASCII)")
        return text_items

    except ImportError:
        print("[OCR-PADDLE] PaddleOCR not installed, trying RapidOCR...")
        return []
    except Exception as e:
        print(f"[OCR-PADDLE] Error ({e}), trying RapidOCR...")
        return []


# ---------------------------------------------------------------------------
# RapidOCR  (local fallback 1)
# ---------------------------------------------------------------------------

def _run_rapidocr(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """Run RapidOCR (ONNX, fast, local). Returns list of text items."""
    try:
        engine = _get_rapidocr()
        print("[OCR] Running RapidOCR on sketch...")

        MAX_DIM = 3000
        h, w = sketch.shape[:2]
        scale = 1.0
        ocr_img = sketch.copy()
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            ocr_img = cv2.resize(ocr_img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

        result, _ = engine(ocr_img)
        if not result:
            return []

        text_items = []
        for item in result:
            bbox_pts = item[0]
            text = str(item[1]).strip()
            conf = float(item[2])
            if conf < 0.35 or not text:
                continue
            xs = [p[0] / scale for p in bbox_pts]
            ys = [p[1] / scale for p in bbox_pts]
            tx, ty = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
            text_items.append({
                "text": text,
                "upper": text.upper(),
                "center": (tx, ty),
                "conf": conf,
            })

        print(f"[OCR-RAPID] Found {len(text_items)} text items")
        return text_items

    except ImportError:
        print("[OCR] RapidOCR not installed, trying EasyOCR...")
        return []
    except Exception as e:
        print(f"[OCR-RAPID] Error ({e}), trying EasyOCR...")
        return []


# ---------------------------------------------------------------------------
# Tesseract  (local fallback 2)
# ---------------------------------------------------------------------------

def _run_tesseract(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """Run free local Tesseract OCR. Returns list of text items."""
    try:
        _configure_tesseract_cmd()
        import pytesseract
        print("[OCR] Running Tesseract on sketch (local fallback)...")

        MAX_DIM = 2600
        h, w = sketch.shape[:2]
        scale = 1.0
        ocr_img = sketch.copy()
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            ocr_img = cv2.resize(ocr_img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        data = pytesseract.image_to_data(
            gray,
            config="--psm 11 --oem 3",
            output_type=pytesseract.Output.DICT,
        )
        text_items = []
        for idx, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text or "").strip()
            if not text:
                continue
            try:
                conf = float(data.get("conf", ["-1"])[idx])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 25:
                continue
            x = float(data.get("left", [0])[idx]) / scale
            y = float(data.get("top", [0])[idx]) / scale
            bw = float(data.get("width", [0])[idx]) / scale
            bh = float(data.get("height", [0])[idx]) / scale
            text_items.append({
                "text": text,
                "upper": text.upper(),
                "center": (int(x + bw / 2), int(y + bh / 2)),
                "conf": max(0.0, min(conf / 100.0, 1.0)),
            })

        print(f"[OCR-TESSERACT] Found {len(text_items)} text items")
        return text_items

    except ImportError:
        print("[OCR] pytesseract not installed, trying EasyOCR...")
        return []
    except Exception as e:
        print(f"[OCR-TESSERACT] Error ({e}), trying EasyOCR...")
        return []


# ---------------------------------------------------------------------------
# EasyOCR  (local fallback 3)
# ---------------------------------------------------------------------------

def _run_easyocr(sketch: np.ndarray, debug_dir: Optional[str] = None) -> List[Dict]:
    """Run EasyOCR (local, last resort). Returns list of text items."""
    try:
        reader = _get_easyocr_reader()
        print("[OCR] Running EasyOCR on sketch (fallback)...")

        MAX_DIM = 2048
        h, w = sketch.shape[:2]
        scale = 1.0
        ocr_img = sketch.copy()
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            ocr_img = cv2.resize(ocr_img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

        ocr_img = _preprocess_for_easyocr(ocr_img)
        ocr_results = reader.readtext(ocr_img, paragraph=False)

        text_items = []
        for (bbox_pts, text, conf) in ocr_results:
            if conf < 0.35 or not text.strip():
                continue
            xs = [p[0] / scale for p in bbox_pts]
            ys = [p[1] / scale for p in bbox_pts]
            tx, ty = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
            text_items.append({
                "text": text.strip(),
                "upper": text.strip().upper(),
                "center": (tx, ty),
                "conf": conf,
            })

        print(f"[OCR-EASYOCR] Found {len(text_items)} text items")
        return text_items

    except ImportError:
        print("[OCR] EasyOCR not installed - no text labels")
        return []
    except Exception as e:
        print(f"[OCR-EASYOCR] Error ({e}) - continuing without labels")
        return []


def _preprocess_for_easyocr(image: np.ndarray) -> np.ndarray:
    """CLAHE + denoise for better EasyOCR handwriting recognition."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=8, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Floor title extraction from form area
# ---------------------------------------------------------------------------

def _extract_floor_title(form_area: np.ndarray) -> Optional[str]:
    """Extract floor title (Ground/First/etc.) from the survey form area."""
    # Try RapidOCR first
    try:
        engine = _get_rapidocr()
        result, _ = engine(form_area)
        for item in (result or []):
            text = str(item[1]).strip()
            conf = float(item[2])
            if conf < 0.3:
                continue
            t = text.upper()
            if "FLOOR" in t and ":" not in t:
                return text
            if t in ("GROUND", "FIRST", "SECOND", "BASEMENT", "LOFT", "ALL"):
                return text
    except Exception:
        pass

    # Fallback to EasyOCR
    try:
        reader = _get_easyocr_reader()
        MAX_DIM = 2048
        fh, fw = form_area.shape[:2]
        form_ocr = form_area.copy()
        if max(fh, fw) > MAX_DIM:
            s = MAX_DIM / max(fh, fw)
            form_ocr = cv2.resize(form_ocr, (int(fw * s), int(fh * s)),
                                  interpolation=cv2.INTER_AREA)
        for _, text, conf in reader.readtext(form_ocr, paragraph=False):
            if conf < 0.2:
                continue
            t = text.strip().upper()
            if "FLOOR" in t and ":" not in t:
                return text.strip()
            if t in ("GROUND", "FIRST", "SECOND", "BASEMENT", "LOFT", "ALL"):
                return text.strip()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Label-to-room assignment
# ---------------------------------------------------------------------------

def assign_labels_to_rooms(
    rooms: List[DetectedRoom],
    labels: List[Dict],
) -> None:
    """
    Assign OCR-detected labels to rooms (modifies rooms in-place).

    Strategy: Label-centric assignment with "smallest container wins".
    For each label, find the best room it belongs to:
      1. Smallest room whose contour contains the label point (most specific)
      2. Smallest room whose bbox contains the label point
      3. Nearest room by distance (fallback)

    Using smallest room prevents large boundary rooms from stealing labels
    that belong to smaller rooms inside them.

    Labels can be assigned to multiple rooms (e.g. "SHOP FLOOR" written
    twice means both rooms are Shop Floor).
    """
    if not rooms or not labels:
        return

    # Build room area list for "smallest container" logic
    room_areas = [r.bbox[2] * r.bbox[3] for r in rooms]
    max_area = max(room_areas) if room_areas else 1

    # label_to_room: best room index for each label
    label_to_room: Dict[int, int] = {}  # label_idx -> room_idx

    for li, label_info in enumerate(labels):
        lx, ly = label_info["location"]

        best_room_idx = -1
        best_score = float('inf')

        for idx, room in enumerate(rooms):
            rx, ry, rw, rh = room.bbox
            cx, cy = rx + rw // 2, ry + rh // 2
            area_ratio = room_areas[idx] / max_area  # 0-1, larger = worse

            d = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)

            # Use contour-based point-in-polygon when available
            inside_contour = False
            if room.contour is not None:
                try:
                    pip = cv2.pointPolygonTest(room.contour, (float(lx), float(ly)), False)
                    inside_contour = pip >= 0
                except Exception:
                    pass

            # Score: lower is better. Prefer specific (small) containers over large ones.
            if inside_contour:
                # Inside actual contour: strongly prefer smallest containing room
                # area_ratio penalty pushes large rooms to lose even when they contain label
                score = d * 0.1 + area_ratio * 500
            elif rx <= lx <= rx + rw and ry <= ly <= ry + rh:
                # Inside bbox but not contour: less certain, penalise large rooms more
                score = d * 0.5 + area_ratio * 800
            elif (rx - rw * 0.3 <= lx <= rx + rw * 1.3 and
                  ry - rh * 0.3 <= ly <= ry + rh * 1.3):
                score = d * 0.8 + area_ratio * 400
            else:
                score = d + area_ratio * 200  # Distance + area penalty

            if score < best_score:
                best_score = score
                best_room_idx = idx

        if best_room_idx >= 0:
            label_to_room[li] = best_room_idx

    # Separate room names from room numbers
    room_name_map: Dict[int, List[Tuple[int, float]]] = {}   # room_idx -> [(label_idx, dist)]
    room_num_map: Dict[int, List[Tuple[int, float]]] = {}    # room_idx -> [(label_idx, dist)]

    for li, label_info in enumerate(labels):
        if li not in label_to_room:
            continue
        room_idx = label_to_room[li]
        lx, ly = label_info["location"]
        rx, ry, rw, rh = rooms[room_idx].bbox
        cx, cy = rx + rw // 2, ry + rh // 2
        d = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)

        if label_info.get("is_room_number"):
            room_num_map.setdefault(room_idx, []).append((li, d))
        else:
            room_name_map.setdefault(room_idx, []).append((li, d))

    # Assign room names (pick closest label to room centroid)
    for room_idx, label_scores in room_name_map.items():
        best_li, _ = min(label_scores, key=lambda x: x[1])
        rooms[room_idx].label = labels[best_li]["text"]

    # Assign room numbers (pick closest number to room centroid)
    for room_idx, num_scores in room_num_map.items():
        best_li, _ = min(num_scores, key=lambda x: x[1])
        num_text = labels[best_li]["text"].strip()
        rooms[room_idx].room_number = num_text.zfill(2) if num_text.isdigit() else num_text

    # Second pass: assign any remaining unassigned NAME labels to the nearest room
    assigned_labels = set(label_to_room.keys())
    unassigned = [li for li in range(len(labels))
                  if li not in assigned_labels and not labels[li].get("is_room_number")]
    for li in unassigned:
        label_info = labels[li]
        lx, ly = label_info["location"]
        best_room_idx = -1
        best_d = float('inf')
        for idx, room in enumerate(rooms):
            rx, ry, rw, rh = room.bbox
            cx, cy = rx + rw // 2, ry + rh // 2
            d = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            if d < best_d:
                best_d = d
                best_room_idx = idx
        if best_room_idx >= 0 and not rooms[best_room_idx].label:
            rooms[best_room_idx].label = label_info["text"]
