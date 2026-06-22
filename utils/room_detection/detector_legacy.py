"""
Main multi-strategy room detector.

Tries 4 OpenCV strategies, scores each, picks the best.
Room geometry always comes from OpenCV contour detection.
AI (Gemini/GPT-4o) is used ONLY to label each room by cropping the bbox region.
"""

import os
import re
import tempfile
import cv2
import numpy as np
from typing import Optional

from .models import DetectedRoom, FloorPlanAnalysis
from .preprocessing import preprocess_sketch
from .color_analysis import analyze_colors
from .helpers import score_detection_result, merge_overlapping_rooms
from .ocr import run_ocr, assign_labels_to_rooms
from .strategies.dark_threshold import strategy_dark_threshold
from .strategies.adaptive_distance import strategy_adaptive_distance
from .strategies.color_erasure import strategy_color_erasure_flood
from .strategies.watershed import strategy_watershed
from .strategies.edge_tracing import strategy_edge_tracing


def detect_geometry_only(
    image_path: str,
    debug_dir: Optional[str] = None,
) -> list:
    """Geometry-only room detection — no OCR, no AI, no color analysis.

    Returns DetectedRoom objects with only bbox and contour populated.
    Labels, room_type, room_number, and floor are NOT set (those come from API).

    Much faster than detect_rooms_multi_strategy because it skips:
    - OCR (run_ocr)
    - AI layout detection (GPT-4o, Gemini)
    - Color analysis (ACM detection)
    - Label assignment
    - Virtual room creation

    Args:
        image_path: Path to sketch image
        debug_dir: Optional debug output directory

    Returns:
        List of DetectedRoom objects with bbox and contour only
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    print(f"[GEOM] Image size: {w}x{h}")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # Preprocessing (crop form, deskew, normalize)
    sketch, form_area = preprocess_sketch(image, debug_dir)
    gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sketch, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 40, 40), (130, 255, 255))
    dark_pixels = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
    walls_only = cv2.bitwise_and(dark_pixels, cv2.bitwise_not(blue_mask))
    walls_only = cv2.morphologyEx(
        walls_only,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    gray_for_detection = walls_only
    sketch_h, sketch_w = sketch.shape[:2]
    print(f"[GEOM] Sketch area: {sketch_w}x{sketch_h}")

    # OpenCV detection strategies (geometry only)
    strategies = [
        ("A_DarkThreshold", lambda: strategy_dark_threshold(gray_for_detection, debug_dir, color_image=sketch)),
        ("B_AdaptiveDist", lambda: strategy_adaptive_distance(gray_for_detection, debug_dir)),
        ("C_ColorErasure", lambda: strategy_color_erasure_flood(sketch, analyze_colors(sketch, debug_dir), debug_dir)),
        ("D_Watershed", lambda: strategy_watershed(sketch, gray_for_detection, debug_dir)),
    ]

    best_rooms = []
    best_score = 0.0
    best_strategy = "none"

    for name, strategy_fn in strategies:
        try:
            rooms = strategy_fn()
            score = score_detection_result(rooms, gray_for_detection.shape)
            print(f"[GEOM] {name}: {len(rooms)} rooms, score={score:.2f}")

            if score > best_score:
                best_score = score
                best_rooms = rooms
                best_strategy = name

            if score >= 0.85:
                print(f"[GEOM] Good result from {name}, skipping remaining")
                break
        except Exception as e:
            print(f"[GEOM] {name} failed: {e}")
            continue

    # Merge overlapping rooms
    best_rooms = merge_overlapping_rooms(best_rooms, iou_threshold=0.4)

    # Filter building boundary rooms
    best_rooms = _filter_building_boundary_rooms(best_rooms, gray.shape)

    # Filter text-only regions
    best_rooms = _filter_text_only_rooms(best_rooms, gray_for_detection)

    print(f"[GEOM] Result: {best_strategy} -> {len(best_rooms)} rooms (score={best_score:.2f})")
    return best_rooms


def detect_rooms_multi_strategy(
    image_path: str,
    debug_dir: Optional[str] = None,
    ai_fallback: bool = True,
    ai_provider: str = "ollama",
    client_type: Optional[str] = None,
) -> FloorPlanAnalysis:
    """
    Main entry point for room detection.

    Tries multiple strategies and returns the best result.

    Args:
        image_path: Path to sketch image
        debug_dir: Directory to save debug images (None to disable)
        ai_fallback: Whether to use AI if all OpenCV strategies fail
        ai_provider: "anthropic", "openai", or "ollama"

    Returns:
        FloorPlanAnalysis with detected rooms and metadata
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    print(f"[DETECT] Image size: {w}x{h}")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # Step 1: Preprocessing (crop form, deskew, normalize)
    sketch, form_area = preprocess_sketch(image, debug_dir)
    gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sketch, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 40, 40), (130, 255, 255))
    dark_pixels = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
    walls_only = cv2.bitwise_and(dark_pixels, cv2.bitwise_not(blue_mask))
    walls_only = cv2.morphologyEx(
        walls_only,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    gray_for_detection = walls_only
    sketch_h, sketch_w = sketch.shape[:2]
    print(f"[DETECT] Sketch area: {sketch_w}x{sketch_h}")
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "03_walls_only_mask.png"), gray_for_detection)

    # Step 2: Color analysis
    color_analysis = analyze_colors(sketch, debug_dir)

    force_contours = os.environ.get("FORCE_CONTOUR_MODE", "").lower() in {"1", "true", "yes"}
    local_only = os.environ.get("ACORN_LOCAL_OCR_ONLY", "").lower() in {"1", "true", "yes"}
    if force_contours:
        ai_fallback = False

    # Step 2b: OpenCV detection strategies (PRIMARY - accurate room boundaries)
    strategies = [
        ("A_DarkThreshold", lambda: strategy_dark_threshold(gray_for_detection, debug_dir, color_image=sketch)),
        ("B_AdaptiveDist", lambda: strategy_adaptive_distance(gray_for_detection, debug_dir)),
        ("C_ColorErasure", lambda: strategy_color_erasure_flood(sketch, color_analysis, debug_dir)),
        ("D_Watershed", lambda: strategy_watershed(sketch, gray_for_detection, debug_dir)),
    ]
    if force_contours:
        strategies.insert(0, ("E_EdgeTrace", lambda: strategy_edge_tracing(sketch, debug_dir)))

    best_rooms = []
    best_score = 0.0
    best_strategy = "none"

    for name, strategy_fn in strategies:
        try:
            rooms = strategy_fn()
            score = score_detection_result(rooms, gray_for_detection.shape)
            print(f"[DETECT] {name}: {len(rooms)} rooms, score={score:.2f}")

            if score > best_score:
                best_score = score
                best_rooms = rooms
                best_strategy = name

            # Early exit if we have a very good result
            if score >= 0.85:
                print(f"[DETECT] Good result from {name}, skipping remaining strategies")
                break

        except Exception as e:
            print(f"[DETECT] {name} failed: {e}")
            continue

    # Merge overlapping rooms
    best_rooms = merge_overlapping_rooms(best_rooms, iou_threshold=0.4)

    # Filter building boundary rooms first (removes full-sketch outlines while we have enough rooms)
    best_rooms = _filter_building_boundary_rooms(best_rooms, gray.shape)

    # Filter out text-only regions and footer artifacts
    # Use walls_only mask (blue removed) so hatching pixels don't inflate wall density
    best_rooms = _filter_text_only_rooms(best_rooms, gray_for_detection)

    print(f"[DETECT] Best strategy: {best_strategy} ({len(best_rooms)} rooms, score={best_score:.2f})")

    # Step 4b: AI crop-labeling — use model geometry, ask AI only for room names
    # For each room bbox detected by OpenCV, crop that region from the sketch
    # and send the crop to Gemini (free) or GPT-4o (fallback) to identify the room name.
    # This keeps accurate OpenCV geometry and only uses AI for what it's good at: reading text.
    if best_rooms and not force_contours and not local_only:
        _label_rooms_with_ai_crops(sketch, best_rooms)
    elif best_rooms and local_only:
        print("[DETECT] ACORN_LOCAL_OCR_ONLY enabled - skipping AI crop labelling")

    # Step 4c: Fix rooms that completely contain other rooms (overlap filter)
    # GPT-4o sometimes returns a "Loft" room covering the entire first floor area.
    # Instead of removing it, reposition it as a detached room to the right of the plan.
    all_max_x = max((r.bbox[0] + r.bbox[2]) for r in best_rooms) if best_rooms else sketch_w
    all_max_y_top = 0  # top of the plan (y=0 in image coords)
    reposition_gap = int(sketch_w * 0.05)  # small gap between plan and detached room

    for i, ri in enumerate(best_rooms):
        ix, iy, iw, ih = ri.bbox
        contained_count = 0
        for j, rj in enumerate(best_rooms):
            if i == j:
                continue
            jx, jy, jw, jh = rj.bbox
            if jx >= ix and jy >= iy and (jx + jw) <= (ix + iw) and (jy + jh) <= (iy + ih):
                contained_count += 1
        # If room i contains 3+ other rooms, reposition it as a detached room
        if contained_count >= 3:
            # Place it to the right of the plan, same height as a standard room
            new_w = int(sketch_w * 0.20)  # 20% of sketch width
            new_h = int(sketch_h * 0.20)  # 20% of sketch height
            new_x = all_max_x + reposition_gap
            new_y = all_max_y_top
            ri.bbox = (new_x, new_y, new_w, new_h)
            print(f"[FILTER] Repositioned room '{ri.label or '?'}' — was container of {contained_count} rooms, "
                  f"moved to ({new_x},{new_y},{new_w},{new_h})")

    # Step 5: Assign ACM status based on blue overlap + AI hint
    # Two-tier approach:
    #   - High confidence (>= 5% blue overlap): mark as ACM regardless of AI hint
    #   - AI-boosted (any blue > 0.1% AND GPT-4o flagged ACM): trust the AI vision
    #     GPT-4o can see hatching that pixel analysis misses (faint pen, partial overlap)
    #   - No blue (< 0.1%): mark as clear even if AI says ACM (false positive)
    ACM_THRESHOLD_HIGH = 0.05   # Blue overlap alone is sufficient
    ACM_THRESHOLD_AI_BOOST = 0.001  # Minimal blue + AI hint = trust AI
    for room in best_rooms:
        rx, ry, rw, rh = room.bbox
        ry_end = min(ry + rh, color_analysis["blue_mask"].shape[0])
        rx_end = min(rx + rw, color_analysis["blue_mask"].shape[1])

        if room.contour is not None:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [room.contour], 0, 255, -1)
            blue_overlap = cv2.bitwise_and(mask, color_analysis["blue_mask"])
            overlap_pct = cv2.countNonZero(blue_overlap) / max(room.area, 1)
        else:
            # Fallback: check blue mask overlap with bounding box
            if rw > 0 and rh > 0:
                roi = color_analysis["blue_mask"][ry:ry_end, rx:rx_end]
                overlap_pct = cv2.countNonZero(roi) / max(rw * rh, 1)
            else:
                overlap_pct = 0.0

        ai_said_acm = room.room_type == "acm"  # preserve GPT-4o's original ACM flag
        threshold = ACM_THRESHOLD_AI_BOOST if ai_said_acm else ACM_THRESHOLD_HIGH
        print(f"[DETECT] Room bbox=({rx},{ry},{rw},{rh}) blue overlap: {overlap_pct:.1%} "
              f"(AI hint: {'ACM' if ai_said_acm else 'clear'}, threshold: {threshold:.1%})")

        if overlap_pct >= threshold:
            room.room_type = "acm"
            room.color_detected = "blue"
        else:
            room.room_type = "clear"
            room.color_detected = None

    # Step 6: OCR for labels and samples
    ocr_result = run_ocr(sketch, form_area, best_rooms, debug_dir)

    # AI layout strategies (GPT4O, Gemini, Cache) already have labels/numbers
    assign_labels_to_rooms(best_rooms, ocr_result["labels"])
    _improve_labels_with_targeted_ocr(sketch, best_rooms, ocr_result)

    # Create virtual rooms for OCR labels not matched to any detected room
    # (catches rooms that both OpenCV and GPT-4o missed)
    best_rooms = _add_virtual_rooms_for_orphan_labels(
        best_rooms, ocr_result["labels"], sketch_h, sketch_w
    )

    # Step 7: Calculate quality
    if best_score >= 0.7:
        quality = "GOOD"
    elif best_score >= 0.4:
        quality = "FAIR"
    else:
        quality = "POOR"

    # Step 8: Save debug visualization
    if debug_dir and best_rooms:
        _save_debug_visualization(sketch, best_rooms, ocr_result, debug_dir)

    # Merge samples from GPT-4o layout with OCR samples (deduplicate by ID)
    # First, filter out any samples that match room numbers (e.g. S01 when room 01 exists)
    room_nums = {r.room_number for r in best_rooms if r.room_number}
    room_nums_padded = set()
    for rn in room_nums:
        room_nums_padded.add(rn)
        room_nums_padded.add(rn.lstrip("0") or "0")  # "01" -> "1"
        room_nums_padded.add(rn.zfill(2))              # "1" -> "01"
        room_nums_padded.add(rn.zfill(3))              # "1" -> "001"

    def _is_room_number_sample(sid: str) -> bool:
        """Check if a bare digit like '003' is actually a room number, not a sample.
        Only filter IDs that DON'T start with 'S' or 'P' — if OCR found 'S001' or 'P001',
        it's a real sample reference even if room 001 exists."""
        upper = sid.upper()
        if upper.startswith("S") or upper.startswith("P"):
            return False  # Has S/P prefix = real sample, never filter
        return sid in room_nums_padded

    all_samples = [s for s in ocr_result["samples"] if not _is_room_number_sample(s["id"])]

    # Normalize sample IDs for dedup: "301" and "S301" refer to the same sample
    def _normalize_sid(sid: str) -> str:
        upper = sid.upper()
        if upper.startswith("P"):
            num = sid[1:]
            return f"P{num.zfill(2)}" if num.isdigit() else sid
        num = sid.lstrip("S")
        return f"S{num.zfill(2)}" if num.isdigit() else sid
    existing_ids = {_normalize_sid(s["id"]) for s in all_samples}

    # Extract floor info from OCR
    floors = []
    if ocr_result.get("floor_title"):
        floors = [{"title": ocr_result["floor_title"]}]

    acm_count = sum(1 for r in best_rooms if r.room_type == "acm")
    labeled = [r.label for r in best_rooms if r.label]
    nums = [r.room_number for r in best_rooms if r.room_number]
    print(f"[DETECT] Final: {len(best_rooms)} rooms ({acm_count} ACM), "
          f"labels={labeled}, nums={nums}, "
          f"samples={[s['id'] for s in all_samples]}, "
          f"quality={quality}")

    return FloorPlanAnalysis(
        rooms=best_rooms,
        samples=all_samples,
        acm_regions=color_analysis["blue_regions"],
        cable_route=color_analysis["green_path"],
        atm_location=ocr_result["markers"].get("atm_location"),
        db_location=ocr_result["markers"].get("db_location"),
        gas_meter=ocr_result["markers"].get("gas_meter"),
        water_stop_tap=ocr_result["markers"].get("water_stop_tap"),
        text_labels=ocr_result["labels"],
        quality_score=quality,
        detection_method=best_strategy,
        floors=floors,
        caveats=ocr_result.get("caveats", []),
    )


def _label_rooms_with_ai_crops(sketch: np.ndarray, rooms: list) -> None:
    """Label rooms by cropping each bbox from the sketch and asking AI for the room name only.

    Uses Gemini Flash (free) as primary, GPT-4o-mini as fallback.
    AI is NOT asked for bounding boxes or layout — only: "What room is this?"

    Modifies rooms in-place by setting room.label.
    """
    import base64
    import json
    import httpx

    h, w = sketch.shape[:2]

    _CROP_LABEL_PROMPT = (
        "This is a cropped region from a hand-drawn UK asbestos survey floor plan. "
        "What room or area does this show? Reply with ONLY the room name, e.g. "
        "'Kitchen', 'Bedroom 1', 'Landing', 'Bathroom', 'WC', 'Cupboard', 'Hall', "
        "'Shop Floor', 'Till', 'Store', 'Office', 'Corridor', 'Loft', 'Stairs'. "
        "If there is a circled number (like 01, 02, 03), also include it after a pipe: "
        "'Kitchen|01'. If no number is visible, just the name. "
        "If unclear or no text visible, reply 'Unknown'."
    )

    # Use 29-key Gemini rotation from gemini_vision.py
    try:
        from utils.gemini_vision import get_available_key, increment_usage, mark_quota_exceeded
        has_gemini_rotation = True
    except ImportError:
        has_gemini_rotation = False

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")

    labeled_count = 0
    for i, room in enumerate(rooms):
        rx, ry, rw, rh = room.bbox

        # Add 10% padding around the crop for context
        pad_x = max(int(rw * 0.10), 10)
        pad_y = max(int(rh * 0.10), 10)
        cx1 = max(0, rx - pad_x)
        cy1 = max(0, ry - pad_y)
        cx2 = min(w, rx + rw + pad_x)
        cy2 = min(h, ry + rh + pad_y)

        crop = sketch[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf.tobytes()).decode()

        label = None

        # Try Gemini with key rotation (29 keys, free)
        if has_gemini_rotation and not label:
            # Retry up to 3 times with different keys on 429
            for _attempt in range(3):
                key_info = get_available_key()
                if not key_info:
                    print(f"[CROP-LABEL] All Gemini keys exhausted")
                    break
                key_idx, api_key = key_info
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
                    resp = httpx.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{"parts": [
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                                {"text": _CROP_LABEL_PROMPT},
                            ]}],
                            "generationConfig": {"temperature": 0, "maxOutputTokens": 50},
                        },
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    increment_usage(key_idx, feature="crop_label")
                    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    raw = raw.strip("\"' \n.")
                    if raw and raw.lower() != "unknown":
                        label = raw
                    break  # Success (even if "Unknown")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        is_rate = "quota" not in str(e).lower()
                        mark_quota_exceeded(key_idx, is_rate_limit=is_rate)
                        print(f"[CROP-LABEL] Gemini key #{key_idx+1} hit 429, rotating...")
                        continue  # Try next key
                    print(f"[CROP-LABEL] Gemini failed for room {i}: {e}")
                    break
                except Exception as e:
                    print(f"[CROP-LABEL] Gemini failed for room {i}: {e}")
                    break

        # Fallback to GPT-4o-mini
        if not label and openai_key and "sk-" in openai_key:
            try:
                resp = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "temperature": 0,
                        "max_tokens": 50,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}", "detail": "low"
                            }},
                            {"type": "text", "text": _CROP_LABEL_PROMPT},
                        ]}],
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                raw = raw.strip("\"' \n.")
                if raw and raw.lower() != "unknown":
                    label = raw
            except Exception as e:
                print(f"[CROP-LABEL] GPT-4o-mini failed for room {i}: {e}")

        if label:
            # Parse "Kitchen|01" format -> label + room_number
            if "|" in label:
                parts = label.split("|", 1)
                room.label = parts[0].strip()
                num = parts[1].strip()
                if num:
                    room.room_number = num.zfill(3) if num.isdigit() else num
            else:
                room.label = label
            labeled_count += 1
            num_str = f" [{room.room_number}]" if room.room_number else ""
            print(f"[CROP-LABEL] Room {i} bbox=({rx},{ry},{rw},{rh}) -> '{room.label}'{num_str}")
        else:
            print(f"[CROP-LABEL] Room {i} bbox=({rx},{ry},{rw},{rh}) -> Unknown (no AI label)")

    print(f"[CROP-LABEL] Labeled {labeled_count}/{len(rooms)} rooms via AI crop analysis")


def _filter_text_only_rooms(rooms: list, gray: np.ndarray) -> list:
    """
    Remove detected 'rooms' that are actually just text regions with no wall structure.

    Real rooms have thick wall boundaries (high edge density around perimeter).
    Text-only regions (like "Controlled Document 0129") have scattered thin edges.
    Also filter rooms that are too small or in footer-like positions.
    """
    if not rooms:
        return rooms

    h, w = gray.shape
    filtered = []

    for room in rooms:
        rx, ry, rw, rh = room.bbox

        # Skip rooms in the bottom 5% of the image (likely footer text) AND very thin
        if ry + rh > h * 0.95 and rh < h * 0.08:
            print(f"[FILTER] Removed footer region at y={ry} ({rh}px tall)")
            continue

        # Skip very small rooms (< 1.5% of image area)
        room_area_pct = (rw * rh) / max(h * w, 1)
        if room_area_pct < 0.015:
            print(f"[FILTER] Removed tiny region ({room_area_pct:.1%} of image)")
            continue

        # Check wall density: real rooms have thick dark borders
        # Look at the perimeter of the bounding box
        border_w = max(3, min(rw, rh) // 15)
        ry_end = min(ry + rh, h)
        rx_end = min(rx + rw, w)

        # Extract perimeter strips
        top = gray[max(ry, 0):min(ry + border_w, h), rx:rx_end]
        bottom = gray[max(ry_end - border_w, 0):ry_end, rx:rx_end]
        left = gray[ry:ry_end, max(rx, 0):min(rx + border_w, w)]
        right = gray[ry:ry_end, max(rx_end - border_w, 0):rx_end]

        # Count dark pixels on perimeter (wall pixels are dark)
        wall_threshold = 100  # pixels darker than this are "wall"
        perimeter_pixels = 0
        dark_pixels = 0
        for strip in [top, bottom, left, right]:
            if strip.size > 0:
                perimeter_pixels += strip.size
                dark_pixels += np.sum(strip < wall_threshold)

        wall_density = dark_pixels / max(perimeter_pixels, 1)

        # Real rooms typically have > 15% dark perimeter pixels (walls)
        # Text regions have < 10%
        # On the walls_only mask blue pixels are already white, so lower threshold
        if wall_density < 0.05 and room_area_pct < 0.04:
            print(f"[FILTER] Removed text-only region (wall density={wall_density:.1%}, area={room_area_pct:.1%})")
            continue

        filtered.append(room)

    # Safety: never filter down to fewer than 2 rooms if we started with 3+
    if len(filtered) < 2 and len(rooms) >= 2:
        print(f"[FILTER] Safety: keeping all {len(rooms)} rooms (filtering would leave {len(filtered)})")
        return rooms

    if len(filtered) < len(rooms):
        print(f"[FILTER] Kept {len(filtered)}/{len(rooms)} rooms after filtering")
    return filtered


def _filter_building_boundary_rooms(rooms: list, image_shape: tuple) -> list:
    """
    Remove rooms that span nearly the full sketch width or height.

    The watershed sometimes segments the overall building exterior as a "room".
    These overly-large rooms cause incorrect label assignments because every
    OCR label falls inside their bounding box.

    Only removes boundary rooms if smaller rooms still remain (>= 2 kept).
    """
    if len(rooms) <= 2:
        return rooms  # Too few rooms to safely filter

    h, w = image_shape
    FULL_WIDTH_THRESHOLD = 0.88  # More than 88% of sketch width = boundary
    FULL_HEIGHT_THRESHOLD = 0.88

    boundary = []
    real = []
    for room in rooms:
        rx, ry, rw, rh = room.bbox
        is_full_width = rw >= w * FULL_WIDTH_THRESHOLD
        is_full_height = rh >= h * FULL_HEIGHT_THRESHOLD
        if is_full_width or is_full_height:
            boundary.append(room)
        else:
            real.append(room)

    # Only filter if we still have meaningful rooms left
    if len(real) >= 2:
        for r in boundary:
            rx, ry, rw, rh = r.bbox
            print(f"[FILTER] Removed boundary room ({rw}x{rh}, "
                  f"{rw/w:.0%} wide, {rh/h:.0%} tall)")
        return real

    # Not enough real rooms - keep everything
    return rooms


def _add_virtual_rooms_for_orphan_labels(
    rooms: list,
    labels: list,
    sketch_h: int,
    sketch_w: int,
) -> list:
    """
    Create virtual rooms for OCR labels that didn't match any detected room.

    When GPT-4o layout misses smaller rooms (WC, cupboards, stairs),
    the OCR pass may still find their labels. Create placeholder rooms
    for these orphaned labels so they appear in the output.
    """
    existing_labels_lower = {r.label.lower() for r in rooms if r.label}
    valid_room_tokens = {
        "SHOP", "FLOOR", "STORE", "STORAGE", "KITCHEN", "BED", "BEDROOM", "BATH",
        "BATHROOM", "WC", "TOILET", "CORRIDOR", "HALL", "HALLWAY", "OFFICE", "TILL",
        "CUPBOARD", "STAIR", "STAIRS", "STAIRCASE", "STAIRWELL", "STEP", "STEPS",
        "LANDING", "LOFT", "ATTIC", "BASEMENT", "CELLAR", "GARAGE", "LOUNGE",
        "LIVING", "DINING", "RECEPTION", "LOBBY", "ENTRANCE", "UTILITY",
    }
    virtual_rooms = []

    # Separate room numbers from room names in the labels
    room_number_labels = {}  # {(x,y): "003"} for nearby matching
    for label_info in labels:
        if label_info.get("is_room_number"):
            room_number_labels[label_info["location"]] = label_info["text"]

    for label_info in labels:
        text = label_info["text"]

        # Skip room number labels (these are not room names)
        if label_info.get("is_room_number"):
            continue

        # Skip very short labels (but allow known short room names like WC)
        stripped = text.strip()
        upper_text = stripped.upper()
        if len(stripped) < 2:
            continue
        if len(stripped) <= 3 and upper_text not in {"WC"}:
            continue

        # Skip pure numeric labels (room numbers misclassified as labels)
        if re.fullmatch(r'\d+', stripped):
            continue

        # Skip labels with long digit sequences (likely OCR noise from numbers)
        if re.search(r"\d{3,}", stripped):
            continue

        # Skip obvious non-room tokens
        if re.search(r"\b(ref|tc|bt|issue|revision|document|controlled|scale|pt|ft)\b",
                      stripped, re.IGNORECASE):
            continue

        # Only create virtual rooms for plausible room/location labels.
        # This blocks OCR noise like "BSP", "X", arrows, and partial tokens.
        if not any(tok in upper_text for tok in valid_room_tokens):
            continue

        # Skip if already matched to an existing room (case-insensitive)
        if stripped.lower() in existing_labels_lower:
            continue

        # Check for fuzzy match against existing rooms (avoid near-duplicates)
        from difflib import SequenceMatcher
        is_dup = False
        for existing in existing_labels_lower:
            ratio = SequenceMatcher(None, stripped.lower(), existing).ratio()
            if ratio > 0.7:
                is_dup = True
                break
        if is_dup:
            continue

        lx, ly = label_info["location"]

        # Skip labels that fall INSIDE an existing room's bounding box.
        # These are sub-labels (e.g. "TILL" inside "SHOP FLOOR", "FRONT OF SHOP"
        # within the main shop area) — not separate enclosed rooms.
        inside_existing = False
        for room in rooms:
            rx, ry, rw, rh = room.bbox
            if rx <= lx <= rx + rw and ry <= ly <= ry + rh:
                inside_existing = True
                break
        if inside_existing:
            continue

        # Create a virtual room centered on the label
        vw = max(int(sketch_w * 0.10), 100)
        vh = max(int(sketch_h * 0.08), 80)
        vx = max(0, lx - vw // 2)
        vy = max(0, ly - vh // 2)

        from .models import DetectedRoom
        virtual = DetectedRoom(
            bbox=(vx, vy, vw, vh),
            contour=None,
            area=vw * vh,
        )
        virtual.label = stripped

        # Try to find a nearby room number label
        best_dist = float('inf')
        best_num = None
        for (nx, ny), num_text in room_number_labels.items():
            dist = ((lx - nx) ** 2 + (ly - ny) ** 2) ** 0.5
            if dist < best_dist and dist < sketch_w * 0.15:  # Within 15% of width
                best_dist = dist
                best_num = num_text
        if best_num:
            virtual.room_number = best_num.zfill(3) if best_num.isdigit() else best_num

        virtual_rooms.append(virtual)
        num_str = f" [{virtual.room_number}]" if virtual.room_number else ""
        print(f"[VIRTUAL] Created room '{stripped}'{num_str} at ({lx},{ly})")
        existing_labels_lower.add(stripped.lower())

    return rooms + virtual_rooms


def _improve_labels_with_targeted_ocr(
    sketch: np.ndarray,
    rooms: list,
    ocr_result: dict,
) -> None:
    """
    Improve garbled OCR labels using fuzzy matching against known room names.

    EasyOCR struggles with handwriting but gets close: "Shof" for "Shop",
    "TLL" for "Till", "Flbol" for "Floor". Use difflib to find the best
    match from a known vocabulary of room names in asbestos surveys.
    """
    # Known room names for fuzzy matching
    # Covers common RapidOCR misreads: SHOPFLOON->Shop Floor, CUPROARD->Cupboard, FRONTOF->Front of Shop
    KNOWN_ROOMS = [
        "Shop Floor", "Front of Shop", "Till", "Cupboard", "Kitchen",
        "Bathroom", "Bath", "Bedroom", "Hallway", "Hall", "Corridor",
        "Living Room", "Lounge", "Dining Room", "Toilet", "WC",
        "Utility", "Office", "Store", "Store Room", "Storage",
        "Landing", "Stairs", "Lobby", "Reception", "Garage",
        "Loft", "Attic", "Cellar", "Basement", "Porch", "Entrance",
        "Boiler Room", "Plant Room", "Server Room", "Staff Room",
        "Back Room", "Counter", "Shop", "Floor",
        "Dis Board", "Distribution Board", "Electrical",
        "Electrical Dis Board", "Electrical Cupboard",
        "Meter Cupboard", "Comms Room", "Airing Cupboard",
        "Casino", "Stock Area", "Stock Room", "Vestibule",
        "Ground Floor", "First Floor", "Pantry", "Laundry",
        "Lobby Entrance", "Fire Exit", "Loading Bay",
        # Common short names from GPT-4o layout detection
        "Bed 1", "Bed 2", "Bed 3", "Bed 4", "Bed 5",
        "Bedroom 1", "Bedroom 2", "Bedroom 3", "Bedroom 4",
    ]

    # Two-pass: compute match scores for all rooms first, then assign by best score
    # This ensures "FRONTiOF" beats "SHoe" for "Front of Shop" even if SHoe comes first
    from difflib import SequenceMatcher

    known_set = {n.upper() for n in KNOWN_ROOMS}

    room_matches = []  # (score, room_idx, matched_name)
    for idx, room in enumerate(rooms):
        if not room.label:
            continue
        # Skip rooms whose label is already a known room name (no improvement needed)
        if room.label.upper() in known_set:
            continue
        best_match, best_score = _fuzzy_match_room_name_scored(room.label, KNOWN_ROOMS)
        if best_match:
            room_matches.append((best_score, idx, best_match))

    # Apply matches - allow duplicate names (same label on multiple rooms is valid)
    # e.g. GPT-4o may detect "SHOP FLOOR" written in multiple spots of the same open-plan area
    for score, idx, matched_name in room_matches:
        room = rooms[idx]
        old = room.label
        room.label = matched_name
        if old != matched_name:
            print(f"[FUZZY-MATCH] '{old}' -> '{matched_name}' (score={score:.2f})")

    # Remove residual numeric/noise labels that survived matching
    for room in rooms:
        if not room.label:
            continue
        label = room.label.strip()
        if re.fullmatch(r"[0-9\s./-]+", label):
            room.label = ""
        elif re.search(r"\b(ref|tc|bt|issue|revision|document|controlled)\b", label, re.IGNORECASE):
            room.label = ""


def _fuzzy_match_room_name(ocr_text: str, known_rooms: list) -> Optional[str]:
    """
    Match garbled OCR text to the closest known room name.

    Uses character-level similarity (SequenceMatcher ratio).
    Strategies:
    1. Full string vs full room name
    2. Single OCR word vs each word in room names ("SHOF" -> "SHOP" in "Shop Floor")
    3. OCR with small words stripped vs room name consonants ("FRONTiOF" -> "Front of Shop")
    Returns the best match if similarity is high enough.
    """
    from difflib import SequenceMatcher

    ocr_clean = ocr_text.strip()
    ocr_upper = ocr_clean.upper()
    if not ocr_upper or len(ocr_upper) < 2:
        return None

    # Strip common small join words that OCR merges: "OF", "THE", "AND"
    # e.g. "FRONTiOF" -> remove "OF" -> "FRONTi" to better match "FRONT"
    ocr_stripped = re.sub(r'(OF|THE|AND|IN|ON)$', '', ocr_upper).strip()

    best_ratio = 0.0
    best_name = None

    for name in known_rooms:
        name_upper = name.upper()
        # Remove "OF/THE/AND" from name too for comparison
        name_stripped = re.sub(r'\b(OF|THE|AND|IN|ON)\b', '', name_upper).replace('  ', ' ').strip()

        # 1. Full string comparison
        ratio = SequenceMatcher(None, ocr_upper, name_upper).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name

        # 2. Stripped comparison (removes small words before comparing)
        if ocr_stripped and name_stripped:
            ratio2 = SequenceMatcher(None, ocr_stripped, name_stripped).ratio()
            if ratio2 > best_ratio:
                best_ratio = ratio2
                best_name = name

        # 3. For single-word OCR text, compare against each word in room name
        # e.g., "SHOF" vs "SHOP" in "Shop Floor", "CUPROARD" vs "CUPBOARD"
        # BUT penalise multi-word names when the OCR text is a single short word
        # to prefer "Floor" over "Shop Floor" for the OCR word "FLoon"
        if " " not in ocr_upper:
            name_words = [w for w in name_upper.split() if len(w) >= 3]
            for name_word in name_words:
                wr = SequenceMatcher(None, ocr_upper, name_word).ratio()
                # Penalise multi-word room names slightly (favour simple names)
                penalty = 0.02 * (len(name_words) - 1)
                adjusted = wr - penalty
                if adjusted > best_ratio:
                    best_ratio = adjusted
                    best_name = name

    # Require fairly high confidence for short words to avoid false matches
    min_ratio = 0.55 if len(ocr_upper) >= 4 else 0.65
    if best_ratio >= min_ratio:
        return best_name

    # If no good match, return the original text cleaned up
    return ocr_clean.title() if len(ocr_clean) > 2 else None


def _fuzzy_match_room_name_scored(ocr_text: str, known_rooms: list):
    """Same as _fuzzy_match_room_name but also returns the confidence score."""
    from difflib import SequenceMatcher

    ocr_clean = ocr_text.strip()
    ocr_upper = ocr_clean.upper()
    if not ocr_upper or len(ocr_upper) < 2:
        return None, 0.0

    ocr_stripped = re.sub(r'(OF|THE|AND|IN|ON)$', '', ocr_upper).strip()

    best_ratio = 0.0
    best_name = None

    for name in known_rooms:
        name_upper = name.upper()
        name_stripped = re.sub(r'\b(OF|THE|AND|IN|ON)\b', '', name_upper).replace('  ', ' ').strip()

        ratio = SequenceMatcher(None, ocr_upper, name_upper).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name

        if ocr_stripped and name_stripped:
            ratio2 = SequenceMatcher(None, ocr_stripped, name_stripped).ratio()
            if ratio2 > best_ratio:
                best_ratio = ratio2
                best_name = name

        if " " not in ocr_upper:
            name_words = [w for w in name_upper.split() if len(w) >= 3]
            for name_word in name_words:
                wr = SequenceMatcher(None, ocr_upper, name_word).ratio()
                penalty = 0.02 * (len(name_words) - 1)
                adjusted = wr - penalty
                if adjusted > best_ratio:
                    best_ratio = adjusted
                    best_name = name

    min_ratio = 0.55 if len(ocr_upper) >= 4 else 0.65
    if best_ratio >= min_ratio:
        return best_name, best_ratio

    return ocr_clean.title() if len(ocr_clean) > 2 else None, best_ratio


def _save_debug_visualization(
    sketch: np.ndarray,
    rooms: list,
    ocr_result: dict,
    debug_dir: str,
) -> None:
    """Save a debug image with detected rooms overlaid."""
    debug_img = sketch.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255)]

    for i, room in enumerate(rooms):
        x, y, w, h = room.bbox
        c = colors[i % len(colors)]
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), c, 3)
        label = room.label or f"Room {i + 1}"
        if room.room_type == "acm":
            label += " [ACM]"
        cv2.putText(debug_img, label, (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    # Draw sample locations
    for sample in ocr_result.get("samples", []):
        sx, sy = sample["location"]
        cv2.circle(debug_img, (sx, sy), 15, (0, 255, 255), 3)
        cv2.putText(debug_img, sample["id"], (sx + 20, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(os.path.join(debug_dir, "99_final_detection.png"), debug_img)
