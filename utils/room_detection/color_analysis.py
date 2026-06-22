"""Color analysis: detect blue ACM hatching, green cable routes, red markers."""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional


def analyze_colors(image: np.ndarray, debug_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect and extract colored regions from the sketch.

    Returns dict with blue_mask, blue_regions, green_mask, green_path,
    red_mask, has_colors.
    """
    import os
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ----- Blue detection (ACM hatching) -----
    # Only detect clear blue PEN marks (sat>=70), not grid paper tint (sat 10-40).
    # Real blue hatching uses strong-saturation pen/marker strokes.
    # Grid paper under variable lighting can reach sat 30-50, so we stay well above.
    blue_ranges = [
        (np.array([100, 70, 50]), np.array([130, 255, 255])),   # Standard blue pen
        (np.array([90, 70, 40]), np.array([140, 255, 255])),    # Wider hue, same sat floor
    ]
    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower, upper in blue_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        blue_mask = cv2.bitwise_or(blue_mask, mask)

    # Clean up blue mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Sanity check: if >40% of image is "blue", it's grid paper tint, not real hatching.
    # Real ACM hatching covers specific rooms, never the entire sketch.
    total_px = max(image.shape[0] * image.shape[1], 1)
    blue_pct_raw = cv2.countNonZero(blue_mask) / total_px
    if blue_pct_raw > 0.40:
        print(f"[COLOR] Blue mask covers {blue_pct_raw:.0%} of image - likely grid paper, resetting")
        blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Find blue region contours
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.shape[0] * image.shape[1] * 0.001
    blue_regions = [c for c in blue_contours if cv2.contourArea(c) > min_area]

    # ----- Green detection (cable route) -----
    green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    green_path = None
    has_cable = cv2.countNonZero(green_mask) > (image.shape[0] * image.shape[1] * 0.002)
    if has_cable:
        gc, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if gc:
            largest = max(gc, key=cv2.contourArea)
            green_path = largest

    # ----- Red detection (sample labels, markers) -----
    red_mask1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 60, 60]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    has_colors = (cv2.countNonZero(blue_mask) > min_area * 10 or has_cable)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "02_blue_mask.png"), blue_mask)
        cv2.imwrite(os.path.join(debug_dir, "02_green_mask.png"), green_mask)
        cv2.imwrite(os.path.join(debug_dir, "02_red_mask.png"), red_mask)

    blue_pct = cv2.countNonZero(blue_mask) / max(image.shape[0] * image.shape[1], 1) * 100
    print(f"[COLOR] Blue: {blue_pct:.1f}%, regions: {len(blue_regions)}, cable: {has_cable}")

    return {
        "blue_mask": blue_mask,
        "blue_regions": blue_regions,
        "green_mask": green_mask,
        "green_path": green_path,
        "red_mask": red_mask,
        "has_colors": has_colors,
    }


def erase_colors(image: np.ndarray, color_analysis: Dict[str, Any]) -> np.ndarray:
    """Erase all colored pixels from image, leaving only black pen lines."""
    clean = image.copy()
    all_colors = color_analysis["blue_mask"].copy()
    all_colors = cv2.bitwise_or(all_colors, color_analysis["green_mask"])
    all_colors = cv2.bitwise_or(all_colors, color_analysis["red_mask"])
    # Dilate to catch edges of colored regions
    all_colors = cv2.dilate(all_colors, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    clean[all_colors > 0] = [255, 255, 255]
    return clean


def erase_colors_aggressive(image: np.ndarray) -> np.ndarray:
    """Aggressively erase ALL coloured pixels (sat≥40) for wall detection.

    Uses a lower saturation threshold than analyze_colors() to strip blue
    hatching that sits at sat 40-70 which the standard analysis misses.
    Grid paper tint (sat 10-30) is preserved.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clean = image.copy()

    # Blue at sat≥40 (catches light blue hatching that sat≥70 misses)
    blue = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([140, 255, 255]))
    # Green cable routes
    green = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
    # Red markers
    red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))

    all_color = cv2.bitwise_or(blue, green)
    all_color = cv2.bitwise_or(all_color, cv2.bitwise_or(red1, red2))

    # Sanity: if >50%, it's grid paper tint, not hatching
    h, w = image.shape[:2]
    if cv2.countNonZero(all_color) / (h * w) > 0.50:
        return image

    # Dilate aggressively to catch edges
    all_color = cv2.dilate(all_color,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
                           iterations=1)
    pct = cv2.countNonZero(all_color) / (h * w) * 100
    print(f"[COLOR] Aggressive erasure: {pct:.1f}% colored pixels removed")
    clean[all_color > 0] = [255, 255, 255]
    return clean
