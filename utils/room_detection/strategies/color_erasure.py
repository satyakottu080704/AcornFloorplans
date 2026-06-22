"""Strategy C: Color Erasure + Flood Fill - best for sketches with blue hatching.

After erasing colors, uses a low global threshold (not OTSU) to avoid
catching blue-tinted grid paper as walls. OTSU fails on grid paper because
the bimodal distribution includes the grid lines at intensity 165-191.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from ..models import DetectedRoom
from ..color_analysis import erase_colors
from ..helpers import find_rooms_from_walls


def strategy_color_erasure_flood(
    image: np.ndarray,
    color_analysis: Dict[str, Any],
    debug_dir: Optional[str] = None
) -> List[DetectedRoom]:
    """
    Detect rooms by erasing colors first, then flood fill.
    Blue hatching creates dense patterns that look like walls.
    Erasing all colored pixels reveals the actual wall structure.
    """
    import os

    # Erase all color pixels to white
    clean = erase_colors(image, color_analysis)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "C1_colors_erased.png"), clean)

    # Threshold on cleaned image
    # Use global threshold at 65 (NOT OTSU) — OTSU catches grid paper
    # on blue-tinted grid paper because its bimodal split includes grid lines
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

    # Histogram-adaptive: use 8th percentile as threshold
    # After CLAHE + color erasure, pen walls may be at grayscale 100-160
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    cumsum = np.cumsum(hist)
    total = cumsum[-1]
    ink_5pct = int(np.searchsorted(cumsum, total * 0.05))
    ink_8pct = int(np.searchsorted(cumsum, total * 0.08))
    ink_15pct = int(np.searchsorted(cumsum, total * 0.15))
    threshold = max(min(ink_8pct, ink_15pct), 40)

    _, walls = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    wall_pct = cv2.countNonZero(walls) / max(gray.shape[0] * gray.shape[1], 1)
    print(f"[Strategy C] Threshold={threshold} (5pct={ink_5pct}, 8pct={ink_8pct}), "
          f"wall density={wall_pct:.1%}")

    # If still too dense, lower threshold further
    while wall_pct > 0.15 and threshold > 40:
        threshold -= 10
        _, walls = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        wall_pct = cv2.countNonZero(walls) / max(gray.shape[0] * gray.shape[1], 1)
        print(f"[Strategy C] Lowered to {threshold} (density={wall_pct:.1%})")

    # Clean walls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, kernel)

    # Close gaps and dilate
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                             iterations=2)
    walls = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "C2_walls.png"), walls)

    rooms = find_rooms_from_walls(walls, gray.shape, debug_dir, "C")
    print(f"[Strategy C] Found {len(rooms)} rooms")
    return rooms
