"""Strategy B: Adaptive Threshold + Distance Transform - best for variable lighting.

Uses adaptive threshold with aggressive morphological opening to remove
grid paper lines while preserving thick pen walls. The opening kernel
must be large enough to kill 1-2px grid lines but small enough to keep
4-8px pen strokes.
"""

import cv2
import numpy as np
from typing import List, Optional

from ..models import DetectedRoom
from ..helpers import labels_to_rooms


def strategy_adaptive_distance(gray: np.ndarray, debug_dir: Optional[str] = None) -> List[DetectedRoom]:
    """
    Detect rooms using adaptive threshold + distance transform.
    Adaptive threshold handles uneven lighting.
    Distance transform finds room centers far from walls.
    """
    import os

    h, w = gray.shape

    # Adaptive threshold — use larger block size (101px) and higher C constant (15)
    # to be more selective about what counts as "dark enough to be a wall"
    # Small blockSize catches grid lines; large blockSize only catches thick pen walls
    walls = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=101, C=15
    )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "B1_adaptive_thresh.png"), walls)

    # Aggressive morphological opening with larger kernel
    # Grid lines are 1-2px thin; pen walls are 4-8px thick
    # A 5x5 ellipse opening erodes 2px from each side: kills 4px grid lines,
    # preserves 8px+ pen walls
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, kernel_open, iterations=2)

    wall_pct = cv2.countNonZero(walls) / (h * w)
    print(f"[Strategy B] After adaptive+opening: {wall_pct:.1%} wall pixels")

    # If still too dense (>20%), the grid paper is bleeding through
    # Fall back to histogram-adaptive global threshold
    if wall_pct > 0.20:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        ink_8pct = int(np.searchsorted(cumsum, total * 0.08))
        fb_threshold = max(ink_8pct, 40)
        print(f"[Strategy B] Too dense ({wall_pct:.1%}), fallback threshold={fb_threshold}")
        _, walls = cv2.threshold(gray, fb_threshold, 255, cv2.THRESH_BINARY_INV)
        walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, kernel_open)
        wall_pct = cv2.countNonZero(walls) / (h * w)
        print(f"[Strategy B] Fallback: {wall_pct:.1%} wall pixels")

    # Close gaps in walls
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "B2_walls_cleaned.png"), walls)

    # Distance transform on room space (inverse of walls)
    room_space = cv2.bitwise_not(walls)
    dist = cv2.distanceTransform(room_space, cv2.DIST_L2, 5)

    if debug_dir:
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "B3_distance.png"), dist_vis)

    # Threshold: keep pixels far from walls (room centers)
    dist_max = dist.max()
    if dist_max < 10:
        print(f"[Strategy B] Distance max too low ({dist_max:.1f}), no rooms")
        return []

    dist_threshold = min(max(dist_max * 0.15, 10), 150)
    _, sure_fg = cv2.threshold(dist, dist_threshold, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "B4_sure_foreground.png"), sure_fg)

    print(f"[Strategy B] Distance max={dist_max:.1f}, threshold={dist_threshold:.1f}")

    # Connected components on sure foreground = room seeds
    num_labels, labels = cv2.connectedComponents(sure_fg)
    print(f"[Strategy B] Components: {num_labels - 1}")

    rooms = labels_to_rooms(labels, gray.shape)
    print(f"[Strategy B] Found {len(rooms)} rooms")
    return rooms
