"""Strategy A: Dark Threshold - best for clear black pen on light grid paper.

After CLAHE normalization, pen walls are at grayscale 100-155 and grid paper
intersections at 150-165. The separation is tight but workable.

Pipeline:
1. Erase colored regions (blue hatching, green cable, red markers) to white
2. Find histogram knee (where grid paper density accelerates)
3. Threshold just below the knee to catch pen walls only
4. Aggressive morphological opening to remove residual grid dots
5. Distance transform + flood fill to find enclosed rooms
"""

import cv2
import numpy as np
from typing import List, Optional

from ..models import DetectedRoom
from ..helpers import find_rooms_from_walls, labels_to_rooms
from ..color_analysis import erase_colors_aggressive


def strategy_dark_threshold(
    gray: np.ndarray,
    debug_dir: Optional[str] = None,
    color_image: Optional[np.ndarray] = None,
) -> List[DetectedRoom]:
    """
    Detect rooms by thresholding only the darkest pixels (pen ink).

    If color_image is provided, erases blue/green/red markings first
    so hatching doesn't get caught as walls.

    Works on the CLAHE-processed gray (better contrast for pen walls)
    but uses color_image for HSV-based color erasure.
    """
    import os

    h, w = gray.shape
    working_gray = gray

    # --- Step 0: Erase colored regions if color image available ---
    # Uses aggressive erasure (sat≥40) to strip blue hatching that
    # standard analysis (sat≥70) misses, giving OpenCV clean walls.
    if color_image is not None:
        clean = erase_colors_aggressive(color_image)
        working_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "A0_colors_erased.png"), clean)

    # --- Step 1: Histogram knee detection for threshold ---
    # Find where grid paper pixel density accelerates sharply.
    # Threshold just BELOW the knee catches pen walls but not grid paper.
    hist = cv2.calcHist([working_gray], [0], None, [256], [0, 256]).flatten()
    cumsum = np.cumsum(hist)
    total = cumsum[-1]

    bin_size = 5
    bin_counts = []
    for v in range(40, 220, bin_size):
        count = int(cumsum[min(v + bin_size, 255)] - cumsum[v])
        bin_counts.append((v, count))

    # Find the threshold where bin density doubles AND is significant (>0.5% of image)
    threshold = 150  # default if no clear knee found
    for i in range(1, len(bin_counts)):
        prev_val, prev_count = bin_counts[i - 1]
        curr_val, curr_count = bin_counts[i]
        if prev_count > 0 and curr_count > prev_count * 2 and curr_count > total * 0.005:
            threshold = prev_val + bin_size
            break

    # Safety: at most capture 8% of pixels (beyond that = grid paper)
    wall_density = np.sum(working_gray < threshold) / total
    while wall_density > 0.08 and threshold > 40:
        threshold -= 5
        wall_density = np.sum(working_gray < threshold) / total

    # Safety: at least 40
    threshold = max(threshold, 40)
    wall_density = np.sum(working_gray < threshold) / total
    print(f"[Strategy A] Threshold: {threshold} (knee-detect), density: {wall_density:.1%}")

    _, walls = cv2.threshold(working_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # --- Step 2: Morphological cleanup ---
    # Opening removes thin grid remnants while preserving thick pen walls.
    # Use progressive kernel sizes until density drops below 5%.
    for kernel_size in [3, 5, 7]:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        walls_opened = cv2.morphologyEx(walls, cv2.MORPH_OPEN, kernel_open, iterations=1)
        wall_pct = cv2.countNonZero(walls_opened) / (h * w)
        if wall_pct <= 0.05:
            walls = walls_opened
            print(f"[Strategy A] Opening k={kernel_size}: {wall_pct:.1%} walls (good)")
            break
        elif kernel_size == 7:
            # Last resort: use this even if >5%
            walls = walls_opened
            print(f"[Strategy A] Opening k={kernel_size}: {wall_pct:.1%} walls (max kernel)")
        else:
            print(f"[Strategy A] Opening k={kernel_size}: {wall_pct:.1%} walls (trying larger)")
            walls = walls_opened

    wall_pct = cv2.countNonZero(walls) / (h * w)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "A1_walls_raw.png"), walls)

    # Close gaps in hand-drawn walls (pen strokes have breaks)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    walls_closed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Dilate slightly to make walls solid barriers for flood fill
    walls_closed = cv2.dilate(walls_closed,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                              iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "A2_walls_closed.png"), walls_closed)

    # --- Step 3: Distance transform to find room cores ---
    room_space = cv2.bitwise_not(walls_closed)
    dist = cv2.distanceTransform(room_space, cv2.DIST_L2, 5)

    if debug_dir:
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "A3_distance_transform.png"), dist_vis)

    dist_max = dist.max()
    if dist_max < 10:
        print(f"[Strategy A] Distance max too low ({dist_max:.1f}), no rooms found")
        return []

    # Cap distance threshold at 150px max
    dist_thresh = min(max(dist_max * 0.15, 10), 150)
    _, sure_fg = cv2.threshold(dist, dist_thresh, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "A4_sure_foreground.png"), sure_fg)

    print(f"[Strategy A] Distance max={dist_max:.1f}, threshold={dist_thresh:.1f}")

    # --- Step 4: Find rooms via flood fill + connected components ---
    flood_rooms = find_rooms_from_walls(walls_closed, gray.shape, debug_dir, "A")

    num_labels, labels_map = cv2.connectedComponents(sure_fg)
    dist_rooms = labels_to_rooms(labels_map, gray.shape)

    # Use whichever method found more rooms
    if len(dist_rooms) > len(flood_rooms):
        rooms = dist_rooms
        method = "distance"
    else:
        rooms = flood_rooms
        method = "flood"

    print(f"[Strategy A] flood={len(flood_rooms)}, dist={len(dist_rooms)}, "
          f"using {method} -> {len(rooms)} rooms")
    return rooms
