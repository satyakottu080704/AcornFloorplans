"""Shared helper functions for room detection strategies."""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .models import DetectedRoom


def find_rooms_from_walls(
    walls: np.ndarray,
    image_shape: Tuple[int, int],
    debug_dir: Optional[str] = None,
    prefix: str = ""
) -> List[DetectedRoom]:
    """
    Find rooms by flood-filling from borders (exterior) and inverting.
    Whatever is NOT filled and NOT wall = enclosed rooms.
    """
    import os
    h, w = walls.shape

    # Flood fill from borders to find exterior
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    exterior = walls.copy()

    # Fill from border pixels at regular intervals
    for x in range(0, w, 10):
        if exterior[0, x] == 0:
            cv2.floodFill(exterior, flood_mask, (x, 0), 128)
        if exterior[h - 1, x] == 0:
            cv2.floodFill(exterior, flood_mask, (x, h - 1), 128)
    for y in range(0, h, 10):
        if exterior[y, 0] == 0:
            cv2.floodFill(exterior, flood_mask, (0, y), 128)
        if exterior[y, w - 1] == 0:
            cv2.floodFill(exterior, flood_mask, (w - 1, y), 128)

    # Interior = not wall (255) and not exterior (128)
    interior = (exterior == 0).astype(np.uint8) * 255

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}5_flood_filled.png"), exterior)
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}5_interior.png"), interior)

    # Connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior)
    return _labels_to_rooms_with_stats(labels, stats, centroids, image_shape)


def labels_to_rooms(labels: np.ndarray, image_shape: Tuple[int, int]) -> List[DetectedRoom]:
    """Convert connected component label map to room objects (without stats)."""
    h, w = image_shape
    min_area = h * w * 0.005
    max_area = h * w * 0.8

    rooms = []
    for label_id in range(1, labels.max() + 1):
        region = (labels == label_id).astype(np.uint8) * 255
        area = cv2.countNonZero(region)
        if area < min_area or area > max_area:
            continue
        coords = cv2.findNonZero(region)
        if coords is None:
            continue
        x, y, bw, bh = cv2.boundingRect(coords)
        if max(bw, bh) / max(min(bw, bh), 1) > 10:
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else None
        rooms.append(DetectedRoom(
            bbox=(x, y, bw, bh), contour=contour, area=area
        ))

    return rooms


def _labels_to_rooms_with_stats(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    image_shape: Tuple[int, int]
) -> List[DetectedRoom]:
    """Convert connected component labels+stats to room objects."""
    h, w = image_shape
    min_area = h * w * 0.005
    max_area = h * w * 0.8

    rooms = []
    for i in range(1, labels.max() + 1):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter extreme aspect ratios
        if max(bw, bh) / max(min(bw, bh), 1) > 10:
            continue

        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else None

        rooms.append(DetectedRoom(
            bbox=(x, y, bw, bh), contour=contour, area=area
        ))

    return rooms


def merge_overlapping_rooms(rooms: List[DetectedRoom], iou_threshold: float = 0.3) -> List[DetectedRoom]:
    """
    Merge rooms that overlap significantly (IoU > threshold).
    Keeps the larger room when two overlap.
    """
    if len(rooms) <= 1:
        return rooms

    # Sort by area descending (keep larger rooms)
    rooms = sorted(rooms, key=lambda r: r.area, reverse=True)
    keep = [True] * len(rooms)

    for i in range(len(rooms)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(rooms)):
            if not keep[j]:
                continue
            iou = _bbox_iou(rooms[i].bbox, rooms[j].bbox)
            if iou > iou_threshold:
                # Check if smaller room is mostly inside larger
                containment = _bbox_containment(rooms[i].bbox, rooms[j].bbox)
                if containment > 0.5:
                    keep[j] = False  # Remove smaller room
                elif iou > 0.5:
                    keep[j] = False  # High overlap, remove smaller

    merged = [r for r, k in zip(rooms, keep) if k]
    if len(merged) < len(rooms):
        print(f"[MERGE] Reduced {len(rooms)} -> {len(merged)} rooms (removed {len(rooms) - len(merged)} overlapping)")
    return merged


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union for two bboxes (x,y,w,h)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / max(union, 1)


def _bbox_containment(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> float:
    """What fraction of 'inner' bbox is contained within 'outer'."""
    ox1, oy1, ow, oh = outer
    ix1, iy1, iw, ih = inner
    ox2, oy2 = ox1 + ow, oy1 + oh
    ix2, iy2 = ix1 + iw, iy1 + ih

    cx1, cy1 = max(ox1, ix1), max(oy1, iy1)
    cx2, cy2 = min(ox2, ix2), min(oy2, iy2)

    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0

    clipped_area = (cx2 - cx1) * (cy2 - cy1)
    inner_area = max(iw * ih, 1)
    return clipped_area / inner_area


def score_detection_result(rooms: List[DetectedRoom], image_shape: Tuple[int, int]) -> float:
    """
    Score a detection result for quality (0.0 to 1.0).

    Good detection: 2-15 rooms, reasonable sizes, 20-90% total coverage.
    """
    if not rooms:
        return 0.0

    h, w = image_shape
    image_area = h * w

    # Room count score (best: 3-8 rooms)
    n = len(rooms)
    if n < 2:
        count_score = 0.2
    elif n <= 8:
        count_score = 1.0
    elif n <= 15:
        count_score = 0.7
    else:
        count_score = 0.3

    # Room size score
    sizes = [r.area / image_area for r in rooms]
    good_sizes = sum(1 for s in sizes if 0.01 <= s <= 0.5)
    size_score = good_sizes / len(rooms)

    # Total coverage score
    total_coverage = sum(sizes)
    if 0.2 <= total_coverage <= 0.9:
        coverage_score = 1.0
    elif total_coverage < 0.2:
        coverage_score = total_coverage / 0.2
    else:
        coverage_score = max(0, 1.0 - (total_coverage - 0.9))

    return 0.4 * count_score + 0.3 * size_score + 0.3 * coverage_score
