"""Strategy D: Watershed Segmentation - best for complex plans with shared walls."""

import cv2
import numpy as np
from typing import List, Optional

from ..models import DetectedRoom
from ..helpers import labels_to_rooms


def strategy_watershed(
    image: np.ndarray,
    gray: np.ndarray,
    debug_dir: Optional[str] = None
) -> List[DetectedRoom]:
    """
    Detect rooms using watershed segmentation.
    Uses distance transform to find room center seeds,
    then watershed grows regions until they hit walls.
    """
    import os
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage

    # Threshold to get walls
    _, walls = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "D1_walls.png"), walls)

    # Distance transform
    room_space = cv2.bitwise_not(walls)
    dist = cv2.distanceTransform(room_space, cv2.DIST_L2, 5)

    if debug_dir:
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "D2_distance.png"), dist_vis)

    # Find local maxima (room centers)
    # Higher min_distance and threshold to avoid over-segmentation
    min_distance = int(min(gray.shape) * 0.08)
    local_max = peak_local_max(dist, min_distance=max(min_distance, 40),
                               threshold_rel=0.4, exclude_border=True)

    if len(local_max) == 0:
        print(f"[Strategy D] No local maxima found")
        return []

    print(f"[Strategy D] Found {len(local_max)} local maxima")

    # Create markers for watershed
    markers = np.zeros_like(gray, dtype=np.int32)
    for i, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = i

    # Expand markers slightly
    markers = ndimage.grey_dilation(markers, size=(20, 20))

    # Run watershed
    labels = watershed(-dist, markers, mask=room_space)

    if debug_dir:
        label_vis = np.zeros((*gray.shape, 3), dtype=np.uint8)
        for i in range(1, labels.max() + 1):
            color = np.random.randint(50, 255, 3).tolist()
            label_vis[labels == i] = color
        cv2.imwrite(os.path.join(debug_dir, "D3_watershed.png"), label_vis)

    rooms = labels_to_rooms(labels, gray.shape)
    print(f"[Strategy D] Found {len(rooms)} rooms")
    return rooms
