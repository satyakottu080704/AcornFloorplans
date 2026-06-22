"""Strategy E: Edge-based wall tracing using Canny + morphology."""

import cv2
import numpy as np
from typing import List, Optional

from ..models import DetectedRoom
from ..helpers import find_rooms_from_walls


def strategy_edge_tracing(
    image: np.ndarray,
    debug_dir: Optional[str] = None
) -> List[DetectedRoom]:
    """
    Detect rooms by tracing wall edges, then flood fill to find enclosed spaces.
    Useful when sketches have faint walls but clear edges.
    """
    import os

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Strong edge enhancement
    sobelx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.bitwise_or(edges, sobel)

    # Strengthen edges into thicker walls
    walls = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    walls = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "E1_edges.png"), edges)
        cv2.imwrite(os.path.join(debug_dir, "E2_walls.png"), walls)

    rooms = find_rooms_from_walls(walls, gray.shape, debug_dir, "E")
    print(f"[Strategy E] Found {len(rooms)} rooms")
    return rooms
