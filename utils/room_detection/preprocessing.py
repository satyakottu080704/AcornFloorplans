"""Image preprocessing: auto-rotate, crop Acorn form, deskew, normalize lighting."""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_sketch(image: np.ndarray, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Full preprocessing pipeline for Acorn surveyor sketch photos.

    Handles:
    - Landscape templates photographed in portrait (rotated)
    - Acorn survey form on left/top side
    - Uneven lighting from phone camera
    - Slight rotation/skew

    Returns:
        (sketch_area, form_area)
        sketch_area has a `_raw_sketch` attribute: same crop WITHOUT CLAHE normalization.
        Strategy A uses raw sketch because CLAHE darkens grid intersections into
        the same range as pen walls, making them inseparable.
    """
    import os

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "00_original.png"), image)

    # Step 1: Auto-rotate if the template is landscape but photo is portrait
    image, was_rotated = _auto_rotate_to_landscape(image, debug_dir)

    # Keep a copy BEFORE CLAHE for strategies that need raw grayscale
    image_raw = image.copy()

    # Step 2: Normalize lighting (CLAHE)
    image = normalize_lighting(image)

    # Step 3: Crop out the survey form (same crop coords for both raw and CLAHE)
    sketch, form_area = detect_and_crop_form(image, debug_dir, was_rotated=was_rotated)

    # Apply the same crop to the raw image
    # detect_and_crop_form returns a cropped view — replicate the crop on raw
    raw_sketch = _apply_same_crop(image_raw, image, sketch)

    # Step 4: Trim footer/margins
    sketch_before_trim = sketch
    sketch = _trim_footer_and_margins(sketch, debug_dir)

    # Apply same trim to raw
    raw_sketch = _apply_same_trim(raw_sketch, sketch_before_trim, sketch)

    # Step 5: Deskew slight rotation
    sketch = deskew_image(sketch)
    raw_sketch = deskew_image(raw_sketch)

    # Step 6: Suppress graph paper grid lines so they don't confuse the model
    sketch = _suppress_grid_lines(sketch, debug_dir)
    raw_sketch = _suppress_grid_lines(raw_sketch, None)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01_preprocessed.png"), sketch)

    # Attach raw sketch as attribute (numpy arrays support arbitrary attributes via subclass)
    # Instead, store in module-level cache for access by strategies
    _raw_sketch_cache["latest"] = raw_sketch

    return sketch, form_area


def _suppress_grid_lines(image: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Aggressively suppress graph paper grid lines from the survey sheet.

    Grid lines are thin, regularly-spaced lines that the model classifies
    as room interior (50%+ room pixels). This uses morphological line
    detection to find thin horizontal/vertical structures and whites them
    out, while preserving thick pen strokes (walls, text, annotations).
    """
    import os

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    is_color = len(image.shape) == 3
    h, w = gray.shape

    # Adaptive kernel size based on image resolution (longer kernels = more selective for lines)
    line_len = max(25, min(w, h) // 40)

    # Step 1: adaptive threshold to binarize ALL dark marks (grid + pen + text).
    # Grid paper grid lines are light-gray; pen is dark. Adaptive threshold handles
    # uneven lighting much better than a fixed threshold.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=10,
    )

    # Step 2: isolate long horizontal / vertical structures from the binary mask.
    # A morphological OPEN with a long 1-pixel-thick kernel only passes structures
    # that are at least `line_len` long in that direction — exactly what grid lines
    # (and, unfortunately, long wall strokes) look like.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine and dilate slightly to cover full grid line width
    grid_mask = cv2.bitwise_or(h_mask, v_mask)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=1)

    # PROTECT thick pen strokes (walls, text, annotations). Pen ink is distinctly
    # darker than light-gray grid lines — use a tighter threshold so we actually
    # protect walls. Dilate generously so edges survive.
    pen_mask = (gray < 100).astype(np.uint8) * 255
    pen_protect = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    grid_mask = cv2.bitwise_and(grid_mask, cv2.bitwise_not(pen_protect))

    grid_pct = np.sum(grid_mask > 0) / (h * w) * 100
    print(f"[PREPROCESS] Grid suppression: {grid_pct:.1f}% of pixels identified as grid lines")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "05_grid_mask.png"), grid_mask)

    # White out grid lines completely
    result = image.copy()
    if is_color:
        result[grid_mask > 0] = [255, 255, 255]
    else:
        result[grid_mask > 0] = 255

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "06_grid_suppressed.png"), result)

    return result


# Module-level cache for raw (un-CLAHE) sketch
_raw_sketch_cache = {}


def get_raw_sketch() -> Optional[np.ndarray]:
    """Get the latest raw (un-CLAHE) sketch from preprocessing."""
    return _raw_sketch_cache.get("latest")


def _apply_same_crop(raw_image: np.ndarray, clahe_image: np.ndarray, cropped: np.ndarray) -> np.ndarray:
    """Apply the same crop that was applied to clahe_image to get cropped, but on raw_image."""
    clahe_h, clahe_w = clahe_image.shape[:2]
    crop_h, crop_w = cropped.shape[:2]

    if crop_h == clahe_h and crop_w == clahe_w:
        return raw_image  # No crop happened

    # Find the crop offset by matching dimensions
    # The crop removes either left/right or top/bottom
    dy = clahe_h - crop_h
    dx = clahe_w - crop_w

    raw_h, raw_w = raw_image.shape[:2]

    # Try to find matching corner — check if crop starts at (0,0) or offset
    # Most common: form on LEFT removed, so crop starts at (dx, 0)
    # Or form at TOP removed, so crop starts at (0, dy)
    if dx > 0 and dy <= 10:
        # Horizontal crop — form on left or right
        # Check if the left edge was removed
        return raw_image[:, dx:][:crop_h, :crop_w]
    elif dy > 0 and dx <= 10:
        # Vertical crop — form on top or bottom
        return raw_image[dy:, :][:crop_h, :crop_w]
    else:
        # Both dimensions changed — use the crop dimensions from the end
        return raw_image[:crop_h, :crop_w]


def _apply_same_trim(raw_sketch: np.ndarray, before_trim: np.ndarray, after_trim: np.ndarray) -> np.ndarray:
    """Apply the same footer/margin trim to raw sketch."""
    bh, bw = before_trim.shape[:2]
    ah, aw = after_trim.shape[:2]

    if bh == ah and bw == aw:
        return raw_sketch  # No trim happened

    # Trim is bottom and/or left
    dy = bh - ah  # bottom trimmed
    dx = bw - aw  # left trimmed

    rh, rw = raw_sketch.shape[:2]
    y_end = rh - dy if dy > 0 else rh
    x_start = dx if dx > 0 else 0

    return raw_sketch[:y_end, x_start:]


def _auto_rotate_to_landscape(image: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Acorn survey templates are LANDSCAPE. If the photo is portrait,
    detect the form position and rotate to match.

    The Acorn form has:
    - Dense table lines (horizontal + vertical) in one region
    - The Acorn logo (green circle) at one corner
    - Form text fields (Client, N-, Date, Site, Floor, etc.)

    Logic:
    - If image is portrait (h > w * 1.1), it's rotated
    - Find where the dense form region is (top/bottom/left/right)
    - Rotate so the form ends up on the LEFT side (standard Acorn layout)
    """
    import os
    h, w = image.shape[:2]

    # Detect where the form is by analyzing edge density in quadrants
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Split into 4 quadrants and measure density
    mid_y, mid_x = h // 2, w // 2
    q_tl = np.mean(edges[:mid_y, :mid_x])   # top-left
    q_tr = np.mean(edges[:mid_y, mid_x:])    # top-right
    q_bl = np.mean(edges[mid_y:, :mid_x])    # bottom-left
    q_br = np.mean(edges[mid_y:, mid_x:])    # bottom-right

    # Also check for green pixels (Acorn logo) in corners
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 25, 25]), np.array([85, 255, 255]))
    corner_size = min(h, w) // 6
    green_tl = np.sum(green_mask[:corner_size, :corner_size] > 0)
    green_tr = np.sum(green_mask[:corner_size, -corner_size:] > 0)
    green_bl = np.sum(green_mask[-corner_size:, :corner_size] > 0)
    green_br = np.sum(green_mask[-corner_size:, -corner_size:] > 0)

    max_green = max(green_tl, green_tr, green_bl, green_br)
    top_density = (q_tl + q_tr) / 2
    bottom_density = (q_bl + q_br) / 2
    left_density = (q_tl + q_bl) / 2
    right_density = (q_tr + q_br) / 2

    # Landscape survey photos are already in the required orientation. Never
    # rotate them 90 degrees based only on colour/logo heuristics: clipboard,
    # clothing, and lighting regularly create false green detections.
    if h <= w * 1.1:
        print(f"[PREPROCESS] Already landscape/square ({w}x{h}) - keeping landscape orientation")
        return image, False

    # Legacy orientation analysis below is only used for portrait inputs.
    if h <= w * 1.1:
        # Form is sideways (top or bottom) if:
        # 1. Logo is in TR or BR corner
        # 2. Top or bottom edge has significantly higher density than left
        is_sideways = (
            (max_green > 100 and (max_green == green_tr or max_green == green_br)) or
            (bottom_density > left_density * 1.25) or (top_density > left_density * 1.25)
        )

        if not is_sideways:
            # Check if form is on the right side
            is_form_on_right = (
                (max_green > 100 and (max_green == green_tr or max_green == green_br)) or
                (right_density > left_density * 1.2)
            )
            if is_form_on_right:
                print(f"[PREPROCESS] Already landscape/square ({w}x{h}) with form on RIGHT - rotating 180°")
                return cv2.rotate(image, cv2.ROTATE_180), True
            else:
                print(f"[PREPROCESS] Already landscape/square ({w}x{h}) with form on LEFT - no rotation needed")
                return image, False

    print(f"[PREPROCESS] Checking rotation for {w}x{h} image...")
    print(f"[PREPROCESS] Edge density - TL:{q_tl:.1f} TR:{q_tr:.1f} BL:{q_bl:.1f} BR:{q_br:.1f}")
    print(f"[PREPROCESS] Green logo - TL:{green_tl} TR:{green_tr} BL:{green_bl} BR:{green_br}")

    # For portrait inputs, compare both rotations and prefer the one that
    # places the structurally dense form panel on the left. This works for
    # photos taken in either phone orientation and does not depend on colour.
    cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _left_form_score(candidate):
        candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        candidate_edges = cv2.Canny(candidate_gray, 50, 150)
        candidate_w = candidate_edges.shape[1]
        left = float(np.mean(candidate_edges[:, :candidate_w // 3]))
        right = float(np.mean(candidate_edges[:, 2 * candidate_w // 3:]))
        return left / max(right, 0.1)

    cw_score = _left_form_score(cw)
    ccw_score = _left_form_score(ccw)
    if max(cw_score, ccw_score) >= 1.15 and abs(cw_score - ccw_score) >= 0.10:
        if cw_score > ccw_score:
            print(f"[PREPROCESS] Portrait structure selects 90 degrees clockwise "
                  f"(left-form score {cw_score:.2f} vs {ccw_score:.2f})")
            rotated = cw
        else:
            print(f"[PREPROCESS] Portrait structure selects 90 degrees counter-clockwise "
                  f"(left-form score {ccw_score:.2f} vs {cw_score:.2f})")
            rotated = ccw
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "00b_rotated.png"), rotated)
        return rotated, True

    # Determine rotation based on where the form is
    if max_green == green_tl and max_green > 100:
        # Logo at top-left in portrait = needs 90° clockwise rotation
        # After rotation: form on left (correct Acorn layout)
        print(f"[PREPROCESS] Logo at TOP-LEFT, rotating 90° clockwise")
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif max_green == green_tr and max_green > 100:
        # Logo at top-right in portrait photo.
        # Try both rotations — pick the one where the form (dense area) ends up on LEFT.
        cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cw_gray = cv2.cvtColor(cw, cv2.COLOR_BGR2GRAY)
        ccw_gray = cv2.cvtColor(ccw, cv2.COLOR_BGR2GRAY)
        cw_w = cw_gray.shape[1]
        ccw_w = ccw_gray.shape[1]
        cw_left = np.mean(cw_gray[:, :cw_w // 3] < 128)
        cw_right = np.mean(cw_gray[:, 2 * cw_w // 3:] < 128)
        ccw_left = np.mean(ccw_gray[:, :ccw_w // 3] < 128)
        ccw_right = np.mean(ccw_gray[:, 2 * ccw_w // 3:] < 128)
        cw_score = cw_left / max(cw_right, 0.001)
        ccw_score = ccw_left / max(ccw_right, 0.001)
        if cw_score > ccw_score:
            print(f"[PREPROCESS] Logo at TOP-RIGHT, rotating 90° clockwise (form-left: {cw_score:.1f} vs {ccw_score:.1f})")
            rotated = cw
        else:
            print(f"[PREPROCESS] Logo at TOP-RIGHT, rotating 90° counter-clockwise (form-left: {ccw_score:.1f} vs {cw_score:.1f})")
            rotated = ccw
    elif max_green == green_bl and max_green > 100:
        # Logo at bottom-left = needs 90° counter-clockwise
        print(f"[PREPROCESS] Logo at BOTTOM-LEFT, rotating 90° counter-clockwise")
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif max_green == green_br and max_green > 100:
        # Logo at bottom-right = needs 90° clockwise
        print(f"[PREPROCESS] Logo at BOTTOM-RIGHT, rotating 90° clockwise")
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif top_density > bottom_density * 1.3:
        # Form at top, rotate clockwise to put it on left
        print(f"[PREPROCESS] Dense region at TOP, rotating 90° clockwise")
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif bottom_density > top_density * 1.3:
        # Form at bottom, rotate counter-clockwise
        print(f"[PREPROCESS] Dense region at BOTTOM, rotating 90° counter-clockwise")
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Default: rotate clockwise (most common phone orientation)
        print(f"[PREPROCESS] Default rotation 90° clockwise")
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Post-rotation verification: form should be on LEFT side
    # If form ended up on RIGHT, we rotated wrong way — flip 180°
    rot_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    rot_h, rot_w = rot_gray.shape
    left_density = np.mean(rot_gray[:, :rot_w // 4] < 128)
    right_density = np.mean(rot_gray[:, 3 * rot_w // 4:] < 128)
    if right_density > left_density * 1.5 and right_density > 0.05:
        print(f"[PREPROCESS] Post-check: form on RIGHT (L={left_density:.3f}, R={right_density:.3f}), flipping 180°")
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "00b_rotated.png"), rotated)

    return rotated, True


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Correct uneven lighting using CLAHE on luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l)
    lab_corrected = cv2.merge([l_corrected, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def detect_and_crop_form(
    image: np.ndarray,
    debug_dir: Optional[str] = None,
    was_rotated: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Detect and remove the Acorn survey form.

    After rotation, the form should be on the LEFT side.
    Strategy:
    1. Analyze column-wise edge density (form has dense table borders)
    2. Find the column where density drops sharply = form boundary
    3. Crop everything left of that = form, right of that = sketch
    """
    import os
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Analyze column-wise density in strips
    strip_w = max(w // 40, 5)
    col_densities = []
    for x in range(0, w - strip_w, strip_w):
        strip = edges[:, x:x + strip_w]
        col_densities.append(float(np.mean(strip)))

    if not col_densities:
        print(f"[PREPROCESS] No density data, using full image")
        return image, None

    # Also check row-wise for top/bottom form
    row_densities = []
    strip_h = max(h // 40, 5)
    for y in range(0, h - strip_h, strip_h):
        strip = edges[y:y + strip_h, :]
        row_densities.append(float(np.mean(strip)))

    # Find form boundary: look for the biggest density DROP
    # The form side has high density, sketch side has lower
    left_avg = np.mean(col_densities[:len(col_densities) // 3])
    right_avg = np.mean(col_densities[len(col_densities) // 3:])
    top_avg = np.mean(row_densities[:len(row_densities) // 3])
    bottom_avg = np.mean(row_densities[len(row_densities) // 3:])

    # Check for green pixels (Acorn logo) to resolve form position
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 25, 25]), np.array([85, 255, 255]))
    green_left = int(np.sum(green_mask[:, :w // 2] > 0))
    green_right = int(np.sum(green_mask[:, w // 2:] > 0))
    print(f"[PREPROCESS] Density - Left:{left_avg:.1f} Right:{right_avg:.1f} Top:{top_avg:.1f} Bottom:{bottom_avg:.1f}")
    print(f"[PREPROCESS] Green logo pixels - Left: {green_left}, Right: {green_right}")

    forced_left = False
    forced_right = False

    if left_avg > right_avg * 1.2 and left_avg > 3:
        print("[PREPROCESS] Form structure detected on LEFT, overriding colour heuristic")
        forced_left = True
    elif right_avg > left_avg * 1.2 and right_avg > 3:
        print("[PREPROCESS] Form structure detected on RIGHT, overriding colour heuristic")
        forced_right = True
    elif green_left > 100 or green_right > 100:
        # Colour is supporting evidence only. Clipboard, lighting, clothing,
        # and compression regularly create green false positives, especially
        # on low-resolution images. Preserve the full image when structure is
        # not strong enough to identify the form side.
        print("[PREPROCESS] Green pixels found without structural form evidence; "
              "not forcing a crop")

    margin = 10
    form_area = None

    # Form on LEFT (most common after rotation)
    if left_avg > right_avg * 1.2 and left_avg > 3:
        cut_x = _find_boundary_from_densities(col_densities, strip_w, max_pct=0.55)
        if cut_x > 0:
            form_area = image[:, :cut_x]
            sketch = image[:, min(cut_x + margin, w):]
            print(f"[PREPROCESS] Form on LEFT, cut at x={cut_x} ({cut_x*100//w}% of width)")
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "01a_form_area.png"), form_area)
                cv2.imwrite(os.path.join(debug_dir, "01a_sketch_crop.png"), sketch)
            return sketch, form_area

    # Form at TOP - Disabled to prevent false-positive cropping on clean drawings
    # if top_avg > bottom_avg * 1.3 and top_avg > 3:
    #     cut_y = _find_boundary_from_densities(row_densities, strip_h, max_pct=0.5)
    #     if cut_y > 0:
    #         form_area = image[:cut_y, :]
    #         sketch = image[min(cut_y + margin, h):, :]
    #         print(f"[PREPROCESS] Form at TOP, cut at y={cut_y}")
    #         if debug_dir:
    #             cv2.imwrite(os.path.join(debug_dir, "01a_sketch_crop.png"), sketch)
    #         return sketch, form_area

    # Form at BOTTOM - Disabled to prevent false-positive cropping on clean drawings
    # if bottom_avg > top_avg * 1.3 and bottom_avg > 3:
    #     cut_y_from_bottom = _find_boundary_from_densities(
    #         list(reversed(row_densities)), strip_h, max_pct=0.5)
    #     if cut_y_from_bottom > 0:
    #         cut_y = h - cut_y_from_bottom
    #         form_area = image[cut_y:, :]
    #         sketch = image[:max(cut_y - margin, 0), :]
    #         print(f"[PREPROCESS] Form at BOTTOM, cut at y={cut_y}")
    #         if debug_dir:
    #             cv2.imwrite(os.path.join(debug_dir, "01a_sketch_crop.png"), sketch)
    #         return sketch, form_area

    # Form on RIGHT: mirror the same structural boundary detection used for
    # left-side forms. This supports reversed/mirrored survey photographs.
    if right_avg > left_avg * 1.2 and right_avg > 3:
        cut_x_from_right = _find_boundary_from_densities(
            list(reversed(col_densities)), strip_w, max_pct=0.55)
        if cut_x_from_right > 0:
            cut_x = w - cut_x_from_right
            form_area = image[:, cut_x:]
            sketch = image[:, :max(cut_x - margin, 0)]
            print(f"[PREPROCESS] Form on RIGHT, cut at x={cut_x}")
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "01a_form_area.png"), form_area)
                cv2.imwrite(os.path.join(debug_dir, "01a_sketch_crop.png"), sketch)
            return sketch, form_area

    # Fallback: try contour-based detection
    sketch, form_area = _crop_form_by_contour(image, debug_dir)
    if form_area is not None:
        return sketch, form_area

    print("[PREPROCESS] No reliable form boundary detected; preserving full image")
    return image, None


def _find_boundary_from_densities(densities: list, strip_size: int, max_pct: float = 0.55) -> int:
    """
    Find the boundary position from a density profile.
    Looks for the biggest density drop within the first max_pct of the profile.
    """
    max_idx = int(len(densities) * max_pct)
    if max_idx < 2:
        return 0

    # Calculate drops between consecutive strips
    drops = []
    for i in range(min(max_idx, len(densities) - 1)):
        drop = densities[i] - densities[i + 1]
        drops.append(drop)

    if not drops:
        return 0

    # Find the position of the biggest drop
    max_drop_idx = int(np.argmax(drops))

    # Verify this is a real boundary (not just noise)
    before_avg = np.mean(densities[:max(max_drop_idx + 1, 1)])
    after_avg = np.mean(densities[max_drop_idx + 1:max(max_drop_idx + 5, max_drop_idx + 2)])

    if before_avg > after_avg * 1.2:
        return (max_drop_idx + 1) * strip_size
    return 0


def _crop_form_by_contour(image: np.ndarray, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fallback: detect form as a dense rectangular contour region."""
    import os
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.5:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Form spans nearly full height -> left/right side
        if ch > h * 0.6 and cw < w * 0.5:
            margin = 20
            if x < w * 0.5:
                form_area = image[:, :x + cw + margin]
                sketch = image[:, min(x + cw + margin, w):]
                print(f"[PREPROCESS] Form contour on LEFT (x=0..{x + cw})")
            else:
                form_area = image[:, max(x - margin, 0):]
                sketch = image[:, :max(x - margin, 0)]
                print(f"[PREPROCESS] Form contour on RIGHT")
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, "01a_sketch_crop.png"), sketch)
            return sketch, form_area

        # Do not contour-crop top/bottom regions. A floor-plan outer wall or
        # long room boundary commonly spans most of the page width and was
        # repeatedly mistaken for a horizontal form, deleting real rooms.
        # Side forms remain structurally distinctive and safe to crop above.

    return image, None


def _trim_footer_and_margins(image: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Trim footer text and thin margins from the sketch area.

    Acorn forms have footer text like "Controlled Document 0129",
    "Issue 1: 07 January 2007", "Revision 1: 05 November 2013" at the bottom.
    Also trim thin form column remnants on the left edge.
    """
    import os
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # --- Trim bottom footer ---
    # Scan from bottom upward: footer rows have sparse, text-like edge patterns
    # (horizontal lines of small text vs. thick pen walls of sketch)
    strip_h = max(h // 30, 8)
    bottom_crop = h

    # Check bottom 25% of image for footer-like rows
    check_from = int(h * 0.75)
    row_densities = []
    for y in range(check_from, h - strip_h, strip_h):
        strip = edges[y:y + strip_h, :]
        density = float(np.mean(strip))
        row_densities.append((y, density))

    if row_densities:
        # Find the sketch area: rows with substantial drawing content
        # Footer text is typically much lower density than pen-drawn walls
        # Look for a gap (very low density row) followed by sparse text
        sketch_densities = []
        for y in range(0, check_from, strip_h):
            sketch_densities.append(float(np.mean(edges[y:y + strip_h, :])))

        if sketch_densities:
            sketch_avg = np.mean(sketch_densities)
            # Footer is typically < 30% of sketch density
            footer_threshold = max(sketch_avg * 0.3, 1.5)

            # Find first row from bottom that has meaningful content
            for y, density in reversed(row_densities):
                if density > footer_threshold * 2:
                    # This row has real sketch content
                    bottom_crop = min(y + strip_h + 10, h)
                    break

    # --- Trim left margin remnants ---
    # If form was on left, sometimes a thin strip of form cells remains
    left_crop = 0
    check_width = min(int(w * 0.08), 60)  # Check first 8% of width

    if check_width > 10:
        left_strip = edges[:, :check_width]
        left_density = float(np.mean(left_strip))
        main_strip = edges[:, check_width:check_width * 3]
        main_density = float(np.mean(main_strip))

        # If left edge is denser than the area next to it, it's form remnant
        if left_density > main_density * 1.5 and left_density > 3:
            # Find where the form remnant ends
            for x in range(check_width, 0, -2):
                col = edges[:, x:x + 4]
                if float(np.mean(col)) < left_density * 0.4:
                    left_crop = x + 5
                    break

    # --- Apply crops ---
    if left_crop > 0 or bottom_crop < h:
        old_h, old_w = h, w
        image = image[:bottom_crop, left_crop:]
        new_h, new_w = image.shape[:2]
        if left_crop > 0:
            print(f"[PREPROCESS] Trimmed left margin: {left_crop}px")
        if bottom_crop < old_h:
            print(f"[PREPROCESS] Trimmed bottom footer: {old_h - bottom_crop}px")
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "01b_trimmed.png"), image)

    return image


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Correct slight rotation/skew using Hough line detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        angles.append(angle)

    if angles:
        median_angle = float(np.median(angles))
        if 0.5 < abs(median_angle) < 10:  # Only correct small angles, not 90° rotations
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return image
