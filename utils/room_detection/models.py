"""Data models for floor plan room detection."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectedRoom:
    """A single detected room from a floor plan sketch."""
    bbox: Tuple[int, int, int, int]       # (x, y, width, height) in pixels
    contour: Optional[np.ndarray] = None  # OpenCV contour points
    area: int = 0                         # Pixel area
    label: Optional[str] = None           # OCR-detected room name
    room_type: str = "clear"              # "acm", "clear", "no_access"
    color_detected: Optional[str] = None  # "blue", "red", "none"
    room_number: Optional[str] = None     # Surveyor's 3-digit number: "001", "017"
    floor: Optional[str] = None           # Floor assignment: "Ground Floor", "First Floor"


@dataclass
class FloorPlanAnalysis:
    """Complete analysis result from floor plan detection."""
    rooms: List[DetectedRoom] = field(default_factory=list)
    samples: List[Dict] = field(default_factory=list)          # [{id, location, material}]
    acm_regions: List[np.ndarray] = field(default_factory=list)
    cable_route: Optional[np.ndarray] = None
    atm_location: Optional[Tuple[int, int]] = None
    db_location: Optional[Tuple[int, int]] = None
    gas_meter: Optional[Tuple[int, int]] = None
    water_stop_tap: Optional[Tuple[int, int]] = None
    text_labels: List[Dict] = field(default_factory=list)      # [{text, location}]
    quality_score: str = "POOR"           # "GOOD", "FAIR", "POOR"
    detection_method: str = "none"        # Which strategy succeeded
    floors: List[Dict] = field(default_factory=list)            # [{title, y1_pct, y2_pct}]
    caveats: List[Dict] = field(default_factory=list)           # [{text, location}]
