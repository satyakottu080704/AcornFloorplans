# ============================================
# ACORN ATLAS FLOOR PLAN - CONFIGURATION
# ============================================

import os

# Project root resolves from this file's location, so the project folder can
# be renamed or moved without breaking these paths.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================
# MODEL — YOLOv11s-seg, trained on acorn_clean_merged (919 imgs, 9 classes)
# ============================================
# Re-enabled 2026-05-13 after retraining on the leak-fixed dataset.
# Box mAP50 = 0.59 (peak ep 126), room mask AP50 = 0.49 (peak ep 114).
# The old ResNet+UNet checkpoint (acorn_atlas_v5_NEW.pth) is kept on disk
# but no longer referenced.
USE_MODEL = os.environ.get("USE_MODEL", "true").strip().lower() in ("true", "1", "yes")
DEBUG_MODE = os.environ.get("DEBUG_MODE", "true").strip().lower() in ("true", "1", "yes")
MODEL_PATH = os.environ.get(
    "ACORN_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "training", "Training", "weights", "best.pt"),
)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)
MODEL_IMGSZ = int(os.environ.get("MODEL_IMGSZ", "1280"))  # match current training imgsz
MODEL_CONF_THRESHOLD = float(os.environ.get("MODEL_CONF_THRESHOLD", "0.15"))
                                # Tuned 2026-05-13 from 0.25 -> 0.15 to recover
                                # rooms / stairs the model predicts in the
                                # 0.15-0.24 range (missed rooms on prod sketches)
# ============================================
# FOLDERS
# ============================================
SKETCHES_FOLDER = os.environ.get("SKETCHES_FOLDER") or os.path.join(PROJECT_ROOT, "Input")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER") or os.path.join(PROJECT_ROOT, "output")

# ============================================
# RENDER PAGE + COORDINATE SPACE (env-overridable)
# ============================================
# A3 landscape in inches; the 0-1000 grid the layout/exporters work in.
PAGE_WIDTH_IN = float(os.environ.get("PAGE_WIDTH_IN", "16.54"))
PAGE_HEIGHT_IN = float(os.environ.get("PAGE_HEIGHT_IN", "11.69"))
COORD_MAX = float(os.environ.get("COORD_MAX", "1000"))

# ============================================
# PIXEL-SCALE FALLBACK (env-overridable)
# ============================================
# Estimated real-world building WIDTH (metres) by room count, used ONLY when the
# surveyor wrote no measurements. Override per deployment without touching code.
EST_WIDTH_M_SMALL = float(os.environ.get("EST_WIDTH_M_SMALL", "10"))    # <= 5 rooms
EST_WIDTH_M_MEDIUM = float(os.environ.get("EST_WIDTH_M_MEDIUM", "15"))  # <= 10 rooms
EST_WIDTH_M_LARGE = float(os.environ.get("EST_WIDTH_M_LARGE", "25"))    # <= 20 rooms
EST_WIDTH_M_XLARGE = float(os.environ.get("EST_WIDTH_M_XLARGE", "40"))  # > 20 rooms

# ============================================
# GPT-4o — Set in .env file:
#   OPENAI_API_KEY=sk-...
#   OPENAI_VISION_MODEL=gpt-4o
# ============================================

# ============================================
# 11 SEGMENTATION CLASSES
# ============================================
NUM_CLASSES = 6
CLASSES = ['acm', 'door', 'floor', 'room', 'stairs', 'walls']

def ensure_folders():
    """Create output folders."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "visio"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "cache"), exist_ok=True)
