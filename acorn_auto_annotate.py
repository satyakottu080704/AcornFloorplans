"""Auto-annotate Acorn floor-plan images into YOLO labels + review previews.

Reads images from   data/images/
Writes YOLO labels  data/labels/   (one .txt per image, YOLO bbox format)
Writes review PNGs  data/review/   (original + drawn bboxes for visual QA)
Progress JSON       data/annotation_progress.json  (resume-safe)

Why GPT-4o, not autodistill/Grounded-SAM
----------------------------------------
The user's original brief said `pip install autodistill autodistill-grounded-sam`.
For this domain (hand-drawn UK asbestos survey sketches on grid paper) those
open-vocabulary models are out of distribution and tend to miss the
domain-specific symbols (ACM hatching, loft-hatch X, CUP labels, etc).
GPT-4o vision is what produced the labels behind the current YOLO checkpoint
(`models/best_room.pt`, 0.59 box mAP per config.py:18-22), so we use it here.

Flip BACKEND = "groundedsam" to try the Grounded-SAM path; the same review
pipeline applies, so the side-by-side visual diff is meaningful.

Quick start
-----------
    # 1. Make sure data/images/ has some plans (run download_plans.py first).
    # 2. Make sure .env has OPENAI_API_KEY=sk-...
    # 3. Smoke test on 20 images:
    python acorn_auto_annotate.py            # uses MAX_IMAGES below
    # 4. Eyeball data/review/*.png. If quality looks OK, edit MAX_IMAGES = None
    #    and re-run -- already-done images are skipped automatically.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Tunables -- edit these
# ---------------------------------------------------------------------------

MAX_IMAGES: Optional[int] = 500  # set to None to process everything

# "gpt4o" is the proven path for this domain; "groundedsam" is a placeholder
# that prints install instructions and exits unless autodistill is installed.
BACKEND = "gpt4o"

# 7-class ontology -- matches the archived Path B annotator. We drop
# 'background' (negative space) and 'door' (GPT-4o is unreliable on tiny
# hand-drawn door symbols, and labelling them poorly broke an earlier model).
YOLO_CLASSES: List[str] = [
    "room",        # 0
    "acm",         # 1
    "stairs",      # 2
    "CupBoard",    # 3
    "Loft Hatch",  # 4
    "text",        # 5
    "wall",        # 6
]
CLASS_ID = {name.lower(): i for i, name in enumerate(YOLO_CLASSES)}

# Skip boxes thinner than this fraction of the image in either dimension.
MIN_DIM_PCT = 1.0

IMAGES_DIR = PROJECT_ROOT / "data" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
REVIEW_DIR = PROJECT_ROOT / "data" / "review"
PROGRESS_FILE = PROJECT_ROOT / "data" / "annotation_progress.json"

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}
COST_PER_CALL_USD = 0.01  # rough; gpt-4o vision @ detail=high


# ---------------------------------------------------------------------------
# Env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Prompt -- identical structure to the archived annotator, so labels feed
# the same downstream training pipeline.
# ---------------------------------------------------------------------------

MAIN_PROMPT = """This is a hand-drawn UK asbestos survey floor plan on grid paper.
Identify ALL visible elements and return their bounding boxes as percentages
of image dimensions (0-100).

Return ONLY this JSON (no other text, no markdown):
{
  "rooms": [
    {"name": "Kitchen", "x_pct": 10, "y_pct": 20, "w_pct": 30, "h_pct": 25,
     "has_acm": false, "no_access": false}
  ],
  "stairs":       [{"x_pct": 60, "y_pct": 30, "w_pct": 8, "h_pct": 10}],
  "loft_hatches": [{"x_pct": 70, "y_pct": 20, "w_pct": 5, "h_pct": 5}],
  "cupboards":    [{"x_pct": 15, "y_pct": 60, "w_pct": 6, "h_pct": 8}],
  "acm_areas":    [{"x_pct": 20, "y_pct": 30, "w_pct": 15, "h_pct": 12}],
  "walls":        [{"x_pct": 0,  "y_pct": 0,  "w_pct": 100, "h_pct": 2}],
  "text_regions": [{"x_pct": 5,  "y_pct": 2,  "w_pct": 20, "h_pct": 3}]
}

Rules:
- Return empty [] for any class not visible.
- Be conservative -- only mark what you clearly see.
- has_acm = room has diagonal hatching lines (red OR black) across interior.
- no_access = room has an X drawn through it.
- acm_areas = distinct hatched patches that aren't a full room (optional;
  leave [] if ACM is already captured via has_acm on a room).
- walls = line-like bboxes covering main external walls only (skip if unsure).
- text_regions = handwritten labels / sample annotations (S01, room names, ...).
- cupboards = small "CUP" or "CPD" labelled rooms or cabinet shapes.
- loft_hatches = small square symbol with an X or "LH" label.
- Return ONLY valid JSON.
"""

RETRY_PROMPT = (
    "List every room in this hand-drawn floor plan as a bounding box. "
    'Return ONLY JSON: {"rooms":[{"name":"Kitchen","x_pct":10,"y_pct":20,'
    '"w_pct":30,"h_pct":25,"has_acm":false,"no_access":false}]}'
)


# ---------------------------------------------------------------------------
# Backend: GPT-4o via the existing pipeline helpers
# ---------------------------------------------------------------------------

def _call_gpt4o_json(img_bgr: np.ndarray, prompt: str,
                     max_tokens: int = 4000) -> Optional[Dict]:
    from pipeline import _call_gpt4o, _encode_sketch, _parse_json
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [ERR] OPENAI_API_KEY not set in .env")
        return None
    b64 = _encode_sketch(img_bgr)
    raw = _call_gpt4o(api_key, b64, prompt, max_tokens=max_tokens)
    if not raw:
        return None
    return _parse_json(raw)


def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Run the project's preprocessing (landscape rotate + form panel crop)."""
    try:
        from utils.room_detection.preprocessing import preprocess_sketch
    except Exception as e:
        print(f"  [WARN] preprocess_sketch unavailable ({e}); using raw image")
        return img_bgr
    out = preprocess_sketch(img_bgr)
    if isinstance(out, tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Geometry: GPT-4o JSON -> YOLO lines + review boxes
# ---------------------------------------------------------------------------

def _bbox_to_yolo(x_pct: float, y_pct: float,
                  w_pct: float, h_pct: float) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    try:
        x = max(0.0, min(100.0, float(x_pct)))
        y = max(0.0, min(100.0, float(y_pct)))
        w = max(0.0, float(w_pct))
        h = max(0.0, float(h_pct))
    except (TypeError, ValueError):
        return None
    if w < MIN_DIM_PCT or h < MIN_DIM_PCT:
        return None
    if x + w > 100:
        w = 100 - x
    if y + h > 100:
        h = 100 - y
    if w <= 0 or h <= 0:
        return None
    xc, yc = (x + w / 2) / 100.0, (y + h / 2) / 100.0
    wn, hn = w / 100.0, h / 100.0
    return f"{xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}", (x, y, w, h)


def _data_to_yolo(ai: Dict) -> Tuple[List[str], List[Tuple[int, Tuple[float, float, float, float]]], Dict[str, int]]:
    """Return (yolo_lines, vis_boxes [(class_id, (x,y,w,h)_pct)], per-class counts)."""
    lines: List[str] = []
    vis: List[Tuple[int, Tuple[float, float, float, float]]] = []
    counts: Dict[str, int] = {name: 0 for name in YOLO_CLASSES}

    def _add(cls_name: str, box: Dict) -> None:
        cid = CLASS_ID.get(cls_name.lower())
        if cid is None:
            return
        out = _bbox_to_yolo(
            box.get("x_pct", 0), box.get("y_pct", 0),
            box.get("w_pct", 0), box.get("h_pct", 0),
        )
        if not out:
            return
        frag, pct_xywh = out
        lines.append(f"{cid} {frag}")
        vis.append((cid, pct_xywh))
        counts[YOLO_CLASSES[cid]] += 1

    for r in ai.get("rooms", []) or []:
        if bool(r.get("no_access")):
            continue
        _add("acm" if bool(r.get("has_acm")) else "room", r)
    for s in ai.get("stairs", []) or []:
        _add("stairs", s)
    for c in ai.get("cupboards", []) or []:
        _add("CupBoard", c)
    for lh in ai.get("loft_hatches", []) or []:
        _add("Loft Hatch", lh)
    for t in ai.get("text_regions", []) or []:
        _add("text", t)
    for w in ai.get("walls", []) or []:
        _add("wall", w)
    for a in ai.get("acm_areas", []) or []:
        _add("acm", a)
    return lines, vis, counts


# ---------------------------------------------------------------------------
# Review PNG
# ---------------------------------------------------------------------------

_CLASS_COLOR = {
    0: (60, 200, 60),    # room   - green
    1: (40, 40, 230),    # acm    - red
    2: (255, 180, 0),    # stairs - blue
    3: (200, 120, 220),  # CupBoard - magenta
    4: (0, 200, 220),    # Loft Hatch - yellow
    5: (200, 200, 200),  # text   - light grey
    6: (130, 130, 130),  # wall   - dark grey
}


def _draw_review(img: np.ndarray, boxes: List[Tuple[int, Tuple[float, float, float, float]]]) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    for cid, (x_pct, y_pct, w_pct, h_pct) in boxes:
        x1 = int(round(x_pct / 100 * w))
        y1 = int(round(y_pct / 100 * h))
        x2 = int(round((x_pct + w_pct) / 100 * w))
        y2 = int(round((y_pct + h_pct) / 100 * h))
        color = _CLASS_COLOR.get(cid, (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = YOLO_CLASSES[cid]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _load_progress() -> Dict[str, Dict]:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_progress(progress: Dict[str, Dict]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Per-image
# ---------------------------------------------------------------------------

def _process_one(img_path: Path) -> Dict:
    t0 = time.time()
    row: Dict = {
        "filename": img_path.name,
        "status": "failed",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "api_calls": 0,
        "backend": BACKEND,
    }
    img = cv2.imread(str(img_path))
    if img is None:
        row["error"] = "cv2.imread returned None"
        return row

    sketch = _preprocess(img)

    if BACKEND == "gpt4o":
        ai = _call_gpt4o_json(sketch, MAIN_PROMPT)
        row["api_calls"] = 1
        rooms = len(ai.get("rooms", [])) if isinstance(ai, dict) else 0
        # retry once if first call returned nothing on a non-tiny image
        if (not ai or rooms == 0) and img_path.stat().st_size > 50 * 1024:
            ai2 = _call_gpt4o_json(sketch, RETRY_PROMPT, max_tokens=2000)
            row["api_calls"] = 2
            if ai2 and ai2.get("rooms"):
                ai = ai2
    elif BACKEND == "groundedsam":
        row["error"] = (
            "BACKEND='groundedsam' not implemented in this script. "
            "Install: pip install autodistill autodistill-grounded-sam, "
            "then wire a function returning the same dict shape as MAIN_PROMPT."
        )
        return row
    else:
        row["error"] = f"unknown BACKEND={BACKEND!r}"
        return row

    if not ai:
        row["error"] = "no parseable JSON from backend"
        return row

    lines, vis_boxes, counts = _data_to_yolo(ai)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    label_path = LABELS_DIR / (img_path.stem + ".txt")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    review_path = REVIEW_DIR / (img_path.stem + "__review.png")
    cv2.imwrite(str(review_path), _draw_review(sketch, vis_boxes))

    row["status"] = "done" if lines else "failed_no_boxes"
    row["label_path"] = str(label_path)
    row["review_path"] = str(review_path)
    row["counts"] = counts
    row["time_sec"] = round(time.time() - t0, 2)
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _list_images() -> List[Path]:
    out: List[Path] = []
    seen = set()
    for ext in SUPPORTED_EXT:
        for p in sorted(IMAGES_DIR.glob(f"*{ext}")):
            key = os.path.normcase(str(p.resolve()))
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    _load_env()

    if not IMAGES_DIR.exists():
        print(f"[ERR] No image folder: {IMAGES_DIR}")
        print("      Run download_plans.py first.")
        sys.exit(1)

    print("=" * 72)
    print("  ACORN AUTO-ANNOTATE")
    print("=" * 72)
    print(f"  Backend:      {BACKEND}")
    print(f"  Images:       {IMAGES_DIR}")
    print(f"  Labels:       {LABELS_DIR}")
    print(f"  Review PNGs:  {REVIEW_DIR}")
    print(f"  Classes ({len(YOLO_CLASSES)}): {YOLO_CLASSES}")
    print(f"  MAX_IMAGES:   {MAX_IMAGES if MAX_IMAGES is not None else 'all'}")

    progress = _load_progress()
    all_imgs = _list_images()
    pending = [p for p in all_imgs if p.name not in progress
               or progress[p.name].get("status") != "done"]

    print(f"  Total found:  {len(all_imgs)}")
    print(f"  Already done: {len(all_imgs) - len(pending)}")
    print(f"  Pending:      {len(pending)}")
    print("=" * 72)

    if not pending:
        print("  Nothing to do.")
        return

    batch = pending if MAX_IMAGES is None else pending[:MAX_IMAGES]
    print(f"\n  Processing {len(batch)} image(s)...")

    total_cost = 0.0
    done = failed = 0
    for i, p in enumerate(batch, 1):
        print(f"\n  [{i}/{len(batch)}] {p.name}")
        try:
            row = _process_one(p)
        except KeyboardInterrupt:
            print("\n  [INTERRUPTED]")
            _save_progress(progress)
            raise
        except Exception:
            row = {
                "filename": p.name,
                "status": "failed",
                "error": traceback.format_exc(limit=2),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "api_calls": 0,
                "backend": BACKEND,
            }
        progress[p.name] = row
        _save_progress(progress)
        total_cost += row.get("api_calls", 0) * COST_PER_CALL_USD

        if row["status"] == "done":
            done += 1
            counts = row.get("counts") or {}
            summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v) or "(no boxes)"
            print(f"        OK  {summary}  ({row.get('time_sec', 0)}s)")
        else:
            failed += 1
            print(f"        {row['status']}  error={row.get('error', '-')}")

    print("\n" + "=" * 72)
    print("  RUN COMPLETE")
    print("=" * 72)
    print(f"  Annotated OK:    {done}/{len(batch)}")
    print(f"  Failed:          {failed}/{len(batch)}")
    print(f"  Est. GPT-4o $:   ${total_cost:.2f}")
    print(f"  Review here:     {REVIEW_DIR}")
    print(f"  Labels here:     {LABELS_DIR}")
    print(f"  Progress:        {PROGRESS_FILE}")
    print("=" * 72)
    print("\n  Eyeball a few PNGs in data/review/. If quality looks good,")
    print("  set MAX_IMAGES = None at the top of this file and re-run.")


if __name__ == "__main__":
    main()
