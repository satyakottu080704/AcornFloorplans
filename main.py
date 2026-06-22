#!/usr/bin/env python3
"""
Acorn Floor Plan — single consolidated entry point
==================================================
This is THE file to run for testing and local production of the BOX PIPELINE
(YOLO geometry + GPT-4o labels -> professional Visio). It replaces the old
``run_detector.py`` + ``test_local.py`` split: every option lives here.

Why the box pipeline (and not the GPT free-hand layout extractor): GPT-4o
cannot produce metrically accurate wall coordinates, so the container/Windows
"layout_extractor" path drifts ("geometry doesn't match the sketch"). The box
pipeline instead takes ROOM RECTANGLES from the trained YOLO model and lets the
renderer DERIVE walls from those rectangle edges — which is what keeps the
geometry aligned with the sketch.

Usage
-----
    # Single sketch -> .vsdx (YOLO geometry + GPT labels, COM renderer on Windows)
    python main.py --image sketch.jpg
    python main.py --image sketch.jpg --output out.vsdx --renderer com

    # Pure-GPT layout, no YOLO (use ONLY to compare/diagnose — geometry is worse)
    python main.py --image sketch.jpg --no-model

    # Keep the original sketch as a locked background, overlay labels
    python main.py --image sketch.jpg --overlay

    # Batch a folder (resumable, crash-safe results file)
    python main.py --batch sketches/ --resume

    # YOLO-only diagnostic: print detection counts + save an annotated preview
    python main.py --image sketch.jpg --model-only

    # Housekeeping
    python main.py --clear-cache
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path

# --- Project root on sys.path + .env, before importing the pipeline ----------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


# ----------------------------------------------------------------------------
# Geometry health: surface WHERE each room's geometry came from.
# ----------------------------------------------------------------------------
def _geometry_source_breakdown(plan) -> dict:
    """Count rooms by geometry_source ('model' = YOLO, 'ai_bbox' = GPT)."""
    counts = {}
    for room in plan.rooms:
        src = getattr(room, "geometry_source", "unknown") or "unknown"
        counts[src] = counts.get(src, 0) + 1
    return counts


def _print_geometry_health(plan):
    """Tell the user if YOLO actually drove the geometry — the alignment signal.

    If most rooms are 'ai_bbox', YOLO under-detected and the layout fell back to
    GPT positions, which is the exact cause of "geometry doesn't match". This
    print makes that visible instead of silent.
    """
    counts = _geometry_source_breakdown(plan)
    total = max(1, len(plan.rooms))
    model_n = counts.get("model", 0)
    ai_n = counts.get("ai_bbox", 0)
    print(f"  Geometry source: {model_n} from YOLO, {ai_n} from GPT, "
          f"others {total - model_n - ai_n}")
    if model_n < ai_n:
        print("  [WARN] More rooms came from GPT than YOLO — geometry may not "
              "match the sketch. Check the YOLO model with --model-only.")


def _check_yolo_model():
    """Warn loudly if the YOLO weights are missing.

    A missing model means the pipeline silently falls back to GPT-only geometry
    — the very failure mode we are trying to avoid — so make it loud.
    """
    try:
        from config import USE_MODEL, MODEL_PATH
    except Exception:
        return
    if not USE_MODEL:
        print("[WARN] config.USE_MODEL is False — running GPT-only geometry "
              "(no YOLO). Set USE_MODEL=True for accurate room geometry.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] YOLO weights not found at {MODEL_PATH} — geometry will "
              f"fall back to GPT only. Set ACORN_MODEL_PATH or run git lfs pull.")


# ----------------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------------
def _batch_process(folder, output_dir, resume, no_model, overlay):
    from pipeline import process_sketch

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(sorted(glob.glob(os.path.join(folder, ext))))
    if not images:
        print(f"No images found in {folder}")
        return

    results_path = os.path.join(output_dir, "batch_results.json")
    results = {}
    if resume and os.path.exists(results_path):
        try:
            results = json.loads(open(results_path, encoding="utf-8").read())
            print(f"Resuming: {len(results)} already processed")
        except Exception:
            results = {}

    total = len(images)
    skip = success = failed = 0
    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  ACORN — BATCH ({total} images)  ->  {output_dir}")
    print(f"{'=' * 60}")

    for i, img_path in enumerate(images, 1):
        base = os.path.splitext(os.path.basename(img_path))[0]
        if resume and base in results and results[base].get("success"):
            skip += 1
            continue

        print(f"\n[{i}/{total}] {os.path.basename(img_path)}")
        result = {"file": os.path.basename(img_path), "success": False,
                  "rooms": 0, "samples": 0, "time_sec": 0, "error": None}
        t0 = time.time()
        try:
            vsdx, plan = process_sketch(
                img_path,
                output_path=os.path.join(output_dir, f"{base}.vsdx"),
                no_model=no_model,
                overlay=overlay,
            )
            result.update(
                success=True,
                time_sec=round(time.time() - t0, 1),
                output=vsdx,
                rooms=len(plan.rooms),
                samples=len(plan.samples),
                acm_rooms=sum(1 for r in plan.rooms if r.has_acm),
                no_access_rooms=sum(1 for r in plan.rooms if r.no_access),
                geometry_source=_geometry_source_breakdown(plan),
            )
            _print_geometry_health(plan)
            success += 1
        except Exception as e:
            result["error"] = str(e)
            result["time_sec"] = round(time.time() - t0, 1)
            print(f"  FAILED: {e}")
            failed += 1

        results[base] = result
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE — {success} ok, {failed} failed, {skip} skipped")
    print(f"  Time: {(time.time() - t_start) / 60:.1f} min   Results: {results_path}")
    print(f"{'=' * 60}")


# ----------------------------------------------------------------------------
# YOLO-only diagnostic
# ----------------------------------------------------------------------------
def _diagnose_model_only(image_path):
    import cv2
    from collections import Counter
    from config import MODEL_PATH, MODEL_IMGSZ, MODEL_CONF_THRESHOLD, OUTPUT_FOLDER
    from utils.room_detection.preprocessing import preprocess_sketch
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"\n{'=' * 60}\n  YOLO-ONLY DIAGNOSTIC\n  Model: {MODEL_PATH}\n{'=' * 60}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Could not load image: {image_path}")
        sys.exit(1)

    sketch, _ = preprocess_sketch(image)
    h, w = sketch.shape[:2]
    print(f"  Preprocessed: {w}x{h}")

    model = YOLO(MODEL_PATH)
    result = model.predict(sketch, imgsz=MODEL_IMGSZ,
                           conf=MODEL_CONF_THRESHOLD, verbose=False)[0]
    names = model.names
    cls_ids = result.boxes.cls.int().tolist() if result.boxes is not None else []
    confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    counts = Counter(names[int(c)] for c in cls_ids)

    print("\n  PER-CLASS DETECTIONS:")
    for name, count in sorted(counts.items()) or [("(none)", 0)]:
        print(f"    {name:<14} {count:>4}")
    if confs:
        print(f"\n  Confidence: min={min(confs):.2f}, "
              f"mean={sum(confs) / len(confs):.2f}, max={max(confs):.2f}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    debug_path = os.path.join(OUTPUT_FOLDER, f"{Path(image_path).stem}__model_only.jpg")
    result.save(filename=debug_path)
    print(f"\n  Saved YOLO overlay: {debug_path}")


# ----------------------------------------------------------------------------
# Single image
# ----------------------------------------------------------------------------
def _single(image_path, output_path, no_model, overlay):
    from pipeline import process_sketch

    print(f"\n{'=' * 60}\n  ACORN — {os.path.basename(image_path)}\n{'=' * 60}")
    vsdx, plan = process_sketch(
        image_path, output_path=output_path, no_model=no_model, overlay=overlay,
    )
    acm = sum(1 for r in plan.rooms if r.has_acm)
    na = sum(1 for r in plan.rooms if r.no_access)
    print(f"\n  Output: {vsdx}")
    print(f"  Rooms: {len(plan.rooms)} ({acm} ACM, {na} no-access)   "
          f"Samples: {len(plan.samples)}")
    _print_geometry_health(plan)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Acorn floor-plan generator (box pipeline: YOLO + GPT-4o).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", help="Path to a single sketch image (JPG/PNG)")
    p.add_argument("--batch", metavar="FOLDER", help="Process every image in FOLDER")
    p.add_argument("--output", "-o", help="Output .vsdx path (single) or directory (batch)")
    p.add_argument("--renderer", choices=["com", "aspose"],
                   default=("com" if sys.platform == "win32" else "aspose"),
                   help="com = Windows + Visio (clean); aspose = Linux/no-Visio "
                        "(watermarked unless licensed). Default: com on Windows.")
    p.add_argument("--no-model", action="store_true",
                   help="Skip YOLO; use GPT-only layout (worse geometry — diagnostic only)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--overlay", action="store_true",
                      help="Lock the source sketch as background and overlay labels")
    mode.add_argument("--vector", action="store_true",
                      help="Force reconstructed vector geometry (default behaviour)")
    p.add_argument("--resume", action="store_true", help="Batch: skip already-processed images")
    p.add_argument("--clear-cache", action="store_true", help="Clear the GPT-4o response cache")
    p.add_argument("--model-only", action="store_true",
                   help="Diagnostic: run ONLY YOLO, print counts, save annotated preview")
    args = p.parse_args()

    from config import OUTPUT_FOLDER, ensure_folders
    ensure_folders()

    # Renderer selection is read from the env by pipeline.export_visio.
    os.environ["RENDERER"] = args.renderer

    if args.clear_cache:
        import shutil
        cache_dir = os.path.join(OUTPUT_FOLDER, "cache")
        if os.path.exists(cache_dir):
            n = len([f for f in os.listdir(cache_dir) if f.endswith(".json")])
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Cleared {n} cached responses")
        if not args.image and not args.batch:
            return

    if args.model_only:
        if not args.image or not os.path.isfile(args.image):
            p.error("--model-only requires a valid --image")
        _diagnose_model_only(args.image)
        return

    effective_overlay = True if args.overlay else (False if args.vector else None)

    # Surface YOLO availability up front unless the user opted out of it.
    if not args.no_model:
        _check_yolo_model()

    if args.batch:
        if not os.path.isdir(args.batch):
            p.error(f"--batch folder not found: {args.batch}")
        output_dir = args.output or os.path.join(OUTPUT_FOLDER, "visio")
        os.makedirs(output_dir, exist_ok=True)
        _batch_process(args.batch, output_dir, args.resume, args.no_model, effective_overlay)
    elif args.image:
        if not os.path.isfile(args.image):
            p.error(f"image not found: {args.image}")
        _single(args.image, args.output, args.no_model, effective_overlay)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
