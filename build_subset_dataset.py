"""Build a YOLO training dataset from the GPT-4o auto-annotation output.

The annotator (acorn_auto_annotate.py) writes labels whose coordinates are
relative to the *preprocessed* sketch (form panel cropped, grid suppressed),
but it only saved the label .txt and a review PNG -- not the clean
preprocessed image. preprocess_sketch is deterministic, so we regenerate the
preprocessed image here and pair it with its label.

Output layout (Ultralytics YOLO standard):
    data/subset_dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml

Only images with a NON-EMPTY label file are included (empty = nothing
detected, no training signal). 90/10 train/val split, deterministic.
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGES_DIR = PROJECT_ROOT / "data" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
OUT_DIR = PROJECT_ROOT / "data" / "subset_dataset"

# Must match YOLO_CLASSES order in acorn_auto_annotate.py.
CLASSES = ["room", "acm", "stairs", "CupBoard", "Loft Hatch", "text", "wall"]
VAL_FRACTION = 0.10
SEED = 42


def _preprocess(img):
    from utils.room_detection.preprocessing import preprocess_sketch
    out = preprocess_sketch(img)
    return out[0] if isinstance(out, tuple) else out


def main() -> None:
    label_files = sorted(LABELS_DIR.glob("*.txt"))
    if not label_files:
        print(f"[ERR] No labels in {LABELS_DIR}. Run acorn_auto_annotate.py first.")
        sys.exit(1)

    # Keep only labels with content + a locatable source image.
    pairs = []
    missing_img = 0
    empty = 0
    for lf in label_files:
        if lf.stat().st_size == 0 or not lf.read_text().strip():
            empty += 1
            continue
        # Find the source image (label stem -> image with any supported ext).
        src = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            cand = IMAGES_DIR / (lf.stem + ext)
            if cand.exists():
                src = cand
                break
        if src is None:
            missing_img += 1
            continue
        pairs.append((src, lf))

    print(f"  Label files:        {len(label_files)}")
    print(f"  Empty (skipped):    {empty}")
    print(f"  Missing image:      {missing_img}")
    print(f"  Usable pairs:       {len(pairs)}")

    if not pairs:
        print("[ERR] No usable image/label pairs.")
        sys.exit(1)

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    print(f"  Train / Val:        {len(splits['train'])} / {len(splits['val'])}")

    # Fresh output tree.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    for split, items in splits.items():
        for src, lf in items:
            img = cv2.imread(str(src))
            if img is None:
                failed += 1
                continue
            try:
                pre = _preprocess(img)
            except Exception as e:
                print(f"  [WARN] preprocess failed for {src.name}: {e}")
                failed += 1
                continue
            stem = src.stem
            out_img = OUT_DIR / "images" / split / f"{stem}.jpg"
            out_lbl = OUT_DIR / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(out_img), pre)
            shutil.copyfile(lf, out_lbl)
            written += 1
            if written % 50 == 0:
                print(f"  ... {written} images prepared")

    # data.yaml
    yaml_path = OUT_DIR / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml_path.write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("  DATASET BUILT")
    print("=" * 60)
    print(f"  Images written:  {written}")
    print(f"  Failed:          {failed}")
    print(f"  data.yaml:       {yaml_path}")
    print(f"  Classes ({len(CLASSES)}):    {CLASSES}")
    print("=" * 60)


if __name__ == "__main__":
    main()
