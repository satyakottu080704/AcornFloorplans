"""Resize the subset_dataset images down to <=1280px for a small Colab upload.

Training runs at imgsz=1280, so images larger than that carry no benefit --
they just bloat the zip (1.3 GB -> ~300 MB). YOLO labels are normalized
(0-1), so resizing the image requires NO change to the label files.

Output: data/subset_dataset_1280/  (same train/val split, same labels,
just smaller JPEGs) + subset_dataset_1280.zip ready for files.upload().
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "data" / "subset_dataset"
DST = PROJECT_ROOT / "data" / "subset_dataset_1280"
MAX_DIM = 1280
JPEG_QUALITY = 90


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"[ERR] {SRC} not found - run build_subset_dataset.py first.")

    if DST.exists():
        shutil.rmtree(DST)

    resized = 0
    copied_labels = 0
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in sorted((SRC / "images" / split).glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] unreadable: {img_path.name}")
                continue
            h, w = img.shape[:2]
            scale = MAX_DIM / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            out_img = DST / "images" / split / img_path.name
            cv2.imwrite(str(out_img), img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            resized += 1

            lbl = SRC / "labels" / split / (img_path.stem + ".txt")
            if lbl.exists():
                shutil.copyfile(lbl, DST / "labels" / split / lbl.name)
                copied_labels += 1

            if resized % 100 == 0:
                print(f"  ... {resized} images resized")

    # data.yaml -- path is rewritten on Colab anyway, keep it relative-friendly.
    (DST / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "nc: 7\nnames:\n"
        "  0: room\n  1: acm\n  2: stairs\n  3: CupBoard\n"
        "  4: Loft Hatch\n  5: text\n  6: wall\n",
        encoding="utf-8",
    )

    print(f"\n  Resized images:  {resized}")
    print(f"  Labels copied:   {copied_labels}")

    print("  Zipping...")
    zip_base = PROJECT_ROOT / "data" / "subset_dataset_1280"
    shutil.make_archive(str(zip_base), "zip", str(DST))
    mb = (zip_base.with_suffix(".zip")).stat().st_size / 1024 / 1024
    print(f"  Created: data/subset_dataset_1280.zip ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
