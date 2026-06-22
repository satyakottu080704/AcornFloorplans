#!/usr/bin/env python3
"""
Train a YOLOv11 instance-segmentation model on the FloorPlans v1 dataset.

Dataset: Roboflow export "FloorPlans.v1i.yolov11" — 16,632 images, 6 classes
(acm, door, floor, room, stairs, walls). Polygon/segmentation labels.

Usage:
    python train_floorplans.py                       # train, sensible defaults
    python train_floorplans.py --epochs 300 --model yolo11m-seg.pt
    python train_floorplans.py --data datasets/floorplans_v1   # custom location

NOTE ON HARDWARE: this is a 16k-image segmentation dataset. It needs a GPU.
On CPU it is effectively untrainable (days per epoch). Run on a CUDA machine
or Colab. The script auto-detects the device.

----------------------------------------------------------------------------
POST-TRAINING INTEGRATION (do NOT do this before the model is trained — it
would break the current 9-class best_room.pt):

  1. Copy runs/.../weights/best.pt  ->  models/best_floorplans.pt
  2. In config.py set:
         CLASSES = ['acm', 'door', 'floor', 'room', 'stairs', 'walls']
         NUM_CLASSES = 6
         MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_floorplans.pt")
  3. Update _yolo_class_id() in pipeline.py to map names to the new 6-class
     order above (the old map assumed 9 classes incl CupBoard/Loft Hatch/text).
  4. A/B the new model vs best_room.pt on a held-out set before committing.
----------------------------------------------------------------------------
"""
import argparse
import sys
from pathlib import Path


def _build_data_yaml(dataset_dir: Path) -> Path:
    """
    Roboflow's data.yaml uses paths like '../train/images', which only
    resolve correctly from inside a sub-folder. Ultralytics resolves dataset
    paths relative to the yaml's own directory, so we rewrite a clean yaml
    with absolute paths to avoid silent "0 images found" failures.
    """
    for split in ("train", "valid", "test"):
        if not (dataset_dir / split / "images").is_dir():
            sys.exit(f"ERROR: missing {split}/images under {dataset_dir} — "
                     f"extract the dataset there first.")

    fixed = dataset_dir / "floorplans.yaml"
    fixed.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: 6\n"
        f"names: ['acm', 'door', 'floor', 'room', 'stairs', 'walls']\n",
        encoding="utf-8",
    )
    print(f"[data] wrote resolved dataset config: {fixed}")
    return fixed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/floorplans_v1",
                    help="Dataset root (contains train/ valid/ test/).")
    ap.add_argument("--model", default="yolo11s-seg.pt",
                    help="Base seg model: yolo11n/s/m/l-seg.pt (bigger = "
                         "slower but more accurate).")
    ap.add_argument("--epochs", type=int, default=300,
                    help="Epoch CEILING. Early-stopping (--patience) keeps "
                         "the best checkpoint well before this.")
    ap.add_argument("--patience", type=int, default=50,
                    help="Stop if val metrics don't improve for N epochs.")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="Training image size (dataset is 1024x1024).")
    ap.add_argument("--batch", type=int, default=-1,
                    help="Batch size. -1 = auto-fit to GPU memory.")
    ap.add_argument("--name", default="floorplans_v1",
                    help="Run name under runs/segment/.")
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"ERROR: {e}. Install with: pip install ultralytics")

    dataset_dir = Path(args.data).resolve()
    data_yaml = _build_data_yaml(dataset_dir)

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU detected. Segmentation training on CPU "
              "is impractically slow — use a GPU machine or Colab.")
    else:
        print(f"[device] CUDA GPU: {torch.cuda.get_device_name(0)}")

    print(f"[train] model={args.model} epochs<={args.epochs} "
          f"patience={args.patience} imgsz={args.imgsz} batch={args.batch}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        # Roboflow already baked in flip / rotate / brightness augmentation,
        # so keep extra augmentation modest to avoid over-distorting.
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        plots=True,
    )

    print("\n[done] Best weights: runs/segment/"
          f"{args.name}/weights/best.pt")
    print("Next: validate on the test split, then see the POST-TRAINING "
          "INTEGRATION notes at the top of this file.")


if __name__ == "__main__":
    main()
