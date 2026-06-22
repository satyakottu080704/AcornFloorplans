"""Train YOLOv11n on the 500-image GPT-4o-labelled subset (CPU).

Detection (not segmentation): the GPT-4o annotator produces bounding boxes,
so this trains a detect model. best_room.pt is a seg model, so the later
comparison is detect-vs-seg on boxes only -- good enough to answer "are the
GPT-4o labels good enough to train on".

CPU, imgsz=640, nano model -- a proof-of-concept run, ~4-8 h.
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "subset_dataset" / "data.yaml"

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # auto-downloads pretrained nano weights
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=4,
        patience=40,          # early-stop if no val improvement for 40 epochs
                              # (raised for the slow OOD warmup)
        project=str(PROJECT_ROOT / "runs" / "subset"),
        name="yolo11n_500",
        exist_ok=True,
        plots=True,
    )
    print("\nTraining complete. Best weights:")
    print(f"  {PROJECT_ROOT / 'runs' / 'subset' / 'yolo11n_500' / 'weights' / 'best.pt'}")
