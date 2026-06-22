import os
import re
import json
import logging
import requests
import cv2
import numpy as np
from pathlib import Path

log = logging.getLogger("auto-training")

# Local directories for saving training data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_IMAGES_DIR = PROJECT_ROOT / "training" / "auto_datasets" / "images"
LOCAL_LABELS_DIR = PROJECT_ROOT / "training" / "auto_datasets" / "labels"

# SharePoint target path (env-overridable; defaults kept for backward-compat).
SHAREPOINT_DRIVE_ID = os.environ.get("AUTO_TRAINING_DRIVE_ID", "b!zN4eSsWB8E6plWioOQ7Li-FhLR__cQJEu-YN7mSw8APbsu2dgNFISrbh3ef7KRSD")
SHAREPOINT_RELATIVE_FOLDER = os.environ.get("AUTO_TRAINING_FOLDER", "Research and Development Claims/2025 - 2026 New System Creation/AI Automation/AcornPlanGenration/AutoTrainingData")

# Acceptance thresholds for auto-collected samples (env-overridable).
MIN_CONF = float(os.environ.get("AUTO_TRAINING_MIN_CONF", "0.82"))
MIN_ROOMS = int(os.environ.get("AUTO_TRAINING_MIN_ROOMS", "3"))
MAX_ROOMS = int(os.environ.get("AUTO_TRAINING_MAX_ROOMS", "25"))
MAX_ROOM_IOU = float(os.environ.get("AUTO_TRAINING_MAX_IOU", "0.20"))
MIN_COVERAGE = float(os.environ.get("AUTO_TRAINING_MIN_COVERAGE", "0.15"))
MAX_COVERAGE = float(os.environ.get("AUTO_TRAINING_MAX_COVERAGE", "0.95"))

def _acquire_graph_token():
    """Acquire Microsoft Graph access token using credentials from environment."""
    tenant = (os.environ.get("SP_TENANT_ID") or os.environ.get("ACORN_TENANT_ID") or os.environ.get("OUTLOOK_TENANT_ID") or "").strip()
    client_id = (os.environ.get("SP_CLIENT_ID") or os.environ.get("ACORN_CLIENT_ID") or os.environ.get("OUTLOOK_CLIENT_ID") or "").strip()
    secret = (os.environ.get("SP_CLIENT_SECRET") or os.environ.get("ACORN_CLIENT_SECRET") or os.environ.get("OUTLOOK_CLIENT_SECRET") or "").strip()
    
    if not (tenant and client_id and secret):
        log.warning("[Auto-Training] SharePoint credentials not fully configured in .env; skipping cloud sync")
        return None
        
    url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    data = {
        "client_id": client_id,
        "client_secret": secret,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    try:
        r = requests.post(url, data=data, timeout=30)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        log.error("[Auto-Training] Failed to acquire MS Graph token: %s", e)
        return None

def upload_file_to_sharepoint(local_file_path: Path, subfolder: str) -> bool:
    """Upload local file to the SharePoint site default document library."""
    token = _acquire_graph_token()
    if not token:
        return False
        
    filename = local_file_path.name
    # Target URL via specific Drive ID
    url = f"https://graph.microsoft.com/v1.0/drives/{SHAREPOINT_DRIVE_ID}/root:/{SHAREPOINT_RELATIVE_FOLDER}/{subfolder}/{filename}:/content"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream"
    }
    
    try:
        if not local_file_path.exists():
            log.error("[Auto-Training] Local file not found for upload: %s", local_file_path)
            return False
            
        with open(local_file_path, "rb") as f:
            data = f.read()
            
        r = requests.put(url, headers=headers, data=data, timeout=60)
        if r.status_code in (200, 201):
            web_url = r.json().get("webUrl")
            log.info("[Auto-Training] Successfully uploaded %s to SharePoint: %s", filename, web_url)
            return True
        else:
            log.error("[Auto-Training] SharePoint upload failed with status %s: %s", r.status_code, r.text)
            return False
    except Exception as e:
        log.error("[Auto-Training] SharePoint upload exception: %s", e)
        return False

def _bbox_iou(box1, box2) -> float:
    """Compute IoU of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def validate_predictions(result) -> bool:
    """
    Autonomous geometric solver to audit YOLO model predictions:
    1. Average confidence must be high (> 0.82).
    2. Room count must be realistic (between 3 and 25 rooms).
    3. Room overlaps must be low (no duplicated room boxes overlapping heavily, IoU < 0.20).
    4. Coverage of rooms relative to the building footprint must be reasonable.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return False
        
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.int().tolist()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    
    # 1. Average confidence check (only for rooms/stairs/acm classes)
    # Mapping for floorplans classes: 0=acm, 1=door, 2=floor, 3=room, 4=stairs, 5=walls
    target_cids = {0, 3, 4} # ACM, Room, Stairs
    filtered_confs = [confs[i] for i, cid in enumerate(cls_ids) if cid in target_cids]
    
    if not filtered_confs:
        return False
        
    avg_conf = sum(filtered_confs) / len(filtered_confs)
    log.info("[Auto-Training] Average prediction confidence: %.3f", avg_conf)
    if avg_conf < MIN_CONF:
        log.info("[Auto-Training] Rejected: Average confidence below threshold (%.2f)", MIN_CONF)
        return False
        
    # 2. Room count check
    room_count = sum(1 for cid in cls_ids if cid == 3) # Room class ID
    log.info("[Auto-Training] Room count: %d", room_count)
    if room_count < MIN_ROOMS or room_count > MAX_ROOMS:
        log.info("[Auto-Training] Rejected: Room count outside valid range [%d, %d]", MIN_ROOMS, MAX_ROOMS)
        return False
        
    # 3. Room overlaps check (no two room boxes overlapping heavily)
    room_boxes = [boxes_xyxy[i] for i, cid in enumerate(cls_ids) if cid == 3]
    for i in range(len(room_boxes)):
        for j in range(i + 1, len(room_boxes)):
            iou = _bbox_iou(room_boxes[i], room_boxes[j])
            if iou > MAX_ROOM_IOU:
                log.info("[Auto-Training] Rejected: Heavily overlapping room boxes detected (IoU: %.3f)", iou)
                return False
                
    # 4. Coverage check
    sketch_h, sketch_w = result.orig_shape[:2]
    total_area = sketch_w * sketch_h
    union_area = 0
    for r_box in room_boxes:
        w = r_box[2] - r_box[0]
        h = r_box[3] - r_box[1]
        union_area += w * h
    coverage = union_area / total_area
    log.info("[Auto-Training] Ground coverage ratio: %.3f", coverage)
    if coverage < MIN_COVERAGE or coverage > MAX_COVERAGE:
        log.info("[Auto-Training] Rejected: Coverage outside valid bounds [%.2f, %.2f]", MIN_COVERAGE, MAX_COVERAGE)
        return False
        
    log.info("[Auto-Training] Accepted: YOLO prediction passed all geometric audit rules.")
    return True

def collect_training_sample(sketch: np.ndarray, result, project_number: str) -> bool:
    """Extract and save verified YOLO labels and preprocessed sketch image locally and to SharePoint."""
    try:
        # High-confidence passes are saved as verified training data. Failures
        # (low confidence / odd geometry) are the cases that actually IMPROVE the
        # model, so when AUTO_TRAINING_CAPTURE_FAILURES=true save them to a
        # "needs_review" subfolder for human correction instead of discarding.
        verified = validate_predictions(result)
        if verified:
            sub_img, sub_lbl = "images", "labels"
        elif os.getenv("AUTO_TRAINING_CAPTURE_FAILURES", "false").strip().lower() in ("true", "1", "yes"):
            sub_img, sub_lbl = "needs_review/images", "needs_review/labels"
            log.info("[Auto-Training] Capturing FAILED/low-confidence sample for human review: %s", project_number)
        else:
            return False

        pn_clean = re.sub(r'[^\w\-]', '_', project_number)
        
        # 1. Ensure local folders exist
        LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)
        
        local_img_path = LOCAL_IMAGES_DIR / f"{pn_clean}.jpg"
        local_lbl_path = LOCAL_LABELS_DIR / f"{pn_clean}.txt"
        
        # 2. Save preprocessed sketch image locally
        cv2.imwrite(str(local_img_path), sketch)
        
        # 3. Save YOLO labels (segmentation polygons preferred, fall back to boxes)
        lines = []
        cls_ids = result.boxes.cls.int().tolist()
        
        if result.masks is not None and len(result.masks.xyn) > 0:
            polygons = result.masks.xyn
            for idx, poly in enumerate(polygons):
                cls_id = cls_ids[idx]
                if len(poly) > 0:
                    coords_str = " ".join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly)
                    lines.append(f"{cls_id} {coords_str}")
        else:
            # Bounding box fallback (normalized xywh)
            boxes_xywhn = result.boxes.xywhn.cpu().numpy()
            for idx, box in enumerate(boxes_xywhn):
                cls_id = cls_ids[idx]
                lines.append(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                
        local_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        log.info("[Auto-Training] Saved verified training sample locally: %s", pn_clean)
        
        # 4. Sync files to SharePoint
        img_ok = upload_file_to_sharepoint(local_img_path, sub_img)
        lbl_ok = upload_file_to_sharepoint(local_lbl_path, sub_lbl)
        
        if img_ok and lbl_ok:
            log.info("[Auto-Training] Sync to SharePoint complete: %s", pn_clean)
            # Clean up local files to prevent server disk space issues
            try:
                if local_img_path.exists():
                    local_img_path.unlink()
                if local_lbl_path.exists():
                    local_lbl_path.unlink()
                log.info("[Auto-Training] Cleaned up local files for: %s", pn_clean)
            except Exception as clean_ex:
                log.error("[Auto-Training] Failed to clean up local files: %s", clean_ex)
            return True
        else:
            log.warning("[Auto-Training] Local files saved, but SharePoint sync failed.")
            return False
            
    except Exception as e:
        log.error("[Auto-Training] Error collecting training sample: %s", e)
        return False
