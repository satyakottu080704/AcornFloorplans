#!/usr/bin/env python3
"""
Windows Visio Agent for Acorn Floor Plans
=========================================
Runs in the background on the Windows host.
Polls SharePoint folder 'General/AI Automation/Pending_Draw' for layout JSON files.
Generates premium Visio floor plans using local MS Visio COM automation.
Uploads the finished VSDX back to SharePoint 'General/AI Automation/Generated_Plans'.
"""

import os
import sys
import json
import time
import math
import re
import argparse
import requests
import tempfile
from pathlib import Path
import urllib.parse

# Setup path and import dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# Read configuration from .env
CLIENT_ID = os.getenv("ACORN_CLIENT_ID") or os.getenv("SP_CLIENT_ID") or os.getenv("OUTLOOK_CLIENT_ID")
CLIENT_SECRET = os.getenv("ACORN_CLIENT_SECRET") or os.getenv("SP_CLIENT_SECRET") or os.getenv("OUTLOOK_CLIENT_SECRET")
TENANT_ID = os.getenv("ACORN_TENANT_ID") or os.getenv("SP_TENANT_ID") or os.getenv("OUTLOOK_TENANT_ID")
DRIVE_ID = (os.getenv("SP_DRIVE_ID") or "").strip()  # required; no hardcoded fallback

PENDING_FOLDER = os.getenv("SHAREPOINT_PENDING_FOLDER", "General/AI Automation/Pending_Draw").strip().strip("/")
OUTPUT_FOLDER = os.getenv("SHAREPOINT_OUTPUT_FOLDER", "General/AI Automation/Generated_Plans").strip().strip("/")
USE_MODEL = os.getenv("DRAW_USE_MODEL", "true").strip().lower() in ("true", "1", "yes")

def get_graph_token():
    """Acquire MS Graph Token using client credentials."""
    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]):
        print("[ERROR] Missing SharePoint Microsoft Graph API credentials in .env")
        return None
        
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    try:
        r = requests.post(url, data=data, timeout=15)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print(f"[ERROR] Failed to acquire access token: {e}")
        return None

def list_pending_drawings(token):
    """List all pending layout and image files in the Pending_Draw folder."""
    escaped_folder = urllib.parse.quote(PENDING_FOLDER)
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{escaped_folder}:/children"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 404:
            return [] # Folder empty or not created yet
        r.raise_for_status()
        items = r.json().get("value", [])
        pending = []
        for item in items:
            name = item.get("name", "").lower()
            if (name.endswith(".json") or 
                name.endswith(".jpg") or 
                name.endswith(".jpeg") or 
                name.endswith(".png")):
                pending.append(item)
        return pending
    except Exception as e:
        print(f"[ERROR] Failed to list pending drawings: {e}")
        return []

def download_layout_file(token, item_id):
    """Download and parse a JSON layout file."""
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Failed to download layout file {item_id}: {e}")
        return None

def download_binary_file(token, item_id, local_path):
    """Download a file from SharePoint as binary and write it to local_path."""
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download file {item_id} to {local_path}: {e}")
        return False

def upload_vsdx_to_sharepoint(token, local_path, project_number):
    """Upload generated VSDX to SharePoint Generated_Plans folder."""
    filename = f"{project_number} AI Draft.vsdx"
    escaped_folder = urllib.parse.quote(OUTPUT_FOLDER)
    escaped_filename = urllib.parse.quote(filename)
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{escaped_folder}/{escaped_filename}:/content"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream"
    }
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        r = requests.put(url, headers=headers, data=data, timeout=60)
        r.raise_for_status()
        print(f"[SUCCESS] Uploaded {filename} to SharePoint.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to upload VSDX to SharePoint: {e}")
        return False

def delete_pending_file(token, item_id):
    """Delete the JSON layout file from Pending_Draw folder."""
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/items/{item_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.delete(url, headers=headers, timeout=15)
        r.raise_for_status()
        print(f"[INFO] Deleted pending JSON layout file {item_id} from SharePoint.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete pending file {item_id}: {e}")
        return False

def draw_visio_plan(layout, project_number, output_path):
    """Draw floor plan using Microsoft Visio COM API."""
    import win32com.client
    import pythoncom
    
    print(f"[INFO] Drawing Visio floor plan for {project_number}...")
    pythoncom.CoInitialize()
    
    visio = None
    try:
        visio = win32com.client.Dispatch("Visio.Application")
        try:
            visio.Visible = False
            visio.AlertResponse = 7 # Auto-dismiss popups
            visio.Settings.ShowSmartTags = False
        except Exception:
            pass
            
        doc = visio.Documents.Add("")
        page = doc.Pages.Item(1)
        page.Name = "Floor Plan"
        
        # A3 Landscape dimensions
        PAGE_W = 16.54
        PAGE_H = 11.69
        page.PageSheet.Cells("PageWidth").FormulaU = f"{PAGE_W} in"
        page.PageSheet.Cells("PageHeight").FormulaU = f"{PAGE_H} in"
        
        # Convert 1000x1000 grid coordinates to page inches
        SCALE_X = PAGE_W / 1000.0
        SCALE_Y = PAGE_H / 1000.0
        
        def to_inches(x, y):
            # Scale coordinates and invert Y axis (Visio is Y-up)
            return x * SCALE_X, PAGE_H - (y * SCALE_Y)
            
        # 1. Draw Page Border Frame
        border_margin = 0.4
        bx1, by1 = border_margin, border_margin
        bx2, by2 = PAGE_W - border_margin, PAGE_H - border_margin
        border = page.DrawRectangle(bx1, by1, bx2, by2)
        border.Cells("FillPattern").FormulaU = "0"
        border.Cells("LineColor").FormulaU = "RGB(180,180,180)"
        border.Cells("LineWeight").FormulaU = "1.0 pt"
        
        # 2. Draw Title Block (Top-Left)
        title_text = f"Floor Plan - Project: {project_number}"
        tx1, ty1 = border_margin + 0.2, PAGE_H - border_margin - 0.7
        tx2, ty2 = tx1 + 6.0, ty1 + 0.4
        title_shape = page.DrawRectangle(tx1, ty1, tx2, ty2)
        title_shape.Cells("FillPattern").FormulaU = "0"
        title_shape.Cells("LinePattern").FormulaU = "0"
        title_shape.Text = title_text
        title_shape.Cells("Char.Size").FormulaU = "14 pt"
        title_shape.Cells("Char.Style").FormulaU = "5" # Bold + Underline
        title_shape.Cells("Char.Color").FormulaU = "RGB(0,0,0)"
        title_shape.Cells("Para.HorzAlign").FormulaU = "0" # Left
        
        # 3. Draw Windows (Nice thick blue lines)
        for win in layout.get("windows", []):
            x1, y1 = to_inches(win.get("x1", 0), win.get("y1", 0))
            x2, y2 = to_inches(win.get("x2", 0), win.get("y2", 0))
            line = page.DrawLine(x1, y1, x2, y2)
            line.Cells("LineWeight").FormulaU = "4.0 pt"
            line.Cells("LineColor").FormulaU = "RGB(41,128,185)" # Premium Blue
            
        # 4. Draw Doors (Panels + dashed arcs)
        for door in layout.get("doors", []):
            hx, hy = to_inches(door.get("hinge_x", 0), door.get("hinge_y", 0))
            ox, oy = to_inches(door.get("open_x", 0), door.get("open_y", 0))
            cx, cy = to_inches(door.get("closed_x", 0), door.get("closed_y", 0))
            
            # Draw panel line
            panel = page.DrawLine(hx, hy, ox, oy)
            panel.Cells("LineWeight").FormulaU = "2.0 pt"
            panel.Cells("LineColor").FormulaU = "RGB(36,94,168)"
            
            # Draw swing arc (calculate midpoint of 90 deg swing)
            r = math.sqrt((ox - hx)**2 + (oy - hy)**2)
            if r > 0.05:
                a_closed = math.atan2(cy - hy, cx - hx)
                a_open = math.atan2(oy - hy, ox - hx)
                
                diff = a_open - a_closed
                if diff > math.pi:
                    diff -= 2 * math.pi
                elif diff < -math.pi:
                    diff += 2 * math.pi
                a_mid = a_closed + diff / 2
                
                mx = hx + r * math.cos(a_mid)
                my = hy + r * math.sin(a_mid)
                
                try:
                    arc = page.DrawArcByThreePoints(cx, cy, ox, oy, mx, my)
                    arc.Cells("LineWeight").FormulaU = "1.2 pt"
                    arc.Cells("LineColor").FormulaU = "RGB(52,152,219)"
                    arc.Cells("LinePattern").FormulaU = "2" # Dashed
                except Exception:
                    # Fallback to straight dashed line if arc fails
                    fallback = page.DrawLine(cx, cy, ox, oy)
                    fallback.Cells("LineWeight").FormulaU = "1.2 pt"
                    fallback.Cells("LineColor").FormulaU = "RGB(52,152,219)"
                    fallback.Cells("LinePattern").FormulaU = "2"
                    
        # 5. Draw Walls (exterior = bold weight, interior = thin weight)
        for wall in layout.get("walls", []):
            x1, y1 = to_inches(wall.get("x1", 0), wall.get("y1", 0))
            x2, y2 = to_inches(wall.get("x2", 0), wall.get("y2", 0))
            w_type = wall.get("type", "interior")
            weight = "4.5 pt" if w_type == "exterior" else "2.0 pt"
            line = page.DrawLine(x1, y1, x2, y2)
            line.Cells("LineWeight").FormulaU = weight
            line.Cells("LineColor").FormulaU = "RGB(32,32,32)"
            
        # 6. Draw Rooms (Centered Bold Label + White Background Mask Rectangle)
        for room in layout.get("rooms", []):
            name = room.get("name", "Room")
            rx, ry = to_inches(room.get("x", 0), room.get("y", 0))
            
            text_w = max(1.5, len(name) * 0.12)
            text_h = 0.4
            
            tx1 = rx - text_w / 2
            ty1 = ry - text_h / 2
            tx2 = rx + text_w / 2
            ty2 = ry + text_h / 2
            
            rect = page.DrawRectangle(tx1, ty1, tx2, ty2)
            rect.Cells("FillForegnd").FormulaU = "RGB(255,255,255)" # White bg mask
            rect.Cells("FillPattern").FormulaU = "1"
            rect.Cells("LinePattern").FormulaU = "0" # No border outline
            rect.Text = name
            rect.Cells("Char.Size").FormulaU = "12 pt"
            rect.Cells("Char.Style").FormulaU = "1" # Bold
            rect.Cells("Char.Color").FormulaU = "RGB(44,62,80)" # Premium Slate Blue
            rect.Cells("Para.HorzAlign").FormulaU = "1" # Center
            
        # 7. Draw Samples (Red Pins with White Borders + White-masked ID Labels)
        for sample in layout.get("samples", []):
            sid = sample.get("id", "S001")
            sx, sy = to_inches(sample.get("x", 0), sample.get("y", 0))
            
            # Red circle pin
            pin_r = 0.12
            circle = page.DrawOval(sx - pin_r, sy - pin_r, sx + pin_r, sy + pin_r)
            circle.Cells("FillForegnd").FormulaU = "RGB(231,76,60)"
            circle.Cells("FillPattern").FormulaU = "1"
            circle.Cells("LineColor").FormulaU = "RGB(255,255,255)" # White border
            circle.Cells("LineWeight").FormulaU = "1.5 pt"
            
            # Label beside pin
            lx, ly = sx + 0.25, sy
            label_w = max(0.7, len(sid) * 0.12)
            label_h = 0.35
            
            ltx1 = lx - label_w / 2
            lty1 = ly - label_h / 2
            ltx2 = lx + label_w / 2
            lty2 = ly + label_h / 2
            
            lbl_rect = page.DrawRectangle(ltx1, lty1, ltx2, lty2)
            lbl_rect.Cells("FillForegnd").FormulaU = "RGB(255,255,255)"
            lbl_rect.Cells("FillPattern").FormulaU = "1"
            lbl_rect.Cells("LinePattern").FormulaU = "0"
            lbl_rect.Text = sid
            lbl_rect.Cells("Char.Size").FormulaU = "11 pt"
            lbl_rect.Cells("Char.Style").FormulaU = "1" # Bold
            lbl_rect.Cells("Char.Color").FormulaU = "RGB(192,57,43)" # Dark Red
            lbl_rect.Cells("Para.HorzAlign").FormulaU = "0" # Left
            
        # Save VSDX document
        doc.SaveAs(output_path)
        print(f"[SUCCESS] Saved generated Visio file locally to: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to draw plan in Visio: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if visio:
            try:
                visio.Quit()
            except Exception:
                pass

def process_single_drawing(token, item):
    """Download pending drawing, generate VSDX (using professional pipeline or layout helper), upload VSDX, delete pending."""
    filename = item.get("name", "")
    item_id = item.get("id")
    
    # Extract project number from filename, e.g. N-108457_sketch.jpg or N-108457_layout.json
    m = re.search(r"N-?\d{4,10}", filename, re.IGNORECASE)
    if not m:
        print(f"[WARNING] Skipping file: {filename} (no valid project number found)")
        return
        
    pn = m.group(0).upper()
    if "-" not in pn:
        pn = "N-" + pn[1:]
        
    print(f"\n[INFO] Processing file: {filename} for project: {pn}")
    
    is_json = filename.lower().endswith(".json")
    temp_dir = Path(os.environ.get("TEMP") or tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_input = temp_dir / filename
    
    if local_input.exists():
        try:
            local_input.unlink()
        except OSError:
            pass
            
    # Download file content
    dl_success = download_binary_file(token, item_id, str(local_input))
    if not dl_success or not local_input.exists():
        print(f"[ERROR] Skipping project {pn} due to download failure.")
        return
        
    # Target locally drawn VSDX path
    local_vsdx = temp_dir / f"{pn}_plan.vsdx"
    if local_vsdx.exists():
        try:
            local_vsdx.unlink()
        except OSError:
            pass
            
    success = False
    if is_json:
        # Load JSON and draw using basic COM helper
        try:
            with open(local_input, "r", encoding="utf-8") as f:
                layout = json.load(f)
            success = draw_visio_plan(layout, pn, str(local_vsdx))
        except Exception as e:
            print(f"[ERROR] Failed to process layout JSON: {e}")
    else:
        # Run professional pipeline locally
        try:
            from pipeline import process_sketch
            print(f"[INFO] Invoking local professional Visio pipeline on {local_input}...")
            vsdx, plan = process_sketch(
                str(local_input),
                output_path=str(local_vsdx),
                no_model=not USE_MODEL,
            )
            success = vsdx and os.path.exists(vsdx)
        except Exception as e:
            print(f"[ERROR] Professional Visio pipeline invocation failed: {e}")
            import traceback
            traceback.print_exc()

    # Clean up local input
    try:
        local_input.unlink()
    except OSError:
        pass
        
    if not success or not local_vsdx.exists():
        print(f"[ERROR] Skipped project {pn} due to Visio generation failure.")
        return
        
    # Upload generated vsdx
    uploaded = upload_vsdx_to_sharepoint(token, str(local_vsdx), pn)
    
    # Clean up local vsdx
    try:
        local_vsdx.unlink()
    except OSError:
        pass
        
    if uploaded:
        # Delete pending file on success
        delete_pending_file(token, item_id)
        print(f"[SUCCESS] Project {pn} successfully processed end-to-end!")

def main():
    parser = argparse.ArgumentParser(description="Process SharePoint Pending_Draw sketches with Visio")
    parser.add_argument("--project", help="Only process this project number, e.g. N-108451")
    parser.add_argument("--once", action="store_true", help="Process the current queue once, then exit")
    args = parser.parse_args()

    project_filter = None
    if args.project:
        match = re.search(r"N-?\d{4,10}", args.project, re.IGNORECASE)
        if not match:
            parser.error("--project must contain a valid N-number")
        project_filter = match.group(0).upper()
        if "-" not in project_filter:
            project_filter = "N-" + project_filter[1:]

    print("======================================================")
    print("      Acorn Windows Visio Drawing Agent Active")
    print("======================================================")
    print(f"Watching SharePoint folder: {PENDING_FOLDER}")
    print(f"Uploading VSDX files to:  {OUTPUT_FOLDER}")
    
    # Ensure Visio is installed
    try:
        import win32com.client
    except ImportError:
        print("[CRITICAL] pywin32 library not found. Install with: pip install pywin32")
        sys.exit(1)

    # SharePoint drive id must come from the environment (no hardcoded default).
    if not DRIVE_ID:
        print("[CRITICAL] SP_DRIVE_ID is not set. Add it to .env / the environment.")
        sys.exit(1)
        
    while True:
        token = get_graph_token()
        if not token:
            print("[ERROR] Token authentication failed. Retrying in 15 seconds...")
            time.sleep(15)
            continue
            
        items = list_pending_drawings(token)
        if project_filter:
            items = [
                item for item in items
                if project_filter.lower() in item.get("name", "").lower()
            ]
        if items:
            print(f"[INFO] Found {len(items)} pending layout file(s).")
            for item in items:
                try:
                    process_single_drawing(token, item)
                except Exception as ex:
                    print(f"[ERROR] Exception during layout processing: {ex}")
        else:
            # Subtle indicator of active polling
            print(".", end="", flush=True)

        if args.once:
            break
        time.sleep(6) # Poll every 6 seconds

if __name__ == "__main__":
    main()
