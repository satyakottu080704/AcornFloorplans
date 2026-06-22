"""Visio overlay exporter.

For complex photographed plans, preserving the source geometry is better than
trying to reconstruct walls from imperfect room boxes. This exporter places the
preprocessed sketch image as a locked background. AI/model labels, room
highlights, and sample markers are opt-in review annotations.
"""
from __future__ import annotations

import gc
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _should_render_overlay_room(room: Dict[str, Any]) -> bool:
    """Render room labels only when explicitly enabled and evidence-backed."""
    if os.environ.get("OVERLAY_DRAW_ROOM_LABELS", "false").strip().lower() != "true":
        return False
    geometry_source = room.get("geometry_source") or "unknown"
    label = str(room.get("label") or "")
    if room.get("is_fallback") or geometry_source == "synthesized":
        return False
    return not (geometry_source == "model" and label.lower().startswith("room "))


def _should_render_overlay_samples() -> bool:
    """Keep uncertain AI sample detections off source-faithful overlays."""
    return os.environ.get("OVERLAY_DRAW_SAMPLE_MARKERS", "false").strip().lower() == "true"


def _publish_saved_vsdx(temp_path: str, requested_path: str) -> str:
    """Move a completed temporary VSDX to its requested path.

    Visio cannot overwrite a VSDX that is currently open. In that case keep
    the generated file under a timestamped name instead of losing the run.
    """
    if os.path.exists(requested_path):
        try:
            os.remove(requested_path)
        except PermissionError:
            base, ext = os.path.splitext(requested_path)
            requested_path = f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
            print(f"[VISIO-OVERLAY] Requested file is open; saving as {os.path.basename(requested_path)}")
    # Visio can keep the completed temporary file locked for a few seconds
    # after Document.Close/Quit. Retry the atomic publish before falling back.
    last_error: Optional[OSError] = None
    for _attempt in range(10):
        try:
            os.replace(temp_path, requested_path)
            return requested_path
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5)
        except OSError:
            break
    try:
        shutil.move(temp_path, requested_path)
    except OSError as move_error:
        # Some agent/output directories allow create/write but deny rename or
        # delete. A completed VSDX can still be published by copying it.
        try:
            shutil.copy2(temp_path, requested_path)
            try:
                os.remove(temp_path)
            except OSError:
                pass
        except OSError:
            if last_error is not None:
                raise last_error
            raise move_error
    return requested_path


def generate_overlay_visio(
    detected: Dict[str, Any],
    background_image_path: str,
    project_number: str,
    output_path: str,
) -> Optional[str]:
    try:
        import win32com.client
        import pythoncom
    except ImportError:
        print("[VISIO-OVERLAY] win32com not available - cannot generate Visio files")
        return None

    if not os.path.exists(background_image_path):
        print(f"[VISIO-OVERLAY] Background not found: {background_image_path}")
        return None

    rooms = detected.get("rooms", []) or []
    samples = detected.get("sample_details", []) or []
    sketch_w, sketch_h = (detected.get("sketch_size") or [0, 0])[:2]
    if not sketch_w or not sketch_h:
        print("[VISIO-OVERLAY] Missing sketch_size; cannot align overlay.")
        return None

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    temp_output = os.path.join(
        os.path.dirname(abs_output),
        f".{Path(abs_output).stem}.{uuid.uuid4().hex}.tmp.vsdx",
    )

    visio = None
    doc = None
    saved_doc = None
    page = bg = title = legend = rect = text = marker = label = None
    try:
        pythoncom.CoInitialize()
        # Use an isolated automation process. Dispatch may attach to an
        # operator's open Visio session, leaving generated files locked.
        visio = win32com.client.DispatchEx("Visio.Application")
        try:
            visio.Visible = False
            visio.AlertResponse = 7
        except Exception:
            pass

        doc = visio.Documents.Add("")
        page = doc.Pages.Item(1)
        page.Name = detected.get("floor_title") or "Overlay"

        PAGE_W = 16.54
        PAGE_H = 11.69
        MARGIN_L = 0.35
        MARGIN_R = 0.35
        MARGIN_T = 0.65
        MARGIN_B = 0.45
        page.PageSheet.Cells("PageWidth").FormulaU = f"{PAGE_W} in"
        page.PageSheet.Cells("PageHeight").FormulaU = f"{PAGE_H} in"

        draw_w = PAGE_W - MARGIN_L - MARGIN_R
        draw_h = PAGE_H - MARGIN_T - MARGIN_B
        scale = min(draw_w / sketch_w, draw_h / sketch_h)
        img_w = sketch_w * scale
        img_h = sketch_h * scale
        off_x = MARGIN_L + (draw_w - img_w) / 2
        off_y = MARGIN_B + (draw_h - img_h) / 2

        def to_visio(px: float, py: float):
            return off_x + px * scale, off_y + img_h - py * scale

        # Background image.
        bg = page.Import(os.path.abspath(background_image_path))
        bg.Cells("PinX").FormulaU = f"{off_x + img_w / 2} in"
        bg.Cells("PinY").FormulaU = f"{off_y + img_h / 2} in"
        bg.Cells("Width").FormulaU = f"{img_w} in"
        bg.Cells("Height").FormulaU = f"{img_h} in"
        try:
            bg.SendToBack()
            bg.Cells("LockWidth").FormulaU = "1"
            bg.Cells("LockHeight").FormulaU = "1"
            bg.Cells("LockMoveX").FormulaU = "1"
            bg.Cells("LockMoveY").FormulaU = "1"
        except Exception:
            pass

        title = page.DrawRectangle(0.3, PAGE_H - 0.35, 5.0, PAGE_H - 0.05)
        title.Text = f"{project_number} - Overlay Review"
        title.Cells("LinePattern").FormulaU = "0"
        title.Cells("FillPattern").FormulaU = "0"
        title.Cells("Char.Size").FormulaU = "10 pt"
        title.Cells("Char.Style").FormulaU = "1"

        fill_by_type = {
            "acm": ("RGB(220,50,50)", "RGB(170,0,0)", 65),
            "no_access": ("RGB(50,100,210)", "RGB(0,60,170)", 55),
            "clear": ("RGB(80,190,80)", "RGB(0,120,0)", 88),
        }
        draw_room_boxes = os.environ.get("OVERLAY_DRAW_ROOM_BOXES", "false").strip().lower() == "true"

        for idx, room in enumerate(rooms, 1):
            if not _should_render_overlay_room(room):
                continue
            label_value = str(room.get("label") or "")
            bbox = room.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = bbox
            x1, y1 = to_visio(x, y + h)
            x2, y2 = to_visio(x + w, y)
            room_type = room.get("type") or "clear"
            if room.get("no_access"):
                room_type = "no_access"
            fill, line, transparency = fill_by_type.get(room_type, fill_by_type["clear"])

            # AI/model boxes are approximate. Do not draw them as surveyed
            # walls unless an operator explicitly enables diagnostic boxes.
            if draw_room_boxes:
                rect = page.DrawRectangle(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                rect.Cells("FillForegnd").FormulaU = fill
                rect.Cells("FillPattern").FormulaU = "1"
                rect.Cells("Transparency").FormulaU = f"{transparency}%"
                rect.Cells("LineColor").FormulaU = line
                rect.Cells("LineWeight").FormulaU = "1.25 pt"

            cx, cy = to_visio(x + w / 2, y + h / 2)
            label = label_value or f"Room {idx}"
            num = room.get("room_number") or f"{idx:03d}"
            if room_type == "no_access" and "NO ACCESS" not in label.upper():
                label = "NO ACCESS\n" + label
            text = page.DrawRectangle(cx - 0.65, cy - 0.23, cx + 0.65, cy + 0.23)
            text.Text = f"{num}\n{label}"
            text.Cells("FillForegnd").FormulaU = "RGB(255,255,255)"
            text.Cells("FillPattern").FormulaU = "1"
            text.Cells("Transparency").FormulaU = "35%"
            text.Cells("LineColor").FormulaU = line
            text.Cells("LineWeight").FormulaU = "0.5 pt"
            text.Cells("Char.Size").FormulaU = "6.5 pt"
            text.Cells("Char.Style").FormulaU = "1"

        for sample in samples if _should_render_overlay_samples() else []:
            loc = sample.get("location")
            if not loc or len(loc) < 2:
                continue
            sx, sy = to_visio(loc[0], loc[1])
            marker = page.DrawOval(sx - 0.08, sy - 0.08, sx + 0.08, sy + 0.08)
            marker.Cells("FillForegnd").FormulaU = "RGB(220,0,0)"
            marker.Cells("LineColor").FormulaU = "RGB(120,0,0)"
            label = page.DrawRectangle(sx + 0.10, sy + 0.02, sx + 1.15, sy + 0.32)
            sid = sample.get("id") or "S?"
            mat = sample.get("material") or ""
            label.Text = f"{sid} {mat}".strip()
            label.Cells("FillForegnd").FormulaU = "RGB(255,245,245)"
            label.Cells("FillPattern").FormulaU = "1"
            label.Cells("LineColor").FormulaU = "RGB(180,0,0)"
            label.Cells("Char.Color").FormulaU = "RGB(180,0,0)"
            label.Cells("Char.Size").FormulaU = "6 pt"
            label.Cells("Char.Style").FormulaU = "1"

        legend = page.DrawRectangle(PAGE_W - 3.1, 0.20, PAGE_W - 0.35, 0.95)
        room_labels_enabled = (
            os.environ.get("OVERLAY_DRAW_ROOM_LABELS", "false").strip().lower() == "true"
        )
        sample_markers_enabled = _should_render_overlay_samples()
        legend.Text = (
            "Source-faithful survey overlay\nAI annotations hidden"
            if not room_labels_enabled and not sample_markers_enabled
            else "AI Draft overlay: original survey sketch preserved\nLabels and sample markers require review"
        )
        legend.Cells("FillForegnd").FormulaU = "RGB(255,255,255)"
        legend.Cells("FillPattern").FormulaU = "1"
        legend.Cells("Transparency").FormulaU = "5%"
        legend.Cells("LineColor").FormulaU = "RGB(160,160,160)"
        legend.Cells("Char.Size").FormulaU = "7 pt"

        try:
            doc.SaveAs(temp_output)
        except Exception:
            # Some Visio COM wrappers expose a stale Documents.Add handle.
            visio.ActiveDocument.SaveAs(temp_output)
        saved_doc = visio.ActiveDocument
        try:
            saved_doc.Close()
            visio.Quit()
        except Exception:
            pass
        # COM proxies keep Visio alive even after Quit(), which keeps the
        # freshly saved VSDX locked. Release every shape/page reference before
        # publishing the temporary file.
        page = bg = title = legend = rect = text = marker = label = None
        doc = saved_doc = visio = None
        pythoncom.CoUninitialize()
        gc.collect()
        published_output = _publish_saved_vsdx(temp_output, abs_output)
        print(f"[VISIO-OVERLAY] Generated overlay floor plan: {published_output}")
        return published_output
    except Exception as e:
        print(f"[VISIO-OVERLAY] Generation error: {e}")
        import traceback
        traceback.print_exc()
        if visio:
            try:
                if doc:
                    doc.Close()
                visio.Quit()
            except Exception:
                pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass
        try:
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except OSError:
            pass
        return None
