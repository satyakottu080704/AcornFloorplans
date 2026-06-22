#!/usr/bin/env python3
"""
Aspose.Diagram Renderer for Acorn Floor Plans
=============================================
Natively generates premium floor plan .vsdx drawings from layout coordinates
inside Linux Docker containers, bypassing Visio COM automation.
"""

import os
import math
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Aspose.Diagram runs on .NET. On minimal Linux images no compatible ICU is
# present, so enable invariant globalization before the import (the container
# Dockerfile sets this too, but a bare Linux/dev machine will not).
os.environ.setdefault("DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "1")

# Try to import aspose.diagram
try:
    import aspose.diagram as ad
    ASPOSE_AVAILABLE = True
except ImportError:
    ASPOSE_AVAILABLE = False

logger = logging.getLogger("aspose_renderer")

PAGE_W = 16.54  # A3 in inches
PAGE_H = 11.69
SCALE_X = PAGE_W / 1000.0
SCALE_Y = PAGE_H / 1000.0

def to_inches(x, y):
    """Convert 1000x1000 coordinates to page inches, flipping Y-axis for Visio (Y-up)."""
    return x * SCALE_X, PAGE_H - (y * SCALE_Y)

def render_plan_to_vsdx(
    detected: Dict[str, Any],
    project_number: str,
    template_path: str,
    output_path: str
) -> Optional[str]:
    """
    Render floor plan natively inside Linux container (or Windows fallback) via Aspose.Diagram.
    
    Args:
        detected: Layout dictionary containing rooms, walls, doors, windows, samples.
        project_number: e.g. 'N-108434'
        template_path: Path to 'Plan template.vsd' or 'template.vsdx'
        output_path: Target path to write final '.vsdx'
    """
    if not ASPOSE_AVAILABLE:
        logger.error("aspose-diagram-python is not installed. Cannot use Aspose renderer.")
        return None

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    try:
        logger.info(f"[ASPOSE] Loading template from {template_path}...")
        diagram = ad.Diagram(template_path)
        
        # Group everything by floor_idx
        rooms_by_floor = {}
        for room in detected.get("rooms", []):
            f_idx = int(room.get("floor_idx", 0) or 0)
            rooms_by_floor.setdefault(f_idx, []).append(room)
            
        walls_by_floor = {}
        for wall in detected.get("walls", []):
            f_idx = int(wall.get("floor_idx", 0) or 0)
            walls_by_floor.setdefault(f_idx, []).append(wall)

        doors_by_floor = {}
        for door in detected.get("doors", []):
            f_idx = int(door.get("floor_idx", 0) or 0)
            doors_by_floor.setdefault(f_idx, []).append(door)

        windows_by_floor = {}
        for win in detected.get("windows", []):
            f_idx = int(win.get("floor_idx", 0) or 0)
            windows_by_floor.setdefault(f_idx, []).append(win)

        samples_by_floor = {}
        for sample in detected.get("samples", []):
            f_idx = int(sample.get("floor_idx", 0) or 0)
            samples_by_floor.setdefault(f_idx, []).append(sample)

        # Get unique floor indices in ascending order
        all_floor_indices = sorted(list(set(
            list(rooms_by_floor.keys()) + 
            list(walls_by_floor.keys()) + 
            list(doors_by_floor.keys()) + 
            list(windows_by_floor.keys()) + 
            list(samples_by_floor.keys())
        )))

        if not all_floor_indices:
            all_floor_indices = [0]

        logger.info(f"[ASPOSE] Rendering {len(all_floor_indices)} page(s) for floors: {all_floor_indices}")

        # Map floor_idx to page names
        floor_names_map = {}
        for r in detected.get("rooms", []):
            f_idx = int(r.get("floor_idx", 0) or 0)
            f_name = r.get("floor") or "Floor Plan"
            floor_names_map[f_idx] = f_name

        # Process each floor
        for idx_pos, f_idx in enumerate(all_floor_indices):
            f_name = floor_names_map.get(f_idx) or f"Floor {f_idx}"
            
            # Find or copy page
            if idx_pos == 0:
                # Use the template's first page
                page = diagram.pages[0]
                page.name = f_name
            else:
                # Copy from first page to keep borders and title block
                new_page = ad.Page()
                new_page.copy(diagram.pages[0])
                new_page.name = f_name
                page_idx = diagram.pages.add(new_page)
                page = diagram.pages.get_page(page_idx)

            logger.info(f"[ASPOSE] Rendering page '{f_name}'...")

            # 1. Draw Walls
            for wall in walls_by_floor.get(f_idx, []):
                x1, y1 = to_inches(wall.get("x1", 0), wall.get("y1", 0))
                x2, y2 = to_inches(wall.get("x2", 0), wall.get("y2", 0))
                w_type = wall.get("type", "interior")
                
                id_line = page.draw_line(x1, y1, x2, y2)
                s = page.shapes.get_shape(id_line)
                s.line.line_color.value = "#202020"
                s.line.line_weight.value = 0.035 if w_type == "exterior" else 0.015
                s.line.line_pattern.value = ad.LinePatternValue.SOLID

            # 2. Draw Windows
            for win in windows_by_floor.get(f_idx, []):
                x1, y1 = to_inches(win.get("x1", 0), win.get("y1", 0))
                x2, y2 = to_inches(win.get("x2", 0), win.get("y2", 0))
                
                id_line = page.draw_line(x1, y1, x2, y2)
                s = page.shapes.get_shape(id_line)
                s.line.line_color.value = "#2980B9"  # Thick premium blue
                s.line.line_weight.value = 0.045
                s.line.line_pattern.value = ad.LinePatternValue.SOLID

            # 3. Draw Doors (panels + dashed swings)
            for door in doors_by_floor.get(f_idx, []):
                hx, hy = to_inches(door.get("hinge_x", 0), door.get("hinge_y", 0))
                ox, oy = to_inches(door.get("open_x", 0), door.get("open_y", 0))
                cx, cy = to_inches(door.get("closed_x", 0), door.get("closed_y", 0))

                # Panel
                id_panel = page.draw_line(hx, hy, ox, oy)
                s_panel = page.shapes.get_shape(id_panel)
                s_panel.line.line_color.value = "#34495E"
                s_panel.line.line_weight.value = 0.02
                s_panel.line.line_pattern.value = ad.LinePatternValue.SOLID

                # Swing path
                id_swing = page.draw_line(cx, cy, ox, oy)
                s_swing = page.shapes.get_shape(id_swing)
                s_swing.line.line_color.value = "#3498DB"
                s_swing.line.line_weight.value = 0.012
                s_swing.line.line_pattern.value = ad.LinePatternValue.DASH

            # 4. Draw Rooms (centered white text box mask)
            for room in rooms_by_floor.get(f_idx, []):
                name = room.get("label") or room.get("name") or "Room"
                bbox = room.get("bbox") or [0, 0, 100, 100]
                min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
                
                x1, y1 = to_inches(min_x, min_y)
                x2, y2 = to_inches(max_x, max_y)
                
                rx = (x1 + x2) / 2
                ry = (y1 + y2) / 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                # Make label box size
                lw = max(1.5, len(name) * 0.12)
                lh = 0.4
                
                id_rect = page.draw_rectangle(rx, ry, lw, lh)
                s_rect = page.shapes.get_shape(id_rect)
                s_rect.fill.fill_foregnd.value = "#FFFFFF"
                s_rect.fill.fill_pattern.value = 1
                s_rect.line.line_pattern.value = ad.LinePatternValue.NONE
                
                # Text value
                s_rect.text.value.set_whole_text(name)
                
                # Font size/bold
                if s_rect.chars.count == 0:
                    s_rect.chars.add(ad.Char())
                s_rect.chars[0].size.value = 11.5 / 72.0  # ~11.5pt
                s_rect.chars[0].style.value = ad.StyleValue.BOLD
                s_rect.chars[0].color.value = "#2C3E50"  # premium slate blue

                # Center align
                if s_rect.paras.count == 0:
                    s_rect.paras.add(ad.Para())
                s_rect.paras[0].horz_align.value = ad.HorzAlignValue.CENTER

            # 5. Draw Sample Pins (red circle + ID label)
            for sample in samples_by_floor.get(f_idx, []):
                sid = sample.get("id", "S001")
                sx, sy = to_inches(sample.get("x", 0), sample.get("y", 0))

                # Pin circle
                pin_r = 0.12
                id_pin = page.draw_ellipse(sx, sy, pin_r * 2, pin_r * 2)
                s_pin = page.shapes.get_shape(id_pin)
                s_pin.fill.fill_foregnd.value = "#E74C3C"
                s_pin.fill.fill_pattern.value = 1
                s_pin.line.line_color.value = "#FFFFFF"
                s_pin.line.line_weight.value = 0.02

                # ID label beside pin
                try:
                    page.add_text(sx + 0.25, sy, 0.7, 0.35, sid)
                except Exception:
                    pass
                s_lbl = page.shapes[page.shapes.count - 1]
                s_lbl.fill.fill_pattern.value = 0
                s_lbl.line.line_pattern.value = ad.LinePatternValue.NONE
                if s_lbl.chars.count == 0:
                    s_lbl.chars.add(ad.Char())
                s_lbl.chars[0].size.value = 11.0 / 72.0
                s_lbl.chars[0].style.value = ad.StyleValue.BOLD
                s_lbl.chars[0].color.value = "#C0392B"  # dark red text

            # 6. Draw Floor Title Block (Top-Left under border)
            walls = walls_by_floor.get(f_idx, [])
            if walls:
                min_w_x = min([w.get("x1", 500) for w in walls] + [w.get("x2", 500) for w in walls])
                max_w_y = min([w.get("y1", 500) for w in walls] + [w.get("y2", 500) for w in walls])
                title_x, title_y = to_inches(min_w_x, max_w_y)
                title_y += 0.45
            else:
                title_x, title_y = 0.6, PAGE_H - 1.2
                
            try:
                page.add_text(title_x + 1.5, title_y, 3.0, 0.4, f"{f_name}:")
            except Exception:
                pass
            s_title = page.shapes[page.shapes.count - 1]
            s_title.fill.fill_pattern.value = 0
            s_title.line.line_pattern.value = ad.LinePatternValue.NONE
            if s_title.chars.count == 0:
                s_title.chars.add(ad.Char())
            s_title.chars[0].size.value = 12.0 / 72.0
            s_title.chars[0].style.value = ad.StyleValue.BOLD | ad.StyleValue.UNDERLINE
            s_title.chars[0].color.value = "#000000"

        # Save VSDX drawing
        diagram.save(abs_output, ad.SaveFileFormat.VSDX)
        logger.info(f"[ASPOSE] SUCCESS: Saved generated Visio file locally to: {abs_output}")
        return abs_output
        
    except Exception as e:
        logger.error(f"[ASPOSE] Failed to render plan via Aspose: {e}", exc_info=True)
        return None
