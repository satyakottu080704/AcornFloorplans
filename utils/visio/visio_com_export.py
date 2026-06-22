"""
Visio COM Export - Creates valid .vsdx files using Windows COM automation
=========================================================================
This is the ONLY reliable way to create Visio files that open without errors.
XML-based approaches cause "missing parts" errors in Visio.

Requirements:
    - Microsoft Visio installed
    - pip install pywin32
"""

import os
import sys
import subprocess
import time

_HAS_COM = False

if sys.platform == 'win32':
    try:
        import win32com.client
        import pythoncom
        _HAS_COM = True
    except ImportError:
        print("[VISIO] WARNING: pywin32 not installed. Run: pip install pywin32")
else:
    print("[VISIO] WARNING: COM export requires Windows with Microsoft Visio installed")


def _kill_stale_visio():
    """Kill any orphaned Visio processes that block COM automation."""
    if os.environ.get("ALLOW_VISIO_TASKKILL", "false").strip().lower() != "true":
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq VISIO.EXE"],
            capture_output=True, text=True, timeout=5
        )
        if "VISIO.EXE" in result.stdout:
            count = result.stdout.count("VISIO.EXE")
            print(f"[VISIO] Found {count} stale Visio process(es), killing...")
            subprocess.run(
                ["taskkill", "/F", "/IM", "VISIO.EXE"],
                capture_output=True, timeout=10
            )
            time.sleep(1)  # Brief pause for process cleanup
            return True
    except Exception as e:
        print(f"[VISIO] Warning: Could not check for stale processes: {e}")
    return False


class VisioComExporter:
    """
    Create professional floor plan Visio files using COM automation.

    Usage:
        exporter = VisioComExporter()
        exporter.create_document()
        exporter.add_room((100, 100, 200, 150), has_acm=True, label="Kitchen")
        exporter.add_room((300, 100, 180, 150), has_acm=False, label="Living Room")
        exporter.save("output/floor_plan.vsdx")
        exporter.close()
    """

    # A3 Landscape page size in inches
    PAGE_WIDTH = 16.54   # ~420mm
    PAGE_HEIGHT = 11.69  # ~297mm

    def __init__(self):
        self.visio = None
        self.doc = None
        self.page = None
        self._initialized = False

    def create_document(self, page_name="Floor Plan"):
        """Initialize Visio and create a new document."""
        if not _HAS_COM:
            raise RuntimeError(
                "Visio COM export requires Windows with pywin32 and Microsoft Visio installed. "
                "Install pywin32: pip install pywin32"
            )

        # Try COM, and if it fails, kill stale processes and retry once
        for attempt in range(2):
            try:
                pythoncom.CoInitialize()

                self.visio = win32com.client.Dispatch("Visio.Application")
                try:
                    self.visio.Visible = False
                except Exception:
                    pass  # Some Visio versions don't allow this

                self.doc = self.visio.Documents.Add("")
                self.page = self.doc.Pages.Item(1)
                self.page.Name = page_name

                # Set page size to A3 Landscape
                self.page.PageSheet.Cells("PageWidth").FormulaU = f"{self.PAGE_WIDTH} in"
                self.page.PageSheet.Cells("PageHeight").FormulaU = f"{self.PAGE_HEIGHT} in"

                self._initialized = True
                print(f"[VISIO] Created document: {page_name}")
                return

            except Exception as e:
                if attempt == 0:
                    print(f"[VISIO] COM failed ({e}), killing stale Visio processes and retrying...")
                    self._cleanup_com()
                    _kill_stale_visio()
                else:
                    print(f"[VISIO] ERROR creating document after retry: {e}")
                    self._cleanup_com()
                    raise

    def _cleanup_com(self):
        """Safely release COM objects."""
        try:
            if self.doc:
                self.doc.Close()
        except Exception:
            pass
        try:
            if self.visio:
                self.visio.Quit()
        except Exception:
            pass
        self.visio = None
        self.doc = None
        self.page = None
        self._initialized = False
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def add_room(self, bbox, has_acm=False, label=None, image_size=(3000, 4000)):
        """
        Add a room rectangle to the floor plan.

        Args:
            bbox: (x, y, width, height) in pixels from original image
            has_acm: True if room contains asbestos
            label: Room name/label text
            image_size: (width, height) of original image for scaling
        """
        if not self._initialized:
            raise RuntimeError("Call create_document() first")

        x, y, w, h = bbox
        img_w, img_h = image_size

        # Scale pixel coordinates to Visio page inches
        scale_x = self.PAGE_WIDTH / img_w
        scale_y = self.PAGE_HEIGHT / img_h

        # Convert to Visio coordinates (origin at bottom-left, Y increases upward)
        left = x * scale_x
        bottom = self.PAGE_HEIGHT - (y + h) * scale_y
        right = (x + w) * scale_x
        top = self.PAGE_HEIGHT - y * scale_y

        # Draw rectangle
        shape = self.page.DrawRectangle(left, bottom, right, top)

        # Apply styling based on ACM status
        if has_acm:
            # RED fill for ACM rooms
            shape.Cells("FillForegnd").FormulaU = "RGB(255,200,200)"
            shape.Cells("LineColor").FormulaU = "RGB(200,0,0)"
            shape.Cells("LineWeight").FormulaU = "2 pt"
        else:
            # GREEN fill for clear rooms
            shape.Cells("FillForegnd").FormulaU = "RGB(200,255,200)"
            shape.Cells("LineColor").FormulaU = "RGB(0,0,0)"
            shape.Cells("LineWeight").FormulaU = "1 pt"

        # Add label text
        if label:
            shape.Text = label
            shape.Cells("Char.Size").FormulaU = "10 pt"
            shape.Cells("VerticalAlign").FormulaU = "1"  # Center
            shape.Cells("Para.HorzAlign").FormulaU = "1"  # Center

        return shape

    def add_title(self, title_text, subtitle_text=None):
        """Add title at top of page."""
        if not self._initialized:
            raise RuntimeError("Call create_document() first")

        # Title position (top center)
        title_x = self.PAGE_WIDTH / 2
        title_y = self.PAGE_HEIGHT - 0.5

        title_shape = self.page.DrawRectangle(
            title_x - 3, title_y - 0.3,
            title_x + 3, title_y + 0.3
        )
        title_shape.Text = title_text
        title_shape.Cells("Char.Size").FormulaU = "16 pt"
        title_shape.Cells("Char.Style").FormulaU = "1"  # Bold
        title_shape.Cells("LinePattern").FormulaU = "0"  # No border
        title_shape.Cells("FillPattern").FormulaU = "0"  # No fill

        if subtitle_text:
            sub_shape = self.page.DrawRectangle(
                title_x - 4, title_y - 0.7,
                title_x + 4, title_y - 0.4
            )
            sub_shape.Text = subtitle_text
            sub_shape.Cells("Char.Size").FormulaU = "10 pt"
            sub_shape.Cells("LinePattern").FormulaU = "0"
            sub_shape.Cells("FillPattern").FormulaU = "0"

    def add_legend(self):
        """Add color legend at bottom of page."""
        if not self._initialized:
            raise RuntimeError("Call create_document() first")

        legend_x = 1.0
        legend_y = 0.8
        box_size = 0.3

        # ACM legend
        acm_box = self.page.DrawRectangle(
            legend_x, legend_y,
            legend_x + box_size, legend_y + box_size
        )
        acm_box.Cells("FillForegnd").FormulaU = "RGB(255,200,200)"
        acm_box.Cells("LineColor").FormulaU = "RGB(200,0,0)"

        acm_label = self.page.DrawRectangle(
            legend_x + 0.4, legend_y,
            legend_x + 2.0, legend_y + box_size
        )
        acm_label.Text = "ACM Detected"
        acm_label.Cells("Char.Size").FormulaU = "9 pt"
        acm_label.Cells("LinePattern").FormulaU = "0"
        acm_label.Cells("FillPattern").FormulaU = "0"

        # Clear legend
        clear_box = self.page.DrawRectangle(
            legend_x + 2.5, legend_y,
            legend_x + 2.5 + box_size, legend_y + box_size
        )
        clear_box.Cells("FillForegnd").FormulaU = "RGB(200,255,200)"
        clear_box.Cells("LineColor").FormulaU = "RGB(0,0,0)"

        clear_label = self.page.DrawRectangle(
            legend_x + 2.9, legend_y,
            legend_x + 4.5, legend_y + box_size
        )
        clear_label.Text = "No ACM"
        clear_label.Cells("Char.Size").FormulaU = "9 pt"
        clear_label.Cells("LinePattern").FormulaU = "0"
        clear_label.Cells("FillPattern").FormulaU = "0"

    def save(self, output_path):
        """Save the document as .vsdx file."""
        if not self._initialized:
            raise RuntimeError("Call create_document() first")

        if not output_path.endswith('.vsdx'):
            output_path += '.vsdx'

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.doc.SaveAs(output_path)
        print(f"[VISIO] Saved: {output_path}")

        return output_path

    def close(self):
        """Close document and quit Visio."""
        self._cleanup_com()


def create_visio_plan(rooms, output_path, image_size, title=None):
    """
    Convenience function to create a complete floor plan.

    Args:
        rooms: List of DetectedRoom objects with bbox, has_acm, label
        output_path: Output .vsdx file path
        image_size: (width, height) of source image
        title: Optional title text

    Returns:
        Path to created .vsdx file
    """
    exporter = VisioComExporter()

    try:
        exporter.create_document()

        # Add title
        if title:
            exporter.add_title(title)

        # Add all rooms
        for i, room in enumerate(rooms):
            is_acm = getattr(room, 'has_acm', False) or getattr(room, 'room_type', '') == 'acm'
            label = getattr(room, 'label', None) or f"Room {i+1}"
            if is_acm and "[ACM]" not in label:
                label += " [ACM]"

            exporter.add_room(
                bbox=room.bbox,
                has_acm=is_acm,
                label=label,
                image_size=image_size
            )

        # Add legend
        exporter.add_legend()

        # Save
        return exporter.save(output_path)

    finally:
        exporter.close()


# Simple test
if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Tuple

    @dataclass
    class TestRoom:
        bbox: Tuple[int, int, int, int]
        has_acm: bool
        label: str = None

    # Create test rooms
    test_rooms = [
        TestRoom(bbox=(200, 200, 400, 300), has_acm=False, label="Living Room"),
        TestRoom(bbox=(650, 200, 350, 300), has_acm=True, label="Kitchen"),
        TestRoom(bbox=(200, 550, 350, 250), has_acm=False, label="Bedroom 1"),
        TestRoom(bbox=(600, 550, 400, 250), has_acm=True, label="Bathroom"),
    ]

    output = create_visio_plan(
        rooms=test_rooms,
        output_path="output/visio/test_plan.vsdx",
        image_size=(1200, 900),
        title="Test Floor Plan"
    )

    print(f"\nTest complete! Open: {output}")
