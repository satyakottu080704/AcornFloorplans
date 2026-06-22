#!/usr/bin/env python3
"""
AI Floor Plan Vector Generator
==============================
Processes hand-drawn sketches or images of floor plans and generates
clean, layered vector SVG files suitable for importing into Visio.
Also directly generates Visio .vsdx files natively without paid libraries.

Usage:
    python src/plans/generate_plan.py N-xxxxx --image path/to/sketch.jpg
"""

import os
import sys
import json
import base64
import argparse
import logging
import zipfile
import tempfile
import shutil
import time
import re
import uuid
import requests
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to python path to resolve imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from src.utils.layout_extractor import extract_floor_plan_layout as run_layout_extract
except ImportError:
    try:
        # Repo layout (no src/ prefix) — e.g. local test runs.
        from utils.layout_extractor import extract_floor_plan_layout as run_layout_extract
    except ImportError:
        logger.warning("Could not import extract_floor_plan_layout from src.utils.layout_extractor or utils.layout_extractor")
        run_layout_extract = None

# Try to import svgwrite
try:
    import svgwrite
    SVGWRITE_AVAILABLE = True
except ImportError:
    SVGWRITE_AVAILABLE = False
    logger.warning("svgwrite not installed. Install with: pip install svgwrite")


def extract_floor_plan_layout(image_path: Path) -> Dict[str, Any]:
    """
    Call AI Vision API to extract structured floor plan layout.
    """
    if not run_layout_extract:
        raise ValueError("Layout extraction function is not imported/available.")
    
    logger.info(f"Extracting layout coordinates from image: {image_path.name} using AI Vision...")
    return run_layout_extract(str(image_path))


class NativeVsdxExporter:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.pages = {}  # {page_name: [ET.Element]}
        self.page_names = []
        self.current_page = None
        self.shape_id = 1

        # Read page1.xml from template
        import xml.etree.ElementTree as ET
        ET.register_namespace('', 'http://schemas.microsoft.com/office/visio/2012/main')
        ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
        
        with zipfile.ZipFile(str(template_path.absolute()), 'r') as zin:
            page1_str = zin.read('visio/pages/page1.xml').decode('utf-8')
            pages_str = zin.read('visio/pages/pages.xml').decode('utf-8')
            
        self.template_page1_root = ET.fromstring(page1_str)
        pages_root = ET.fromstring(pages_str)

        # Use the template's own coordinate system. The corporate template is
        # metric and stores an A3 page at roughly 1655 x 1169 units; drawing
        # new shapes at 2-14 "inch-like" coordinates makes them microscopic.
        ns = {'v': 'http://schemas.microsoft.com/office/visio/2012/main'}
        page_sheet = next(
            (element for element in pages_root.iter() if element.tag.endswith("PageSheet")),
            None,
        )

        def page_value(name: str, fallback: float) -> float:
            if page_sheet is not None:
                for cell in list(page_sheet):
                    if cell.tag.endswith('Cell') and cell.get('N') == name:
                        try:
                            return float(cell.get('V'))
                        except (TypeError, ValueError):
                            break
            return fallback

        self.page_width = page_value("PageWidth", 1655.0)
        self.page_height = page_value("PageHeight", 1169.0)
        self.margin_l = self.page_width * 0.15
        self.margin_r = self.page_width * 0.15
        self.margin_t = self.page_height * 0.09
        self.margin_b = self.page_height * 0.10
        self.scale_x = (self.page_width - self.margin_l - self.margin_r) / 1000.0
        self.scale_y = (self.page_height - self.margin_t - self.margin_b) / 1000.0
        
        # Find existing template shapes on page1.xml (border, logo, etc.)
        self.template_shapes = []
        shapes_el = self.template_page1_root.find('v:Shapes', ns)
        if shapes_el is None:
            shapes_el = self.template_page1_root.find('Shapes')
            
        if shapes_el is not None:
            for s in shapes_el.findall('v:Shape', ns) + shapes_el.findall('Shape'):
                if self._shape_intersects_page(s):
                    self.template_shapes.append(s)
                
        # Find the true maximum shape ID recursively (including grouped shapes)
        # to prevent duplicate ID collisions with custom-drawn elements.
        max_id = 0
        for s in self.template_page1_root.iter():
            if s.tag.endswith('Shape'):
                sid_str = s.get('ID')
                if sid_str and sid_str.isdigit():
                    max_id = max(max_id, int(sid_str))
                    
        self.initial_shape_id = max_id + 1

    def _shape_intersects_page(self, shape) -> bool:
        """Keep page furniture, but exclude the template's off-page stencil palette."""
        cells = {
            cell.get("N"): cell.get("V")
            for cell in list(shape)
            if cell.tag.endswith("Cell") and cell.get("V") is not None
        }
        try:
            pin_x = float(cells["PinX"])
            pin_y = float(cells["PinY"])
            width = abs(float(cells.get("Width", 0)))
            height = abs(float(cells.get("Height", 0)))
        except (KeyError, TypeError, ValueError):
            return False
        return not (
            pin_x + width / 2 < 0
            or pin_y + height / 2 < 0
            or pin_x - width / 2 > self.page_width
            or pin_y - height / 2 > self.page_height
        )

    def set_page(self, name: str):
        self.current_page = name
        if name not in self.pages:
            import copy
            self.pages[name] = [copy.deepcopy(s) for s in self.template_shapes]
            self.page_names.append(name)
        self.shape_id = self.initial_shape_id + len(self.pages[name]) - len(self.template_shapes)

    def to_inches(self, x, y):
        # Convert 1000x1000 layout coordinates into the template's page units.
        ix = self.margin_l + x * self.scale_x
        iy = self.margin_b + (1000.0 - y) * self.scale_y
        return ix, iy

    def reposition_title(self, floor_name: str, min_x: float, min_y: float):
        """
        Finds the matching template title shape (e.g. Ground Floor, First Floor, Loft)
        and positions it at the top left of the floor plan drawing (slightly above min_y).
        """
        ns = {'v': 'http://schemas.microsoft.com/office/visio/2012/main'}
        
        # Place title 40 coordinate units above the top of the drawing (min_y - 40)
        # and aligned with the left edge (min_x)
        ix, iy = self.to_inches(min_x, max(0.0, min_y - 40.0))
        
        # Determine target search text based on page floor_name
        search_term = ""
        clean_floor = floor_name.lower()
        if "ground" in clean_floor:
            search_term = "Ground Floor:"
        elif "first" in clean_floor:
            search_term = "First Floor:"
        elif "loft" in clean_floor or "attic" in clean_floor:
            search_term = "Loft:"
        else:
            search_term = "Ground Floor:"  # Fallback to Ground Floor template shape, and we will update its text
            
        # Find the shape on the current page
        for s in self.pages[self.current_page]:
            text_el = s.find('v:Text', ns)
            if text_el is None:
                text_el = s.find('Text')
            if text_el is not None:
                text_content = "".join(text_el.itertext()).lower()
                # If we match our search term or any floor title shapes
                if "ground floor:" in text_content or "first floor:" in text_content or "loft:" in text_content or "external:" in text_content:
                    # If this is the one we want to move
                    if search_term.lower().replace(":", "") in text_content:
                        # Move it to our target coordinates
                        for cell in s.findall('v:Cell', ns) + s.findall('Cell'):
                            if cell.get('N') == 'PinX':
                                cell.set('V', str(ix + 1.5))  # Add half-width offset because Visio uses center PinX
                            elif cell.get('N') == 'PinY':
                                cell.set('V', str(iy))
                        
                        # If the name is custom, update its text
                        if clean_floor not in ["ground floor", "first floor", "loft"]:
                            # Remove all child elements of Text (like cp, pp)
                            for child in list(text_el):
                                text_el.remove(child)
                            text_el.text = f"{floor_name}:"
                        return True
        return False

    def add_shape_xml(self, xml_str, name="Generated Shape"):
        import xml.etree.ElementTree as ET
        shape_el = ET.fromstring(xml_str)
        # Visio Desktop tolerates anonymous native shapes, but Visio Online
        # drops them from its rendered view. Give every generated shape the
        # same identity attributes Visio writes for normal page shapes.
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "Generated_Shape"
        shape_name = f"{safe_name}.{self.shape_id}"
        shape_el.set("NameU", shape_name)
        shape_el.set("Name", shape_name)
        shape_el.set("UniqueID", "{" + str(uuid.uuid4()).upper() + "}")
        self.pages[self.current_page].append(shape_el)
        self.shape_id += 1

    def add_rect(self, x1, y1, x2, y2, fill_color="#FFFFFF", line_color="#000000", line_weight="1.0", name="Rect"):
        ix1, iy1 = self.to_inches(x1, y1)
        ix2, iy2 = self.to_inches(x2, y2)
        
        cx = (ix1 + ix2) / 2
        cy = (iy1 + iy2) / 2
        w = abs(ix2 - ix1)
        h = abs(iy2 - iy1)
        
        if not fill_color.startswith('#'):
            fill_color = f"#{fill_color}"
        if not line_color.startswith('#'):
            line_color = f"#{line_color}"
            
        try:
            pt_weight = float(line_weight)
        except ValueError:
            pt_weight = 1.0
        inch_weight = pt_weight / 72.0
            
        xml = f'''    <Shape xmlns="http://schemas.microsoft.com/office/visio/2012/main" ID="{self.shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">
      <Cell N="PinX" V="{cx}"/>
      <Cell N="PinY" V="{cy}"/>
      <Cell N="Width" V="{w}"/>
      <Cell N="Height" V="{h}"/>
      <Cell N="LocPinX" V="{w/2}" F="Width*0.5"/>
      <Cell N="LocPinY" V="{h/2}" F="Height*0.5"/>
      <Cell N="Angle" V="0"/>
      <Cell N="FlipX" V="0"/>
      <Cell N="FlipY" V="0"/>
      <Cell N="ResizeMode" V="0"/>
      <Cell N="FillForegnd" V="{fill_color}"/>
      <Cell N="FillPattern" V="1"/>
      <Cell N="LineColor" V="{line_color}"/>
      <Cell N="LineWeight" V="{inch_weight}" U="PT"/>
      <Cell N="LinePattern" V="1"/>
      <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="0"/>
        <Cell N="NoLine" V="0"/>
        <Cell N="NoShow" V="0"/>
        <Cell N="NoSnap" V="0"/>
        <Cell N="NoQuickDrag" V="0"/>
        <Row T="MoveTo" IX="1"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>
        <Row T="LineTo" IX="2"><Cell N="X" V="{w}" F="Width*1"/><Cell N="Y" V="0" F="Height*0"/></Row>
        <Row T="LineTo" IX="3"><Cell N="X" V="{w}" F="Width*1"/><Cell N="Y" V="{h}" F="Height*1"/></Row>
        <Row T="LineTo" IX="4"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="{h}" F="Height*1"/></Row>
        <Row T="LineTo" IX="5"><Cell N="X" V="0" F="Geometry1.X1"/><Cell N="Y" V="0" F="Geometry1.Y1"/></Row>
      </Section>
    </Shape>'''
        self.add_shape_xml(xml, name)

    def add_line(self, x1, y1, x2, y2, line_color="#000000", line_weight="1.0", name="Line", dashed=False):
        ix1, iy1 = self.to_inches(x1, y1)
        ix2, iy2 = self.to_inches(x2, y2)
        
        cx = (ix1 + ix2) / 2
        cy = (iy1 + iy2) / 2
        w = max(0.01, abs(ix2 - ix1))
        h = max(0.01, abs(iy2 - iy1))
        
        slope_up = ((x2 - x1) >= 0) == ((iy2 - iy1) >= 0)
        
        if slope_up:
            sx, sy = 0, 0
            ex, ey = w, h
            sx_formula, sy_formula = "Width*0", "Height*0"
            ex_formula, ey_formula = "Width*1", "Height*1"
        else:
            sx, sy = 0, h
            ex, ey = w, 0
            sx_formula, sy_formula = "Width*0", "Height*1"
            ex_formula, ey_formula = "Width*1", "Height*0"
            
        line_pattern = "2" if dashed else "1"
        
        if not line_color.startswith('#'):
            line_color = f"#{line_color}"
            
        try:
            pt_weight = float(line_weight)
        except ValueError:
            pt_weight = 1.0
        inch_weight = pt_weight / 72.0
            
        xml = f'''    <Shape xmlns="http://schemas.microsoft.com/office/visio/2012/main" ID="{self.shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">
      <Cell N="PinX" V="{cx}"/>
      <Cell N="PinY" V="{cy}"/>
      <Cell N="Width" V="{w}"/>
      <Cell N="Height" V="{h}"/>
      <Cell N="LocPinX" V="{w/2}" F="Width*0.5"/>
      <Cell N="LocPinY" V="{h/2}" F="Height*0.5"/>
      <Cell N="Angle" V="0"/>
      <Cell N="FlipX" V="0"/>
      <Cell N="FlipY" V="0"/>
      <Cell N="ResizeMode" V="0"/>
      <Cell N="FillPattern" V="0"/>
      <Cell N="LineColor" V="{line_color}"/>
      <Cell N="LineWeight" V="{inch_weight}" U="PT"/>
      <Cell N="LinePattern" V="{line_pattern}"/>
      <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="1"/>
        <Cell N="NoLine" V="0"/>
        <Cell N="NoShow" V="0"/>
        <Cell N="NoSnap" V="0"/>
        <Cell N="NoQuickDrag" V="0"/>
        <Row T="MoveTo" IX="1"><Cell N="X" V="{sx}" F="{sx_formula}"/><Cell N="Y" V="{sy}" F="{sy_formula}"/></Row>
        <Row T="LineTo" IX="2"><Cell N="X" V="{ex}" F="{ex_formula}"/><Cell N="Y" V="{ey}" F="{ey_formula}"/></Row>
      </Section>
    </Shape>'''
        self.add_shape_xml(xml, name)

    def add_circle(self, x, y, r=9, fill_color="#E74C3C", line_color="#FFFFFF", name="Circle"):
        ix, iy = self.to_inches(x, y)
        d = r * 2 * self.scale_x
        
        if not fill_color.startswith('#'):
            fill_color = f"#{fill_color}"
        if not line_color.startswith('#'):
            line_color = f"#{line_color}"
            
        xml = f'''    <Shape xmlns="http://schemas.microsoft.com/office/visio/2012/main" ID="{self.shape_id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">
      <Cell N="PinX" V="{ix}"/>
      <Cell N="PinY" V="{iy}"/>
      <Cell N="Width" V="{d}"/>
      <Cell N="Height" V="{d}"/>
      <Cell N="LocPinX" V="{d/2}" F="Width*0.5"/>
      <Cell N="LocPinY" V="{d/2}" F="Height*0.5"/>
      <Cell N="Angle" V="0"/>
      <Cell N="FlipX" V="0"/>
      <Cell N="FlipY" V="0"/>
      <Cell N="ResizeMode" V="0"/>
      <Cell N="FillForegnd" V="{fill_color}"/>
      <Cell N="FillPattern" V="1"/>
      <Cell N="LineColor" V="{line_color}"/>
      <Cell N="LineWeight" V="0.020833333333333332" U="PT"/>
      <Cell N="LinePattern" V="1"/>
      <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="0"/>
        <Cell N="NoLine" V="0"/>
        <Cell N="NoShow" V="0"/>
        <Cell N="NoSnap" V="0"/>
        <Cell N="NoQuickDrag" V="0"/>
        <Row T="Ellipse" IX="1">
          <Cell N="X" V="{d/2}" F="Width*0.5"/>
          <Cell N="Y" V="{d/2}" F="Height*0.5"/>
          <Cell N="A" V="{d}" F="Width*1"/>
          <Cell N="B" V="{d/2}" F="Height*0.5"/>
          <Cell N="C" V="{d/2}" F="Width*0.5"/>
          <Cell N="D" V="{d}" F="Height*1"/>
        </Row>
      </Section>
    </Shape>'''
        self.add_circle_xml = xml
        self.add_shape_xml(xml, name)

    def add_text(self, x, y, text, font_size=12, name="Text"):
        ix, iy = self.to_inches(x, y)
        # Text boxes must use the template's metric coordinate system too.
        # Inch-sized boxes make every character wrap onto a separate line.
        w = max(self.page_width * 0.08, len(text) * self.page_width * 0.006)
        h = self.page_height * 0.025
        
        char_size = font_size / 72.0
        
        xml = f'''    <Shape xmlns="http://schemas.microsoft.com/office/visio/2012/main" ID="{self.shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="3">
      <Cell N="PinX" V="{ix}"/>
      <Cell N="PinY" V="{iy}"/>
      <Cell N="Width" V="{w}"/>
      <Cell N="Height" V="{h}"/>
      <Cell N="LocPinX" V="{w/2}" F="Width*0.5"/>
      <Cell N="LocPinY" V="{h/2}" F="Height*0.5"/>
      <Cell N="Angle" V="0"/>
      <Cell N="FlipX" V="0"/>
      <Cell N="FlipY" V="0"/>
      <Cell N="ResizeMode" V="0"/>
      <Cell N="FillPattern" V="0"/>
      <Cell N="LineColor" V="#FFFFFF"/>
      <Cell N="LineWeight" V="0"/>
      <Cell N="LinePattern" V="0"/>
      <Section N="Character">
        <Row IX="0">
          <Cell N="Color" V="#1E1E1E"/>
          <Cell N="Size" V="{char_size}" U="PT"/>
          <Cell N="LangID" V="en-US"/>
        </Row>
      </Section>
      <Section N="Paragraph">
        <Row IX="0">
          <Cell N="HorzAlign" V="1"/>
        </Row>
      </Section>
      <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="0"/>
        <Cell N="NoLine" V="1"/>
        <Cell N="NoShow" V="0"/>
        <Cell N="NoSnap" V="0"/>
        <Cell N="NoQuickDrag" V="0"/>
        <Row T="MoveTo" IX="1"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>
        <Row T="LineTo" IX="2"><Cell N="X" V="{w}" F="Width*1"/><Cell N="Y" V="0" F="Height*0"/></Row>
        <Row T="LineTo" IX="3"><Cell N="X" V="{w}" F="Width*1"/><Cell N="Y" V="{h}" F="Height*1"/></Row>
        <Row T="LineTo" IX="4"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="{h}" F="Height*1"/></Row>
        <Row T="LineTo" IX="5"><Cell N="X" V="0" F="Geometry1.X1"/><Cell N="Y" V="0" F="Geometry1.Y1"/></Row>
      </Section>
      <Text><cp IX="0"/><pp IX="0"/>{text}</Text>
    </Shape>'''
        self.add_shape_xml(xml, name)

    def export(self, output_path: str):
        import xml.etree.ElementTree as ET
        import copy

        ET.register_namespace('', 'http://schemas.microsoft.com/office/visio/2012/main')
        ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')

        with zipfile.ZipFile(str(self.template_path.absolute()), 'r') as zin:
            content_types_str = zin.read('[Content_Types].xml').decode('utf-8')
            ET.register_namespace('', 'http://schemas.openxmlformats.org/package/2006/content-types')
            root_ct = ET.fromstring(content_types_str)

            pages_str = zin.read('visio/pages/pages.xml').decode('utf-8')
            ET.register_namespace('', 'http://schemas.microsoft.com/office/visio/2012/main')
            root_pages = ET.fromstring(pages_str)

            ns = {'v': 'http://schemas.microsoft.com/office/visio/2012/main',
                  'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
            first_page_el = root_pages.find('v:Page', ns)
            if first_page_el is None:
                first_page_el = root_pages.find('Page')

            rels_str = zin.read('visio/pages/_rels/pages.xml.rels').decode('utf-8')
            ET.register_namespace('', 'http://schemas.openxmlformats.org/package/2006/relationships')
            root_rels = ET.fromstring(rels_str)

            pages_xml_data = {}
            for page_idx, page_name in enumerate(self.page_names, start=1):
                page_content_root = copy.deepcopy(self.template_page1_root)
                shapes_el = page_content_root.find('v:Shapes', ns)
                if shapes_el is None:
                    shapes_el = page_content_root.find('Shapes')

                if shapes_el is not None:
                    shapes_el.clear()
                    for s in self.pages[page_name]:
                        shapes_el.append(s)

                pages_xml_data[f'visio/pages/page{page_idx}.xml'] = ET.tostring(
                    page_content_root, encoding='utf-8'
                )

                if page_idx == 1:
                    first_page_el.set('Name', page_name)
                    first_page_el.set('NameU', page_name)
                else:
                    new_page_el = copy.deepcopy(first_page_el)
                    new_page_el.set('ID', str(page_idx - 1))
                    new_page_el.set('Name', page_name)
                    new_page_el.set('NameU', page_name)
                    rel_child = new_page_el.find('v:Rel', ns)
                    if rel_child is None:
                        rel_child = new_page_el.find('Rel')
                    if rel_child is not None:
                        rel_child.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id', f'rId{page_idx}')
                    root_pages.append(new_page_el)

                    override_el = ET.Element('{http://schemas.openxmlformats.org/package/2006/content-types}Override')
                    override_el.set('PartName', f'/visio/pages/page{page_idx}.xml')
                    override_el.set('ContentType', 'application/vnd.ms-visio.page+xml')
                    root_ct.append(override_el)

                    rel_el = ET.Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
                    rel_el.set('Id', f'rId{page_idx}')
                    rel_el.set('Type', 'http://schemas.microsoft.com/visio/2010/relationships/page')
                    rel_el.set('Target', f'page{page_idx}.xml')
                    root_rels.append(rel_el)

            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if item.filename == '[Content_Types].xml':
                        zout.writestr(item, ET.tostring(root_ct, encoding='utf-8'))
                    elif item.filename == 'visio/pages/pages.xml':
                        zout.writestr(item, ET.tostring(root_pages, encoding='utf-8'))
                    elif item.filename == 'visio/pages/_rels/pages.xml.rels':
                        zout.writestr(item, ET.tostring(root_rels, encoding='utf-8'))
                    elif item.filename == 'visio/pages/page1.xml':
                        zout.writestr(item, pages_xml_data['visio/pages/page1.xml'])
                    elif item.filename.startswith('visio/pages/page') and item.filename.endswith('.xml'):
                        pass
                    else:
                        zout.writestr(item, zin.read(item.filename))

                for f_path, f_data in pages_xml_data.items():
                    if f_path != 'visio/pages/page1.xml':
                        zout.writestr(f_path, f_data)


def validate_native_vsdx(vsdx_path: Path, first_custom_shape_id: int) -> None:
    """Reject malformed or effectively blank native VSDX output."""
    import xml.etree.ElementTree as ET

    required_namespaces = {
        "[Content_Types].xml": "http://schemas.openxmlformats.org/package/2006/content-types",
        "visio/pages/pages.xml": "http://schemas.microsoft.com/office/visio/2012/main",
        "visio/pages/_rels/pages.xml.rels": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(vsdx_path) as package:
        if package.testzip() is not None:
            raise RuntimeError("Native VSDX package contains a corrupt part")
        for part, namespace in required_namespaces.items():
            root = ET.fromstring(package.read(part))
            if root.tag != f"{{{namespace}}}{root.tag.split('}')[-1]}":
                raise RuntimeError(f"Native VSDX part has invalid namespace: {part}")

        page_parts = sorted(
            name for name in package.namelist()
            if name.startswith("visio/pages/page")
            and name.endswith(".xml")
            and name != "visio/pages/pages.xml"
        )
        if not page_parts:
            raise RuntimeError("Native VSDX contains no drawing pages")

        for page_part in page_parts:
            page = ET.fromstring(package.read(page_part))
            custom_pins = []
            anonymous_custom_shapes = []
            for shape in page.iter():
                if not shape.tag.endswith("Shape"):
                    continue
                try:
                    shape_id = int(shape.get("ID", "0"))
                except ValueError:
                    continue
                if shape_id < first_custom_shape_id:
                    continue
                if not all(shape.get(attr) for attr in ("Name", "NameU", "UniqueID")):
                    anonymous_custom_shapes.append(shape_id)
                cells = {
                    cell.get("N"): float(cell.get("V"))
                    for cell in shape
                    if cell.tag.endswith("Cell")
                    and cell.get("N") in {"PinX", "PinY"}
                    and cell.get("V") is not None
                }
                if {"PinX", "PinY"} <= cells.keys():
                    custom_pins.append((cells["PinX"], cells["PinY"]))

            if not custom_pins:
                raise RuntimeError(f"Native VSDX page has no generated shapes: {page_part}")
            if anonymous_custom_shapes:
                raise RuntimeError(
                    f"Native VSDX page has anonymous generated shapes that Visio Online "
                    f"will not render: {page_part}"
                )
            x_span = max(x for x, _ in custom_pins) - min(x for x, _ in custom_pins)
            y_span = max(y for _, y in custom_pins) - min(y for _, y in custom_pins)
            if x_span < 100 and y_span < 100:
                raise RuntimeError(
                    f"Native VSDX generated content is microscopic on page: {page_part}"
                )


def _layout_points(layout: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Every coordinate in the layout, for bounding-box / fit calculations."""
    pts: List[Tuple[float, float]] = []
    for w in layout.get("walls", []):
        pts.append((float(w.get("x1", 0)), float(w.get("y1", 0))))
        pts.append((float(w.get("x2", 0)), float(w.get("y2", 0))))
    for d in layout.get("doors", []):
        pts.append((float(d.get("hinge_x", 0)), float(d.get("hinge_y", 0))))
        pts.append((float(d.get("open_x", 0)), float(d.get("open_y", 0))))
        pts.append((float(d.get("closed_x", 0)), float(d.get("closed_y", 0))))
    for win in layout.get("windows", []):
        pts.append((float(win.get("x1", 0)), float(win.get("y1", 0))))
        pts.append((float(win.get("x2", 0)), float(win.get("y2", 0))))
    for r in layout.get("rooms", []):
        pts.append((float(r.get("x", 0)), float(r.get("y", 0))))
    for s in layout.get("samples", []):
        pts.append((float(s.get("x", 0)), float(s.get("y", 0))))
    return pts


def _is_renderable_floor(fl: Dict[str, Any]) -> bool:
    """Whether a floor split has enough distinct geometry to render on its own tab.

    A floor reduced to a single point (e.g. one Loft room with no walls) cannot be
    fitted to the page — ``_fit_layout_to_canvas`` leaves a zero-span layout
    untouched, so it renders microscopically and trips the visibility guard,
    killing the whole render. Such a floor must not get its own page.
    """
    pts = _layout_points(fl)
    if len(pts) < 2:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) > 1.0 or (max(ys) - min(ys)) > 1.0


def _fit_layout_to_canvas(layout: Dict[str, Any], margin: float = 70.0,
                          coord_max: float = 1000.0) -> Dict[str, Any]:
    """Scale and centre the drawing so it fills the page instead of sitting tiny
    in one corner. Returns a new layout; the original is not mutated.

    Aspect ratio is preserved, so multiple floors keep their relative positions
    (proper per-floor pages are a separate roadmap item that needs floor tags).
    """
    pts = _layout_points(layout)
    if not pts:
        return layout

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x, span_y = max_x - min_x, max_y - min_y
    if span_x <= 0 and span_y <= 0:
        return layout

    usable = coord_max - 2 * margin
    scale = min(
        usable / span_x if span_x > 0 else usable,
        usable / span_y if span_y > 0 else usable,
    )
    off_x = margin + (usable - span_x * scale) / 2 - min_x * scale
    off_y = margin + (usable - span_y * scale) / 2 - min_y * scale

    def tx(v: Any) -> float:
        return float(v) * scale + off_x

    def ty(v: Any) -> float:
        return float(v) * scale + off_y

    fitted: Dict[str, Any] = {
        "walls": [
            {**w, "x1": tx(w.get("x1", 0)), "y1": ty(w.get("y1", 0)),
             "x2": tx(w.get("x2", 0)), "y2": ty(w.get("y2", 0))}
            for w in layout.get("walls", [])
        ],
        "doors": [
            {**d, "hinge_x": tx(d.get("hinge_x", 0)), "hinge_y": ty(d.get("hinge_y", 0)),
             "open_x": tx(d.get("open_x", 0)), "open_y": ty(d.get("open_y", 0)),
             "closed_x": tx(d.get("closed_x", 0)), "closed_y": ty(d.get("closed_y", 0))}
            for d in layout.get("doors", [])
        ],
        "windows": [
            {**win, "x1": tx(win.get("x1", 0)), "y1": ty(win.get("y1", 0)),
             "x2": tx(win.get("x2", 0)), "y2": ty(win.get("y2", 0))}
            for win in layout.get("windows", [])
        ],
        "rooms": [
            {**r, "x": tx(r.get("x", 0)), "y": ty(r.get("y", 0))}
            for r in layout.get("rooms", [])
        ],
        "samples": [
            {**s, "x": tx(s.get("x", 0)), "y": ty(s.get("y", 0))}
            for s in layout.get("samples", [])
        ],
    }
    return fitted


def _number_rooms(rooms: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """Pair each room with a display number (001, 002, ...).

    Honours an existing ``room_number`` if the extractor supplied one; otherwise
    auto-numbers in reading order (top-to-bottom, left-to-right), matching the
    professional_visio convention. Returns ``(room, number_str)`` pairs in the
    original list order.
    """
    order = sorted(
        range(len(rooms)),
        key=lambda i: (round(float(rooms[i].get("y", 0)), -1), float(rooms[i].get("x", 0))),
    )
    numbers: Dict[int, str] = {}
    seq = 1
    for i in order:
        existing = str(rooms[i].get("room_number") or "").strip()
        if existing:
            numbers[i] = existing.zfill(3) if existing.isdigit() else existing
        else:
            numbers[i] = f"{seq:03d}"
            seq += 1
    return [(rooms[i], numbers[i]) for i in range(len(rooms))]


def split_layout_by_floor(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group layout elements dynamically: only separate Loft/Attic onto its own tab.
    All other floors (Ground, First, etc.) are combined onto a single tab called 'Floor Plans'.
    """
    import copy
    rooms = layout.get("rooms", [])
    if not rooms:
        return [layout]

    # Check if there is any loft/attic room
    has_loft = any("loft" in str(r.get("name", "")).lower() or "attic" in str(r.get("name", "")).lower() for r in rooms)

    if not has_loft:
        # If no loft, just put everything on a single page named "Floor Plans"
        layout_copy = copy.deepcopy(layout)
        for r in layout_copy.get("rooms", []):
            r["floor"] = "Floor Plans"
            r["floor_idx"] = 0
        return [layout_copy]

    # Partition rooms into main and loft
    loft_rooms = []
    main_rooms = []
    for r in rooms:
        r_name = str(r.get("name", "")).lower()
        if "loft" in r_name or "attic" in r_name:
            loft_rooms.append(r)
        else:
            main_rooms.append(r)

    # If we have only loft rooms or only main rooms, we don't need to split
    if not loft_rooms or not main_rooms:
        layout_copy = copy.deepcopy(layout)
        f_name = "Loft" if loft_rooms else "Floor Plans"
        for r in layout_copy.get("rooms", []):
            r["floor"] = f_name
            r["floor_idx"] = 0
        return [layout_copy]

    # Create two layouts: one for main floors, one for loft
    main_layout = {
        "walls": [],
        "doors": [],
        "windows": [],
        "rooms": [copy.deepcopy(r) for r in main_rooms],
        "samples": [],
    }
    loft_layout = {
        "walls": [],
        "doors": [],
        "windows": [],
        "rooms": [copy.deepcopy(r) for r in loft_rooms],
        "samples": [],
    }

    # Set floor name and floor_idx for rooms
    for r in main_layout["rooms"]:
        r["floor"] = "Floor Plans"
        r["floor_idx"] = 0
    for r in loft_layout["rooms"]:
        r["floor"] = "Loft"
        r["floor_idx"] = 1

    # Helper to assign elements based on proximity to rooms
    def get_closest_floor_by_proximity(x, y):
        # We compare distance to all rooms in main_rooms vs loft_rooms
        min_main_dist = float('inf')
        for r in main_rooms:
            dist = (float(r.get("x", 0)) - x)**2 + (float(r.get("y", 0)) - y)**2
            if dist < min_main_dist:
                min_main_dist = dist

        min_loft_dist = float('inf')
        for r in loft_rooms:
            dist = (float(r.get("x", 0)) - x)**2 + (float(r.get("y", 0)) - y)**2
            if dist < min_loft_dist:
                min_loft_dist = dist

        if min_loft_dist < min_main_dist:
            return 1 # Loft
        else:
            return 0 # Floor Plans

    # Assign walls
    for w in layout.get("walls", []):
        mx = (float(w.get("x1", 0)) + float(w.get("x2", 0))) / 2
        my = (float(w.get("y1", 0)) + float(w.get("y2", 0))) / 2
        f_idx = get_closest_floor_by_proximity(mx, my)
        if f_idx == 0:
            main_layout["walls"].append(copy.deepcopy(w))
        else:
            loft_layout["walls"].append(copy.deepcopy(w))

    # Assign doors
    for d in layout.get("doors", []):
        mx = (float(d.get("hinge_x", 0)) + float(d.get("open_x", 0)) + float(d.get("closed_x", 0))) / 3
        my = (float(d.get("hinge_y", 0)) + float(d.get("open_y", 0)) + float(d.get("closed_y", 0))) / 3
        f_idx = get_closest_floor_by_proximity(mx, my)
        if f_idx == 0:
            main_layout["doors"].append(copy.deepcopy(d))
        else:
            loft_layout["doors"].append(copy.deepcopy(d))

    # Assign windows
    for win in layout.get("windows", []):
        mx = (float(win.get("x1", 0)) + float(win.get("x2", 0))) / 2
        my = (float(win.get("y1", 0)) + float(win.get("y2", 0))) / 2
        f_idx = get_closest_floor_by_proximity(mx, my)
        if f_idx == 0:
            main_layout["windows"].append(copy.deepcopy(win))
        else:
            loft_layout["windows"].append(copy.deepcopy(win))

    # Assign samples
    for s in layout.get("samples", []):
        sx = float(s.get("x", 0))
        sy = float(s.get("y", 0))
        f_idx = get_closest_floor_by_proximity(sx, sy)
        if f_idx == 0:
            main_layout["samples"].append(copy.deepcopy(s))
        else:
            loft_layout["samples"].append(copy.deepcopy(s))

    return [main_layout, loft_layout]



def generate_vsdx_natively(layout: Dict[str, Any], vsdx_path: Path):
    """
    Directly generates a valid .vsdx file from the layout dictionary,
    bypassing Aspose.Diagram entirely (100% free and watermarks-free).
    """
    logger.info(f"Generating Visio .vsdx natively for {vsdx_path.name}...")

    # Log what the layout actually contains.
    logger.info(
        "Layout received -> walls:%d doors:%d windows:%d rooms:%d samples:%d",
        len(layout.get("walls", [])), len(layout.get("doors", [])),
        len(layout.get("windows", [])), len(layout.get("rooms", [])),
        len(layout.get("samples", [])),
    )
    if not (layout.get("walls") or layout.get("rooms")):
        logger.warning(
            "Layout has no walls or rooms; the .vsdx will be nearly empty. "
            "Check that the extractor returned the expected schema."
        )

    # Split layout into separate floor plans
    floor_layouts = split_layout_by_floor(layout)

    # A separate floor tab needs enough distinct geometry to render visibly. If a
    # split produced a degenerate floor (e.g. a Loft with a single room and no
    # walls), it would render microscopically and fail the visibility guard,
    # aborting the whole render — so fall back to one combined page instead.
    if len(floor_layouts) > 1 and not all(_is_renderable_floor(fl) for fl in floor_layouts):
        logger.info(
            "A floor split was too sparse to render on its own tab; "
            "combining all floors onto a single page."
        )
        import copy
        combined = copy.deepcopy(layout)
        for r in combined.get("rooms", []):
            r["floor"] = "Floor Plans"
            r["floor_idx"] = 0
        floor_layouts = [combined]

    logger.info(f"Split layout into {len(floor_layouts)} floor plan(s).")

    # Determine template path
    template_path = Path(__file__).resolve().parent / "template.vsdx"
    if not template_path.exists():
        template_path = Path(__file__).resolve().parents[2] / "template.vsdx"
        
    if not template_path.exists():
        raise FileNotFoundError(f"Visio template not found at {template_path}")

    exporter = NativeVsdxExporter(template_path)

    # Render each floor layout to its own page
    for idx_pos, fl in enumerate(floor_layouts):
        rooms = fl.get("rooms", [])
        if not rooms and not fl.get("walls"):
            continue
            
        f_name = rooms[0].get("floor") if rooms else f"Floor {idx_pos + 1}"
        exporter.set_page(f_name)
        
        # Fit/scale the geometry of this floor layout to fill the page canvas margins
        fitted_fl = _fit_layout_to_canvas(fl)
        
        # Reposition and update the floor title shape from template
        pts = _layout_points(fitted_fl)
        if pts:
            min_x = min(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
        else:
            min_x = 70.0
            min_y = 70.0
        if not exporter.reposition_title(f_name, min_x, min_y):
            exporter.add_text(min_x + 80, max(20.0, min_y - 40.0), f"{f_name}:",
                              font_size=13, name="Floor Title")
        
        # 1. Windows
        for win in fitted_fl.get("windows", []):
            x1, y1 = win.get("x1", 0), win.get("y1", 0)
            x2, y2 = win.get("x2", 0), win.get("y2", 0)
            exporter.add_line(x1, y1, x2, y2, line_color="#34495E", line_weight="2.0", name="Window")
            
        # 2. Doors
        for door in fitted_fl.get("doors", []):
            hx, hy = door.get("hinge_x", 0), door.get("hinge_y", 0)
            ox, oy = door.get("open_x", 0), door.get("open_y", 0)
            cx, cy = door.get("closed_x", 0), door.get("closed_y", 0)
            exporter.add_line(hx, hy, ox, oy, line_color="#2980B9", line_weight="1.5", name="Door Panel")
            exporter.add_line(cx, cy, ox, oy, line_color="#3498DB", line_weight="1.0", name="Door Swing", dashed=True)
            
        # 3. Walls
        for wall in fitted_fl.get("walls", []):
            x1, y1 = wall.get("x1", 0), wall.get("y1", 0)
            x2, y2 = wall.get("x2", 0), wall.get("y2", 0)
            w_type = wall.get("type", "interior")
            weight = "3.0" if w_type == "exterior" else "1.5"
            exporter.add_line(x1, y1, x2, y2, line_color="#1E1E24", line_weight=weight, name="Wall")
            
        # 4. Rooms
        for room, number in _number_rooms(fitted_fl.get("rooms", [])):
            name = room.get("name", "Room")
            rx, ry = room.get("x", 0), room.get("y", 0)
            exporter.add_text(rx, ry - 11, number, font_size=13, name="Room Number")
            exporter.add_text(rx, ry + 9, name, font_size=11, name="Room Label")
            
        # 5. Samples
        for sample in fitted_fl.get("samples", []):
            sid = sample.get("id", "S001")
            sx, sy = sample.get("x", 0), sample.get("y", 0)
            exporter.add_circle(sx, sy, r=9, fill_color="#E74C3C", line_color="#FFFFFF", name="Sample Pin")
            exporter.add_text(sx + 25, sy - 5, sid, font_size=12, name="Sample Label")

    vsdx_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.export(str(vsdx_path.absolute()))
    validate_native_vsdx(vsdx_path, exporter.initial_shape_id)
    logger.info(f"Successfully generated native multi-page VSDX plan: {vsdx_path.name}")


def draw_svg_plan(layout: Dict[str, Any], output_path: Path):
    """
    Renders the floor plan layout into a clean, professional vector SVG file.
    Uses svgwrite to group and style layers for seamless Visio editing.
    """
    if not SVGWRITE_AVAILABLE:
        raise ImportError("svgwrite is required to generate the vector plan.")

    # Scale/centre the geometry to fill the page (matches the VSDX output).
    layout = _fit_layout_to_canvas(layout)

    # A4 printable aspect ratio at 1000px scale
    dwg = svgwrite.Drawing(str(output_path), size=('1000px', '1000px'), profile='full')

    # Plain white canvas (no grid — keeps it reading as a finished drawing).
    dwg.add(dwg.rect(insert=(0, 0), size=('1000', '1000'), fill='#FFFFFF'))

    # Layer 1: Windows (Subtle background layer for structural wall gaps)
    g_windows = dwg.add(dwg.g(id='windows', stroke='#34495E', stroke_width=2, fill='#E0F7FA', opacity=0.9))
    for win in layout.get("windows", []):
        x1, y1 = win.get("x1", 0), win.get("y1", 0)
        x2, y2 = win.get("x2", 0), win.get("y2", 0)
        # Draw window as parallel lines (rectangular frame)
        g_windows.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke_width=6))
        # Draw inner dividing line
        g_windows.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='white', stroke_width=2))

    # Layer 2: Doors (Pivot hinge, panel, and 90-degree swing path)
    g_doors = dwg.add(dwg.g(id='doors', stroke='#3498DB', stroke_width=2, fill='none'))
    for door in layout.get("doors", []):
        hx, hy = door.get("hinge_x", 0), door.get("hinge_y", 0)
        ox, oy = door.get("open_x", 0), door.get("open_y", 0)
        cx, cy = door.get("closed_x", 0), door.get("closed_y", 0)
        
        # 1. Door panel (open door line)
        g_doors.add(dwg.line(start=(hx, hy), end=(ox, oy), stroke='#2980B9', stroke_width=3))
        
        # 2. Door swing arc (90-degree curve from closed to open position)
        dx_open = ox - hx
        dy_open = oy - hy
        dx_closed = cx - hx
        dy_closed = cy - hy
        
        cross_product = dx_closed * dy_open - dy_closed * dx_open
        sweep = 1 if cross_product >= 0 else 0
        
        radius = ((ox - hx)**2 + (oy - hy)**2)**0.5
        if radius > 0:
            path_d = f"M {cx} {cy} A {radius} {radius} 0 0 {sweep} {ox} {oy}"
            g_doors.add(dwg.path(d=path_d, stroke_dasharray='4,4', opacity=0.7))

    # Layer 3: Walls (Dark premium charcoal lines, rounded caps for clean joins)
    g_walls = dwg.add(dwg.g(id='walls', stroke='#1E1E24', stroke_linecap='round'))
    for wall in layout.get("walls", []):
        x1, y1 = wall.get("x1", 0), wall.get("y1", 0)
        x2, y2 = wall.get("x2", 0), wall.get("y2", 0)
        w_type = wall.get("type", "interior")
        width = 7 if w_type == "exterior" else 4
        g_walls.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke_width=width))

    # Layer 4: Room number (001, 002...) above the name, like the template.
    g_rooms = dwg.add(dwg.g(id='rooms', font_family='Arial, sans-serif', font_weight='bold'))
    for room, number in _number_rooms(layout.get("rooms", [])):
        name = room.get("name", "Room")
        rx, ry = room.get("x", 0), room.get("y", 0)

        room_group = g_rooms.add(dwg.g())
        label_len = max(len(name), len(number)) * 9
        room_group.add(dwg.rect(insert=(rx - label_len/2 - 4, ry - 20), size=(label_len + 8, 36), rx=3, ry=3, fill='white', fill_opacity=0.85))
        room_group.add(dwg.text(number, insert=(rx, ry - 4), text_anchor='middle', font_size=13, fill='#2C3E50'))
        room_group.add(dwg.text(name, insert=(rx, ry + 12), text_anchor='middle', font_size=11, fill='#2C3E50'))

    # Layer 5: Asbestos Samples (Circular red pin drops with white outline + bold label)
    g_samples = dwg.add(dwg.g(id='samples', font_family='Arial, sans-serif', font_weight='bold'))
    for sample in layout.get("samples", []):
        sid = sample.get("id", "S001")
        sx, sy = sample.get("x", 0), sample.get("y", 0)
        
        g_samples.add(dwg.circle(center=(sx, sy), r=9, fill='#E74C3C', stroke='white', stroke_width=2))
        g_samples.add(dwg.text(sid, insert=(sx + 12, sy + 4), font_size=12, fill='#C0392B'))
        
    dwg.save()
    logger.info(f"Vector SVG plan successfully saved: {output_path}")


def generate_dummy_layout() -> Dict[str, Any]:
    """Generates a simple box placeholder layout if API or processing fails."""
    return {
        "walls": [
            {"x1": 100, "y1": 100, "x2": 900, "y2": 100, "type": "exterior"},
            {"x1": 900, "y1": 100, "x2": 900, "y2": 900, "type": "exterior"},
            {"x1": 900, "y1": 900, "x2": 100, "y2": 900, "type": "exterior"},
            {"x1": 100, "y1": 900, "x2": 100, "y2": 100, "type": "exterior"},
            {"x1": 500, "y1": 100, "x2": 500, "y2": 900, "type": "interior"}
        ],
        "doors": [
            {"hinge_x": 500, "hinge_y": 450, "open_x": 450, "open_y": 450, "closed_x": 500, "closed_y": 400, "label": "D1"}
        ],
        "windows": [
            {"x1": 250, "y1": 100, "x2": 350, "y2": 100},
            {"x1": 650, "y1": 100, "x2": 750, "y2": 100}
        ],
        "rooms": [
            {"name": "Office A", "x": 300, "y": 500},
            {"name": "Office B", "x": 700, "y": 500}
        ],
        "samples": [
            {"id": "S001", "x": 300, "y": 600}
        ]
    }


def delegate_drawing_to_windows_agent(image_path: Path, project_number: str, vsdx_path: Path) -> bool:
    """
    Delegate the drawing of the VSDX file to the Windows background agent.
    If DELEGATION_MODE is 'sharepoint', uses legacy SharePoint upload-and-poll.
    If 'http', calls the FastAPI drawing service directly over Tailscale,
    then downloads the generated VSDX from SharePoint in a single Graph request.
    """
    tenant = (os.environ.get("SP_TENANT_ID") or os.environ.get("ACORN_TENANT_ID") or os.environ.get("OUTLOOK_TENANT_ID") or "").strip()
    client_id = (os.environ.get("SP_CLIENT_ID") or os.environ.get("ACORN_CLIENT_ID") or os.environ.get("OUTLOOK_CLIENT_ID") or "").strip()
    secret = (os.environ.get("SP_CLIENT_SECRET") or os.environ.get("ACORN_CLIENT_SECRET") or os.environ.get("OUTLOOK_CLIENT_SECRET") or "").strip()
    drive_id = (os.environ.get("SP_DRIVE_ID") or "").strip()
    
    if not (tenant and client_id and secret and drive_id):
        logger.error("SharePoint credentials or SP_DRIVE_ID not found. Cannot delegate Visio drawing.")
        return False

    # Existing production infrastructure uses SharePoint Pending_Draw and the
    # Windows agent. HTTP delegation must be explicitly configured.
    mode = os.environ.get("DELEGATION_MODE", "sharepoint").strip().lower()
    
    # Acquire Graph Token (needed for both HTTP download and legacy SharePoint delegation)
    logger.info("Acquiring Microsoft Graph token for Visio agent delegation...")
    token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    token_data = {
        "client_id": client_id,
        "client_secret": secret,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    
    try:
        r = requests.post(token_url, data=token_data, timeout=15)
        r.raise_for_status()
        token = r.json().get("access_token")
    except Exception as e:
        logger.error(f"Failed to acquire MS Graph token: {e}")
        return False

    output_folder = os.environ.get(
        "SHAREPOINT_OUTPUT_FOLDER", "General/AI Automation/Generated_Plans"
    ).strip().strip("/")
    vsdx_filename = f"{project_number} AI Draft.vsdx"
    escaped_out_folder = urllib.parse.quote(output_folder)
    escaped_vsdx_filename = urllib.parse.quote(vsdx_filename)
    check_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{escaped_out_folder}/{escaped_vsdx_filename}"
    download_url = f"{check_url}/content"

    if mode == "http":
        draw_url = os.environ.get("DRAW_SERVICE_URL", "http://localhost:8088/draw")
        draw_key = os.environ.get("DRAW_SERVICE_API_KEY", "acorn-drawing-service-secret-key-2026")
        
        logger.info(f"Delegating drawing directly to Windows FastAPI Service at: {draw_url}")
        try:
            with open(image_path, "rb") as f:
                files = {"sketch": (image_path.name, f, f"image/{image_path.suffix.strip('.') or 'jpeg'}")}
                data = {"project_number": project_number, "api_key": draw_key}
                
                # FastAPI draw service acquires single worker lock, so set a 5-minute timeout
                r_draw = requests.post(draw_url, files=files, data=data, timeout=300)
                r_draw.raise_for_status()
                resp_json = r_draw.json()
                logger.info(f"FastAPI drawing succeeded in {resp_json.get('duration_seconds')}s. Downloading VSDX...")
                
            # Direct Graph GET to download final VSDX, with retries for replication/indexing delay
            logger.info("Downloading generated plan from SharePoint...")
            download_success = False
            for attempt in range(5):
                try:
                    r_dl = requests.get(download_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
                    if r_dl.status_code == 200:
                        vsdx_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(vsdx_path, "wb") as f:
                            f.write(r_dl.content)
                        logger.info(f"Successfully downloaded beautiful Visio plan to: {vsdx_path.name}")
                        download_success = True
                        break
                    elif r_dl.status_code == 404:
                        logger.warning(f"SharePoint reported 404 (not indexed yet) on attempt {attempt+1}/5. Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        r_dl.raise_for_status()
                except Exception as ex:
                    if attempt == 4:
                        raise ex
                    logger.warning(f"SharePoint download attempt {attempt+1}/5 failed: {ex}. Retrying...")
                    time.sleep(2)
            
            if download_success:
                return True
            else:
                logger.error("Failed to download generated VSDX from SharePoint after 5 attempts.")
                return False
            
        except Exception as e:
            logger.error(f"HTTP drawing delegation failed or timed out: {e}")
            return False

    else:
        # A previous run may have left a VSDX with the same project filename.
        # Remove it before submitting the new sketch so polling cannot mistake
        # stale output (including an old placeholder) for this run's result.
        logger.info(f"Removing any existing generated plan before submitting {project_number}...")
        try:
            r_existing = requests.delete(
                check_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if r_existing.status_code not in (204, 404):
                logger.error(
                    "Could not clear existing SharePoint output for %s: HTTP %s %s",
                    project_number,
                    r_existing.status_code,
                    r_existing.text,
                )
                return False
        except Exception as e:
            logger.error(f"Could not clear existing SharePoint output for {project_number}: {e}")
            return False

        # 1. Upload Sketch Image to 'General/AI Automation/Pending_Draw'
        pending_folder = "General/AI Automation/Pending_Draw"
        ext = image_path.suffix.lower()
        sketch_filename = f"{project_number}_sketch{ext}"
        escaped_folder = urllib.parse.quote(pending_folder)
        escaped_filename = urllib.parse.quote(sketch_filename)
        upload_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{escaped_folder}/{escaped_filename}:/content"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream"
        }
        
        logger.info(f"Uploading sketch image to SharePoint for Windows Agent (SharePoint Mode): {sketch_filename}")
        try:
            with open(image_path, "rb") as f:
                file_data = f.read()
            r = requests.put(upload_url, headers=headers, data=file_data, timeout=90)
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to upload sketch image to SharePoint: {e}")
            return False

        # 2. Poll 'General/AI Automation/Generated_Plans' for the VSDX file
        poll_headers = {"Authorization": f"Bearer {token}"}
        timeout_seconds = 180
        poll_interval = 3
        elapsed = 0
        
        logger.info(f"Waiting up to {timeout_seconds}s for Windows Visio Agent to generate floor plan...")
        while elapsed < timeout_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval
            try:
                r = requests.get(check_url, headers=poll_headers, timeout=10)
                if r.status_code == 200:
                    logger.info(f"Found generated VSDX in SharePoint! Downloading...")
                    r_dl = requests.get(download_url, headers=poll_headers, timeout=60)
                    r_dl.raise_for_status()
                    vsdx_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(vsdx_path, "wb") as f:
                        f.write(r_dl.content)
                    logger.info(f"Successfully downloaded beautiful Visio plan to: {vsdx_path.name}")
                    return True
            except Exception:
                pass
                
        logger.error(f"Timed out waiting for Windows Visio Agent to generate {vsdx_filename} after {timeout_seconds} seconds.")
        return False


def _write_quality_sidecar(layout: Dict[str, Any], layout_is_real: bool, vsdx_path: Path) -> None:
    """Emit a self-contained quality summary next to the VSDX so the orchestrator
    can gate auto-publish WITHOUT ground truth.

    STRUCTURAL sanity only — it proves the plan is not broken/empty/placeholder.
    It does NOT prove the plan matches the sketch (that needs the eval harness /
    a human). Catches: placeholder fallbacks, zero rooms, unlabeled rooms, no
    enclosing walls, out-of-bounds coordinates, unlabeled samples.
    """
    rooms = layout.get("rooms", []) or []
    labels = [str(r.get("name", "")).strip() for r in rooms]
    walls = layout.get("walls", []) or []
    samples = layout.get("samples", []) or []

    def _in_bounds(v: Any) -> bool:
        try:
            return 0.0 <= float(v) <= 1000.0
        except (TypeError, ValueError):
            return False

    coords: List[Any] = []
    for w in walls:
        coords += [w.get("x1"), w.get("y1"), w.get("x2"), w.get("y2")]
    for r in rooms:
        coords += [r.get("x"), r.get("y")]
    coords_in_bounds = bool(coords) and all(_in_bounds(c) for c in coords)

    summary = {
        "layout_is_real": bool(layout_is_real),
        "room_count": len(rooms),
        "rooms": labels,
        "blank_label_count": sum(1 for lbl in labels if not lbl),
        "wall_count": len(walls),
        "door_count": len(layout.get("doors", []) or []),
        "window_count": len(layout.get("windows", []) or []),
        "sample_count": len(samples),
        "blank_sample_count": sum(1 for s in samples if not str(s.get("id", "")).strip()),
        "coords_in_bounds": coords_in_bounds,
    }
    side = Path(vsdx_path).with_suffix(".quality.json")
    try:
        side.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Quality sidecar written: %s", side.name)
    except Exception as exc:
        logger.warning("Could not write quality sidecar: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Extract hand-drawn floor plan sketch into Visio-ready vector SVG and native VSDX.")
    parser.add_argument("project_number", help="Alpha Tracker project number (e.g. N-12345)")
    parser.add_argument("--image", help="Absolute path to the surveyor floor plan sketch image file", required=False)
    parser.add_argument("--output", help="Optional custom absolute output file path", required=False)
    
    args = parser.parse_args()
    project_number = args.project_number
    
    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        logger.info(f"Looking for default sketches in project workspace for {project_number}...")
        workspace_root = Path(__file__).resolve().parents[2]
        possible_sketches = [
            workspace_root / "downloads" / f"{project_number}_sketch.jpg",
            workspace_root / "downloads" / f"{project_number}_plan.jpg",
            workspace_root / "temp" / f"{project_number}_plan.jpg",
            workspace_root / "temp" / f"sketch_{project_number}.jpg",
        ]
        for path in possible_sketches:
            if path.is_file():
                image_path = path
                break
                
    if not image_path or not image_path.is_file():
        logger.error(f"Sketch image file not found. Please provide path via --image parameter.")
        layout = generate_dummy_layout()
        layout_is_real = False
    else:
        try:
            layout = extract_floor_plan_layout(image_path)
            layout_is_real = True
        except Exception as e:
            logger.error(f"AI Plan extraction failed: {e}. Generating fallback placeholder plan.")
            layout = generate_dummy_layout()
            layout_is_real = False

    if args.output:
        output_path = Path(args.output)
    else:
        workspace_root = Path(__file__).resolve().parents[2]
        output_dir = workspace_root / "src" / "output" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{project_number} AI Draft.svg"

    try:
        # Draw SVG first as requested
        draw_svg_plan(layout, output_path)
        print(f"SUCCESS: Generated vector floor plan layout at {output_path.absolute()}")
        
        # Generate VSDX directly and natively (no Aspose or Visio required)
        if output_path.suffix.lower() == ".svg":
            vsdx_path = output_path.with_suffix(".vsdx")
            allow_placeholder = os.environ.get(
                "ALLOW_PLACEHOLDER_PLAN", "false"
            ).strip().lower() in ("true", "1", "yes")

            # VSDX_RENDER_MODE controls how the .vsdx is produced:
            #   native        -> render the real layout on Linux with the
            #                     built-in exporter (no Windows, no Aspose, no
            #                     SharePoint round-trip). Default.
            #   windows_agent -> legacy: upload the sketch and poll the Windows
            #                     Visio agent via SharePoint (waits up to 180s).
            render_mode = os.environ.get("VSDX_RENDER_MODE", "native").strip().lower()

            if render_mode == "native":
                # Only the *dummy* layout is a placeholder; rendering a real
                # extracted layout natively is the genuine professional output.
                if not layout_is_real and not allow_placeholder:
                    raise RuntimeError(
                        "Floor plan extraction failed; refusing to render a "
                        "placeholder plan (set ALLOW_PLACEHOLDER_PLAN=true to override)"
                    )
                generate_vsdx_natively(layout, vsdx_path)
            else:
                # Legacy: delegate drawing to the Windows Visio agent first.
                if image_path and image_path.is_file():
                    success = delegate_drawing_to_windows_agent(image_path, project_number, vsdx_path)
                else:
                    success = False
                if not success:
                    if not allow_placeholder:
                        raise RuntimeError(
                            "Windows drawing delegation failed; refusing to create/upload a placeholder plan"
                        )
                    logger.warning(
                        "Drawing delegation failed; ALLOW_PLACEHOLDER_PLAN is enabled, "
                        "falling back to native VSDX exporter"
                    )
                    generate_vsdx_natively(layout, vsdx_path)

            # Emit a quality summary next to the VSDX for the publish gate.
            _write_quality_sidecar(layout, layout_is_real, vsdx_path)

    except Exception as e:
        logger.error(f"Failed to render vector SVG/VSDX file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
