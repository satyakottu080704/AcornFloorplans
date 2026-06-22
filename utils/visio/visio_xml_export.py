"""
Acorn Atlas - Visio Export Module

Converts detected rooms to Microsoft Visio (.vsdx) format.

Usage:
    from utils.visio_export import create_visio_plan
    
    create_visio_plan(
        rooms=detected_rooms,
        output_path='floor_plan.vsdx',
        image_path='original_sketch.jpg'
    )
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import shutil
import tempfile
import uuid


@dataclass
class VisioShape:
    """Represents a shape in Visio."""
    id: int
    name: str
    x: float  # Center X in inches
    y: float  # Center Y in inches
    width: float  # Width in inches
    height: float  # Height in inches
    fill_color: str = "#FFFFFF"
    line_color: str = "#000000"
    text: str = ""
    shape_type: str = "rectangle"


class VisioExporter:
    """
    Export floor plan to Visio .vsdx format.
    
    VSDX is a ZIP archive containing XML files.
    """
    
    # Standard page size (A3 landscape in inches)
    PAGE_WIDTH = 16.54  # ~420mm
    PAGE_HEIGHT = 11.69  # ~297mm
    
    # Pixels to inches conversion (assuming 96 DPI)
    PX_TO_INCH = 1 / 96.0
    
    # Color mapping
    COLORS = {
        'room': '#E8F5E9',      # Light green
        'acm': '#FFCDD2',       # Light red
        'wall': '#9E9E9E',      # Gray
        'door': '#8D6E63',      # Brown
        'window': '#90CAF9',    # Light blue
        'stairs': '#CE93D8',    # Purple
        'text': '#FFF9C4',      # Light yellow
    }
    
    def __init__(self):
        self.shapes: List[VisioShape] = []
        self.shape_id_counter = 1
    
    def add_room(
        self,
        bbox: Tuple[int, int, int, int],
        room_type: str = 'room',
        label: str = None,
        image_size: Tuple[int, int] = (3000, 4000),
    ):
        """
        Add a room shape.
        
        Args:
            bbox: (x, y, width, height) in pixels
            room_type: 'room', 'acm', etc.
            label: Room label text
            image_size: Original image (width, height) for scaling
        """
        x, y, w, h = bbox
        img_w, img_h = image_size
        
        # Scale to page size
        scale_x = self.PAGE_WIDTH / img_w
        scale_y = self.PAGE_HEIGHT / img_h
        
        # Convert to inches (center coordinates)
        center_x = (x + w / 2) * scale_x
        center_y = self.PAGE_HEIGHT - (y + h / 2) * scale_y  # Flip Y axis
        width = w * scale_x
        height = h * scale_y
        
        # Get color
        fill_color = self.COLORS.get(room_type, '#FFFFFF')
        line_color = '#FF0000' if room_type == 'acm' else '#000000'
        
        shape = VisioShape(
            id=self.shape_id_counter,
            name=label or f"Room {self.shape_id_counter}",
            x=center_x,
            y=center_y,
            width=width,
            height=height,
            fill_color=fill_color,
            line_color=line_color,
            text=label or "",
        )
        
        self.shapes.append(shape)
        self.shape_id_counter += 1
        
        return shape
    
    def add_rooms_from_detector(
        self,
        rooms: List[Any],  # List of DetectedRoom
        image_size: Tuple[int, int],
    ):
        """Add all rooms from detector output."""
        for i, room in enumerate(rooms):
            # Support both has_acm (PlanGeneration) and room_type (ReportingAutomation)
            is_acm = getattr(room, 'has_acm', False) or getattr(room, 'room_type', '') == 'acm'
            room_type = 'acm' if is_acm else 'room'
            label = getattr(room, 'label', None) or f"Room {i + 1}"
            if is_acm and "[ACM]" not in label:
                label += " [ACM]"
            
            self.add_room(
                bbox=room.bbox,
                room_type=room_type,
                label=label,
                image_size=image_size,
            )
    
    def export(self, output_path: str) -> str:
        """
        Export to VSDX file.
        
        Args:
            output_path: Path for output .vsdx file
            
        Returns:
            Path to created file
        """
        if not output_path.endswith('.vsdx'):
            output_path += '.vsdx'
            
        # Find template
        template_path = os.path.join(os.path.dirname(__file__), 'template.vsdx')
        if not os.path.exists(template_path):
            # Fallback to parent workspace folder
            template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'template.vsdx'))
            
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Visio template not found at {template_path}")
            
        shapes_xml = ""
        for shape in self.shapes:
            shapes_xml += self._shape_to_xml(shape)
            
        page_content = f'''<?xml version="1.0" encoding="utf-8" ?>
<PageContents xmlns="http://schemas.microsoft.com/office/visio/2012/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xml:space="preserve">
  <Shapes>
{shapes_xml}
  </Shapes>
</PageContents>'''

        with zipfile.ZipFile(template_path, 'r') as zin:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if item.filename == 'visio/pages/page1.xml':
                        zout.writestr(item, page_content)
                    else:
                        zout.writestr(item, zin.read(item.filename))
                        
        print(f"[VisioExport] Created: {output_path}")
        return output_path
    
    def _shape_to_xml(self, shape: VisioShape) -> str:
        """Convert shape to Visio XML."""
        
        # Convert hex color to Visio format
        fill_rgb = self._hex_to_rgb(shape.fill_color)
        line_rgb = self._hex_to_rgb(shape.line_color)
        
        fill_color_str = f"#{fill_rgb}"
        line_color_str = f"#{line_rgb}"
        
        char_section = ""
        text_element = ""
        if shape.text:
            char_section = '''
      <Section N="Character">
        <Row IX="0">
          <Cell N="Color" V="#1E1E1E"/>
          <Cell N="Size" V="0.1666666666666667" U="PT"/>
          <Cell N="LangID" V="en-US"/>
        </Row>
      </Section>
      <Section N="Paragraph">
        <Row IX="0">
          <Cell N="HorzAlign" V="1"/>
        </Row>
      </Section>'''
            text_element = f'<Text><cp IX="0"/><pp IX="0"/>{shape.text}</Text>'
            
        return f'''    <Shape ID="{shape.id}" Type="Shape" LineStyle="3" FillStyle="3" TextStyle="3">
      <Cell N="PinX" V="{shape.x}"/>
      <Cell N="PinY" V="{shape.y}"/>
      <Cell N="Width" V="{shape.width}"/>
      <Cell N="Height" V="{shape.height}"/>
      <Cell N="LocPinX" V="{shape.width/2}" F="Width*0.5"/>
      <Cell N="LocPinY" V="{shape.height/2}" F="Height*0.5"/>
      <Cell N="Angle" V="0"/>
      <Cell N="FlipX" V="0"/>
      <Cell N="FlipY" V="0"/>
      <Cell N="ResizeMode" V="0"/>
      <Cell N="FillForegnd" V="{fill_color_str}"/>
      <Cell N="FillPattern" V="1"/>
      <Cell N="LineColor" V="{line_color_str}"/>
      <Cell N="LineWeight" V="0.013888888888888888" U="PT"/>
      <Cell N="LinePattern" V="1"/>{char_section}
      <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="0"/>
        <Cell N="NoLine" V="0"/>
        <Cell N="NoShow" V="0"/>
        <Cell N="NoSnap" V="0"/>
        <Cell N="NoQuickDrag" V="0"/>
        <Row T="MoveTo" IX="1"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="0" F="Height*0"/></Row>
        <Row T="LineTo" IX="2"><Cell N="X" V="{shape.width}" F="Width*1"/><Cell N="Y" V="0" F="Height*0"/></Row>
        <Row T="LineTo" IX="3"><Cell N="X" V="{shape.width}" F="Width*1"/><Cell N="Y" V="{shape.height}" F="Height*1"/></Row>
        <Row T="LineTo" IX="4"><Cell N="X" V="0" F="Width*0"/><Cell N="Y" V="{shape.height}" F="Height*1"/></Row>
        <Row T="LineTo" IX="5"><Cell N="X" V="0" F="Geometry1.X1"/><Cell N="Y" V="0" F="Geometry1.Y1"/></Row>
      </Section>
      {text_element}
    </Shape>
'''
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert #RRGGBB to RRGGBB."""
        return hex_color.replace('#', '')


# ============================================================================
# Convenience Function
# ============================================================================

def create_visio_plan(
    rooms: List[Any],
    output_path: str,
    image_size: Tuple[int, int] = (3000, 4000),
    title: str = "Floor Plan",
) -> str:
    """
    Create Visio floor plan from detected rooms.
    
    Args:
        rooms: List of DetectedRoom objects
        output_path: Output .vsdx file path
        image_size: Original image (width, height)
        title: Document title
        
    Returns:
        Path to created file
    """
    exporter = VisioExporter()
    exporter.add_rooms_from_detector(rooms, image_size)
    return exporter.export(output_path)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    # Test with dummy data
    @dataclass
    class DummyRoom:
        bbox: Tuple[int, int, int, int]
        has_acm: bool
        label: Optional[str] = None
    
    rooms = [
        DummyRoom(bbox=(100, 100, 300, 200), has_acm=False, label="Living Room"),
        DummyRoom(bbox=(450, 100, 200, 200), has_acm=True, label="Kitchen"),
        DummyRoom(bbox=(100, 350, 200, 150), has_acm=False, label="Bedroom"),
    ]
    
    output = create_visio_plan(
        rooms=rooms,
        output_path='test_floor_plan.vsdx',
        image_size=(800, 600),
    )
    print(f"Created: {output}")
