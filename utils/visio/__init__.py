"""
Visio Export Module

Creates .vsdx floor plan files from detected rooms.
Tries COM-based export first (requires Visio installed), falls back to XML.
"""

from .visio_xml_export import create_visio_plan as _xml_export, VisioExporter

_com_export = None
try:
    from .visio_com_export import create_visio_plan as _com_candidate
    _com_export = _com_candidate
except Exception:
    pass


def create_visio_plan(rooms, output_path, image_size, title=None):
    """Create Visio .vsdx file. Tries COM first, falls back to XML."""
    if _com_export is not None:
        try:
            return _com_export(rooms=rooms, output_path=output_path,
                               image_size=image_size, title=title)
        except Exception as e:
            print(f"[VISIO] COM failed ({e}), using XML export")
    return _xml_export(rooms=rooms, output_path=output_path,
                       image_size=image_size, title=title)


try:
    from .professional_visio import generate_visio_from_detected
except ImportError:
    generate_visio_from_detected = None

__all__ = ['create_visio_plan', 'generate_visio_from_detected', 'VisioExporter']
