"""Overlay window and hint renderer (PySide6 at runtime, guarded imports)."""

from .renderer import HintStyle, placement_cell_rects
from .window import OverlayWindow

__all__ = ["HintStyle", "OverlayWindow", "placement_cell_rects"]
