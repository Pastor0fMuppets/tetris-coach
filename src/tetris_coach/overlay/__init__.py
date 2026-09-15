"""Overlay window and hint renderer (PySide6 at runtime, guarded imports)."""

from .renderer import HintStyle, placement_cell_rects, rotation_badge_rect
from .window import OverlayWindow

__all__ = ["HintStyle", "OverlayWindow", "placement_cell_rects", "rotation_badge_rect"]
