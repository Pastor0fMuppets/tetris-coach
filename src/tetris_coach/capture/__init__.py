"""Screen capture (mss-backed at runtime; protocol-based for tests)."""

from .screen import ArraySource, FrameSource, Rect, ScreenCapture

__all__ = ["ArraySource", "FrameSource", "Rect", "ScreenCapture"]
