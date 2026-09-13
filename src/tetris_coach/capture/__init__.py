"""Screen capture (mss-backed at runtime; protocol-based for tests)."""

from .screen import ArraySource, FrameSource, ImageFileSource, Rect, ScreenCapture

__all__ = ["ArraySource", "FrameSource", "ImageFileSource", "Rect", "ScreenCapture"]
