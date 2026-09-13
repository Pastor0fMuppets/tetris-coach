"""Screen capture (macOS at runtime; interface is platform-neutral).

``FrameSource`` is the protocol the rest of the app depends on, so tests can
inject frames from image files or arrays instead of a live screen. The
mss-backed ``ScreenCapture`` is only instantiated on a real display; importing
this module never touches mss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mss


@dataclass(frozen=True)
class Rect:
    """A screen rectangle in *logical* (points) coordinates."""

    left: int
    top: int
    width: int
    height: int


class FrameSource(Protocol):
    """Anything that can produce a BGR frame for a screen rectangle."""

    def grab(self, rect: Rect) -> NDArray[np.uint8]:
        """Return a BGR uint8 image of ``rect``.

        The returned image may be larger than ``rect`` (Retina scaling);
        callers must treat cell geometry as ratios of the frame size.
        """
        ...


class ScreenCapture:
    """mss-based capture of screen rectangles at native (Retina) scale.

    mss is imported lazily so the package imports cleanly on headless
    hosts; constructing this class without mss raises a clear error.
    """

    def __init__(self) -> None:
        try:
            import mss as mss_module
        except ImportError as exc:  # pragma: no cover - depends on platform
            raise RuntimeError(
                "ScreenCapture requires the 'mss' package and a display; "
                "install tetris-coach with its default dependencies on macOS."
            ) from exc
        self._sct: mss.mss = mss_module.mss()
        self._scale: float | None = None

    @property
    def scale(self) -> float | None:
        """Pixels per logical point (e.g. 2.0 on Retina); None before first grab."""
        return self._scale

    def grab(self, rect: Rect) -> NDArray[np.uint8]:
        shot = self._sct.grab(
            {
                "left": rect.left,
                "top": rect.top,
                "width": rect.width,
                "height": rect.height,
            }
        )
        frame = np.asarray(shot, dtype=np.uint8)[:, :, :3]  # BGRA -> BGR
        if rect.width > 0:
            self._scale = frame.shape[1] / rect.width
        return frame

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ArraySource:
    """FrameSource fed from in-memory frames (testing / development)."""

    def __init__(self, frames: list[NDArray[np.uint8]], loop: bool = False) -> None:
        if not frames:
            raise ValueError("ArraySource needs at least one frame")
        self._frames = frames
        self._loop = loop
        self._index = 0

    def grab(self, rect: Rect) -> NDArray[np.uint8]:
        frame = self._frames[self._index]
        if self._index + 1 < len(self._frames):
            self._index += 1
        elif self._loop:
            self._index = 0
        crop = frame[rect.top : rect.top + rect.height, rect.left : rect.left + rect.width]
        return np.ascontiguousarray(crop)


class ImageFileSource:
    """FrameSource that reads frames from image files on demand.

    Each ``grab`` consumes the next file (the last file repeats, or loops).
    Useful for feeding recorded screenshots through the vision stack.
    """

    def __init__(self, paths: list[str], loop: bool = False) -> None:
        if not paths:
            raise ValueError("ImageFileSource needs at least one path")
        self._paths = paths
        self._loop = loop
        self._index = 0

    def grab(self, rect: Rect) -> NDArray[np.uint8]:
        import cv2

        path = self._paths[self._index]
        if self._index + 1 < len(self._paths):
            self._index += 1
        elif self._loop:
            self._index = 0
        image = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        crop = image[rect.top : rect.top + rect.height, rect.left : rect.left + rect.width]
        return np.ascontiguousarray(crop)
