"""Full-screen drag-rectangle region picker (PySide6, macOS runtime only).

Dims the whole screen and lets the user drag a rectangle; returns the rect
in logical coordinates. Run twice: once for the board, once for the
next-piece box. A selection smaller than the caller's minimum (e.g. a
click without a drag) is rejected with an on-screen message and the user
can retry or press Esc to cancel. Imports cleanly without PySide6.
"""

from __future__ import annotations

from .capture.screen import Rect

# Minimum selection sizes in logical px, as (width, height). Below these
# a 10x20 board grid (or a preview box) is sub-pixel and vision cannot
# work; in particular a click-without-drag must never produce a 1x1 rect.
MIN_BOARD_SIZE = (40, 80)
MIN_PREVIEW_SIZE = (16, 16)

try:  # pragma: no cover - depends on platform
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent
    from PySide6.QtWidgets import QApplication, QWidget

    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False


if HAVE_QT:  # pragma: no cover - macOS only

    class _RegionPicker(QWidget):
        def __init__(self, prompt: str, min_size: tuple[int, int] = (1, 1)) -> None:
            super().__init__(
                None,
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
            )
            self._prompt = prompt
            self._min_size = min_size
            self._warning = ""
            self._origin: QPoint | None = None
            self._current: QPoint | None = None
            self.result: Rect | None = None
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setCursor(Qt.CursorShape.CrossCursor)
            screen = QApplication.primaryScreen()
            if screen is not None:
                self.setGeometry(screen.geometry())

        def paintEvent(self, event: QPaintEvent) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                self.rect().adjusted(0, 40, 0, 0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                self._prompt,
            )
            if self._warning:
                painter.setPen(QColor(255, 120, 120))
                painter.drawText(
                    self.rect().adjusted(0, 70, 0, 0),
                    int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                    self._warning,
                )
            if self._origin is not None and self._current is not None:
                selection = QRect(self._origin, self._current).normalized()
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                painter.fillRect(selection, QColor(0, 0, 0, 0))
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                painter.setPen(QColor(0, 229, 255))
                painter.drawRect(selection)
            painter.end()

        def mousePressEvent(self, event: QMouseEvent) -> None:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

        def mouseMoveEvent(self, event: QMouseEvent) -> None:
            if self._origin is not None:
                self._current = event.position().toPoint()
                self.update()

        def mouseReleaseEvent(self, event: QMouseEvent) -> None:
            if self._origin is None:
                return
            selection = QRect(self._origin, event.position().toPoint()).normalized()
            self._origin = None
            self._current = None
            min_w, min_h = self._min_size
            if selection.width() < min_w or selection.height() < min_h:
                # Too small (typically a stray click without a drag):
                # reject and let the user drag again or Esc-cancel.
                self._warning = (
                    f"Selection {selection.width()}x{selection.height()} is too "
                    f"small — drag at least {min_w}x{min_h} px, or press Esc."
                )
                self.update()
                return
            geo = self.geometry()
            self.result = Rect(
                left=geo.left() + selection.left(),
                top=geo.top() + selection.top(),
                width=selection.width(),
                height=selection.height(),
            )
            self.close()

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if event.key() == Qt.Key.Key_Escape:
                self.result = None
                self.close()


def select_region(prompt: str, min_size: tuple[int, int] = (1, 1)) -> Rect | None:
    """Show the picker and block until a rectangle is chosen (or Esc).

    Selections smaller than ``min_size`` (width, height) are rejected
    on-screen and the user can retry; only Esc returns ``None``.
    """
    if not HAVE_QT:  # pragma: no cover - headless
        raise RuntimeError("Region selection requires PySide6 (macOS).")
    app = QApplication.instance() or QApplication([])  # pragma: no cover
    picker = _RegionPicker(prompt, min_size)  # pragma: no cover
    picker.showFullScreen()  # pragma: no cover
    while picker.isVisible():  # pragma: no cover
        app.processEvents()
    return picker.result  # pragma: no cover
