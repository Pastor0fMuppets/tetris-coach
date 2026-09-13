"""Full-screen drag-rectangle region picker (PySide6, macOS runtime only).

Dims the whole screen and lets the user drag a rectangle; returns the rect
in logical coordinates. Run twice: once for the board, once for the
next-piece box. Imports cleanly without PySide6.
"""

from __future__ import annotations

from .capture.screen import Rect

try:  # pragma: no cover - depends on platform
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent
    from PySide6.QtWidgets import QApplication, QWidget

    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False


if HAVE_QT:  # pragma: no cover - macOS only

    class _RegionPicker(QWidget):
        def __init__(self, prompt: str) -> None:
            super().__init__(
                None,
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
            )
            self._prompt = prompt
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
            if self._origin is not None and self._current is not None:
                selection = QRect(self._origin, self._current).normalized()
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                painter.fillRect(selection, QColor(0, 0, 0, 0))
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_SourceOver
                )
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
            geo = self.geometry()
            self.result = Rect(
                left=geo.left() + selection.left(),
                top=geo.top() + selection.top(),
                width=max(1, selection.width()),
                height=max(1, selection.height()),
            )
            self.close()

        def keyPressEvent(self, event: QKeyEvent) -> None:
            if event.key() == Qt.Key.Key_Escape:
                self.result = None
                self.close()


def select_region(prompt: str) -> Rect | None:
    """Show the picker and block until a rectangle is chosen (or Esc)."""
    if not HAVE_QT:  # pragma: no cover - headless
        raise RuntimeError("Region selection requires PySide6 (macOS).")
    app = QApplication.instance() or QApplication([])  # pragma: no cover
    picker = _RegionPicker(prompt)  # pragma: no cover
    picker.showFullScreen()  # pragma: no cover
    while picker.isVisible():  # pragma: no cover
        app.processEvents()
    return picker.result  # pragma: no cover
