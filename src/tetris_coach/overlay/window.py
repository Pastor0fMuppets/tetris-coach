"""Transparent, click-through, always-on-top overlay window (PySide6).

macOS-only at runtime. The module imports cleanly without PySide6 (the
window class then raises on construction), so headless test runs are safe.
"""

from __future__ import annotations

from ..capture.screen import Rect
from ..core.board import DEFAULT_HEIGHT, WIDTH
from ..solver.search import Move
from .renderer import HintStyle, draw_hint

try:  # pragma: no cover - depends on platform
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPainter, QPaintEvent
    from PySide6.QtWidgets import QWidget

    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False


if HAVE_QT:  # pragma: no cover - macOS only

    class OverlayWindow(QWidget):
        """Frameless transparent window positioned exactly over the board."""

        def __init__(
            self,
            board_rect: Rect,
            style: HintStyle | None = None,
            rows: int = DEFAULT_HEIGHT,
        ) -> None:
            super().__init__(
                None,
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.WindowTransparentForInput
                | Qt.WindowType.NoDropShadowWindowHint
                | Qt.WindowType.Tool,
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self._board_rect = board_rect
            self._style = style or HintStyle()
            self._rows = rows
            self._move: Move | None = None
            self.setGeometry(board_rect.left, board_rect.top, board_rect.width, board_rect.height)

        def set_hint(self, move: Move | None) -> None:
            """Update the displayed placement (None hides the hint)."""
            self._move = move
            self.update()

        def set_style(self, style: HintStyle) -> None:
            self._style = style
            self.update()

        def paintEvent(self, event: QPaintEvent) -> None:
            if self._move is None:
                return
            painter = QPainter(self)
            try:
                draw_hint(
                    painter,
                    self._move,
                    cell_width=self.width() / WIDTH,
                    cell_height=self.height() / self._rows,
                    style=self._style,
                )
            finally:
                painter.end()

else:

    class OverlayWindow:  # type: ignore[no-redef]
        """Placeholder that reports the missing GUI runtime."""

        def __init__(
            self,
            board_rect: Rect,
            style: HintStyle | None = None,
            rows: int = DEFAULT_HEIGHT,
        ) -> None:
            raise RuntimeError(
                "OverlayWindow requires PySide6, which is only installed on "
                "macOS (the overlay is a macOS feature)."
            )
