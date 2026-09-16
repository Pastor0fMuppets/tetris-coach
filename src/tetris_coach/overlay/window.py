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
            second_style: HintStyle | None = None,
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
            # A Qt.Tool window is an NSPanel on macOS, and NSPanels hide
            # whenever their application is inactive — which is ALWAYS for
            # this overlay, since the user is focused on the game. This
            # attribute keeps the panel visible while the app is inactive.
            self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
            self._board_rect = board_rect
            self._style = style or HintStyle()
            # The second hint's style, or None for a session that wants one
            # target only (--no-next-hint). Dashed by default, which is the
            # whole distinction on screen: a stroke style reads at a glance
            # where a colour alone would not, over pieces the two hint
            # colours happen to sit near.
            self._second_style = second_style
            self._rows = rows
            self._move: Move | None = None
            self._second: Move | None = None
            self.setGeometry(board_rect.left, board_rect.top, board_rect.width, board_rect.height)

        def set_hint(self, move: Move | None, second: Move | None = None) -> None:
            """Update the displayed placements (None hides that hint).

            Both are set together because they are one answer: the second
            is where the next piece goes IF the first is followed, and a
            frame that moves one moves the other (see
            ``app.CoachEngine.second_hint``).
            """
            self._move = move
            self._second = second
            self.update()

        def set_style(self, style: HintStyle, second_style: HintStyle | None = None) -> None:
            self._style = style
            self._second_style = second_style
            self.update()

        def paintEvent(self, event: QPaintEvent) -> None:
            if self._move is None:
                return
            painter = QPainter(self)
            cell_width = self.width() / WIDTH
            cell_height = self.height() / self._rows
            try:
                # The second hint first, so that where the two are next to
                # each other the target the player acts on NOW is the one
                # drawn last. They never share a cell, so this is about
                # adjacent borders and nothing more.
                if self._second is not None and self._second_style is not None:
                    draw_hint(
                        painter,
                        self._second,
                        cell_width=cell_width,
                        cell_height=cell_height,
                        style=self._second_style,
                    )
                draw_hint(
                    painter,
                    self._move,
                    cell_width=cell_width,
                    cell_height=cell_height,
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
            second_style: HintStyle | None = None,
        ) -> None:
            raise RuntimeError(
                "OverlayWindow requires PySide6, which is only installed on "
                "macOS (the overlay is a macOS feature)."
            )
