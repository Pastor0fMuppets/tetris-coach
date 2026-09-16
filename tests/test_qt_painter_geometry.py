"""The bridge between the painter that is MEASURED and the one that SHIPS.

``tests/test_hint_invisibility.py`` takes the property the whole design
rests on -- that nothing this tool draws can reach the central patch a
reader samples -- through :func:`~tetris_coach.overlay.renderer.paint_hint`,
the numpy painter, because Qt cannot be driven in an ordinary headless
run. What the overlay actually paints with is
:func:`~tetris_coach.overlay.renderer.draw_hint`. Both fill the same
:func:`~tetris_coach.overlay.renderer.hint_paint_rects` geometry, so what
could drift between them is not WHICH rectangles there are but how each
one is filled -- and a measurement taken on one painter says nothing
about the other until that is checked.

So it is checked here, offscreen through a real ``QPainter``: over every
rotation of every piece, at several cell sizes, in both shipped styles,
the Qt painter must touch NO pixel the measured painter does not.

A SUBSET, not an equality, and deliberately so. The badge is an ellipse
with a digit clipped into it, while the measured painter fills its
bounding rectangle, so the corners of that box are painted by the
measured one alone. Over-stating the paint is the safe direction: every
reader of this geometry is asking whether paint can reach somewhere it
must not.

PySide6 is a macOS-only dependency of ``overlay/``, so this file skips
where it is absent -- which is what keeps a headless run green.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tetris_coach.core.board import Board
from tetris_coach.core.pieces import ROTATIONS
from tetris_coach.overlay.renderer import (
    DEFAULT_HINT_COLOR,
    DEFAULT_NEXT_HINT_COLOR,
    HintStyle,
    draw_hint,
    paint_hint,
)
from tetris_coach.solver.search import Move

ROWS, COLS = 12, 10

#: The styles the overlay ships with (``app.hint_styles``): a solid
#: outline for the piece in play, a dashed one for the piece after it.
CURRENT_STYLE = HintStyle(color=DEFAULT_HINT_COLOR, dashed=False)
NEXT_STYLE = HintStyle(color=DEFAULT_NEXT_HINT_COLOR, dashed=True)

#: A ground colour neither hint is drawn in, so that "this pixel changed"
#: and "this pixel was painted" are the same statement.
GROUND = (17, 23, 29)

ALL_ROTATIONS = [(piece, rot) for piece, rots in ROTATIONS.items() for rot in rots]


@pytest.fixture(scope="module")
def qt() -> Iterator[Any]:
    """A Qt application on the offscreen platform, or a skip.

    The overlay's own import guard is what decides: no PySide6, no test.
    The platform is forced offscreen so that a run with a display and one
    without paint the same pixels.
    """
    gui = pytest.importorskip("PySide6.QtGui", reason="PySide6 is the macOS overlay runtime")
    before = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = gui.QGuiApplication.instance() or gui.QGuiApplication([])
    try:
        yield gui
    finally:
        del app
        if before is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = before


def _move(index: int) -> Move:
    """A placement from the same deterministic walk over all 19 rotations."""
    piece, rotation = ALL_ROTATIONS[index]
    return Move(
        piece=piece,
        rotation=rotation,
        col=index % max(1, COLS - rotation.width + 1),
        row=index % max(1, ROWS - rotation.height + 1),
        score=0.0,
        lines_cleared=0,
        board=Board((0,) * ROWS),
    )


def _qt_painted(
    gui: Any, move: Move, cell_w: float, cell_h: float, style: HintStyle
) -> NDArray[np.bool_]:
    """Pixels ``draw_hint`` touches, as a mask, over a transparent canvas."""
    width, height = round(cell_w * COLS), round(cell_h * ROWS)
    image = gui.QImage(width, height, gui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(gui.QColor(0, 0, 0, 0))
    painter = gui.QPainter(image)
    try:
        draw_hint(painter, move, cell_width=cell_w, cell_height=cell_h, style=style)
    finally:
        painter.end()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    rows = buffer.reshape(height, image.bytesPerLine() // 4, 4)
    return np.asarray(rows[:, :width, 3] > 0)  # any pixel the painter tinted at all


def _numpy_painted(move: Move, cell_w: float, cell_h: float, style: HintStyle) -> NDArray[np.bool_]:
    """Pixels ``paint_hint`` changes, as a mask, over a flat ground."""
    width, height = round(cell_w * COLS), round(cell_h * ROWS)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = GROUND
    return np.asarray(np.any(paint_hint(frame, move, cell_w, cell_h, style) != frame, axis=-1))


@pytest.mark.parametrize("style", [CURRENT_STYLE, NEXT_STYLE], ids=["solid", "dashed"])
@pytest.mark.parametrize(
    "cell",
    [
        (48.0, 48.08),  # spawn_latency, the committed geometry
        (47.6, 47.83),  # ghost_session, the smallest committed cell
        (20.0, 20.0),  # the synthetic frames in tests/colour_frames.py
        (120.0, 120.0),  # a large cell on a high-DPI capture
    ],
)
def test_the_qt_painter_touches_no_pixel_the_measured_one_does_not(
    qt: Any, style: HintStyle, cell: tuple[float, float]
) -> None:
    """Every rotation, both styles: ``draw_hint`` paints inside ``paint_hint``.

    This is what carries the invisibility measurement across to the
    painter the user actually sees. The badge's digit is included, and it
    is the reason ``draw_hint`` clips its text to the badge rectangle: a
    glyph that overflowed would be paint outside the band, which is the
    one thing the geometry may not produce.
    """
    cell_w, cell_h = cell
    for index in range(len(ALL_ROTATIONS)):
        move = _move(index)
        qt_mask = _qt_painted(qt, move, cell_w, cell_h, style)
        measured = _numpy_painted(move, cell_w, cell_h, style)
        stray = np.argwhere(qt_mask & ~measured)
        assert not len(stray), (
            f"{move.piece} r{move.rotation.index} at {cell}: Qt painted "
            f"{len(stray)} pixels the measurement does not, first at "
            f"{stray[0].tolist()}"
        )


def test_the_comparison_can_fail(qt: Any) -> None:
    """The control: a hint Qt draws that the measurement is not told about.

    A subset check is worth nothing if it cannot come out false, and the
    thing it is defending against is exactly this -- the Qt painter
    putting paint somewhere the measured geometry says there is none.
    """
    move = _move(0)
    filled = HintStyle(color=DEFAULT_HINT_COLOR, fill_opacity=0.5)
    qt_mask = _qt_painted(qt, move, 48.0, 48.0, filled)
    measured = _numpy_painted(move, 48.0, 48.0, CURRENT_STYLE)
    assert np.any(qt_mask & ~measured)
