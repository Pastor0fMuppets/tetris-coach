"""THE INVISIBILITY PROPERTY, measured on every committed capture window.

The hints are drawn as outlines in the outer band of each cell, and both
readers name a cell from its central patch, inset ``CELL_MARGIN`` on every
side. So the claim the whole design rests on is not that a rule takes the
coach's own paint back out of the reading -- it is that the paint never
enters it:

    rendering both hints over a frame must not change what the reader
    reads.

That is a measurement, not an argument, and this is where it is taken. For
every frame of every committed window, both trackers read the raw frame
and read the same frame with two hints painted over it at the real cell
geometry, and the two readings must be IDENTICAL -- accepted, the named
falling piece, the derived stack, the next piece, the transitions. Any
difference is a failure of the design rather than a tolerance to widen.

Why it matters here in particular: this project has shipped three
user-visible bugs caused by the coach reading its own drawing, most
recently the stutter, where the rotation badge read as a falling piece and
the hint blanked every other frame at 15 fps
(``tests/fixtures/hint_stutter``, 51 frames of it). Those bugs were all
fixed by teaching the reader to recognize the paint. This design removes
the question instead, and the recognition stays in place behind it.

Two things this file deliberately does NOT do:

* It does not compare a reading against the truth oracle. The committed
  frames were captured while the overlay still drew a translucent FILL, so
  they already contain paint of their own; both runs see it, so it cancels,
  and what is measured is the difference the NEW paint makes.
* It does not assume the hints the engine happens to pick. The first test
  paints a schedule that walks every rotation of every piece across the
  board, so the property is measured over placements no session would
  produce as well as the ones it would; the last one then measures the real
  engine's own pair.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tetris_coach.app import CoachConfig, CoachEngine, hint_styles
from tetris_coach.core.board import Board
from tetris_coach.core.pieces import ROTATIONS
from tetris_coach.overlay.renderer import (
    DEFAULT_HINT_COLOR,
    DEFAULT_NEXT_HINT_COLOR,
    OUTLINE_DEPTH,
    PATCH_CLEARANCE,
    HintStyle,
    hint_paint_rects,
    paint_hint,
)
from tetris_coach.race.runners import next_crops
from tetris_coach.solver.search import Move
from tetris_coach.truth.windows import WINDOWS, WindowSpec, load_window
from tetris_coach.vision.colour_palette import CELL_MARGIN, CLEAN, cell_colors, own_paint_states
from tetris_coach.vision.grid import HINT_FILL_OPACITY, OwnPaint

FIXTURES = Path(__file__).parent / "fixtures"

#: The fill the committed windows were CAPTURED with: they were recorded
#: while the overlay still drew one, so every one of them already contains
#: paint of its own. ``race/engine.py`` tells the reader this number,
#: because the race scores a reading against the truth and a reader that
#: is not told what painted its input cannot get that right.
#:
#: Here it is deliberately NOT used. What is measured below is the
#: difference the NEW paint makes, both runs are handed the same old
#: paint, and the run is configured the way a SHIPPING session is
#: (``hint_fill_opacity`` 0, no fill drawn, nothing to un-composite). The
#: old paint is then read the same wrong way in both runs and cancels; the
#: new paint has to make no difference at all.
CAPTURED_FILL = HINT_FILL_OPACITY

#: What a shipping session paints with, which is what this file measures.
SHIPPED_FILL = 0.0

#: The overlay the captures already contain, as the reader would see it.
#: Used only to find those cells and step around them; see
#: :func:`_already_painted`.
CAPTURED_PAINT = OwnPaint.for_hint_color(DEFAULT_HINT_COLOR, opacity=CAPTURED_FILL)

TRACKERS = ("colour", "shape")

#: Every rotation of every piece, in a fixed order, so the schedule below
#: is deterministic and covers all 19 of them.
ALL_ROTATIONS = [(piece, rot) for piece, rots in ROTATIONS.items() for rot in rots]

#: The styles the overlay ships with: a solid outline for the piece in
#: play, a dashed one in a second colour for the piece after it.
CURRENT_STYLE = HintStyle(color=DEFAULT_HINT_COLOR, dashed=False)
NEXT_STYLE = HintStyle(color=DEFAULT_NEXT_HINT_COLOR, dashed=True)

Reading = tuple[bool, str | None, tuple[int, ...], str | None, tuple[str, ...]]
Pair = tuple[tuple[Move, HintStyle], ...]


def _move(index: int, rows: int, cols: int = 10) -> Move:
    """A placement from a deterministic schedule over all 19 rotations.

    Not a solver answer: the point is to cover placements a session would
    never choose as well as the ones it would, including every rotation at
    every column and against both edges.
    """
    piece, rotation = ALL_ROTATIONS[index % len(ALL_ROTATIONS)]
    col = index % max(1, cols - rotation.width + 1)
    row = index % max(1, rows - rotation.height + 1)
    return Move(
        piece=piece,
        rotation=rotation,
        col=col,
        row=row,
        score=0.0,
        lines_cleared=0,
        board=Board((0,) * rows),
    )


def _painted(
    frame: NDArray[np.uint8], rows: int, hints: tuple[tuple[Move, HintStyle], ...]
) -> NDArray[np.uint8]:
    """``frame`` with every hint painted at the frame's own cell geometry.

    The cell size is the one the READER derives from the image, which is
    also the one the overlay window derives from the board rectangle it is
    placed over. Retina scaling cancels: both are the region divided by the
    grid.
    """
    cell_h = frame.shape[0] / rows
    cell_w = frame.shape[1] / 10
    out = frame
    for move, style in hints:
        out = paint_hint(out, move, cell_w, cell_h, style)
    return out


def _read(
    spec: WindowSpec,
    tracker: str,
    hints: list[tuple[tuple[Move, HintStyle], ...]] | None,
) -> list[Reading]:
    """One window through one tracker, with ``hints`` painted over each frame."""
    config = CoachConfig(rows=spec.rows, tracker=tracker, hint_fill_opacity=SHIPPED_FILL)
    engine = CoachEngine(config, unobservable_cells=spec.geometry().unobservable)
    names, boards = load_window(FIXTURES, spec)
    crops = next_crops(FIXTURES, spec, names)
    out: list[Reading] = []
    for index, (board, crop) in enumerate(zip(boards, crops, strict=True)):
        frame = board if hints is None else _painted(board, spec.rows, hints[index])
        engine.process_frame(frame, crop)
        reading = engine.last_reading
        assert reading is not None
        out.append(
            (
                reading.accepted,
                reading.falling_piece,
                reading.stack_rows,
                reading.next_piece,
                tuple(event.name for event in reading.events),
            )
        )
    return out


@cache
def _raw(name: str, tracker: str) -> list[Reading]:
    return _read(_spec(name), tracker, None)


def _spec(name: str) -> WindowSpec:
    for spec in WINDOWS:
        if spec.name == name:
            return spec
    raise KeyError(name)


WINDOW_NAMES = tuple(spec.name for spec in WINDOWS)


# -- the geometry, before any pixels -----------------------------------


def _patch(index: int, size: float) -> tuple[float, float]:
    """The pixel span of one cell's sampled patch along one axis.

    The readers' own arithmetic (``vision.grid._cell_colors``,
    ``vision.colour_palette._inset``): the cell's edges rounded inward by
    ``CELL_MARGIN`` of the cell, to whole pixels.
    """
    low, high = index * size, (index + 1) * size
    return round(low + size * CELL_MARGIN), max(
        round(low + size * CELL_MARGIN) + 1, round(high - size * CELL_MARGIN)
    )


@pytest.mark.parametrize("style", [CURRENT_STYLE, NEXT_STYLE])
@pytest.mark.parametrize(
    "cell",
    [
        (48.0, 48.08),  # spawn_latency, the committed geometry
        (47.6, 47.83),  # ghost_session, the smallest committed cell
        (30.0, 25.0),  # a small non-square cell
        (20.0, 20.0),  # the synthetic frames in tests/colour_frames.py
        (120.0, 120.0),  # a large cell on a high-DPI capture
    ],
)
def test_no_painted_pixel_can_reach_a_sampled_patch(
    style: HintStyle, cell: tuple[float, float]
) -> None:
    """The property, as geometry: dilated by a pixel, the paint still misses.

    This is the part that does not depend on which frames are committed.
    Every rectangle either painter fills, for every rotation of every
    piece, grown a pixel on all four sides to stand in for an antialiased
    edge, must miss the sampled patch of EVERY cell on the board -- its
    own cells included, which is the case that matters, and its
    neighbours', which is the one the badge used to get wrong.
    """
    cell_w, cell_h = cell
    rows, cols = 12, 10
    patches_x = [_patch(c, cell_w) for c in range(cols)]
    patches_y = [_patch(r, cell_h) for r in range(rows)]
    for index in range(len(ALL_ROTATIONS)):
        move = _move(index, rows, cols)
        for x, y, w, h in hint_paint_rects(move, cell_w, cell_h, style):
            x0, y0, x1, y1 = x - 1, y - 1, x + w + 1, y + h + 1
            for px0, px1 in patches_x:
                for py0, py1 in patches_y:
                    overlaps = x0 < px1 and x1 > px0 and y0 < py1 and y1 > py0
                    assert not overlaps, (move.piece, move.rotation.index, (x, y, w, h))


def test_the_outline_uses_the_band_and_leaves_the_clearance() -> None:
    # The two constants are the design: paint the invisible band, keep a
    # slice of it back for rounding and antialiasing.
    assert OUTLINE_DEPTH + PATCH_CLEARANCE == pytest.approx(CELL_MARGIN)
    assert 0.0 < PATCH_CLEARANCE < OUTLINE_DEPTH


def test_the_shipped_styles_draw_no_fill() -> None:
    # The fill is the one thing a reader can see, so it is off in both.
    assert CURRENT_STYLE.fill_opacity == 0.0
    assert NEXT_STYLE.fill_opacity == 0.0
    assert HintStyle().fill_opacity == 0.0
    # ... and the two hints are told apart by stroke, not by fading one.
    assert CURRENT_STYLE.dashed is False
    assert NEXT_STYLE.dashed is True
    assert CURRENT_STYLE.color != NEXT_STYLE.color


# -- the measurement, over the committed windows -----------------------


def _already_painted(spec: WindowSpec, frame: NDArray[np.uint8]) -> set[tuple[int, int]]:
    """Cells of ``frame`` that already carry an EARLIER overlay's paint.

    The committed windows were captured with this tool running, so the
    coach's own hint of the day is in the pixels -- in the hint colour, at
    the fill it drew then. Painting a new hint on top of one of those cells
    OVERWRITES part of the old outline, which breaks a signature the reader
    is entitled to read (``own_paint_states`` asks whether the hint colour
    in a cell runs all the way round it), and the cell changes state.

    That is a property of replaying old captures, not of the design: a live
    session has exactly one overlay, and it does not paint over itself. So
    the schedule below steps around these cells rather than pretending the
    collision means something. There are never many -- one hint's worth per
    frame, four or five cells of a hundred and twenty.
    """
    states = own_paint_states(frame, spec.rows, 10, CAPTURED_PAINT)
    return {(r, c) for r, c in zip(*np.nonzero(states != CLEAN), strict=True)}


def _schedule(spec: WindowSpec, boards: list[NDArray[np.uint8]]) -> list[Pair]:
    """One (current, next) pair per frame, clear of the capture's own paint."""
    out: list[Pair] = []
    for index, frame in enumerate(boards):
        taken = _already_painted(spec, frame)
        picks: list[Move] = []
        candidate = 2 * index
        while len(picks) < 2:
            move = _move(candidate, spec.rows)
            candidate += 1
            if taken.isdisjoint(move.cells) and all(
                set(move.cells).isdisjoint(chosen.cells) for chosen in picks
            ):
                picks.append(move)
        out.append(((picks[0], CURRENT_STYLE), (picks[1], NEXT_STYLE)))
    return out


@pytest.mark.parametrize("name", WINDOW_NAMES)
def test_no_hint_changes_one_sampled_patch(name: str) -> None:
    """The property at its source, with nothing excluded: the patches are equal.

    Both readers name a cell from the mean colour of its central patch and
    from nothing else (``vision.grid._cell_colors``, which
    ``vision.colour_palette`` imports rather than reimplements). So if no
    patch changes, no occupancy, no piece name and no stack can. This runs
    over EVERY frame of every window and over the placements the schedule
    picks whether or not they land on the capture's own old paint -- the
    exclusion the reading test below needs does not apply here, because
    covering an old outline cannot reach a patch either.

    Exact equality, not a tolerance: a float32 mean over the same pixels.
    """
    spec = _spec(name)
    _names, boards = load_window(FIXTURES, spec)
    for index, frame in enumerate(boards):
        hints = (
            (_move(2 * index, spec.rows), CURRENT_STYLE),
            (_move(2 * index + 1, spec.rows), NEXT_STYLE),
        )
        painted = _painted(frame, spec.rows, hints)
        before = cell_colors(frame, spec.rows, 10, CELL_MARGIN)
        after = cell_colors(painted, spec.rows, 10, CELL_MARGIN)
        moved = np.argwhere(np.any(before != after, axis=-1))
        assert not len(moved), f"{name} frame {index}: patches changed at {moved.tolist()[:5]}"


@pytest.mark.parametrize("tracker", TRACKERS)
@pytest.mark.parametrize("name", WINDOW_NAMES)
def test_both_hints_change_nothing_the_reader_reads(name: str, tracker: str) -> None:
    """The property end to end, per window, per tracker.

    Two hints over every frame -- a solid one and a dashed one, from a
    schedule that walks all 19 rotations across the board -- and the
    reading afterwards must be the one from the bare frame, field for
    field: accepted, the named falling piece, the derived stack, the next
    piece, the transitions.
    """
    spec = _spec(name)
    _names, boards = load_window(FIXTURES, spec)
    raw = _raw(name, tracker)
    painted = _read(spec, tracker, _schedule(spec, boards))
    differences = [
        (index, before, after)
        for index, (before, after) in enumerate(zip(raw, painted, strict=True))
        if before != after
    ]
    assert not differences, (
        f"{name}/{tracker}: {len(differences)} frames changed: {differences[:3]}"
    )


@pytest.mark.parametrize("name", WINDOW_NAMES)
def test_the_engine_own_pair_of_hints_changes_no_patch_either(name: str) -> None:
    """The same measurement, on the placements the engine really picks.

    The schedule above covers placements no session would produce, which is
    the point of it; this covers the ones it does. Each frame is painted
    with the hints that were ON SCREEN when it was captured -- the previous
    frame's pair -- in the styles ``app.hint_styles`` builds from the
    shipped configuration, which is the real feedback loop.

    It measures the patch rather than the whole reading, because these
    captures already contain an earlier generation of this overlay and the
    engine's hints land on it constantly (same solver, same frames). That
    collision is not something a live session can produce and
    :func:`_already_painted` says why; what it cannot do either way is
    reach a patch, and that is what is asserted.
    """
    spec = _spec(name)
    config = CoachConfig(rows=spec.rows, hint_fill_opacity=SHIPPED_FILL)
    current_style, second_style = hint_styles(config)
    assert second_style is not None
    engine = CoachEngine(config, unobservable_cells=spec.geometry().unobservable)
    names, boards = load_window(FIXTURES, spec)
    crops = next_crops(FIXTURES, spec, names)
    on_screen: Pair = ()
    pairs = 0
    for index, (board, crop) in enumerate(zip(boards, crops, strict=True)):
        frame = _painted(board, spec.rows, on_screen) if on_screen else board
        before = cell_colors(board, spec.rows, 10, CELL_MARGIN)
        after = cell_colors(frame, spec.rows, 10, CELL_MARGIN)
        moved = np.argwhere(np.any(before != after, axis=-1))
        assert not len(moved), f"{name} frame {index}: patches changed at {moved.tolist()[:5]}"
        hint = engine.process_frame(frame, crop)
        second = engine.second_hint
        on_screen = tuple(
            (move, style)
            for move, style in ((hint, current_style), (second, second_style))
            if move is not None
        )
        pairs += len(on_screen) == 2
    # A guard against measuring nothing: on a genuine run of consecutive
    # captures the coach has a plan on most frames. ``pale_preview`` is a
    # SAMPLED window (see truth/windows.py) -- the tracker diffs frames
    # against each other, so it can barely follow it at all -- and one
    # frame is all it is asked for.
    floor = len(boards) // 3 if spec.consecutive else 1
    assert pairs >= floor, (
        f"{name}: only {pairs} of {len(boards)} frames drew both hints -- "
        "a measurement of one hint is not the measurement asked for"
    )


@pytest.mark.parametrize("tracker", TRACKERS)
def test_the_measurement_would_notice_a_fill(tracker: str) -> None:
    """The control: paint inside the patch and the reading DOES change.

    A property measured by comparing two runs is worth nothing if the
    comparison cannot fail. So here is the same comparison with the one
    thing the design removes put back -- a fill, which is paint inside the
    very patch the reader samples -- and it changes the reading on this
    window. ONE hint, at that, where the test above paints two.

    Which is also what "the fill re-opens the feedback path" means in
    practice: with a fill, a correct reading depends on a rule recognizing
    the paint and taking it back out again, and that rule is one more
    thing that has to be right about every theme, every background and
    every piece underneath. Without one, nothing has to be.
    """
    spec = _spec("live_session")
    _names, boards = load_window(FIXTURES, spec)
    filled = HintStyle(color=DEFAULT_HINT_COLOR, fill_opacity=0.5)
    schedule = [((_move(i, spec.rows), filled),) for i in range(len(boards))]
    assert _read(spec, tracker, schedule) != _raw("live_session", tracker)
