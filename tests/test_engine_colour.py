"""The engine end to end on the colour tracker: what reaches the screen.

``test_engine_light.py`` and ``test_headless.py`` drive the same class on
the shipped tracker. This file drives the DEFAULT one, on synthetic frames
drawn from the colours measured off the real fixtures, and it exists to
pin the app-level behaviours rather than the tracker's own reading (that is
``test_colour_tracker*.py``): the hint is held while its piece is in
flight, it comes down when vision can no longer justify it, this tool's own
paint never becomes a piece, and nothing is ever planned into a cell the
capture cannot see.

Every one of those was written in answer to something the user reported,
so a change of tracker has to carry them over, not inherit them by luck.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris_coach.app import CoachConfig, CoachEngine
from tetris_coach.vision.readers import ColourVision, ShapeVision

from .colour_frames import BLUE_I, CELL, COLS, GREEN_O, PALE_T, ROWS, preview, render

O_CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]
I_CELLS = [(0, 0), (0, 1), (0, 2), (0, 3)]
T_CELLS = [(0, 1), (1, 0), (1, 1), (1, 2)]

# The board cells ROAS Stacker's NEXT panel floats over, as
# app.compute_overlap_mask derives them for the committed windows.
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})


def engine(**overrides: object) -> CoachEngine:
    """A coach reading with the default tracker, on a 12-row board."""
    cells = overrides.pop("unobservable_cells", frozenset())
    config = CoachConfig(rows=ROWS, **overrides)  # type: ignore[arg-type]
    return CoachEngine(config, unobservable_cells=cells)  # type: ignore[arg-type]


def test_the_default_engine_reads_with_the_colour_tracker() -> None:
    assert CoachConfig().tracker == "colour"
    assert isinstance(engine().vision, ColourVision)
    assert isinstance(engine(tracker="shape").vision, ShapeVision)


def test_an_unknown_tracker_is_refused_where_it_is_named() -> None:
    with pytest.raises(ValueError, match="unknown tracker"):
        engine(tracker="wobble")


def test_a_hint_appears_on_the_first_frame_the_piece_does() -> None:
    # Two cells of an O at the top edge: shape cannot name that (an O, an
    # S, a Z, a J and an L all fit), and colour does not have to.
    coach = engine()
    box = preview(O_CELLS, GREEN_O)
    assert coach.process_frame(render({}), box) is None
    hint = coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    assert hint is not None
    assert hint.piece == "O"


def test_the_target_is_held_while_its_piece_is_in_flight() -> None:
    # The user's report was "the overlay moves around a lot for the same
    # piece". The piece is dragged across the board (this game has no
    # gravity) and the target must not follow it about.
    coach = engine()
    box = preview(O_CELLS, GREEN_O)
    coach.process_frame(render({}), box)
    hint = coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    assert hint is not None
    for col in (3, 2, 1, 0, 1, 2):
        moved = coach.process_frame(render({(0, col): GREEN_O, (0, col + 1): GREEN_O}), box)
        assert moved is not None
        assert moved.cells == hint.cells


def test_a_hint_vision_cannot_justify_comes_down() -> None:
    # Frames that are not a board at all: the colour reading rests on cells
    # being drawn flat, and noise is refused by that premise. A brief patch
    # is held (a flash must not make the overlay flicker); a sustained one
    # takes the hint off the screen.
    coach = engine(max_stale_frames=8)
    box = preview(O_CELLS, GREEN_O)
    coach.process_frame(render({}), box)
    assert coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box) is not None
    noise = np.random.default_rng(0).integers(
        0, 256, size=(ROWS * CELL, COLS * CELL, 3), dtype=np.uint8
    )
    for _ in range(8):
        assert coach.process_frame(noise, box) is not None
    for _ in range(2):
        coach.process_frame(noise, box)
    assert coach.current_hint is None
    # ... and it comes back the moment a frame can justify one again.
    back = coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    assert back is not None and back.piece == "O"


def test_the_user_is_told_why_the_coach_went_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """A refusal is invisible: the overlay is simply empty, once per frame.

    Every wholesale vision failure looks the same from outside -- a board
    rectangle that is not over the board, a selection that takes in the
    game's own panels, a game that has been closed -- so the frame's own
    reason is the only thing that tells them apart, and it is free to
    print. Once per silence, at the same point the hint comes down, and
    reset by the first frame that reads.
    """
    coach = engine(max_stale_frames=4, poll_rate=15.0)
    box = preview(O_CELLS, GREEN_O)
    coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    noise = np.random.default_rng(0).integers(
        0, 256, size=(ROWS * CELL, COLS * CELL, 3), dtype=np.uint8
    )
    for _ in range(12):
        coach.process_frame(noise, box)
    said = capsys.readouterr().err
    assert said.count("no frame has been readable") == 1, "once per silence, not per frame"
    assert "not drawn as flat cells" in said, "and it says which premise refused"

    # A readable frame arms it again, so a second silence is reported too.
    coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    for _ in range(12):
        coach.process_frame(noise, box)
    assert capsys.readouterr().err.count("no frame has been readable") == 1


def test_the_shape_reader_says_why_it_refused_too(capsys: pytest.CaptureFixture[str]) -> None:
    # The message is the engine's, so it has to be true of either reader:
    # the shipped one refuses on its confidence gate and says so. The gate
    # is lifted above what any frame can score rather than a frame found
    # that it happens to dislike -- what is under test is the reporting.
    coach = CoachEngine(
        CoachConfig(rows=ROWS, tracker="shape", min_confidence=0.9, max_stale_frames=4)
    )
    noise = np.random.default_rng(0).integers(
        0, 256, size=(ROWS * CELL, COLS * CELL, 3), dtype=np.uint8
    )
    for _ in range(12):
        coach.process_frame(noise, None)
    assert "the confidence gate scored this frame" in capsys.readouterr().err


def test_the_coach_never_reads_its_own_paint_as_the_board() -> None:
    # The coach captures its own overlay. Drawing the hint it just chose
    # over the board must change nothing it believes: same piece, same
    # target, no new piece arriving out of its own paint.
    from .colour_frames import paint_hint

    coach = engine()
    box = preview(I_CELLS, BLUE_I)
    coach.process_frame(render({}), box)
    falling = {(0, 4): BLUE_I, (0, 5): BLUE_I, (0, 6): BLUE_I, (0, 7): BLUE_I}
    hint = coach.process_frame(render(falling), box)
    assert hint is not None and hint.piece == "I"
    painted = paint_hint(render(falling), list(hint.cells))
    for _ in range(4):
        again = coach.process_frame(painted, box)
        assert again is not None
        assert (again.piece, again.cells) == (hint.piece, hint.cells)
    assert coach.last_reading is not None
    assert coach.last_reading.falling_piece == "I"
    assert coach.last_reading.stack_rows == (0,) * ROWS


def test_nothing_is_planned_into_a_cell_the_capture_cannot_see() -> None:
    # The NEXT panel floats over four board cells. They are unknown, not
    # empty: a hint drawn there would be drawn UNDER the panel hiding it.
    coach = engine(unobservable_cells=COVERED)
    box = preview(T_CELLS, PALE_T)
    coach.process_frame(render({}), box)
    for _ in range(3):
        hint = coach.process_frame(render({(0, 4): PALE_T, (1, 3): PALE_T, (1, 4): PALE_T}), box)
    assert hint is not None
    assert not set(hint.cells) & COVERED


def test_a_covered_cell_resting_on_the_stack_is_handed_over_as_filled() -> None:
    # ... and the way that is achieved must not invent holes elsewhere:
    # only covered cells with support under them are filled in.
    coach = engine(unobservable_cells=COVERED)
    resting = (0,) * (ROWS - 1) + (0b1100000000,)  # columns 8-9 on the floor
    board = coach._solver_board(resting)
    assert board.rows[ROWS - 1] == 0b1100000000
    assert board.rows[0] == 0  # nothing is filled in with air beneath it
    coach_without = engine()
    assert coach_without._solver_board(resting).rows == resting


def test_the_debug_view_draws_the_piece_it_is_advising_about(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # This reader separates the piece from the stack as it reads, so a
    # debug view that printed only the stack would leave the thing the
    # coach is advising about out of the picture.
    coach = engine(debug=True)
    box = preview(O_CELLS, GREEN_O)
    coach.process_frame(render({}), box)
    capsys.readouterr()
    coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    out = capsys.readouterr().out
    assert "[vision] frame 2" in out
    assert "falling O (2 cells, floating)" in out
    assert "next O" in out
    assert "PIECE_SPAWNED" in out
    assert "....oo...." in out  # the piece itself, where it is
    # A quiet session says nothing until something changes.
    coach.process_frame(render({(0, 4): GREEN_O, (0, 5): GREEN_O}), box)
    assert capsys.readouterr().out == ""


def test_a_lock_moves_the_hint_on_to_the_next_piece() -> None:
    coach = engine()
    box = preview(I_CELLS, BLUE_I)
    coach.process_frame(render({}), box)
    floor = ROWS - 1
    falling = {(0, 4): GREEN_O, (0, 5): GREEN_O, (1, 4): GREEN_O, (1, 5): GREEN_O}
    green_box = preview(O_CELLS, GREEN_O)
    coach.process_frame(render(falling), green_box)
    hint = coach.process_frame(render(falling), green_box)
    assert hint is not None and hint.piece == "O"
    # The O is dropped and settles; an I arrives at the top edge.
    settled = {
        (floor - 1, 4): GREEN_O,
        (floor - 1, 5): GREEN_O,
        (floor, 4): GREEN_O,
        (floor, 5): GREEN_O,
    }
    for _ in range(4):
        coach.process_frame(render(settled), box)
    arriving = dict(settled)
    arriving.update({(0, c): BLUE_I for c in range(3, 7)})
    for _ in range(2):
        moved = coach.process_frame(render(arriving), box)
    assert moved is not None
    assert moved.piece == "I"
    assert coach.last_reading is not None
    assert coach.last_reading.stack_rows[floor] == 0b110000
