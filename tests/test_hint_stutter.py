"""The flashing hint the user reported, replayed over the frames it is in.

    "sometimes the recommendation stutters ... it would quickly flash from
    one location to another and back and forth between the two"

``hint_stutter`` (51 frames) and ``stray_after_clear`` (41 frames) are that
session. Everything here is measured twice over: once at the tracker, which
is where the mechanism is, and once through :class:`CoachEngine` itself,
which is what the user was looking at.

THE MECHANISM, traced on hint_stutter 00355-00399. Two candidates alternate
and they share no cell:

    even frames   the real J entering, (0,3) (0,4) (0,5), cut by the top edge
    odd frames    one cell at (8, 8), mid-board, nothing near it

On the frames the stray wins, the J is not the piece in flight, so it falls
into the settled stack (22 -> 25 cells), ``falling.piece`` reads None and
the overlay blanks; ``_events`` sees a piece stop flying over a stack that
grew and calls it a LOCK. The next frame it flips back. Seventeen locks for
one J.

The stray cell is THE COACH'S OWN ROTATION BADGE. The cell holds no pixel
of board content -- it reads (252.0, 251.9, 250.9), the background, exactly
-- and the badge sits in its top margin, straddling the boundary. Covering
part of one edge and no patch, it read as our translucent FILL rather than
as our opaque drawing, and un-compositing a fill that is not there turns
bare ground into a colour 55 units off it: content, floating, one cell,
re-interned every frame and so always age 0, which outranks everything.

Three separate things had to be true for that to reach the screen, and all
three are now false:

* ``rotation_badge_rect`` hung the badge at the hint's BOUNDING-BOX corner,
  a cell the piece does not occupy in 6 of the 19 rotations -- which is why
  the J and the Z dominate the count and the I and the O never appear in it.
* ``own_paint_states`` told our fill from our opaque mark by the sampled
  patch alone, which cannot see a mark that hugs an edge.
* ``_rank`` accepted a sub-tetromino candidate anywhere on the board.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tetris_coach.race.engine import ShownFrame, replay
from tetris_coach.truth.windows import by_name
from tetris_coach.vision.colour_tracker import ColourTracker, Event, FrameReport

FIXTURES = Path(__file__).parent / "fixtures"
WINDOWS = ("hint_stutter", "stray_after_clear")

#: How far apart two sightings of the same target may be and still read as
#: a flicker rather than as the player moving the piece back. The measured
#: oscillation is period 2; twelve frames is 0.8 s at 15 fps.
NEARBY = 12


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


@lru_cache(maxsize=len(WINDOWS))
def read(window: str) -> tuple[tuple[str, FrameReport, int], ...]:
    """Every frame of a window through the tracker alone, in order.

    Frame number, report, and how many frames running the piece has been
    believed on its history rather than on its own sighting -- the loan in
    ``_rank``, which is what the last test here is about.
    """
    spec = by_name(window)
    track = ColourTracker(rows=spec.rows, cols=10, unobservable_cells=spec.geometry().unobservable)
    out = []
    for board in sorted((FIXTURES / window).glob("board_*.png")):
        crop = board.with_name(board.name.replace("board_", "next_"))
        report = track.update(load(board), load(crop) if crop.exists() else None)
        out.append((board.stem.split("_")[1], report, track._on_loan))
    return tuple(out)


@lru_cache(maxsize=len(WINDOWS))
def shown(window: str) -> tuple[ShownFrame, ...]:
    """...and through the engine the app runs, which is what was on screen."""
    return tuple(replay(FIXTURES, by_name(window), "colour"))


def oscillations(values: list[object]) -> list[int]:
    """Indices where a value comes back within NEARBY after changing away."""
    out = []
    for i, value in enumerate(values):
        if value is None:
            continue
        for j in range(i + 2, min(i + NEARBY + 1, len(values))):
            if values[j] == value and any(values[k] != value for k in range(i + 1, j)):
                out.append(i)
                break
    return out


@pytest.mark.parametrize("window", WINDOWS)
def test_the_hint_never_flashes_between_two_targets(window: str) -> None:
    """The user's sentence, as a measurement, on the frames it was said of.

    Was 31 on hint_stutter and 18 on stray_after_clear -- the hint landing
    somewhere, moving, and coming back inside 12 frames, over and over, for
    most of every piece's descent.
    """
    targets = [frame.hint for frame in shown(window)]
    assert oscillations(targets) == []


@pytest.mark.parametrize("window", WINDOWS)
def test_no_piece_blinks_out_and_back_while_it_is_in_flight(window: str) -> None:
    """X -> nothing -> X again. Was 16 (all J) and 11 (all Z).

    This is the same event one layer down: the frame the stray outranked
    the real piece is the frame the piece was not in flight at all, and the
    board handed to the solver had the piece in the stack.
    """
    names = [None if r.falling is None else r.falling.piece for _, r, _ in read(window)]
    flicker = [
        (frame, names[i])
        for i, (frame, _, _) in enumerate(read(window)[:-2])
        if names[i] is not None and names[i + 1] is None and names[i + 2] == names[i]
    ]
    assert flicker == []


@pytest.mark.parametrize("window", WINDOWS)
def test_the_settled_board_does_not_gain_and_lose_the_same_cells(window: str) -> None:
    """The stack was oscillating 22 -> 25 -> 22 with the hint: 34 and 23.

    Worth asserting apart from the hint, because this is the half the
    player cannot see: on every other frame the solver was handed a board
    with the piece they are still holding already built into it.
    """
    sizes = [sum(mask.bit_count() for mask in r.stack_rows) for _, r, _ in read(window)]
    flapping = [i for i in range(len(sizes) - 2) if sizes[i] == sizes[i + 2] != sizes[i + 1]]
    assert flapping == []


@pytest.mark.parametrize(("window", "locks"), [("hint_stutter", 1), ("stray_after_clear", 1)])
def test_a_piece_locks_once(window: str, locks: int) -> None:
    """Seventeen locks for one J, and twelve for one Z, is what it was.

    A lock is a structural event -- it re-solves, it moves the hint, it
    advances what the coach thinks the next piece is -- so a phantom one
    every other frame is not a cosmetic problem.
    """
    counted = sum(1 for _, r, _ in read(window) for e in r.events if e is Event.PIECE_LOCKED)
    assert counted == locks


def test_the_stray_cell_is_bare_board_and_the_badge_is_ours() -> None:
    """The evidence for what the phantom was, kept where it can be re-read.

    (8, 8) on hint_stutter 00357 holds no content: its sampled patch is the
    board's own background to a tenth of a uint8 unit. What is in the cell
    is 122 hint-coloured pixels across its top margin -- the rotation badge
    for a hint whose cells are (8,9) (9,9) (10,8) (10,9), hung at the
    bounding box's corner, which that J does not occupy.
    """
    from tetris_coach.vision.colour_palette import OURS, board_colours, own_paint_states

    board = load(FIXTURES / "hint_stutter" / "board_00357.png")
    colours = board_colours(board, 12, 10)
    background = colours[6, 0]  # a cell of plain board, far from anything
    assert float(np.linalg.norm(colours[8, 8] - background)) < 0.1

    states = own_paint_states(board, 12, 10)
    assert states[8, 8] == OURS, "ours and opaque: skipped, never un-composited"
    assert [(r, c) for r in range(12) for c in range(10) if states[r, c] == OURS] == [(8, 8)]


def test_the_hint_is_on_screen_for_every_frame_of_both_windows() -> None:
    """Nothing here is bought by showing less.

    A tracker that refused these frames would score zero on every measure
    above. Both windows hint on every frame the engine accepts, and the
    only frames it does not accept are stray_after_clear 00546-00553, the
    line-clear animation, where a completed row is still on screen.
    """
    expected = {"hint_stutter": [], "stray_after_clear": [f"005{n}" for n in range(46, 54)]}
    for window in WINDOWS:
        frames = shown(window)
        assert [f.frame for f in frames if f.accepted and f.hint is None] == []
        assert [f.frame for f in frames if not f.accepted] == expected[window]


def test_no_hint_is_lost_for_a_single_frame_anywhere_in_the_corpus() -> None:
    """Why ``app.py`` is not given patience for a blank frame, having asked.

    The obvious second fix was to let the coach ride out a one-frame loss
    of the piece the way it rides out a REFUSED frame
    (``CoachConfig.max_stale_frames``). Measured over all eight windows
    once the tracker was fixed, there is nothing left for it to ride out:
    not one accepted frame anywhere names no piece with the SAME piece
    named on the frames either side of it.

    Every blank stretch that remains is several frames long and has a
    cause the coach cannot see past:

        spawn_latency  00154-00157   the piece is entirely behind the
                                     NEXT panel
        ghost_session  00056         a lock gap: the piece either side is
                                     not the same piece
        absorbed_piece 00280-00293   the opening T, in a colour the
                                     palette has not learned and too
                                     clipped to spell itself

    Holding the last hint across those would be drawing a placement for a
    piece the player does not have, which is the failure this tool's own
    user reported and the reason ``_update_hint`` withdraws instead. So
    the patience would buy nothing here and cost the one thing that
    matters most; it is deliberately not added.
    """
    from tetris_coach.truth.windows import CONSECUTIVE

    for spec in CONSECUTIVE:
        frames = replay(FIXTURES, spec, "colour")
        names = [frame.piece for frame in frames]
        lost = [
            frames[i + 1].frame
            for i in range(len(frames) - 2)
            if names[i] is not None
            and names[i + 1] is None
            and names[i + 2] == names[i]
            and frames[i + 1].accepted
        ]
        assert lost == [], spec.name


@pytest.mark.parametrize("window", WINDOWS)
def test_no_frame_of_this_session_is_read_on_credit(window: str) -> None:
    """Every frame here stands on its own sighting, not on the one before.

    The piece already in flight may skip the admission test in ``_rank``,
    which is how a piece survives a cell flickering out for a frame --
    parked or being dragged. That exemption is a LOAN OF ONE FRAME,
    because belief handed on a frame at a time is belief that drifts: a
    one-cell stray stepping down column 7 was otherwise reported as the
    piece four rows below the last row anything would have admitted it at.

    What the bound costs the corpus is nothing, and this is where that is
    checked rather than asserted: no frame of either window -- and,
    measured the same way, no frame of any of the nine committed windows
    -- needs the loan at all.
    """
    assert [frame for frame, _, loan in read(window) if loan] == []
