"""The parallel colour-first tracker replayed over the committed sessions.

Same frames the shipped tracker's own regression tests use, read by
:class:`~tetris_coach.vision.colour_tracker.ColourTracker` instead. Nothing
here touches the shipped path; these are the measurements the design is
claimed on, pinned so they cannot quietly rot.

What the windows cost the shipped tracker, from their READMEs:

    spawn_latency       no hint for the first ~13 frames of every piece
    ghost_session       85 frames (~5.7 s) with no hint at all
    absorbed_piece      58 frames with no hint: the piece was committed as
                        stack, and its own descent then read as stack cells
                        vanishing, which reset the board and re-absorbed it
    pale_piece          90 frames with no falling piece: a pale piece read
                        as background
    ghost_beside_stack  the hint alternating between two targets, because
                        the coach's own paint read as a piece arriving

None of those are failure modes this representation has: the piece is named
by colour from its first visible cell, the stack is re-derived from every
frame rather than committed, and the coach's own paint is recognized by the
outline it draws rather than by the composite it happens to make.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tetris_coach.vision.colour_palette import EMPTY_DIST, board_colours
from tetris_coach.vision.colour_tracker import ColourTracker, Event, FrameReport

FIXTURES = Path(__file__).parent / "fixtures"
ROWS, COLS = 12, 10
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})


def load(path: Path) -> np.ndarray:
    """One captured frame as the capture pipeline would hand it over (BGR)."""
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


@cache
def replay(window: str) -> tuple[tuple[str, ...], tuple[FrameReport, ...]]:
    """Every frame of a window, in order, through one tracker."""
    tracker = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED)
    numbers: list[str] = []
    reports: list[FrameReport] = []
    for board in sorted((FIXTURES / window).glob("board_*.png")):
        crop = board.with_name(board.name.replace("board_", "next_"))
        numbers.append(board.stem.split("_")[1])
        reports.append(tracker.update(load(board), load(crop) if crop.exists() else None))
    return tuple(numbers), tuple(reports)


def named(reports: tuple[FrameReport, ...]) -> int:
    return sum(1 for r in reports if r.falling is not None and r.falling.piece is not None)


@pytest.mark.parametrize(
    ("window", "frames", "with_piece"),
    [
        ("spawn_latency", 161, 153),
        ("live_session", 96, 96),
        ("ghost_session", 95, 94),
        ("ghost_beside_stack", 48, 48),
        ("pale_piece", 61, 47),
        ("absorbed_piece", 81, 67),
    ],
)
def test_every_window_names_the_falling_piece(window: str, frames: int, with_piece: int) -> None:
    """How many frames of each window carry a named falling piece.

    The two windows short of their frame count are short for reasons that
    are not the tracker's: ``spawn_latency`` spends 7 frames inside a
    line-clear animation, ``pale_piece`` ends on 14 frames of a game-over
    panel, and ``absorbed_piece`` opens cold on a colour no NEXT box in the
    window ever shows (see the cold-start test below).
    """
    numbers, reports = replay(window)
    assert len(numbers) == frames
    assert named(reports) == with_piece


def test_a_piece_is_named_the_frame_it_appears() -> None:
    """Spawn latency, the measurement the whole design is for.

    The shipped tracker waits for enough of a piece to descend to name it
    by shape: measured median 13 frames (0.87 s) over this session, because
    1-3 cells at the top edge genuinely fit several tetrominoes. Colour has
    no such ambiguity, so the hint can go up on the first sighting.
    """
    _, reports = replay("spawn_latency")
    latency = []
    pending: int | None = None
    for index, report in enumerate(reports):
        if Event.PIECE_SPAWNED in report.events:
            pending = index
        if pending is not None and report.falling is not None and report.falling.piece:
            latency.append(index - pending)
            pending = None
    assert latency == [0] * 7


def test_the_entering_piece_is_named_from_two_cells() -> None:
    """00086: an O enters showing only its bottom half, and is named there."""
    numbers, reports = replay("spawn_latency")
    report = reports[numbers.index("00086")]
    assert report.falling is not None
    assert report.falling.piece == "O"
    assert report.falling.cells == frozenset({(0, 4), (0, 5)})
    assert not report.falling.complete


def test_a_piece_clipped_by_the_top_edge_is_named_after_it_rotates() -> None:
    """absorbed_piece 00353: an I stood up, 3 of its 4 cells on screen."""
    numbers, reports = replay("absorbed_piece")
    report = reports[numbers.index("00353")]
    assert report.falling is not None
    assert report.falling.piece == "I"
    assert report.falling.cells == frozenset({(0, 3), (1, 3), (2, 3)})


def test_a_colour_never_previewed_costs_a_cold_start_and_nothing_else() -> None:
    """absorbed_piece opens on a T the NEXT box never shows in the window.

    Its colour is unknown until the piece is fully on screen and its shape
    names it (frame 00294), which is the honest cost of learning a palette
    from the session rather than being given one. It is paid once per
    colour per session: the T is named on every frame after that.
    """
    numbers, reports = replay("absorbed_piece")
    unnamed = [n for n, r in zip(numbers, reports) if r.falling and r.falling.piece is None]
    assert unnamed == list(numbers[: numbers.index("00294")])
    assert reports[numbers.index("00294")].falling.piece == "T"  # type: ignore[union-attr]


def test_the_piece_the_shipped_tracker_absorbed_is_tracked_all_the_way_down() -> None:
    """absorbed_piece: 58 frames with no hint, because the T became stack.

    Here the T is in flight on every frame of its descent, and the settled
    board under it never gains a cell: 8 cells, the two corner stacks, from
    the first frame of the window to the T's lock.
    """
    numbers, reports = replay("absorbed_piece")
    descent = reports[: numbers.index("00346")]
    assert all(r.falling is not None and r.falling.floating for r in descent)
    assert {sum(row.bit_count() for row in r.stack_rows) for r in descent} == {8}
    assert all(Event.PIECE_LOCKED not in r.events for r in descent)


def test_the_pale_piece_is_seen_at_all() -> None:
    """pale_piece: 90 frames with no falling piece on the shipped reading.

    The T here scores 0.346 against the background where the shipped
    threshold sits at 0.35. Distance from the background is not what names
    it: its DIRECTION is, and a pale colour has just as much of one.
    """
    numbers, reports = replay("pale_piece")
    play = reports[: numbers.index("00687")]  # the window ends on a game-over panel
    assert all(r.falling is not None and r.falling.piece == "T" for r in play)


def test_our_own_paint_never_arrives_as_a_piece() -> None:
    """ghost_beside_stack: the hint alternating between two targets.

    What the window's README calls a ghost is this tool's own hint: a
    4-wide bar that appears on the floor at cols 4-7, jumps to cols 0-3 and
    blinks out, while the piece it belongs to is up at row 1. Not one of
    those cells is content here, so the floor keeps its two real cells all
    48 frames and nothing ever locks.
    """
    _, reports = replay("ghost_beside_stack")
    assert {r.stack_rows[11] for r in reports} == {0b1100000000}
    assert all(Event.PIECE_LOCKED not in r.events for r in reports)
    assert {r.falling.piece for r in reports if r.falling} == {"I"}


def test_the_ghost_windows_hold_no_ghost() -> None:
    """What these windows are NAMED for is this tool's own hint, not a ghost.

    The window holds both, and the repo has confused them in both
    directions. The thing that wanders and blinks -- the thing the READMEs
    describe and the shipped tracker chokes on -- is this tool's own hint,
    drawn where the SOLVER wants the piece. The proof is which rule saves
    the frames: turn the paint rule off and the same frames grow a phantom
    tetromino on the floor.

    The game also draws a real landing preview, in the falling piece's own
    columns, on 41 of this window's 48 frames. It is invisible to both
    trackers and so it saves nothing and costs nothing --
    ``test_the_game_draws_a_ghost_and_it_is_an_outline`` is where that is
    pinned.
    """
    blind = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED, paint=None)
    floors = set()
    for board in sorted((FIXTURES / "ghost_beside_stack").glob("board_*.png")):
        crop = board.with_name(board.name.replace("board_", "next_"))
        floors.add(blind.update(load(board), load(crop) if crop.exists() else None).stack_rows[11])
    assert floors == {0b1100000000, 0b1111110000, 0b1100001111}
    _, reports = replay("ghost_beside_stack")
    assert {r.stack_rows[11] for r in reports} == {0b1100000000}


def test_the_game_draws_a_ghost_and_it_is_an_outline() -> None:
    """ROAS Stacker draws a landing preview. SPEC.md says it does not.

    Read off ghost_beside_stack 00138 with no vision code in the loop, so
    this is a fact about the PNG and not about anything that reads it.
    The falling I is at row 0 columns 4-7; the game draws four rounded
    squares at row 11 columns 4-7 -- where it would land -- outlined in the
    I's own colour at about 26% alpha over the board's ground.

    Two things follow, and they point opposite ways.

    It is real, so SPEC.md (~line 283), ``vision/grid.py`` (~line 1092) and
    two fixture READMEs are wrong to say no committed fixture holds one.
    The oracle found it; every window in the corpus has one, on 482 of the
    542 frames.

    And it is an OUTLINE, covering under a tenth of the cell, with pure
    background in the middle. A centre-patch sampler cannot see it. So the
    colour-first tracker is exactly as blind to it as the shipped
    occupancy reader, its translucency rule never fired on one in its life,
    and "in colour, one threshold separates the ghost from the piece" is
    not a claim this corpus supports.
    """
    from collections import Counter

    image = np.asarray(
        Image.open(FIXTURES / "ghost_beside_stack" / "board_00138.png").convert("RGB"),
        dtype=np.uint8,
    )
    height, width, _ = image.shape
    ys = np.linspace(0, height, ROWS + 1).round().astype(int)
    xs = np.linspace(0, width, COLS + 1).round().astype(int)

    def cell(row: int, col: int) -> np.ndarray:
        return image[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]

    def shares(row: int, col: int) -> list[tuple[tuple[int, ...], float]]:
        patch = cell(row, col)
        total = patch.shape[0] * patch.shape[1]
        counts = Counter(map(tuple, patch.reshape(-1, 3)))
        return [(tuple(int(v) for v in value), n / total) for value, n in counts.most_common(3)]

    ground = np.array([251.0, 252.0, 252.0])
    piece = np.array([45.0, 46.0, 215.0])  # the falling I, RGB
    assert shares(0, 4)[0][0] == (45, 46, 215), "the I is at row 0 col 4"
    assert shares(6, 5)[0][0] == (251, 252, 252), "the board's ground"

    for col in range(4, 8):
        top = dict(shares(11, col))
        assert top[(251, 252, 252)] > 0.6, "most of a ghost cell is bare board"
        # The one colour in the cell that is a long way from the ground.
        outline = max(top, key=lambda value: float(np.linalg.norm(np.array(value) - ground)))
        alpha = (np.array(outline, dtype=float) - ground) / (piece - ground)
        assert 0.2 < float(alpha.min()) and float(alpha.max()) < 0.35, alpha
        assert 0.05 < top[outline] < 0.15, "an outline, not a fill"

    # And the centre of every one of those cells is bare board, which is
    # why no centre-patch sampler will ever report it.
    centres = board_colours(load(FIXTURES / "ghost_beside_stack" / "board_00138.png"), ROWS, COLS)
    background = np.array([252.0, 252.0, 251.0])  # the same ground, in BGR
    for col in range(4, 8):
        assert float(np.linalg.norm(centres[11, col] - background)) < EMPTY_DIST


def test_no_window_ever_loses_the_piece_for_long() -> None:
    """The freeze the user reported: 25 frames of a state that cannot move.

    The longest run of frames with no falling piece in any window is the
    line-clear animation in spawn_latency, which is four frames of a board
    that genuinely holds no piece.
    """
    worst = {}
    for window in ("spawn_latency", "live_session", "ghost_session", "ghost_beside_stack"):
        _, reports = replay(window)
        run = longest = 0
        for report in reports:
            run = run + 1 if report.falling is None else 0
            longest = max(longest, run)
        worst[window] = longest
    assert worst == {
        "spawn_latency": 4,
        "live_session": 0,
        "ghost_session": 0,
        "ghost_beside_stack": 0,
    }


def test_the_palette_a_session_learns_is_exactly_its_pieces() -> None:
    """Colour classes are the game's own colours, not a frame-by-frame zoo.

    A fade animation stays on its piece's ray (that is what a fade is), so
    the 161 frames of spawn_latency — a line clear and its flash included —
    leave three classes behind, one per piece the window shows.
    """
    tracker = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED)
    for board in sorted((FIXTURES / "spawn_latency").glob("board_*.png")):
        crop = board.with_name(board.name.replace("board_", "next_"))
        tracker.update(load(board), load(crop) if crop.exists() else None)
    assert sorted(k.piece or "?" for k in tracker.palette.classes) == ["I", "O", "T"]


def test_the_next_box_is_read_on_almost_every_frame() -> None:
    """The box names the piece AND its colour, which is what teaches the palette."""
    for window, expected in (("spawn_latency", 161), ("pale_piece", 61)):
        _, reports = replay(window)
        assert sum(1 for r in reports if r.next_piece is not None) == expected
