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

from tetris_coach.race.measures import load_truth
from tetris_coach.vision import colour_tracker
from tetris_coach.vision.colour_palette import EMPTY, EMPTY_DIST, board_colours, flat_cells
from tetris_coach.vision.colour_tracker import ColourTracker, Event, FrameReport, _grounded

FIXTURES = Path(__file__).parent / "fixtures"
ROWS, COLS = 12, 10
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})
WINDOWS = (
    "spawn_latency",
    "live_session",
    "ghost_session",
    "absorbed_piece",
    "pale_piece",
    "ghost_beside_stack",
)


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
        ("spawn_latency", 161, 150),
        ("live_session", 96, 96),
        ("ghost_session", 95, 94),
        ("ghost_beside_stack", 48, 48),
        ("pale_piece", 61, 47),
        ("absorbed_piece", 81, 67),
    ],
)
def test_every_window_names_the_falling_piece(window: str, frames: int, with_piece: int) -> None:
    """How many frames of each window carry a named falling piece.

    The windows short of their frame count are short for reasons that are
    not the tracker's: ``spawn_latency`` spends 7 frames inside a
    line-clear animation, which is refused rather than read (see
    ``test_the_clear_animation_is_refused_rather_than_named``), and 4 more
    with the whole piece parked behind the NEXT panel; ``pale_piece`` ends
    on 14 frames of a web page; and ``absorbed_piece`` opens cold on a
    colour no NEXT box in the window ever shows (see the cold-start test
    below).
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

    Six spawns, not the eight this once counted: two of those were the
    line-clear flash being read as a piece arriving, and the frames they
    were raised on are now refused outright.
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
    assert latency == [0] * 6


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
    play = reports[: numbers.index("00687")]  # the window ends on a web page, not a board
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


def test_the_gate_refuses_the_web_page_and_the_clear_and_nothing_else() -> None:
    """Where the board is not on screen, and where the palette stops growing.

    ``pale_piece`` 00687-00700 is not a game-over panel, which is what this
    file used to call it: it is a web page over the whole capture region.
    The prototype used to read it and report a stack of
    0b11111111 / 0b1000 / 0b11111100 off the page's own layout, and intern
    six junk colour classes into a palette that never shrinks -- so one
    occluded second was a permanent corruption of the only cross-frame
    memory this design has.

    The test that separates them is the design's own premise rather than
    anything about Tetris: are the cells drawn as flat rectangles. Every
    real board frame in all six windows scores 0.89 or better; these
    fourteen score 0.51.

    The one other stretch the gate refuses is spawn_latency's line-clear
    animation, on the premise that a completed row is still on screen. The
    precision of that premise is the point of pinning it here: over all
    six windows it fires on those seven frames and on nothing else.
    """
    refused = {}
    classes = {}
    for window in WINDOWS:
        tracker = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED)
        names = []
        for board in sorted((FIXTURES / window).glob("board_*.png")):
            crop = board.with_name(board.name.replace("board_", "next_"))
            report = tracker.update(load(board), load(crop) if crop.exists() else None)
            if not report.board_visible:
                names.append(board.stem.split("_")[1])
        refused[window] = names
        classes[window] = len(tracker.palette.classes)

    assert refused["pale_piece"] == [f"00{n}" for n in range(687, 701)]
    assert refused["spawn_latency"] == [f"00{n}" for n in range(127, 134)]
    assert all(
        not names
        for window, names in refused.items()
        if window not in ("pale_piece", "spawn_latency")
    )
    # Two colours in pale_piece: the pale periwinkle T and the blue I. It
    # was eight once the page had been read.
    assert classes["pale_piece"] == 2
    assert max(classes.values()) == 3  # spawn_latency, which shows a third piece


def test_the_clear_animation_is_refused_rather_than_named() -> None:
    """A line clear is a gap in the evidence, and is now read as one.

    spawn_latency 00127-00133 is ROAS Stacker's row-clear flash. The row
    animates through a pale tint that happens to lie along the blue I's ray
    from the background, and -- worse -- every cell it recolours counts as
    having JUST changed, which is exactly the evidence this tracker names a
    piece from. So it used to read cells that are not a piece and name
    them: an "I" at (10,4)-(10,5) on 00127, drawn on screen for four of the
    seven frames while the player held an O, with a PIECE_SPAWNED and a
    PIECE_LOCKED to go with it.

    Neither of the first two premises can see this: only 10 to 15 cells
    fail the flatness test (against 4 on an ordinary frame, nowhere near a
    refusal), and nothing is resting on nothing. What sees it is the row
    itself. A completed row is never a resting state of a Tetris board, so
    one still on screen means the clear is playing -- the oracle's own rule
    for abstaining here, and over the whole corpus it fires on exactly
    these seven frames and no others (see
    ``test_the_gate_refuses_the_web_page_and_the_clear_and_nothing_else``).

    Refusing per CELL rather than per frame would also have caught it, and
    must not be done: on this very window (0,4) and (0,5) are non-flat on
    00134 and 00100 too, and there they are the real falling O, clipped by
    the top edge. The rule that would silence the flash would delete every
    entering piece -- which is the one thing this design is for.
    """
    numbers, reports = replay("spawn_latency")
    flash = [reports[numbers.index(f"00{n}")] for n in range(127, 134)]
    assert all(not r.board_visible for r in flash)
    # Two premises cover the animation between them, and which one speaks
    # says what the row is doing: while it is still drawn (recoloured) the
    # row reads as complete, and once it has faded most of the way to the
    # background its cells are neither ground nor content.
    assert [r.refused_because for r in flash] == ["a completed row is still on screen"] * 3 + [
        "cells are neither the board's ground nor content"
    ] * 4
    assert all(r.falling is None for r in flash)
    assert all(r.events == () for r in flash), "a flash is not a spawn and not a lock"
    # The stack the coach keeps looking at across the gap is the last real
    # one, and the clear is reported on the frame the board comes back.
    before, after = reports[numbers.index("00126")], reports[numbers.index("00134")]
    assert all(r.stack_rows == before.stack_rows for r in flash)
    assert Event.LINES_CLEARED in after.events
    assert after.falling is not None and after.falling.piece == "O"

    # Why per-cell refusal is not available as a fix: on an ordinary frame
    # the non-flat cells ARE the entering piece.
    def rough(frame: str) -> set[tuple[int, int]]:
        flat = flat_cells(load(FIXTURES / "spawn_latency" / f"board_{frame}.png"), ROWS, COLS)
        return {(r, c) for r in range(ROWS) for c in range(COLS) if not flat[r, c]}

    assert rough("00100") == {(0, 4), (0, 5), (0, 8), (0, 9)}
    assert len(rough("00130")) > 12
    truth = load_truth()["spawn_latency"]
    at_100 = next(f for f in truth.frames if f.frame == "00100")
    assert {(0, 4), (0, 5)} <= at_100.cells, "the non-flat cells of 00100 ARE the falling piece"


def test_no_window_ever_loses_the_piece_for_long() -> None:
    """The freeze the user reported: 25 frames of a state that cannot move.

    The longest run of frames with no falling piece in any window is the
    line-clear animation in spawn_latency: seven frames the tracker refuses
    outright, because a clear is playing over them and nothing read off
    them is the board.
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
        "spawn_latency": 7,
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


def airborne(labels: np.ndarray) -> int:
    """Content cells no chain of content joins to the floor."""
    return int(np.count_nonzero((labels != EMPTY) & ~_grounded(labels)))


def test_the_airborne_budget_costs_the_corpus_nothing() -> None:
    """One tetromino over the void, measured against every frame there is.

    The gate's second premise (:meth:`ColourTracker._blind`) refuses a
    frame showing more than a tetromino of content resting on nothing. A
    premise is only worth having if the real evidence obeys it, so this
    measures the whole corpus: 521 accepted board frames, and the largest
    airborne reading on any of them is 4 cells -- one piece in flight,
    never once more. The line-clear flash, the hint paint, the post-clear
    stack hanging over its own holes, a piece clipped by the top edge: all
    under the budget, because :func:`_grounded` follows sideways links and
    only a piece in flight is cut off from the floor. (The seven frames of
    the line-clear flash are refused before this premise is reached, on
    the completed row still being on screen, so they are not among the
    521.)
    """
    worst: dict[str, int] = {}
    frames = 0
    for window in WINDOWS:
        tracker = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED)
        for board in sorted((FIXTURES / window).glob("board_*.png")):
            crop = board.with_name(board.name.replace("board_", "next_"))
            report = tracker.update(load(board), load(crop) if crop.exists() else None)
            if not report.board_visible:
                continue
            frames += 1
            assert tracker._labels is not None
            worst[window] = max(worst.get(window, 0), airborne(tracker._labels))
    assert frames == 521
    assert worst == dict.fromkeys(WINDOWS, 4)


def test_the_two_premises_refuse_the_web_page_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one real cover in the corpus fails both tests, not just one.

    The flatness premise is what refuses ``pale_piece`` 00687-00700 in
    ordinary running, and the airborne premise was added for the cover it
    CANNOT see (a flat panel). They are only alternatives on paper unless
    the second is checked against real pixels, so here the flatness test is
    stubbed away and the page is fed to the tracker anyway: 58 of its cells
    rest on nothing, IN ONE COMPONENT, against a per-component budget of 4
    and a corpus maximum of 4. The frame is still refused, and the palette
    still ends the window with the window's own two piece colours rather
    than the eight it used to carry away.
    """
    monkeypatch.setattr(colour_tracker, "board_readable", lambda *a, **k: True)

    # Measure, don't refuse: both budgets lifted out of the way.
    monkeypatch.setattr(colour_tracker, "PIECE_CELLS", 10**6)
    monkeypatch.setattr(colour_tracker, "AIRBORNE_CELLS", 10**6)
    tracker = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED)
    measured = {}
    for board in sorted((FIXTURES / "pale_piece").glob("board_*.png")):
        crop = board.with_name(board.name.replace("board_", "next_"))
        tracker.update(load(board), load(crop) if crop.exists() else None)
        assert tracker._labels is not None
        measured[board.stem.split("_")[1]] = airborne(tracker._labels)
    page = [f"00{n}" for n in range(687, 701)]
    assert [measured[n] for n in page] == [58] * 14
    assert max(v for n, v in measured.items() if n not in page) == 4

    monkeypatch.setattr(colour_tracker, "PIECE_CELLS", 4)
    monkeypatch.setattr(colour_tracker, "AIRBORNE_CELLS", 8)
    tracker = ColourTracker(rows=ROWS, cols=COLS, unobservable_cells=COVERED)
    refused = []
    for board in sorted((FIXTURES / "pale_piece").glob("board_*.png")):
        crop = board.with_name(board.name.replace("board_", "next_"))
        report = tracker.update(load(board), load(crop) if crop.exists() else None)
        if not report.board_visible:
            refused.append(board.stem.split("_")[1])
    assert refused == page
    assert len(tracker.palette.classes) == 2
