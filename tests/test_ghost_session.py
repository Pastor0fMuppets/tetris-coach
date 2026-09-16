"""Replay of 95 CONSECUTIVE frames from the session that produced the ghost bug.

``tests/fixtures/ghost_session/`` holds ticks 00056-00150 of a real ROAS
Stacker session (10 x 12, light theme) with its NEXT preview floating over
board cells (0,8), (0,9), (1,8) and (1,9). See the fixtures' README for the
geometry and the frame-by-frame narrative; the frames are RGB PNGs and the
capture pipeline hands the engine BGR, so each is flipped on load. This is a
DIFFERENT window and different rects from ``tests/fixtures/live_session`` —
the two are not interchangeable.

Consecutive is the whole point: the tracker explains every frame as a diff
against the previous committed one, so only a contiguous run replays what
the live session actually did.

The defect: a translucent third level, scoring between the background and a
real piece (measured against the remembered background: background
0.00-0.02, layer 0.32, solid piece 0.57-0.82). Otsu has two classes to
give, so it went to whichever side the rest of the board pushed it.

It was diagnosed as the GHOST — the landing preview under the falling
piece — and it is not one. Every cell of it carries ``HintStyle.color``
``#00e5ff``: it is THIS TOOL'S placement hint, drawn over the game and
still on screen when the next frame was captured, and the "little round
badge" on frame 150 is ``_draw_rotation_badge``'s. See the fixtures'
README and ``vision.grid._own_paint_layer``. The frame-by-frame story
below is unaffected — what changed is which rule names those cells, and
the numbers are identical either way.

Measured on these frames BEFORE the layer was named at all:

    UNEXPLAINED 52 of 95   BOARD_RESET 9   PIECE_LOCKED 1 (a phantom)
    frames with a hint 8   longest run with no hint 85 (00064-00148, ~5.7 s)

Frames 59-60 read the layer — resting on the floor at cols 0-1 while the
real O hung at the top of the board — as occupied, and two identical frames
is exactly the tracker's debounce, so it committed a LOCK. On frame 61 it
moved to cols 2-3 (the solver changing its mind, not the player dragging),
and "locked" cells moved, which a locked piece cannot do. Four
unexplainable frames later the reset debounce
fired, adopted the observed board, and absorbed the real falling O into the
stack; from there every frame read QUIET with no falling piece and no hint —
the user's "on a few pieces it didn't run at all".

AFTER (same frames, same gate):

    UNEXPLAINED 0          BOARD_RESET 0   PIECE_LOCKED 3
    frames with a hint 93  longest run with no hint 2 (the bootstrap)

The O is tracked from frame 57 to 105 as the player drags it down and left,
hinted at cols 0-1 on the floor the whole way, and it is where the O
actually locks.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.solver.search import Move
from tetris_coach.truth.windows import CAPTURED_FILL_OPACITY
from tetris_coach.vision import grid as vision_grid
from tetris_coach.vision.pieces_vision import FrameKind
from tetris_coach.vision.state import GameEvent

from .layers import NamedLayer, layers_named, pen_stroked_cells

FIXTURES = Path(__file__).parent / "fixtures" / "ghost_session"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=238, top=317, width=476, height=574)
SESSION_NEXT = Rect(left=620, top=314, width=97, height=96)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# Where the landing preview stands on the frames that carry one, read off
# the images. Frames 59-60 put it at cols 0-1 and 61-64 at cols 2-3 — the
# drag that made the phantom lock "move". Nothing is on the board below
# row 9 on any of these frames: the real O is still at the top.
GHOST_CELLS: dict[str, frozenset[tuple[int, int]]] = {
    **{n: frozenset({(10, 0), (10, 1), (11, 0), (11, 1)}) for n in ("00059", "00060")},
    **{
        n: frozenset({(10, 2), (10, 3), (11, 2), (11, 3)})
        for n in ("00061", "00062", "00063", "00064")
    },
}

# The one frame whose widget carries a ROTATION BADGE: the hint is a
# vertical I down col 9 at rows 8-11 and the badge floats opaque over
# (7, 9). The badge is this tool's paint too, but it is drawn at full
# opacity with a black digit through it, so it matches no composite and
# nothing can name it; the rule refuses the whole widget rather than
# leave it behind as an added cell nothing explains.
BADGE_FRAMES = frozenset({"00150"})

# The board as it really stands on the last frame of the window: the O the
# coach tracked all session locked at cols 0-1, an I across cols 2-5 on the
# floor, and the next I locked on top of it at cols 0-5.
FINAL_STACK = (0,) * 10 + (0b111111, 0b111111)


def load(name: str) -> np.ndarray:
    """One captured frame as the capture pipeline would hand it over (BGR)."""
    return np.asarray(Image.open(FIXTURES / name))[:, :, ::-1]


def frame_numbers() -> list[str]:
    return [p.stem.split("_")[1] for p in sorted(FIXTURES.glob("board_*.png"))]


class Tick:
    """What one replayed frame did."""

    def __init__(
        self,
        number: str,
        confidence: float,
        kind: FrameKind | None,
        events: list[GameEvent],
        observed: np.ndarray,
        stack_rows: tuple[int, ...],
        falling: str | None,
        next_piece: str | None,
        hint: Move | None,
        layer: NamedLayer | None,
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.kind = kind
        self.events = events
        self.observed = observed  # what the tracker was handed, or None-ish
        self.stack_rows = stack_rows
        self.falling = falling
        self.next_piece = next_piece
        self.hint = hint
        self.layer = layer  # the third level this frame had removed, if any

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE

    def committed(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return bool(self.stack_rows[row] >> col & 1)


@lru_cache(maxsize=1)
def replay() -> tuple[Tick, ...]:
    """Feed the whole window through CoachEngine, wired as app.run wires it."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(
        CoachConfig(rows=ROWS, tracker="shape", hint_fill_opacity=CAPTURED_FILL_OPACITY),
        unobservable_cells=covered,
    )
    update = engine.tracker.update
    seen: list[GameEvent] = []
    handed: list[np.ndarray] = []

    def spy(occupancy, next_piece):  # type: ignore[no-untyped-def]
        handed.append(np.array(occupancy, copy=True))
        events = update(occupancy, next_piece)
        seen.extend(events)
        return events

    engine.tracker.update = spy  # type: ignore[method-assign]
    ticks: list[Tick] = []
    for number in frame_numbers():
        board = load(f"board_{number}.png")
        preview = FIXTURES / f"next_{number}.png"
        seen.clear()
        handed.clear()
        # The classifier is stateful, so this must read the same frame the
        # engine is about to digest — classify() is a pure function of the
        # image plus the memory, and the engine re-runs it identically.
        with layers_named() as named:
            _occupancy, confidence = engine.classifier.classify(board)
            hint = engine.process_frame(board, load(preview.name) if preview.exists() else None)
        committed = engine.tracker.committed
        ticks.append(
            Tick(
                number=number,
                confidence=confidence,
                kind=engine.tracker.last_kind,
                events=list(seen),
                observed=handed[0] if handed else np.zeros((ROWS, 10), dtype=bool),
                stack_rows=committed.stack_rows,
                falling=committed.falling_piece,
                next_piece=committed.next_piece,
                hint=hint,
                layer=named[-1] if named else None,
            )
        )
    return tuple(ticks)


def tick(number: str) -> Tick:
    return next(t for t in replay() if t.number == number)


def test_the_session_geometry_is_the_readme_geometry() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED
    assert len(frame_numbers()) == 95
    assert frame_numbers()[0] == "00056"
    assert frame_numbers()[-1] == "00150"


def test_no_frame_is_unexplained_and_the_board_never_resets() -> None:
    # The headline: the cascade does not start. Nothing in the window is
    # unexplainable, so the reset debounce never has four frames to count.
    kinds = Counter(t.kind for t in replay() if t.kind is not None)
    assert kinds[FrameKind.UNEXPLAINED] == 0, "an unexplainable frame is back"
    events = Counter(e for t in replay() for e in t.events)
    assert events[GameEvent.BOARD_RESET] == 0, "spurious reset"
    assert events[GameEvent.PIECE_LOCKED] == 3
    assert events[GameEvent.PIECE_SPAWNED] == 4


def test_the_ghost_never_locks_a_piece_that_did_not_land() -> None:
    # The phantom itself. Frames 59-60 are the two identical frames the
    # tracker's debounce used to turn into a lock; 61-64 are the drag that
    # made the "locked" cells move. All six are an ordinary falling frame
    # now, and nothing is committed anywhere on the board.
    ghosted = [tick(n) for n in GHOST_CELLS]
    assert all(t.kind is FrameKind.FALLING for t in ghosted), [t.kind for t in ghosted]
    assert all(not t.events for t in ghosted)
    assert all(all(row == 0 for row in t.stack_rows) for t in ghosted)


def test_the_ghost_is_never_board_content() -> None:
    # The rule's whole job, checked where it matters: the cells the preview
    # covers reach the tracker EMPTY, and they are not in the committed
    # stack. Cols 0-1 of rows 10-11 really do fill later — that is where the
    # O locks on frame 106 — so this is asserted per frame, against the
    # preview's position on that frame, not as a blanket ban on the cells.
    for number, cells in GHOST_CELLS.items():
        t = tick(number)
        assert t.accepted, f"{number}: rejected at {t.confidence:.2f}"
        for cell in cells:
            assert not t.observed[cell], f"{number}: ghost {cell} handed over as content"
            assert not t.committed(cell), f"{number}: ghost {cell} committed"


def test_the_committed_stack_only_ever_grows_and_ends_at_the_board() -> None:
    # No cell is ever committed and then lost (this window clears no lines),
    # so an absorbed ghost cannot hide behind a later commit, and the last
    # frame's stack is the board in the image.
    previous: tuple[int, ...] | None = None
    for t in replay():
        assert all(row == 0 for row in t.stack_rows[:10]), f"{t.number}: {t.stack_rows}"
        if previous is not None:
            assert all(old & ~new == 0 for old, new in zip(previous, t.stack_rows, strict=True)), (
                f"{t.number}: committed cells vanished"
            )
        previous = t.stack_rows
    assert replay()[-1].stack_rows == FINAL_STACK


def test_the_real_falling_o_is_tracked_and_hinted_across_the_whole_window() -> None:
    # What the user lost: 85 frames with nothing on screen while an O was
    # plainly falling. The O is named on frame 58 and stays named until it
    # locks on 105, and every one of those frames carries its hint.
    window = [t for t in replay() if "00058" <= t.number <= "00105"]
    assert len(window) == 48
    assert all(t.falling == "O" for t in window), "the falling O was lost"
    assert all(t.hint is not None and t.hint.piece == "O" for t in window)
    # The preview is readable throughout, so the hints are full 2-ply.
    assert all(t.next_piece == "I" for t in window)


def test_the_hint_the_coach_gave_is_where_the_o_actually_lands() -> None:
    # The end-to-end check that the board handed to the solver is the real
    # one: the coach says cols 0-1 on the floor for the whole descent, and
    # the O locks at cols 0-1 on the floor.
    window = [t for t in replay() if "00058" <= t.number <= "00105"]
    assert {(t.hint.row, t.hint.col) for t in window if t.hint is not None} == {(10, 0)}
    assert GameEvent.PIECE_LOCKED in tick("00106").events
    assert tick("00106").stack_rows == (0,) * 10 + (0b11, 0b11)


def test_a_hint_is_on_screen_on_all_but_the_bootstrap_frames() -> None:
    # 8 of 95 before, 93 of 95 now, and the only gap is the two frames
    # before anything has been committed at all.
    ticks = replay()
    assert sum(t.hint is not None for t in ticks) == 93
    assert [t.number for t in ticks if t.hint is None] == ["00056", "00057"]


def test_the_preview_corner_is_never_read_as_board_content() -> None:
    # Unchanged by this fix, and load-bearing for it: the cells the NEXT
    # box floats over are unknown, not empty, and never enter the stack.
    for t in replay():
        for cell in COVERED:
            assert not t.committed(cell)


def test_the_one_refused_frame_is_the_widget_with_the_badge_on_it() -> None:
    # The deliberate limit, pinned. On frame 150 the preview (a vertical I
    # down col 9) carries the game's little round "1" badge one cell above
    # it. The badge is furniture too, but it is a single cell at a full
    # piece color, so nothing can name it — and naming the preview under it
    # would leave the badge as an unexplainable added cell. The rule
    # refuses the whole widget instead, which leaves the frame reading as
    # it always did: below the gate, last hint held.
    rejected = [t for t in replay() if not t.accepted]
    assert [t.number for t in rejected] == ["00150"]
    assert tick("00150").hint is not None  # the held hint is still on screen


def test_the_layer_removed_here_is_the_coach_s_own_hint_overlay() -> None:
    """What these six frames actually contain, read off the pixels.

    They were diagnosed as the game's landing preview and they are not:
    they are THIS TOOL's placement hint, drawn over the game by
    ``overlay.renderer.draw_hint`` and still on screen when the next
    frame was captured. The proof is the color. Every cell named here
    holds ``HintStyle.color`` ``#00e5ff`` composited onto the board's own
    ground at ``HintStyle.fill_opacity``, ringed by the pen at full
    opacity — which is why ``vision.grid._own_paint_layer`` names them by
    arithmetic instead of guessing at them from structure, and why the
    same frames keep reading the same way whatever else is on the board.

    The shape agrees with the old reading because a hint is a placement
    OF THE FALLING PIECE, so it is a copy of it and lands where the piece
    would come to rest: an O layer under a falling O. What did not
    survive is the COLUMN story — frames 61-64 put the layer at cols 2-3
    while the real O hangs at cols 4-5, because the hint sits where the
    SOLVER wants the piece, not under where the player is holding it.
    """
    named = {t.number: t.layer for t in replay() if t.layer is not None}
    assert set(named) == set(GHOST_CELLS), "a different set of frames removes a layer"
    for number, layer in named.items():
        assert layer.rule == "own_paint", f"{number}: named by {layer.rule}"
        assert layer.cells == GHOST_CELLS[number], f"{number}: {sorted(layer.cells)}"
        assert vision_grid._piece_named(sorted(layer.cells)) == "O", f"{number}: not an O"
        assert layer.in_flight == {"O"}, f"{number}: in flight {sorted(layer.in_flight)}"


def test_every_named_cell_carries_this_tool_s_own_paint() -> None:
    """The color claim, checked against the raw PNGs rather than the rule.

    ``#00e5ff`` at full opacity is what ``draw_hint`` strokes every hint
    cell's outline with, so a cell the classifier deletes must have that
    exact value in it — and no cell it leaves alone may. Read straight
    off the images so the rule cannot vouch for itself.

    Frame 150 is the documented exception in both directions: the pen
    stroked five cells there because the widget carries a rotation badge,
    and the badge is one opaque cell that matches no composite, so the
    rule refuses the whole widget rather than leave it behind as an
    unexplainable added cell.
    """
    named = {t.number: t.layer.cells for t in replay() if t.layer is not None}
    for number in frame_numbers():
        painted = pen_stroked_cells(FIXTURES / f"board_{number}.png", ROWS)
        if number in BADGE_FRAMES:
            assert len(painted) == 5, f"{number}: {sorted(painted)} is not a widget + badge"
            assert number not in named, f"{number}: the badge frame must be refused whole"
        else:
            assert painted == named.get(number, frozenset()), f"{number}: {sorted(painted)}"
