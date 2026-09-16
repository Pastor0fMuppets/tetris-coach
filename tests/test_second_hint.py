"""The second hint: where the NEXT piece goes, and when that may be shown.

The engine has always solved this placement -- it pre-solves the upcoming
piece on the board the standing hint would leave, so the hint flips the
instant the current piece locks. Showing it a piece early costs nothing
extra and is the whole feature. What it costs is a way to be WRONG, and
every test here is about one of them:

* a second target with no first target to be conditional on,
* a second target computed for a board the player then did not produce,
* a second target drawn in the coordinates of a board that has shifted
  under a line clear,
* two targets sharing a cell, which would also put one hint's rotation
  badge in a cell belonging to the other.

The readings are scripted rather than rendered, so each case states the
board, the piece and the preview it is about; the policy under test is the
real one (``CoachEngine`` is driven exactly as ``app.run`` drives it, with
a reader handing it readings instead of pixels). The last test is the
counterweight: the same rules over the committed captures, counting how
often a second target is there and what withholds it, so that "conditional"
stays a description of a feature that is usually on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tetris_coach.app import CoachConfig, CoachEngine, FrameWorker
from tetris_coach.capture.screen import ArraySource, Rect
from tetris_coach.core.board import Board
from tetris_coach.race.runners import next_crops
from tetris_coach.solver.search import best_move
from tetris_coach.truth.windows import CAPTURED_FILL_OPACITY, WINDOWS, load_window
from tetris_coach.vision.readers import FrameReading, VisionEvent

ROWS = 12
FULL = (1 << 10) - 1


@dataclass
class ScriptedVision:
    """A reader that hands the engine the readings a test wrote out.

    The engine's hint policy is written against ``FrameReading`` and
    nothing else (that is what lets two very different trackers share it),
    so a script of readings drives the production policy exactly.
    """

    script: list[FrameReading]
    unknown_rows: tuple[int, ...] = field(default_factory=lambda: (0,) * ROWS)
    tracker: object = None
    index: int = 0

    def read(self, board_image: object, next_image: object) -> FrameReading:
        reading = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return reading


def reading(
    piece: str | None,
    stack: tuple[int, ...],
    next_piece: str | None,
    *events: VisionEvent,
    accepted: bool = True,
) -> FrameReading:
    return FrameReading(
        accepted=accepted,
        falling_piece=piece,
        stack_rows=stack,
        next_piece=next_piece,
        events=tuple(events),
    )


def coach(*script: FrameReading, **overrides: object) -> CoachEngine:
    config = CoachConfig(rows=ROWS, **overrides)  # type: ignore[arg-type]
    return CoachEngine(config, vision=ScriptedVision(list(script)))  # type: ignore[arg-type]


def drive(engine: CoachEngine, frames: int) -> None:
    blank = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in range(frames):
        engine.process_frame(blank, None)


EMPTY = (0,) * ROWS


def test_the_second_hint_is_the_next_piece_on_the_board_the_first_one_leaves() -> None:
    engine = coach(reading("I", EMPTY, "O", VisionEvent.PIECE_SPAWNED))
    drive(engine, 1)
    first, second = engine.current_hint, engine.second_hint
    assert first is not None and first.piece == "I"
    assert second is not None and second.piece == "O"
    # Not a second opinion about the same board: it is solved on the board
    # the first placement leaves behind, which is what makes it a PLAN.
    assert second == best_move(first.board, "O")
    assert set(second.cells).isdisjoint(first.cells)


def test_there_is_no_second_hint_without_a_first_one() -> None:
    # A lock gap: the piece is gone, the next one has not appeared. The
    # engine pre-solves against the settled board here -- a different
    # prediction, and not one to draw, because there is nothing on screen
    # the player is being asked to do first.
    engine = coach(reading(None, EMPTY, "O", VisionEvent.PIECE_LOCKED))
    drive(engine, 1)
    assert engine.current_hint is None
    assert engine._precomputed is not None, "the lock-gap pre-solve still happens"
    assert engine.second_hint is None


def test_there_is_no_second_hint_without_a_next_piece() -> None:
    # An unreadable preview box: one target, and the first hint is the
    # 1-ply answer it always was.
    engine = coach(reading("I", EMPTY, None, VisionEvent.PIECE_SPAWNED))
    drive(engine, 1)
    assert engine.current_hint is not None
    assert engine.second_hint is None


def test_the_second_hint_follows_the_first_when_the_player_goes_elsewhere() -> None:
    """The divergence case, which is the one that could mislead.

    The player ignores the hint and drops the piece somewhere else. The
    board that arrives is not the board the second placement was computed
    for, and what must NOT happen is that placement staying on screen.
    """
    spawn = reading("I", EMPTY, "O", VisionEvent.PIECE_SPAWNED)
    engine = coach(spawn)
    drive(engine, 1)
    first, planned = engine.current_hint, engine.second_hint
    assert first is not None and planned is not None

    # A board the hint would not have produced, with the next piece now in
    # flight -- the player put the I somewhere of their own.
    elsewhere = EMPTY[:-1] + (0b1111000000,)
    assert elsewhere != first.board.rows
    engine.vision.script.append(  # type: ignore[attr-defined]
        reading("O", elsewhere, "T", VisionEvent.PIECE_LOCKED)
    )
    drive(engine, 1)

    now, second = engine.current_hint, engine.second_hint
    assert now is not None and now.piece == "O"
    assert second is not None and second.piece == "T"
    # Solved on the board on screen, not on the one that never happened.
    assert second == best_move(now.board, "T")
    assert second != planned
    assert set(second.cells).isdisjoint(now.cells)


def test_a_hint_that_clears_lines_shows_no_second_target() -> None:
    """A clear shifts every row above it, so the coordinates stop agreeing.

    The second placement is computed on the board AFTER the clear, and the
    player is looking at the board before it. Drawing row 7 of one on row 7
    of the other points at the wrong row, and there is no honest way to
    draw it until the clear has happened.
    """
    # Columns 0 and 1 open on the bottom two rows: an O drops in and clears
    # both of them.
    gap = FULL & ~0b11
    stack = (0,) * (ROWS - 2) + (gap, gap)
    engine = coach(reading("O", stack, "I", VisionEvent.PIECE_SPAWNED))
    drive(engine, 1)
    first = engine.current_hint
    assert first is not None and first.lines_cleared == 2
    assert engine._precomputed is not None, "the precompute is still made"
    assert engine.second_hint is None, "but it is not a thing to draw yet"


def test_the_second_hint_comes_down_with_the_first_one() -> None:
    # Vision stops being able to justify the placement: the hint is
    # withdrawn, and a plan that hangs off it cannot outlive it.
    engine = coach(
        reading("I", EMPTY, "O", VisionEvent.PIECE_SPAWNED),
        reading(None, EMPTY, None, accepted=False),
        max_stale_frames=2,
    )
    drive(engine, 1)
    assert engine.second_hint is not None
    drive(engine, 4)
    assert engine.current_hint is None
    assert engine.second_hint is None


def test_the_two_hints_never_share_a_cell_over_a_run_of_boards() -> None:
    """The property both rotation badges depend on, over real solver output.

    Each badge sits inside a cell ITS OWN hint paints. That one cannot
    land in a cell belonging to the other is a fact about the pair: the
    second placement is solved on a board where the first one's cells are
    already filled, and a drop never lands in a filled cell.
    """
    rng = np.random.default_rng(7)
    pieces = "IOTSZJL"
    seen = 0
    for _ in range(200):
        rows = [int(rng.integers(0, 1 << 10)) & FULL for _ in range(ROWS)]
        rows[: ROWS // 2] = [0] * (ROWS // 2)
        stack = tuple(rows)
        piece = pieces[int(rng.integers(0, 7))]
        upcoming = pieces[int(rng.integers(0, 7))]
        engine = coach(reading(piece, stack, upcoming, VisionEvent.PIECE_SPAWNED))
        drive(engine, 1)
        first, second = engine.current_hint, engine.second_hint
        if first is None or second is None:
            continue
        seen += 1
        assert set(first.cells).isdisjoint(second.cells), (stack, piece, upcoming)
        # ... and the second one really is a legal drop on the predicted board.
        assert second.board != first.board
        assert Board(first.board.rows).drop(second.rotation, second.col) is not None
    assert seen > 100, "the run has to actually produce pairs to be worth anything"


def test_the_tick_carries_the_second_hint_to_the_overlay() -> None:
    """The worker hands both hints across, so the GUI thread asks the engine nothing.

    ``CoachEngine`` belongs to the worker thread; a property read from the
    GUI thread while the next tick is running is a race, and the answer it
    returned would belong to a different frame than the hint beside it.
    """
    engine = coach(reading("I", EMPTY, "O", VisionEvent.PIECE_SPAWNED))
    source = ArraySource([np.zeros((8, 8, 3), dtype=np.uint8)] * 2)
    worker = FrameWorker(engine, source, Rect(0, 0, 8, 8), None)
    result = worker.run_tick()
    assert result.ok
    assert result.hint is not None and result.hint.piece == "I"
    assert result.second is not None and result.second.piece == "O"
    assert (result.hint, result.second) == (engine.current_hint, engine.second_hint)


# -- and the same rules over the real captures -------------------------


def test_how_often_the_second_hint_is_there_over_the_committed_windows() -> None:
    """The cost of the conditions above, counted rather than asserted to be small.

    Every rule here withholds a target, and a rule that withheld it most
    of the time would be a feature that is not there. So the nine
    committed windows are replayed through the real engine (wired as
    ``race/`` wires them: the reader is told about the fill these captures
    were taken with) and the dispositions counted.

    The numbers are pinned because an unpinned measurement in SPEC.md goes
    stale silently -- this project has one that did. What they say is that
    the second hint is on screen for 529 of the 627 frames that carry a
    hint at all, and that EVERY one of the other 98 is a first placement
    that clears a line, which is the one case with a coordinate problem
    behind it. Not one frame withholds it because the prediction went
    stale (every re-solve recomputes the pair together) and not one
    because the two placements overlap (they cannot).
    """
    root = Path(__file__).parent / "fixtures"
    frames = hinted = both = no_first = cleared = stale = overlapping = 0
    for spec in WINDOWS:
        config = CoachConfig(rows=spec.rows, hint_fill_opacity=CAPTURED_FILL_OPACITY)
        engine = CoachEngine(config, unobservable_cells=spec.geometry().unobservable)
        names, boards = load_window(root, spec)
        for board, crop in zip(boards, next_crops(root, spec, names), strict=True):
            hint = engine.process_frame(board, crop)
            second = engine.second_hint
            frames += 1
            hinted += hint is not None
            if second is not None:
                both += 1
            elif hint is None:
                no_first += 1
            elif hint.lines_cleared:
                cleared += 1
            elif engine._predicted_board != hint.board:
                stale += 1
            else:
                overlapping += 1
    assert (frames, hinted, both) == (649, 627, 529)
    assert (no_first, cleared) == (22, 98)
    assert (stale, overlapping) == (0, 0)
