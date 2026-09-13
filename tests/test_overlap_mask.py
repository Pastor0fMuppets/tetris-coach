"""Next-piece overlay: pure geometry helper + engine-level behavior.

Some games float the NEXT preview on top of the top corner of the playfield,
inside the region the user selects as the board. The cells under that box
show the NEXT piece, not the board; read as board content they add a second
tetromino's worth of cells every frame, turning the frame UNEXPLAINED and
eventually tripping a spurious BOARD_RESET (the diagnosed ROAS Stacker
failure: 475 UNEXPLAINED frames, 411 resets, 13 tracked spawns over one game).

:func:`compute_overlap_mask` names those cells. They are UNOBSERVABLE, not
empty: the engine discards what the capture read there and tells the tracker
they are unknown, so a piece resting across the boundary is a partly hidden
piece rather than a broken one, and the solver is never handed free space in
a corner the coach cannot see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.core.board import Board
from tetris_coach.solver.search import enumerate_drops
from tetris_coach.vision.state import GameEvent

from .boards import bottom_lines, grid_of, merge, piece_cells, rows_of
from .synthetic import STYLES, render_board

# The real failing session's rects (from the task diagnosis): the NEXT box
# floats over the top-right corner of the board region.
ROAS_BOARD = Rect(left=164, top=314, width=480, height=577)
ROAS_NEXT = Rect(left=547, top=312, width=97, height=99)


class TestComputeOverlapMask:
    def test_real_rects_map_to_top_right_corner_12_rows(self) -> None:
        # The diagnosed mapping: next_rect covers board cols 8-9, rows 0-1.
        mask = compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=12)
        assert mask == frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

    def test_real_rects_cover_more_rows_at_20(self) -> None:
        # Same pixels, a taller grid: the box spans the top three rows.
        mask = compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=20)
        assert mask == frozenset({(0, 8), (0, 9), (1, 8), (1, 9), (2, 8), (2, 9)})

    def test_next_fully_inside_board_top_right(self) -> None:
        # A clean synthetic case: 100x120 board, 10x12 cells (10x10 px each);
        # a next box over the top-right 2x2 cell block.
        board = Rect(0, 0, 100, 120)
        nxt = Rect(80, 0, 20, 20)
        assert compute_overlap_mask(board, nxt, rows=12) == frozenset(
            {(0, 8), (0, 9), (1, 8), (1, 9)}
        )
        # The same rects on a 20-row grid (6 px tall cells) cover one more row.
        assert compute_overlap_mask(board, nxt, rows=20) == frozenset(
            {(0, 8), (0, 9), (1, 8), (1, 9), (2, 8), (2, 9)}
        )

    def test_no_overlap_returns_empty_mask(self) -> None:
        board = Rect(0, 0, 100, 120)
        # A next box entirely to the right of the board (the general case:
        # a preview drawn in a separate area).
        assert compute_overlap_mask(board, Rect(200, 0, 50, 50), rows=12) == frozenset()
        # Entirely above the board.
        assert compute_overlap_mask(board, Rect(0, -60, 100, 40), rows=12) == frozenset()

    def test_next_none_returns_empty_mask(self) -> None:
        assert compute_overlap_mask(Rect(0, 0, 100, 120), None, rows=12) == frozenset()
        assert compute_overlap_mask(Rect(0, 0, 100, 120), None, rows=20) == frozenset()

    def test_partial_overlap_masks_only_covered_cells(self) -> None:
        # A next box grazing only the last column's cell centers.
        board = Rect(0, 0, 100, 120)
        nxt = Rect(90, 0, 40, 20)  # x in [0.9, 1.3]: only col 9's center (0.95)
        assert compute_overlap_mask(board, nxt, rows=12) == frozenset({(0, 9), (1, 9)})

    def test_border_graze_does_not_mask(self) -> None:
        # A next box whose right edge stops just before col 9's center
        # (95 px) but past col 8's (85 px): only col 8 is masked, not col 9.
        board = Rect(0, 0, 100, 120)
        nxt = Rect(80, 0, 10, 20)  # x in [0.80, 0.90]
        assert compute_overlap_mask(board, nxt, rows=12) == frozenset({(0, 8), (1, 8)})

    def test_degenerate_board_rect_returns_empty(self) -> None:
        assert compute_overlap_mask(Rect(0, 0, 0, 120), ROAS_NEXT, rows=12) == frozenset()
        assert compute_overlap_mask(Rect(0, 0, 100, 0), ROAS_NEXT, rows=12) == frozenset()


class TestEngineTreatsPreviewCellsAsUnknown:
    """End-to-end: the same contaminated frame stream, with and without the
    covered cells declared. Undeclared, the frames go UNEXPLAINED and the
    board spuriously resets; declared, the real falling piece is tracked and
    hinted, no reset."""

    ROWS = 12
    STYLE = STYLES[2]  # jstris-like: solid colors, no noise
    CELL = 16
    EMPTY12: tuple[int, ...] = (0,) * 12

    # A bottom stack clear of the masked corner (cols 8-9, rows 0-1).
    STACK = rows_of(bottom_lines("#########.", height=ROWS), height=ROWS)

    @staticmethod
    def _engine_with_spy(covered: bool):  # type: ignore[no-untyped-def]
        cells = compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=12) if covered else frozenset()
        engine = CoachEngine(CoachConfig(rows=12), unobservable_cells=cells)
        events_log: list[GameEvent] = []
        original = engine.tracker.update

        def spy(occupancy, nxt):  # type: ignore[no-untyped-def]
            events = original(occupancy, nxt)
            events_log.extend(events)
            return events

        engine.tracker.update = spy  # type: ignore[method-assign]
        return engine, events_log

    def _image(self, rows: tuple[int, ...]) -> np.ndarray:
        return render_board(grid_of(rows), self.STYLE, cell_size=self.CELL)

    def test_the_contaminated_corner_is_the_declared_one(self) -> None:
        assert compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=self.ROWS) == frozenset(
            {(0, 8), (0, 9), (1, 8), (1, 9)}
        )

    def _run_stream(self, covered: bool):  # type: ignore[no-untyped-def]
        engine, events = self._engine_with_spy(covered)
        # A changing NEXT blob living in the masked corner (cols 8-9, rows
        # 0-1); 2-cell fragments never form a tetromino on their own.
        blob_a = [(0, 8), (0, 9)]
        blob_b = [(1, 8), (1, 9)]
        blob_c = [(0, 9), (1, 9)]
        blob_d = [(0, 8), (1, 9)]

        # Anchor the classifier's background on the empty pre-game board,
        # then attach the stack (with a corner blob) via the reset debounce.
        engine.process_frame(self._image(self.EMPTY12), None)
        attach = merge(self.STACK, blob_a)
        for _ in range(4):
            engine.process_frame(self._image(attach), None)
        events.clear()  # the initial attach resync is expected setup

        # A real T descends on the left; the corner blob keeps changing.
        spawn = merge(self.STACK, piece_cells("T", 0, 0, 3), blob_b)
        mid = merge(self.STACK, piece_cells("T", 0, 4, 3), blob_c)
        rest = merge(self.STACK, piece_cells("T", 0, 7, 3), blob_d)

        hint = None
        for _ in range(2):
            hint = engine.process_frame(self._image(spawn), None)
        kind_after_spawn = engine.tracker.last_kind
        engine.process_frame(self._image(mid), None)
        for _ in range(4):  # a resting piece: 4 identical frames
            hint = engine.process_frame(self._image(rest), None)
        return engine, events, hint, kind_after_spawn

    def test_declared_corner_tracks_spawn_and_never_resets(self) -> None:
        engine, events, hint, _ = self._run_stream(covered=True)
        assert GameEvent.PIECE_SPAWNED in events
        assert GameEvent.BOARD_RESET not in events
        assert hint is not None
        assert hint.piece == "T"
        # The committed stack is the true board, free of the corner blob.
        assert engine.tracker.committed.stack_rows == self.STACK
        assert engine.tracker.committed.falling_piece == "T"

    def test_undeclared_corner_goes_unexplained_and_resets(self) -> None:
        from tetris_coach.vision.pieces_vision import FrameKind

        _engine, events, _, kind_after_spawn = self._run_stream(covered=False)
        # The contaminated spawn frame cannot be explained ...
        assert kind_after_spawn is FrameKind.UNEXPLAINED
        # ... the real T is never tracked as a spawn ...
        assert GameEvent.PIECE_SPAWNED not in events
        # ... and a run of identical contaminated frames wipes the board.
        assert GameEvent.BOARD_RESET in events


class TestRealFixtureFrames:
    """Smoke test on real captured frames from the failing live ROAS Stacker
    session (board region, 10x12). Each frame's raw occupancy carries NEXT-
    preview cells in the top-right corner; the engine discards them."""

    FIXTURES = Path(__file__).parent / "fixtures" / "roas_stacker"
    FRAMES = ("live_board_500", "live_board_600", "live_board_700", "live_board_800")

    def test_preview_readings_are_discarded_on_real_frames(self) -> None:
        from PIL import Image

        from tetris_coach.vision.grid import GridClassifier

        covered = compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=12)
        assert covered == frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})
        # Reuse the engine's own blanking so the fixture pins the real path.
        engine = CoachEngine(CoachConfig(rows=12), unobservable_cells=covered)

        any_contaminated = False
        for name in self.FRAMES:
            rgb = np.asarray(Image.open(self.FIXTURES / f"{name}.png"))
            bgr = rgb[:, :, ::-1]  # match the BGR capture pipeline
            occupancy, _confidence = GridClassifier(rows=12).classify(bgr)
            # The raw reading has NEXT-preview cells in the covered corner ...
            if any(occupancy[r, c] for r, c in covered):
                any_contaminated = True
            blanked = engine._blanked(occupancy)
            # ... and none of those readings survives into the pipeline.
            for r, c in covered:
                assert not blanked[r, c], f"{name}: cell ({r},{c}) not cleared"
            # Blanking touches nothing outside the corner.
            outside = occupancy.copy()
            for r, c in covered:
                outside[r, c] = False
            assert np.array_equal(blanked, outside)
        assert any_contaminated, "fixtures no longer exhibit the corner contamination"


# A near-top-out 12-row board: the right-hand columns reach row 3, so a
# piece can come to rest at rows 1-2 of cols 8-9 — half under the preview.
HIGH_STACK = rows_of(
    bottom_lines(
        "........##",
        "........##",
        "......####",
        "....######",
        "..########",
        ".#########",
        "..########",
        "...#######",
        "....######",
        height=12,
    ),
    height=12,
)


class TestPieceRestingUnderThePreview:
    """The regression the forced-empty mask introduced: a REAL piece coming
    to rest across the boundary read as a 2-cell fragment, which is not a
    tetromino — UNEXPLAINED, and four identical frames of a piece sitting
    still in lock delay then tripped the very BOARD_RESET the masking was
    added to prevent, wiping the hint and the precompute cache."""

    ROWS = 12
    STYLE = STYLES[2]
    CELL = 16
    COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

    def _image(self, rows: tuple[int, ...]) -> np.ndarray:
        return render_board(grid_of(rows), self.STYLE, cell_size=self.CELL)

    def _run(self):  # type: ignore[no-untyped-def]
        engine = CoachEngine(CoachConfig(rows=self.ROWS), unobservable_cells=self.COVERED)
        events: list[GameEvent] = []
        original = engine.tracker.update

        def spy(occupancy, nxt):  # type: ignore[no-untyped-def]
            got = original(occupancy, nxt)
            events.extend(got)
            return got

        engine.tracker.update = spy  # type: ignore[method-assign]

        engine.process_frame(self._image((0,) * self.ROWS), None)  # anchor background
        for _ in range(4):
            engine.process_frame(self._image(HIGH_STACK), None)  # attach
        events.clear()

        # An O spawns in the middle, then is moved right into the corner,
        # where only its bottom two cells are visible.
        for _ in range(2):
            engine.process_frame(self._image(merge(HIGH_STACK, piece_cells("O", 0, 0, 4))), None)
        spawn_hint = engine.current_hint
        engine.process_frame(self._image(merge(HIGH_STACK, piece_cells("O", 0, 0, 6))), None)
        corner = merge(HIGH_STACK, piece_cells("O", 0, 1, 8))
        hint = None
        for _ in range(8):  # stationary through lock delay: identical frames
            hint = engine.process_frame(self._image(corner), None)
        return engine, events, hint, spawn_hint

    def test_resting_in_the_corner_never_resets_the_board(self) -> None:
        engine, events, hint, spawn_hint = self._run()
        assert GameEvent.PIECE_SPAWNED in events  # the O was tracked
        assert GameEvent.BOARD_RESET not in events  # ... and never wiped
        assert engine.tracker.committed.stack_rows == HIGH_STACK
        # The hint (and the precompute cache behind it) survives the piece
        # disappearing under the preview box.
        assert spawn_hint is not None
        assert hint is spawn_hint

    def test_the_frame_is_occluded_not_unexplained(self) -> None:
        from tetris_coach.vision.pieces_vision import FrameKind

        engine, _events, _hint, _spawn = self._run()
        assert engine.tracker.last_kind is FrameKind.OCCLUDED


class TestSolverNeverPlansIntoCoveredCells:
    """The other half of the forced-empty lie: a stack that legitimately
    grows into the corner was handed to the solver as free space, so the
    hint planned into cells the coach cannot see (and the overlay would
    draw it under the panel that hides them)."""

    ROWS = 12
    COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

    def _engine(self) -> CoachEngine:
        return CoachEngine(CoachConfig(rows=self.ROWS), unobservable_cells=self.COVERED)

    @staticmethod
    def _reaches_covered_columns(board: Board, piece: str) -> bool:
        return any(col + rot.width > 8 for rot, col, _result in enumerate_drops(board, piece))

    def test_a_low_stack_leaves_the_corner_playable(self) -> None:
        # Nothing is under the covered cells, so nothing is assumed: the
        # right-hand columns stay available, as they are for most of a game.
        engine = self._engine()
        stack = rows_of(bottom_lines("#####.....", height=self.ROWS), height=self.ROWS)
        board = engine._solver_board(stack)
        assert board.rows == stack
        assert self._reaches_covered_columns(board, "O")

    def test_a_stack_at_the_boundary_closes_the_corner(self) -> None:
        # The stack has grown to the row directly under the preview box.
        # Whether it continues up into it is unobservable — so the solver is
        # handed those cells filled and plans elsewhere.
        engine = self._engine()
        cells = [(r, 8) for r in range(2, self.ROWS)] + [(r, 9) for r in range(2, self.ROWS)]
        stack = rows_of(cells, height=self.ROWS)
        board = engine._solver_board(stack)
        assert board.rows[0] == 0b1100000000
        assert board.rows[1] == 0b1100000000
        for piece in ("O", "I", "L"):
            assert not self._reaches_covered_columns(board, piece)
        # Counterfactual: believed as empty, cols 8-9 read as free space.
        assert self._reaches_covered_columns(Board(stack), "O")

    def test_believed_stack_under_the_box_is_not_re_derived(self) -> None:
        # An engine with no covered cells (the ordinary game) is untouched.
        engine = CoachEngine(CoachConfig(rows=self.ROWS))
        stack = rows_of([(r, 8) for r in range(2, self.ROWS)], height=self.ROWS)
        assert engine._solver_board(stack).rows == stack
