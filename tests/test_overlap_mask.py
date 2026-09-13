"""Next-piece-overlay masking: pure geometry helper + engine-level behavior.

Some games float the NEXT preview on top of the top corner of the playfield,
inside the region the user selects as the board. The cells under that box
show the NEXT piece, not the board; left unmasked they add a second
tetromino's worth of cells every frame, turning the frame UNEXPLAINED and
eventually tripping a spurious BOARD_RESET (the diagnosed ROAS Stacker
failure: 475 UNEXPLAINED frames, 411 resets, 13 tracked spawns over one game).

:func:`compute_overlap_mask` names those cells; the engine forces them empty
before the state tracker sees them.
"""

from __future__ import annotations

import numpy as np

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
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


class TestEngineMasksNextOverlay:
    """End-to-end: the same contaminated frame stream, with and without the
    mask. Without it the frames go UNEXPLAINED and the board spuriously
    resets; with it the real falling piece is tracked and hinted, no reset."""

    ROWS = 12
    STYLE = STYLES[2]  # jstris-like: solid colors, no noise
    CELL = 16
    EMPTY12: tuple[int, ...] = (0,) * 12

    # A bottom stack clear of the masked corner (cols 8-9, rows 0-1).
    STACK = rows_of(bottom_lines("#########.", height=ROWS), height=ROWS)

    @staticmethod
    def _engine_with_spy(masked: bool):  # type: ignore[no-untyped-def]
        mask = compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=12) if masked else frozenset()
        engine = CoachEngine(CoachConfig(rows=12), masked_cells=mask)
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

    def test_mask_covers_the_contaminated_corner(self) -> None:
        assert compute_overlap_mask(ROAS_BOARD, ROAS_NEXT, rows=self.ROWS) == frozenset(
            {(0, 8), (0, 9), (1, 8), (1, 9)}
        )

    def _run_stream(self, masked: bool):  # type: ignore[no-untyped-def]
        engine, events = self._engine_with_spy(masked)
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

    def test_with_mask_tracks_spawn_and_never_resets(self) -> None:
        engine, events, hint, _ = self._run_stream(masked=True)
        assert GameEvent.PIECE_SPAWNED in events
        assert GameEvent.BOARD_RESET not in events
        assert hint is not None
        assert hint.piece == "T"
        # The committed stack is the true board, free of the corner blob.
        assert engine.tracker.committed.stack_rows == self.STACK
        assert engine.tracker.committed.falling_piece == "T"

    def test_without_mask_goes_unexplained_and_resets(self) -> None:
        from tetris_coach.vision.pieces_vision import FrameKind

        _engine, events, _, kind_after_spawn = self._run_stream(masked=False)
        # The contaminated spawn frame cannot be explained ...
        assert kind_after_spawn is FrameKind.UNEXPLAINED
        # ... the real T is never tracked as a spawn ...
        assert GameEvent.PIECE_SPAWNED not in events
        # ... and a run of identical contaminated frames wipes the board.
        assert GameEvent.BOARD_RESET in events
