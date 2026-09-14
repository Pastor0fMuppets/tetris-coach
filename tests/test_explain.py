"""Unit tests for explain_grid: diff-anchored frame classification."""

import pytest

from tetris_coach.core.board import HEIGHT, WIDTH, Board
from tetris_coach.core.pieces import PIECES, ROTATIONS
from tetris_coach.vision.pieces_vision import (
    FrameKind,
    clear_full_rows,
    explain_grid,
)

from .boards import EMPTY, Cell, bottom_lines, fp, merge, piece_cells, rows_of


class TestQuiet:
    def test_quiet_idle(self) -> None:
        stack = rows_of(bottom_lines("####..####"))
        exp = explain_grid(stack, stack, None)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == stack
        assert exp.falling is None

    def test_quiet_empty_board(self) -> None:
        exp = explain_grid(EMPTY, EMPTY, None)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == EMPTY


class TestFlightAnywhere:
    """F1: the falling piece is FALLING wherever it is — no support rules."""

    def test_piece_resting_on_stack(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        t = piece_cells("T", 0, 17, 3)  # resting directly on the stack
        exp = explain_grid(merge(stack, t), stack, fp("T", 0, 16, 3))
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == stack
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert (exp.falling.row, exp.falling.col) == (17, 3)

    def test_piece_on_floor_of_empty_column(self) -> None:
        # The old floor-touch rule merged this piece into the stack.
        stack = rows_of([(18, 0), (19, 0)])
        o = piece_cells("O", 0, 18, 4)
        exp = explain_grid(merge(stack, o), stack, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "O"
        assert exp.stack_rows == stack

    def test_piece_leaning_against_tall_column(self) -> None:
        stack = rows_of([(r, 0) for r in range(10, 20)])
        i = piece_cells("I", 1, 12, 1)  # side-adjacent, mid-air
        exp = explain_grid(merge(stack, i), stack, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "I"
        assert exp.stack_rows == stack

    def test_sliding_during_lock_delay(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        for col in (3, 4, 5):
            t = piece_cells("T", 0, 17, col)
            exp = explain_grid(merge(stack, t), stack, fp("T", 0, 17, col - 1))
            assert exp.kind is FrameKind.FALLING
            assert exp.stack_rows == stack
            assert exp.falling is not None
            assert exp.falling.col == col


class TestDebrisStaysStack:
    """F3: floating debris lives inside the committed stack, never falls."""

    def test_real_piece_above_island(self) -> None:
        island = rows_of(piece_cells("O", 0, 10, 7))
        t = piece_cells("T", 0, 2, 3)
        exp = explain_grid(merge(island, t), island, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert exp.stack_rows == island

    def test_real_piece_below_island(self) -> None:
        # The old topmost-component rule fails exactly here: the island is
        # above the real piece and would have been picked as falling.
        island = rows_of(piece_cells("O", 0, 10, 7))
        t = piece_cells("T", 0, 15, 6)
        exp = explain_grid(merge(island, t), island, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert exp.falling.row == 15
        assert exp.stack_rows == island


class TestLockRevealL1:
    def test_lock_plus_adjacent_spawn_one_blob(self) -> None:
        # Tall stack: the locked piece is adjacent to the spawn (L2 cannot
        # split the single 8-cell blob; L1 must anchor on the last position).
        stack = rows_of([(r, 3) for r in range(6, 20)])
        last = fp("O", 0, 4, 3)  # at rest on the tower
        spawn = piece_cells("I", 0, 3, 3)  # 4-adjacent to the O
        observed = merge(stack, last.cells, spawn)
        exp = explain_grid(observed, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, last.cells)
        assert exp.falling is not None
        assert exp.falling.piece == "I"
        assert (exp.falling.row, exp.falling.col) == (3, 3)


class TestLockRevealL2:
    STACK = rows_of(bottom_lines("###....###"))

    def test_hard_drop_lock_with_spawn(self) -> None:
        # Zero-ARE: last observed mid-air, locked cells are NOT those cells.
        last = fp("T", 0, 1, 3)
        lock = piece_cells("T", 0, 18, 3)
        spawn = piece_cells("J", 0, 0, 4)
        exp = explain_grid(merge(self.STACK, lock, spawn), self.STACK, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(self.STACK, lock)
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def test_no_prior_observation_name_unconstrained(self) -> None:
        lock = piece_cells("T", 0, 18, 3)
        spawn = piece_cells("J", 0, 0, 4)
        exp = explain_grid(merge(self.STACK, lock, spawn), self.STACK, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(self.STACK, lock)
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def _towers(self) -> tuple[int, ...]:
        left = [(r, c) for r in range(4, 20) for c in (0, 1, 2, 3)]
        right = [(r, c) for r in range(4, 20) for c in (6, 7, 8, 9)]
        return rows_of(left, right)

    def test_both_in_spawn_zone_name_tiebreak(self) -> None:
        # Near top-out: both components supported and in the spawn zone;
        # the piece-name constraint disambiguates.
        stack = self._towers()
        lock_o = piece_cells("O", 0, 2, 0)  # rests on the left tower
        spawn_s = piece_cells("S", 0, 2, 6)  # rests on the right tower
        last = fp("O", 0, 2, 3)  # mid-gap: not a subset of added (L1 fails)
        exp = explain_grid(merge(stack, lock_o, spawn_s), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, lock_o)
        assert exp.falling is not None
        assert exp.falling.piece == "S"

    def test_same_name_intersection_tiebreak(self) -> None:
        stack = self._towers()
        left_o = piece_cells("O", 0, 2, 0)
        right_o = piece_cells("O", 0, 2, 6)
        # Overlaps the left component only (and partly the stack, so L1's
        # subset test fails and L2 must break the tie by intersection).
        last = fp("O", 0, 3, 1)
        exp = explain_grid(merge(stack, left_o, right_o), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, left_o)
        assert exp.falling is not None
        assert exp.falling.piece == "O"
        assert exp.falling.col == 6

    def test_same_name_no_overlap_is_unexplained(self) -> None:
        stack = self._towers()
        left_o = piece_cells("O", 0, 2, 0)
        right_o = piece_cells("O", 0, 2, 6)
        last = fp("O", 0, 2, 4)  # mid-gap: intersects neither component
        exp = explain_grid(merge(stack, left_o, right_o), stack, last)
        assert exp.kind is FrameKind.UNEXPLAINED


class TestLockRevealRejectsFullRows:
    """A zero-ARE clear-flash frame — the completed row still fully lit
    while the next piece is already visible — must stay UNEXPLAINED:
    committing it would anchor the tracker on a stack containing a row
    that is about to vanish."""

    STACK = rows_of(bottom_lines("#########."))
    SPAWN = piece_cells("J", 0, 0, 4)

    def test_l1_flash_frame_with_spawn_unexplained(self) -> None:
        last = fp("I", 1, 16, 9)  # observed at rest, completing row 19
        observed = merge(self.STACK, last.cells, self.SPAWN)
        assert explain_grid(observed, self.STACK, last).kind is FrameKind.UNEXPLAINED

    def test_l2_flash_frame_with_spawn_unexplained(self) -> None:
        last = fp("I", 1, 4, 9)  # hard drop: last observed mid-air
        lock = piece_cells("I", 1, 16, 9)
        observed = merge(self.STACK, lock, self.SPAWN)
        assert explain_grid(observed, self.STACK, last).kind is FrameKind.UNEXPLAINED

    def test_settled_post_clear_frame_still_locks(self) -> None:
        # Once the animation settles, the collapsed board plus the
        # (descended) spawn is explained by the clear tiers as usual.
        last = fp("I", 1, 16, 9)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        observed = merge(s2, piece_cells("J", 0, 1, 4))
        exp = explain_grid(observed, self.STACK, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "J"


class TestClearTierC1:
    def test_last_position_completes_row(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 16, 9)  # vertical I resting, completing row 19
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_with_residual_spawn(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 16, 9)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        spawn = piece_cells("J", 0, 0, 4)
        exp = explain_grid(merge(s2, spawn), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def test_t_spin_double_not_gravity_reachable(self) -> None:
        stack = rows_of(
            [(17, 2)],
            [(18, c) for c in (0, 4, 5, 6, 7, 8, 9)],
            [(19, c) for c in (0, 1, 3, 4, 5, 6, 7, 8, 9)],
        )
        last = fp("T", 2, 18, 1)  # point-down, tucked under the roof at (17, 2)
        # Sanity: gravity cannot reach the observed position (the roof).
        dropped = Board(stack).drop(ROTATIONS["T"][2], 1)
        assert dropped is not None and dropped.landing_row != 18
        s2 = rows_of([(19, 2)])  # double clear collapses everything else
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None


class TestClearTierC2:
    def test_gravity_drop_from_last_alignment(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 5, 9)  # mid-air; the drop lands and clears row 19
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_last_instant_shift_found_by_c3(self) -> None:
        # The piece shifted a column after the last observation: C2's drop
        # from the stale alignment misses; C3 retries the same name.
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 5, 8)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None


class TestClearTierC3Unobserved:
    """F2: a line-clearing lock by a never-observed piece is LOCKED, never
    anything reset-shaped."""

    def test_single_clear(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(s2, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_double_clear_to_empty_board(self) -> None:
        stack = rows_of(bottom_lines("########..", "########.."))
        exp = explain_grid(EMPTY, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == EMPTY
        assert exp.falling is None

    def test_quad_clear_to_empty_board(self) -> None:
        stack = rows_of(bottom_lines("#########.", "#########.", "#########.", "#########."))
        exp = explain_grid(EMPTY, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == EMPTY
        assert exp.falling is None

    def test_clear_with_simultaneous_spawn(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        spawn = piece_cells("S", 0, 1, 4)
        exp = explain_grid(merge(s2, spawn), stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "S"


class TestUnexplained:
    STACK = rows_of(bottom_lines("###....###"))

    def test_fragment_in_open_air_below_the_top_edge(self) -> None:
        # A 2-cell blob floating at row 3 touches no edge of anything: no
        # piece could be showing only that, so it stays unexplained (and can
        # still reach the reset debounce). Contrast TestEnteringFromAbove,
        # where the same blob AT ROW 0 is a piece cut by the capture's edge.
        exp = explain_grid(merge(self.STACK, [(3, 4), (3, 5)]), self.STACK, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_five_cell_blob(self) -> None:
        blob = [(5, 2), (5, 3), (5, 4), (6, 3), (6, 4)]
        exp = explain_grid(merge(self.STACK, blob), self.STACK, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_three_components(self) -> None:
        observed = merge(
            self.STACK,
            piece_cells("O", 0, 2, 1),
            piece_cells("O", 0, 2, 5),
            piece_cells("O", 0, 10, 7),
        )
        assert explain_grid(observed, self.STACK, None).kind is FrameKind.UNEXPLAINED

    def test_eight_cells_failing_l1_and_l2(self) -> None:
        # Two mid-air tetrominoes: neither supported, neither in the zone.
        observed = merge(self.STACK, piece_cells("I", 0, 10, 2), piece_cells("I", 0, 12, 5))
        assert explain_grid(observed, self.STACK, None).kind is FrameKind.UNEXPLAINED

    def test_missing_cells_without_clear_explanation(self) -> None:
        stack = rows_of(bottom_lines("#########."))
        observed = tuple(
            row & ~sum(1 << c for c in (2, 3, 4)) if r == 19 else row for r, row in enumerate(stack)
        )
        assert explain_grid(observed, stack, None).kind is FrameKind.UNEXPLAINED

    def test_piece_plus_same_named_ghost(self) -> None:
        # A bright ghost piece at the bottom with the real piece mid-air
        # must hold as UNEXPLAINED, not commit a false lock.
        stack = rows_of(bottom_lines("###...####"))
        real = piece_cells("T", 0, 5, 3)
        ghost = piece_cells("T", 0, 18, 3)
        last = fp("T", 0, 4, 3)
        exp = explain_grid(merge(stack, real, ghost), stack, last)
        assert exp.kind is FrameKind.UNEXPLAINED


class TestOcclusionTolerance:
    STACK = rows_of(bottom_lines("#########."))

    def _occlude(self, rows: tuple[int, ...], cells: list[tuple[int, int]]) -> tuple[int, ...]:
        out = list(rows)
        for r, c in cells:
            out[r] &= ~(1 << c)
        return tuple(out)

    def test_falling_with_two_missing_keeps_stack_memory(self) -> None:
        observed = self._occlude(merge(self.STACK, piece_cells("T", 0, 5, 3)), [(19, 0), (19, 1)])
        exp = explain_grid(observed, self.STACK, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK  # memory wins over vision
        assert exp.falling is not None
        assert exp.falling.piece == "T"

    def test_quiet_with_one_missing(self) -> None:
        observed = self._occlude(self.STACK, [(19, 4)])
        exp = explain_grid(observed, self.STACK, None)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == self.STACK

    def test_three_missing_is_unexplained(self) -> None:
        observed = self._occlude(self.STACK, [(19, 0), (19, 1), (19, 2)])
        assert explain_grid(observed, self.STACK, None).kind is FrameKind.UNEXPLAINED

    def test_tolerance_never_applies_to_lock_reveal(self) -> None:
        stack = rows_of(bottom_lines("###....###"))
        lock = piece_cells("T", 0, 18, 3)
        spawn = piece_cells("J", 0, 0, 4)
        observed = self._occlude(merge(stack, lock, spawn), [(19, 0), (19, 1)])
        exp = explain_grid(observed, stack, fp("T", 0, 18, 3))
        assert exp.kind is FrameKind.UNEXPLAINED


class TestTwelveRowBoards:
    """explain_grid on a 12-row board: every bound comes from len(rows)."""

    ROWS = 12
    EMPTY12: tuple[int, ...] = (0,) * 12

    def test_quiet_and_falling(self) -> None:
        stack = rows_of(bottom_lines("####..####", height=self.ROWS), height=self.ROWS)
        assert explain_grid(stack, stack, None).kind is FrameKind.QUIET
        t = piece_cells("T", 0, 5, 3)
        exp = explain_grid(merge(stack, t), stack, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "T"

    def test_lock_reveal_of_piece_on_12_row_floor(self) -> None:
        # Hard drop to the FLOOR of the 12-row board plus a fresh spawn:
        # the L2 support test must use the board's own bottom row, not the
        # 20-row constant (which would reject or crash on every floor lock).
        last = fp("T", 0, 1, 3)  # last observed mid-air near the top
        lock = piece_cells("T", 0, self.ROWS - 2, 3)  # rests on row 11
        spawn = piece_cells("J", 0, 0, 5)
        observed = merge(self.EMPTY12, lock, spawn)
        exp = explain_grid(observed, self.EMPTY12, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(self.EMPTY12, lock)
        assert exp.falling is not None
        assert exp.falling.piece == "J"

    def test_lock_with_clear_from_last_position(self) -> None:
        # C1: vertical I resting in column 9, completing the bottom row.
        stack = rows_of(bottom_lines("#########.", height=self.ROWS), height=self.ROWS)
        last = fp("I", 1, self.ROWS - 4, 9)
        s2 = rows_of([(self.ROWS - 3, 9), (self.ROWS - 2, 9), (self.ROWS - 1, 9)], height=self.ROWS)
        exp = explain_grid(s2, stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_lock_with_clear_unobserved_piece(self) -> None:
        # C3: a clearing lock by a never-observed piece on 12 rows.
        stack = rows_of(bottom_lines("#########.", height=self.ROWS), height=self.ROWS)
        s2 = rows_of([(self.ROWS - 3, 9), (self.ROWS - 2, 9), (self.ROWS - 1, 9)], height=self.ROWS)
        spawn = piece_cells("S", 0, 1, 4)
        exp = explain_grid(merge(s2, spawn), stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is not None
        assert exp.falling.piece == "S"

    def test_quad_clear_to_empty_12_row_board(self) -> None:
        stack = rows_of(
            bottom_lines("#########.", "#########.", "#########.", "#########.", height=self.ROWS),
            height=self.ROWS,
        )
        exp = explain_grid(self.EMPTY12, stack, None)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == self.EMPTY12
        assert exp.falling is None


class TestUnobservableCells:
    """``unknown_rows``: cells a game UI panel covers are evidence for
    nothing. They never add cells, never count as missing, may stand in for
    a hypothesis' hidden cells — and never make a frame UNEXPLAINED, which
    is what would eventually wipe the board through the reset debounce."""

    ROWS = 12
    EMPTY12: tuple[int, ...] = (0,) * 12
    # The ROAS Stacker corner: the NEXT preview covers rows 0-1, cols 8-9.
    CORNER = rows_of([(0, 8), (0, 9), (1, 8), (1, 9)], height=12)
    STACK = rows_of(bottom_lines("....######", "..########", height=12), height=12)

    def test_preview_pixels_in_the_corner_are_not_added_cells(self) -> None:
        # The NEXT piece drawn over the corner is not a second tetromino.
        observed = merge(self.STACK, [(0, 8), (1, 9)])
        exp = explain_grid(observed, self.STACK, None, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == self.STACK

    def test_believed_stack_under_the_corner_is_not_missing(self) -> None:
        # The committed stack believes (1,8)/(1,9) are filled; the frame
        # cannot show them. That is not a vanished stack.
        believed = merge(self.STACK, [(1, 8), (1, 9)])
        exp = explain_grid(self.STACK, believed, None, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == believed

    def test_piece_straddling_the_corner_is_occluded_not_unexplained(self) -> None:
        # A real O resting at rows 1-2 / cols 8-9 shows only its bottom
        # half. Forcing the covered cells empty made this a 2-cell fragment
        # -> UNEXPLAINED -> (4 identical frames) a spurious BOARD_RESET.
        observed = merge(self.STACK, piece_cells("O", 0, 1, 8))
        exp = explain_grid(observed, self.STACK, None, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.stack_rows == self.STACK  # memory held, nothing committed
        assert exp.falling is None  # O, J and L all fit: name nothing

    def test_unique_completion_identifies_the_piece(self) -> None:
        # One covered cell at (3,9): three cells of a flat I are visible and
        # only the I completes them, so the piece is named and positioned.
        # Clear of row 0, so the panel is the only hypothesis there is.
        unknown = rows_of([(3, 9)], height=self.ROWS)
        observed = merge(self.EMPTY12, [(3, 6), (3, 7), (3, 8)])
        exp = explain_grid(observed, self.EMPTY12, None, unknown_rows=unknown)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert (exp.falling.piece, exp.falling.row, exp.falling.col) == ("I", 3, 6)

    def test_a_panel_completion_does_not_outrank_the_top_edge(self) -> None:
        # The same fragment AT ROW 0 is not the panel's to answer alone: a
        # flat I with its fourth cell under the panel fits, and so do a T, a
        # J and an L entering from above. Four hypotheses about the same
        # three cells is an ambiguity, and it used to read as a confident I
        # — a wrong name AND a wrong position, in this session's own
        # geometry (the live mask covers rows 0-1, cols 8-9).
        observed = merge(self.EMPTY12, [(0, 5), (0, 6), (0, 7)])
        exp = explain_grid(observed, self.EMPTY12, None, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.falling is None

    def test_the_hint_selects_across_both_kinds_of_completion(self) -> None:
        # The tie above is broken by evidence where there is any: the
        # preview says a T is entering, exactly one candidate is a T, and
        # it is one entering from above — which is all the hint is evidence
        # about. The name is flagged as the hint's, not the frame's.
        observed = merge(self.EMPTY12, [(0, 5), (0, 6), (0, 7)])
        exp = explain_grid(
            observed, self.EMPTY12, None, unknown_rows=self.CORNER, entering_hint="T"
        )
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert (exp.falling.piece, exp.falling.row, exp.falling.col) == ("T", -1, 5)
        assert exp.hinted_name
        # "I" is the panel's candidate: a piece already on the board, which
        # the preview says nothing about. Still ambiguous.
        hinted_i = explain_grid(
            observed, self.EMPTY12, None, unknown_rows=self.CORNER, entering_hint="I"
        )
        assert hinted_i.kind is FrameKind.OCCLUDED
        assert hinted_i.falling is None

    def test_lock_reveal_commits_the_hidden_half_of_the_piece(self) -> None:
        # The O above locks and the next piece spawns: only 2 of the locked
        # cells are visible (6 added cells, not 8). The committed stack must
        # carry the hidden half, or the corner reads empty forever.
        last = fp("O", 0, 1, 8)
        observed = merge(self.STACK, [(2, 8), (2, 9)], piece_cells("T", 0, 0, 3))
        exp = explain_grid(observed, self.STACK, last, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(self.STACK, piece_cells("O", 0, 1, 8))
        assert exp.falling is not None
        assert exp.falling.piece == "T"

    def test_clearing_lock_verifies_around_the_covered_cells(self) -> None:
        # A clear shifts rows through the covered corner: the verification
        # must ignore those cells on both sides instead of failing on them.
        stack = rows_of(bottom_lines("#########.", height=self.ROWS), height=self.ROWS)
        last = fp("I", 1, self.ROWS - 4, 9)
        s2 = rows_of([(self.ROWS - 3, 9), (self.ROWS - 2, 9), (self.ROWS - 1, 9)], height=self.ROWS)
        observed = merge(s2, [(0, 8)])  # preview pixels in the corner
        exp = explain_grid(observed, stack, last, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2

    def test_a_new_world_is_still_unexplained(self) -> None:
        # The reset path must survive: an unrelated board shares no
        # explanation, covered corner or not.
        new_world = rows_of(bottom_lines("#.#.#.#.#.", "##.##.##.#", height=12), height=12)
        exp = explain_grid(new_world, self.STACK, None, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_fragment_far_from_the_corner_is_still_unexplained(self) -> None:
        # Occlusion tolerance is local: 2 stray cells that no piece could be
        # hiding behind the corner stay unexplained (and can still reset).
        observed = merge(self.STACK, [(5, 1), (5, 2)])
        exp = explain_grid(observed, self.STACK, None, unknown_rows=self.CORNER)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_without_unknown_rows_nothing_changes(self) -> None:
        # The default path is the fully-observed one: the same frame reads
        # as a plain falling O.
        observed = merge(self.STACK, piece_cells("O", 0, 1, 8))
        exp = explain_grid(observed, self.STACK, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "O"


def clipped_cells(piece: str, rotation_index: int, clip: int, left: int) -> list[Cell]:
    """The on-grid part of a piece whose top ``clip`` rows are above row 0."""
    return [
        (r - clip, left + c) for r, c in ROTATIONS[piece][rotation_index].cells if r - clip >= 0
    ]


class TestEnteringFromAbove:
    """The top edge of the board region cuts a piece that is entering.

    Pieces spawn above row 0, so the frame in which one appears shows only
    its bottom 1-3 cells. In a game with no gravity (drag to move, drag to
    drop) that fragment SITS there for seconds. Read as board content it is
    a broken tetromino -> UNEXPLAINED -> a BOARD_RESET storm that commits
    the fragment as phantom stack cells, which is what made every hint
    after the first one wrong on the real session (see
    tests/test_live_session.py). Read as what it is, it is the same partial
    observation as a piece under a UI panel.
    """

    STACK_CELLS = bottom_lines("###....###")
    STACK = rows_of(STACK_CELLS)

    @pytest.mark.parametrize("piece", PIECES)
    @pytest.mark.parametrize("clip", [1, 2, 3])
    def test_every_tetromino_clipped_by_one_two_or_three_rows(self, piece: str, clip: int) -> None:
        # The rule is total over the shape space: no clipped piece is ever
        # UNEXPLAINED, and none of them ever moves the committed stack.
        for rotation in ROTATIONS[piece]:
            if clip >= rotation.height:
                continue  # entirely above the board: nothing is observed
            visible = clipped_cells(piece, rotation.index, clip, 3)
            exp = explain_grid(merge(self.STACK, visible), self.STACK, None)
            assert exp.kind in (FrameKind.FALLING, FrameKind.OCCLUDED), (
                f"{piece} rot{rotation.index} clipped by {clip}: {exp.kind}"
            )
            assert exp.stack_rows == self.STACK
            if exp.falling is not None:
                # Named only when the fragment admits exactly one piece, and
                # then the name is the true one and the box starts off-grid.
                assert exp.falling.piece == piece
                assert exp.falling.row == -clip

    def test_a_unique_fragment_names_its_piece(self) -> None:
        # Three cells stacked in one column: only a vertical I has that as
        # its bottom, so the piece is named (and hinted) at once, from a
        # bounding box that starts one row above the board.
        visible = clipped_cells("I", 1, 1, 6)
        exp = explain_grid(merge(self.STACK, visible), self.STACK, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert (exp.falling.piece, exp.falling.row, exp.falling.col) == ("I", -1, 6)
        assert exp.stack_rows == self.STACK  # a piece in flight, not stack

    def test_an_ambiguous_fragment_holds_instead_of_guessing(self) -> None:
        # Two cells side by side at row 0: an O, an S, a Z, a J and an L all
        # fit. A guessed name is a guessed hint, so the frame is coherent
        # and nameless; the piece names itself as soon as it descends.
        exp = explain_grid(merge(self.STACK, [(0, 4), (0, 5)]), self.STACK, None)
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.falling is None
        assert exp.stack_rows == self.STACK

    def test_a_row_zero_stack_remnant_is_not_an_entering_piece(self) -> None:
        # The counter-case the airborne guard exists for: a column stacked to
        # the very top. Its row-0 cell RESTS on the stack, so it is board
        # content, not a spawn — never renamed, never deleted, and still
        # unexplained, so a genuinely new world can still reset the board.
        column = [(r, 3) for r in range(1, 20)]
        stack = rows_of(self.STACK_CELLS, column)
        exp = explain_grid(merge(stack, [(0, 3)]), stack, None)
        assert exp.kind is FrameKind.UNEXPLAINED
        assert exp.falling is None

    def test_a_fragment_hanging_over_the_stack_is_still_entering(self) -> None:
        # Airborne is about the cell directly below, not about height: a
        # piece entering over a tall column has air under it.
        column = [(r, 3) for r in range(2, 20)]
        stack = rows_of(self.STACK_CELLS, column)
        exp = explain_grid(merge(stack, [(0, 3)]), stack, None)
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.stack_rows == stack

    def test_lock_revealed_by_a_clipped_spawn_commits_only_the_lock(self) -> None:
        # The live failure, frame by frame: the I hard-drops to the floor and
        # the next piece appears in the same frame with only two cells on the
        # grid. 4 + 2 added cells is not two tetrominoes, so this used to be
        # UNEXPLAINED. The lock must commit; the fragment must NOT.
        lock = piece_cells("I", 0, 19, 0)
        observed = rows_of(lock, [(0, 4), (0, 5)])
        exp = explain_grid(observed, EMPTY, fp("I", 0, 1, 3))
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == rows_of(lock)  # the fragment is not in it
        assert exp.falling is None  # several pieces fit: name nothing

    def test_a_clipped_spawn_is_named_when_one_piece_fits(self) -> None:
        lock = piece_cells("I", 0, 19, 0)
        observed = rows_of(lock, clipped_cells("I", 1, 1, 6))
        exp = explain_grid(observed, EMPTY, fp("I", 0, 1, 3))
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == rows_of(lock)
        assert exp.falling is not None
        assert (exp.falling.piece, exp.falling.row) == ("I", -1)

    def test_a_clearing_lock_tolerates_a_clipped_spawn(self) -> None:
        # Same frame shape on the clear path: the row vanishes, the next
        # piece is already half on the grid. Without the rule the settled
        # post-clear frame is unexplainable and the board resets.
        stack = rows_of(bottom_lines("#########."))
        last = fp("I", 1, 16, 9)
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        exp = explain_grid(merge(s2, [(0, 4), (0, 5)]), stack, last)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == s2
        assert exp.falling is None

    def test_a_covered_panel_does_not_answer_before_the_top_edge(self) -> None:
        # A panel hypothesis and a top-edge one are hypotheses about the
        # SAME cells, so neither outranks the other. Three cells of a flat I
        # with (0,9) behind the panel used to be a named I because the panel
        # was asked first and its completion was unique THERE — while a T, a
        # J and an L cut by the top edge fit the identical cells. Four
        # candidates is an ambiguity, and OCCLUDED holds state.
        unknown = rows_of([(0, 9)], height=12)
        empty12: tuple[int, ...] = (0,) * 12
        observed = merge(empty12, [(0, 6), (0, 7), (0, 8)])
        exp = explain_grid(observed, empty12, None, unknown_rows=unknown)
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.falling is None
        # One row down there is no top-edge hypothesis at all, and the
        # panel's unique completion names the piece exactly as it always did.
        lower = merge(empty12, [(1, 6), (1, 7), (1, 8)])
        deep = explain_grid(lower, empty12, None, unknown_rows=rows_of([(1, 9)], height=12))
        assert deep.kind is FrameKind.FALLING
        assert deep.falling is not None
        assert (deep.falling.piece, deep.falling.row) == ("I", 1)

    def test_the_lock_reveal_reads_a_spawn_the_panel_can_explain(self) -> None:
        # The same ordering, on the path that verifies a lock: the spawn
        # revealing it may be cut by the top edge OR half under the panel,
        # and asking only the top edge names a piece that is not there.
        # Live geometry: a T spawning at (0,6) shows {(0,7), (1,6), (1,7)}
        # (its other cell is behind the panel), which the top-edge rule
        # alone read as a J at row -1 — a name and an off-grid position for
        # a piece sitting squarely on the board.
        corner = rows_of([(0, 8), (0, 9), (1, 8), (1, 9)], height=12)
        stack = rows_of(bottom_lines("####......", height=12), height=12)
        observed = merge(stack, piece_cells("O", 0, 10, 4), [(0, 7), (1, 6), (1, 7)])
        exp = explain_grid(observed, stack, None, unknown_rows=corner)
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == merge(stack, piece_cells("O", 0, 10, 4))
        assert exp.falling is None  # T, S and J all fit: the lock stands, unnamed


class TestTheEnteringPieceHint:
    """Naming a clipped fragment from the piece that LEFT the NEXT preview.

    A game deals the previewed piece and shows the one after it, so a
    preview that changes from X to Y is the game saying "X is the piece
    now entering". That is evidence, not a guess, and it is the only
    evidence there is about a fragment two cells wide: an O, an S, a Z, a
    J and an L all fit it. The hint may only SELECT among the completions
    the structural rule already accepts.
    """

    STACK_CELLS = bottom_lines("###....###")
    STACK = rows_of(STACK_CELLS)
    FRAGMENT = ((0, 4), (0, 5))

    @pytest.mark.parametrize("piece", ["O", "S", "Z", "J", "L"])
    def test_the_departing_preview_names_the_fragment(self, piece: str) -> None:
        observed = merge(self.STACK, self.FRAGMENT)
        exp = explain_grid(observed, self.STACK, None, entering_hint=piece)
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == piece
        # A real completion of THIS fragment, still in flight above row 0.
        assert exp.falling.row < 0
        assert {cell for cell in exp.falling.cells if cell[0] >= 0} == set(self.FRAGMENT)
        assert exp.stack_rows == self.STACK  # never stack content

    @pytest.mark.parametrize("piece", ["I", "T"])
    def test_a_name_no_completion_carries_changes_nothing(self, piece: str) -> None:
        # No I and no T can be cut into two cells side by side, so the
        # hint selects nothing and the frame is ambiguous exactly as before.
        observed = merge(self.STACK, self.FRAGMENT)
        exp = explain_grid(observed, self.STACK, None, entering_hint=piece)
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.falling is None

    def test_a_name_that_does_not_locate_the_piece_changes_nothing(self) -> None:
        # One cell at row 0 admits THREE T placements. The name is right
        # and the position is still unknown, so there is no hint to give.
        exp = explain_grid(merge(self.STACK, [(0, 4)]), self.STACK, None, entering_hint="T")
        assert exp.kind is FrameKind.OCCLUDED
        assert exp.falling is None

    def test_structure_outranks_the_preview(self) -> None:
        # Three cells stacked in one column are a vertical I and nothing
        # else. A preview saying "O" does not make them one.
        visible = clipped_cells("I", 1, 1, 6)
        exp = explain_grid(merge(self.STACK, visible), self.STACK, None, entering_hint="O")
        assert exp.kind is FrameKind.FALLING
        assert exp.falling is not None
        assert exp.falling.piece == "I"

    def test_the_hint_cannot_create_an_explanation(self) -> None:
        # Every guard still holds: a row-0 cell RESTING on the stack is
        # board content, and no preview reading turns it into a spawn.
        column = [(r, 3) for r in range(1, 20)]
        stack = rows_of(self.STACK_CELLS, column)
        exp = explain_grid(merge(stack, [(0, 3)]), stack, None, entering_hint="I")
        assert exp.kind is FrameKind.UNEXPLAINED
        assert exp.falling is None


class TestTheStackCarryingAPiece:
    """The committed stack was holding a piece that had not landed.

    The self-healing half of the absorbed-piece fix. However a falling
    piece gets into the committed stack, it used to be stuck there: its
    own motion reads as settled cells vanishing, which nothing explains,
    so the tracker resets and re-absorbs it one row lower, the whole way
    down (``tests/fixtures/absorbed_piece``). When one tetromino's worth of
    stack goes missing and that same piece is on the board somewhere else,
    the memory is what was wrong — so take the piece back out of the stack
    and read the frame as that piece falling.
    """

    STACK = rows_of(bottom_lines("#.########", "##.#######", "#########."))

    def carrying(self, piece: str, rotation: int, row: int, col: int) -> tuple[int, ...]:
        """The committed stack with a falling piece wrongly frozen into it."""
        return merge(self.STACK, piece_cells(piece, rotation, row, col))

    def test_the_carried_piece_falling_one_row_is_read_as_falling(self) -> None:
        carried = self.carrying("T", 0, 0, 3)
        exp = explain_grid(merge(self.STACK, piece_cells("T", 0, 1, 3)), carried, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK, "the piece is still in the stack"
        assert exp.falling is not None
        assert exp.falling.piece == "T"
        assert exp.falling.row == 1

    def test_the_carried_piece_dragged_sideways_is_read_as_falling(self) -> None:
        # This game drags rather than drops, so sideways is the common case.
        carried = self.carrying("T", 0, 0, 3)
        exp = explain_grid(merge(self.STACK, piece_cells("T", 0, 0, 6)), carried, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK
        assert exp.falling is not None
        assert (exp.falling.piece, exp.falling.col) == ("T", 6)

    def test_the_carried_piece_rotating_is_read_as_falling(self) -> None:
        carried = self.carrying("T", 0, 0, 3)
        exp = explain_grid(merge(self.STACK, piece_cells("T", 1, 0, 3)), carried, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK
        assert exp.falling is not None
        assert (exp.falling.piece, exp.falling.rotation_index) == ("T", 1)

    def test_the_whole_descent_heals_to_the_same_stack(self) -> None:
        # The property that kills the loop: every frame of the descent
        # names the same corrected stack, so the tracker's debounce
        # confirms it and the piece is out after one more frame — rather
        # than the board resetting once a row, all the way down.
        carried = self.carrying("O", 0, 0, 4)
        for row in range(1, 10):
            exp = explain_grid(merge(self.STACK, piece_cells("O", 0, row, 4)), carried, None)
            assert exp.kind is FrameKind.FALLING, row
            assert exp.stack_rows == self.STACK, row
            assert exp.falling is not None and exp.falling.piece == "O"

    def test_a_piece_that_merely_vanished_is_not_healed(self) -> None:
        # The "reappeared elsewhere" test doing its job. Cells the vision
        # lost leave nothing to find, and inventing a lock or a clear to
        # explain them would corrupt the stack. Holding is the answer.
        carried = self.carrying("T", 0, 0, 3)
        exp = explain_grid(self.STACK, carried, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_a_piece_that_reappears_under_another_name_is_not_healed(self) -> None:
        # A T leaving and an S arriving is not one piece moving.
        carried = self.carrying("T", 0, 0, 3)
        exp = explain_grid(merge(self.STACK, piece_cells("S", 0, 4, 3)), carried, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_a_piece_shaped_component_at_rest_is_stack(self) -> None:
        # The guard rail, and the direction this errs in: an O RESTING on
        # the stack landed there, so the memory is right and the frame is
        # what is wrong. Never delete a block the solver has to plan around.
        resting = merge(self.STACK, piece_cells("O", 0, HEIGHT - 5, 4))
        exp = explain_grid(merge(self.STACK, piece_cells("O", 0, 2, 4)), resting, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_more_than_one_tetrominos_worth_missing_is_never_healed(self) -> None:
        # The budget, which is what keeps clears, garbage and a new board
        # out of this rule: five missing cells are not one piece.
        carried = merge(self.carrying("T", 0, 0, 3), [(5, 0)])
        exp = explain_grid(merge(self.STACK, piece_cells("T", 0, 1, 3)), carried, None)
        assert exp.kind is FrameKind.UNEXPLAINED

    def test_a_real_line_clear_still_reads_as_a_clear(self) -> None:
        # Ordering, pinned: the clear rules run FIRST, so a frame a clear
        # explains is a lock with clears and never a healed stack.
        stack = rows_of(bottom_lines("#########."))
        settled = rows_of([(HEIGHT - 3, 9), (HEIGHT - 2, 9), (HEIGHT - 1, 9)])
        exp = explain_grid(settled, stack, fp("I", 1, HEIGHT - 4, 9))
        assert exp.kind is FrameKind.LOCKED
        assert exp.stack_rows == settled

    def test_a_piece_that_reappears_ABOVE_the_missing_cells_is_not_healed(self) -> None:
        # The position half of "it reappeared elsewhere". Pieces fall; a
        # tetromino of the same name eight rows UP and eight columns across
        # is somebody else, and vouching for the deletion with it costs
        # four real locked cells. Here the committed stack holds a genuine
        # post-clear island at rows 8-9 of cols 0-1, the frame loses it,
        # and an O is on the board near the top.
        island = merge(self.STACK, piece_cells("O", 0, 8, 0))
        exp = explain_grid(merge(self.STACK, piece_cells("O", 0, 0, 8)), island, None)
        assert exp.kind is FrameKind.UNEXPLAINED
        assert exp.stack_rows == island, "a real island was deleted"

    def test_one_dropped_cell_does_not_authorise_deleting_four(self) -> None:
        # Only the VANISHED cells need be part of the candidate — the real
        # absorbed frames depend on that, a piece read in part is still the
        # piece — so one dropped cell of a settled vertical I used to
        # delete all four while the other three were still lit on screen,
        # and read them back as a piece that had moved UP a row. Equality
        # is not the fix (it would refuse the real frames); continuity is.
        # Col 1 has a covered hole at row 17, so the I above it is airborne
        # in the stack without it — as any column over a hole is.
        column = [(r, 1) for r in range(HEIGHT - 7, HEIGHT - 3)]
        stack = merge(self.STACK, column)
        observed = merge(self.STACK, column[:-1], [(HEIGHT - 8, 1)])
        exp = explain_grid(observed, stack, None)
        assert exp.kind is FrameKind.UNEXPLAINED
        assert exp.stack_rows == stack

    def test_a_fade_frame_of_a_line_clear_is_not_a_carried_piece(self) -> None:
        # The case ordering cannot reach. _lock_with_clears explains the
        # SETTLED post-clear frame, not a half-faded row, so a clearing row
        # with its middle four cells blanked arrives here as four missing
        # cells that are tetromino-shaped and airborne over the cavity
        # beneath them — vouched for by the very I whose lock caused the
        # clear. Deleting them hands the solver a phantom four-wide gap to
        # aim at, on the frames where the real board is about to collapse.
        stack = rows_of(
            [(HEIGHT - 2, c) for c in range(WIDTH - 1)],
            [(HEIGHT - 1, c) for c in (0, 1, 6, 7, 8, 9)],
        )
        lit = merge(stack, piece_cells("I", 1, HEIGHT - 5, 9))
        fade = tuple(row & ~0b0000111100 if r == HEIGHT - 2 else row for r, row in enumerate(lit))
        exp = explain_grid(fade, stack, fp("I", 1, HEIGHT - 5, 9))
        assert exp.kind is FrameKind.UNEXPLAINED
        assert exp.stack_rows == stack, "the clearing row was deleted from the stack"

    def test_a_piece_dragged_to_the_next_column_is_still_healed(self) -> None:
        # ...and the measurement that keeps the position test honest: one
        # tick moves a piece at most two columns, but a vertical I stepping
        # one column and an O stepping two leave column spans that merely
        # TOUCH. Measured over 364 consecutive same-piece observations
        # across the four session replays, all three such steps are real.
        carried = self.carrying("I", 1, 0, 3)
        exp = explain_grid(merge(self.STACK, piece_cells("I", 1, 0, 2)), carried, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK
        carried_o = self.carrying("O", 0, 0, 6)
        exp = explain_grid(merge(self.STACK, piece_cells("O", 0, 0, 4)), carried_o, None)
        assert exp.kind is FrameKind.FALLING
        assert exp.stack_rows == self.STACK

    def test_covered_cells_are_never_evidence_that_a_piece_moved(self) -> None:
        # Unobservable cells read as empty, and "covered" must never be
        # read as "vanished" — that would let a panel delete the stack
        # under it one tetromino at a time.
        unknown = (0b1100000000, 0b1100000000) + (0,) * (HEIGHT - 2)
        carried = merge(self.STACK, piece_cells("O", 0, 0, 8))
        exp = explain_grid(self.STACK, carried, None, unknown_rows=unknown)
        assert exp.kind is FrameKind.QUIET
        assert exp.stack_rows == carried, "a covered piece was deleted from the stack"


class TestClearFullRowsMatchesCore:
    @pytest.mark.parametrize("piece", PIECES)
    def test_parity_with_board_drop(self, piece: str) -> None:
        stacks = (
            rows_of(bottom_lines("#########.")),
            rows_of(bottom_lines("########..", "########..")),
            rows_of(bottom_lines("#####.....")),
        )
        for stack in stacks:
            board = Board(stack)
            for rotation in ROTATIONS[piece]:
                for col in range(10 - rotation.width + 1):
                    dropped = board.drop(rotation, col)
                    if dropped is None:
                        continue
                    merged = merge(
                        stack,
                        [(dropped.landing_row + r, col + c) for r, c in rotation.cells],
                    )
                    assert clear_full_rows(merged) == dropped.board.rows
