"""GameStateTracker tests: debounce, lock/spawn/reset events, resyncs."""

import pytest

from tetris_coach.core.board import HEIGHT, WIDTH
from tetris_coach.vision.pieces_vision import FrameKind
from tetris_coach.vision.state import (
    MAX_PREVIEW_GAP,
    GameEvent,
    GameStateTracker,
    Snapshot,
)

from .boards import EMPTY, bottom_lines, grid_of, merge, piece_cells, rows_of


def feed(
    tracker: GameStateTracker,
    rows: tuple[int, ...],
    nxt: str | None,
    times: int = 1,
) -> list[GameEvent]:
    """Feed the same observation ``times`` times; return the LAST call's events."""
    events: list[GameEvent] = []
    for _ in range(times):
        events = tracker.update(grid_of(rows), nxt)
    return events


def attach(tracker: GameStateTracker, rows: tuple[int, ...], nxt: str | None = None) -> None:
    """Force-commit an arbitrary stack via the reset rule (test setup)."""
    for i in range(4):
        events = tracker.update(grid_of(rows), nxt)
        if i < 3:
            assert events == []
    assert events == [GameEvent.BOARD_RESET]
    assert tracker.committed.stack_rows == rows


class TestBootstrap:
    def test_initial_committed_is_empty(self) -> None:
        tracker = GameStateTracker()
        assert tracker.committed == Snapshot((0,) * HEIGHT, None, None)

    def test_first_spawn_commits_without_reset(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        spawn = rows_of(piece_cells("T", 0, 0, 3))
        assert feed(tracker, spawn, "I") == []
        events = feed(tracker, spawn, "I")
        assert events == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "T"
        assert tracker.committed.next_piece == "I"
        assert tracker.committed.stack_rows == EMPTY

    def test_confirm_frames_validation(self) -> None:
        with pytest.raises(ValueError):
            GameStateTracker(confirm_frames=0)
        with pytest.raises(ValueError):
            GameStateTracker(reset_confirm_frames=0)


class TestDebounce:
    def test_falling_piece_position_does_not_block_commit(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        for row in range(3):
            events = feed(tracker, rows_of(piece_cells("S", 0, row, 4)), "Z")
        assert tracker.committed.falling_piece == "S"
        assert events == []  # committed on the second frame already

    def test_flicker_frame_is_ignored(self) -> None:
        tracker = GameStateTracker()
        stack = rows_of(bottom_lines("####..##.."))
        attach(tracker, stack)
        observed = merge(stack, piece_cells("T", 0, 2, 4))
        feed(tracker, observed, "I", times=2)
        committed = tracker.committed
        # One-frame glitch (mid-clear capture: part of the stack missing).
        glitch = rows_of(bottom_lines("##........"))
        assert feed(tracker, glitch, "I") == []
        assert feed(tracker, observed, "I") == []
        assert tracker.committed == committed

    def test_raw_falling_is_last_coherent_observation(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        piece = rows_of(piece_cells("L", 0, 7, 3))
        tracker.update(grid_of(piece), None)
        falling = tracker.falling
        assert falling is not None
        assert (falling.piece, falling.row, falling.col) == ("L", 7, 3)
        # A torn frame does not overwrite the coherent observation.
        torn = rows_of([(5, 2), (5, 3), (5, 4), (6, 3), (6, 4)])
        tracker.update(grid_of(torn), None)
        assert tracker.falling == falling

    def test_next_piece_change_alone_commits_quietly(self) -> None:
        tracker = GameStateTracker()
        spawn = rows_of(piece_cells("I", 0, 0, 3))
        feed(tracker, spawn, None, times=2)
        events = feed(tracker, spawn, "T", times=2)
        assert events == []
        assert tracker.committed.next_piece == "T"


class TestLockDelay:
    """F1: a landed-but-unlocked piece stays FALLING; no early PIECE_LOCKED."""

    def test_lock_delay_end_to_end(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        events = feed(tracker, rows_of(piece_cells("T", 0, 1, 3)), "S", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]
        # Descend, then rest on the floor for several frames (lock delay).
        assert feed(tracker, rows_of(piece_cells("T", 0, 10, 3)), "S") == []
        resting = piece_cells("T", 0, 18, 3)
        for _ in range(4):
            assert feed(tracker, rows_of(resting), "S") == []
            assert tracker.committed.stack_rows == EMPTY
            assert tracker.committed.falling_piece == "T"
        falling = tracker.falling
        assert falling is not None
        assert (falling.row, falling.col) == (18, 3)
        # The next spawn reveals the lock: exactly one commit, both events.
        revealed = merge(rows_of(resting), piece_cells("S", 0, 0, 4))
        assert feed(tracker, revealed, "Z") == []
        events = feed(tracker, revealed, "Z")
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == rows_of(resting)
        assert tracker.committed.falling_piece == "S"

    def test_hard_drop_instant_spawn(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        feed(tracker, rows_of(piece_cells("T", 0, 1, 3)), "J", times=2)
        assert feed(tracker, rows_of(piece_cells("T", 0, 5, 3)), "J") == []
        # The T teleports to the bottom and the J spawns, same frame (L2).
        locked = piece_cells("T", 0, 18, 3)
        revealed = merge(rows_of(locked), piece_cells("J", 0, 0, 4))
        events = feed(tracker, revealed, "L", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == rows_of(locked)
        assert tracker.committed.falling_piece == "J"


class TestClearLocks:
    """F2: line-clearing locks are LOCKED, never BOARD_RESET."""

    def test_clear_lock_with_fade_frames_never_reset(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        stack = rows_of(bottom_lines("#########."))
        attach(tracker, stack, "I")
        all_events: list[GameEvent] = []

        def run(rows: tuple[int, ...], nxt: str | None, times: int = 1) -> list[GameEvent]:
            last: list[GameEvent] = []
            for _ in range(times):
                last = tracker.update(grid_of(rows), nxt)
                all_events.extend(last)
            return last

        # The I spawns and comes to rest completing row 19.
        run(rows_of(piece_cells("I", 1, 4, 9)), "S", times=2)
        resting = piece_cells("I", 1, 16, 9)
        run(merge(stack, resting), "S")
        committed = tracker.committed
        # Two morphing fade frames: no commit, committed untouched.
        fade1 = tuple(
            row & ~0b0000111100 if r == 19 else row for r, row in enumerate(merge(stack, resting))
        )
        fade2 = tuple(
            row & ~0b0111111110 if r == 19 else row for r, row in enumerate(merge(stack, resting))
        )
        assert run(fade1, "S") == []
        assert run(fade2, "S") == []
        assert tracker.committed == committed
        # Settled collapsed board plus the next spawn.
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        settled = merge(s2, piece_cells("S", 0, 0, 4))
        events = run(settled, "Z", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == s2
        assert GameEvent.BOARD_RESET not in all_events

    def test_clear_lock_with_unobserved_piece(self) -> None:
        # The piece was never seen in flight: tier C3 still verifies it.
        tracker = GameStateTracker(confirm_frames=2)
        stack = rows_of(bottom_lines("#########."))
        attach(tracker, stack, "I")
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        events = feed(tracker, s2, "S", times=2)
        assert events == [GameEvent.PIECE_LOCKED]
        assert tracker.committed.stack_rows == s2
        assert tracker.committed.falling_piece is None

    def test_zero_are_clear_flash_with_spawn_never_commits_full_row(self) -> None:
        # Zero-ARE game with a flash animation: the completed row is still
        # fully lit while the next piece is already visible. Those frames
        # must not commit (the stack would contain a full row about to
        # vanish); the settled frame then commits the lock exactly once.
        tracker = GameStateTracker(confirm_frames=2)
        stack = rows_of(bottom_lines("#########."))
        attach(tracker, stack, "I")
        feed(tracker, merge(stack, piece_cells("I", 1, 4, 9)), "J", times=2)
        flash = merge(stack, piece_cells("I", 1, 16, 9), piece_cells("J", 0, 0, 4))
        assert feed(tracker, flash, "L", times=2) == []
        assert tracker.committed.stack_rows == stack  # untouched mid-animation
        s2 = rows_of([(17, 9), (18, 9), (19, 9)])
        settled = merge(s2, piece_cells("J", 0, 1, 4))
        events = feed(tracker, settled, "L", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == s2
        assert tracker.committed.falling_piece == "J"

    def test_full_clear_to_empty_board(self) -> None:
        # Port of the old lock-with-line-clear test: stack 6 + 4 - 10 = 0.
        tracker = GameStateTracker(confirm_frames=2)
        stack = rows_of(bottom_lines("######...."))
        attach(tracker, stack, "I")
        feed(tracker, merge(stack, piece_cells("I", 0, 5, 4)), "J", times=2)
        resting = piece_cells("I", 0, 19, 6)
        feed(tracker, merge(stack, resting), "J")
        revealed = rows_of(piece_cells("J", 0, 0, 4))
        events = feed(tracker, revealed, "L", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == EMPTY
        assert tracker.committed.falling_piece == "J"


class TestPendingSurvivesTornFrames:
    def test_torn_frame_preserves_pending_count(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        resting = piece_cells("T", 0, 18, 3)
        feed(tracker, rows_of(resting), "S", times=2)
        revealed = merge(rows_of(resting), piece_cells("S", 0, 0, 4))
        assert feed(tracker, revealed, "Z") == []  # pending count 1
        torn = merge(rows_of(resting), [(5, 2), (5, 3), (5, 4), (6, 3), (6, 4)])
        assert feed(tracker, torn, "Z") == []  # UNEXPLAINED: pending kept
        events = feed(tracker, revealed, "Z")  # third update commits
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]

    def test_different_coherent_candidate_resets_pending(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        resting = piece_cells("T", 0, 18, 3)
        feed(tracker, rows_of(resting), "S", times=2)
        revealed = merge(rows_of(resting), piece_cells("S", 0, 0, 4))
        assert feed(tracker, revealed, "Z") == []  # pending count 1
        # A coherent candidate differing in next: replaces pending (count 1).
        assert feed(tracker, revealed, "L") == []
        events = feed(tracker, revealed, "L")
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.next_piece == "L"


class TestResetRules:
    def _committed_mid_game(self, tracker: GameStateTracker) -> tuple[int, ...]:
        stack = rows_of(bottom_lines("####..####", "#########."))
        attach(tracker, stack, "I")
        feed(tracker, merge(stack, piece_cells("S", 0, 1, 4)), "Z", times=2)
        return stack

    def test_stable_empty_board_resets_on_fourth_frame(self) -> None:
        tracker = GameStateTracker()
        self._committed_mid_game(tracker)
        for _ in range(3):
            assert feed(tracker, EMPTY, None) == []
        events = feed(tracker, EMPTY, None)
        assert events == [GameEvent.BOARD_RESET]
        assert tracker.committed == Snapshot(EMPTY, None, None)
        assert tracker.falling is None
        # A following spawn commits separately.
        events = feed(tracker, rows_of(piece_cells("I", 0, 0, 3)), "O", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]

    def test_three_unexplained_frames_then_recovery(self) -> None:
        tracker = GameStateTracker()
        stack = self._committed_mid_game(tracker)
        for _ in range(3):
            assert feed(tracker, EMPTY, None) == []
        # Back to normal: no events ever, and the counter restarted.
        assert feed(tracker, merge(stack, piece_cells("S", 0, 8, 4)), "Z") == []
        for _ in range(3):
            assert feed(tracker, EMPTY, None) == []
        assert feed(tracker, merge(stack, piece_cells("S", 0, 8, 4)), "Z") == []

    def test_morphing_unexplained_frames_never_reset(self) -> None:
        tracker = GameStateTracker()
        self._committed_mid_game(tracker)
        blob_a = rows_of([(5, 2), (5, 3), (5, 4), (6, 3), (6, 4)])
        blob_b = rows_of([(8, 6), (8, 7), (8, 8), (9, 7), (9, 8)])
        for _ in range(4):
            assert feed(tracker, blob_a, None) == []
            assert feed(tracker, blob_b, None) == []

    def test_rising_garbage_reanchors(self) -> None:
        tracker = GameStateTracker()
        stack = rows_of(bottom_lines("####..####", "#########."))
        attach(tracker, stack, "I")
        shifted = tuple(list(stack[1:]) + [0b1111111101])  # garbage row, hole col 1
        for _ in range(3):
            assert feed(tracker, shifted, "I") == []
        events = feed(tracker, shifted, "I")
        assert events == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == shifted
        events = feed(tracker, merge(shifted, piece_cells("T", 0, 0, 3)), "S", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]


class TestHoldSwap:
    def test_hold_swap_spawns_only(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        stack = rows_of(bottom_lines("#####....."))
        attach(tracker, stack, "I")
        feed(tracker, merge(stack, piece_cells("T", 0, 2, 3)), "I", times=2)
        events = feed(tracker, merge(stack, piece_cells("Z", 0, 1, 4)), "I", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "Z"
        assert tracker.committed.stack_rows == stack

    def test_same_name_swap_is_invisible(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        feed(tracker, rows_of(piece_cells("T", 0, 2, 3)), "I", times=2)
        assert feed(tracker, rows_of(piece_cells("T", 0, 1, 5)), "I", times=2) == []

    def test_same_piece_kind_twice_after_lock_still_spawns(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        feed(tracker, rows_of(piece_cells("T", 0, 1, 3)), "T", times=2)
        resting = piece_cells("T", 0, 18, 3)
        feed(tracker, rows_of(resting), "T")
        revealed = merge(rows_of(resting), piece_cells("T", 0, 0, 4))
        events = feed(tracker, revealed, "I", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]


class TestUnobservedPieceSemantics:
    def test_piece_appearing_at_rest_spawns_then_locks_on_reveal(self) -> None:
        # Deliberate semantics shift: a piece that appears already at rest
        # is FALLING (SPAWNED on commit); PIECE_LOCKED fires when its cells
        # become immutable-and-explained — here, at the next spawn.
        tracker = GameStateTracker(confirm_frames=2)
        stack = rows_of(bottom_lines("#####....."))
        attach(tracker, stack, "T")
        resting = piece_cells("O", 0, 18, 6)
        events = feed(tracker, merge(stack, resting), "S", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == stack
        revealed = merge(stack, resting, piece_cells("S", 0, 0, 4))
        events = feed(tracker, revealed, "Z", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == merge(stack, resting)


class TestTwelveRowTracker:
    """A tracker constructed for a 12-row board: the rows parameter seeds
    the bootstrap snapshot; everything after flows from the frames."""

    ROWS = 12
    EMPTY12: tuple[int, ...] = (0,) * 12

    def test_bootstrap_snapshot_matches_rows(self) -> None:
        tracker = GameStateTracker(rows=self.ROWS)
        assert tracker.committed == Snapshot(self.EMPTY12, None, None)

    def test_rows_validation(self) -> None:
        with pytest.raises(ValueError):
            GameStateTracker(rows=0)

    def test_spawn_lock_reveal_on_12_rows(self) -> None:
        tracker = GameStateTracker(confirm_frames=2, rows=self.ROWS)
        spawn = rows_of(piece_cells("T", 0, 0, 3), height=self.ROWS)
        events = feed(tracker, spawn, "S", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "T"
        # Rest on the 12-row floor through lock delay, then the next spawn
        # reveals the lock (exercises the len(stack_rows) support bound).
        resting = piece_cells("T", 0, self.ROWS - 2, 3)
        assert feed(tracker, rows_of(resting, height=self.ROWS), "S") == []
        assert tracker.committed.stack_rows == self.EMPTY12
        revealed = merge(rows_of(resting, height=self.ROWS), piece_cells("S", 0, 0, 5))
        events = feed(tracker, revealed, "Z", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        assert tracker.committed.stack_rows == rows_of(resting, height=self.ROWS)
        assert tracker.committed.falling_piece == "S"

    def test_board_reset_on_12_rows(self) -> None:
        tracker = GameStateTracker(rows=self.ROWS)
        stack = rows_of(bottom_lines("####..####", height=self.ROWS), height=self.ROWS)
        attach(tracker, stack)
        for _ in range(3):
            assert feed(tracker, self.EMPTY12, None) == []
        assert feed(tracker, self.EMPTY12, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed == Snapshot(self.EMPTY12, None, None)


class TestMidGameAttach:
    def test_attach_resyncs_and_resumes_tracking(self) -> None:
        tracker = GameStateTracker()
        stack = rows_of(bottom_lines("#.########", "##.#######", "#########."))
        # A piece still in flight while we attach: morphing frames, no reset.
        for row in (3, 6, 9):
            assert feed(tracker, merge(stack, piece_cells("L", 0, row, 4)), "T") == []
        # The piece comes to rest; the stable board resyncs (piece folded in).
        resting = merge(stack, piece_cells("L", 0, 15, 4))
        for _ in range(3):
            assert feed(tracker, resting, "T") == []
        assert feed(tracker, resting, "T") == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == resting
        # The successor spawns; normal tracking resumes.
        events = feed(tracker, merge(resting, piece_cells("T", 0, 0, 3)), "S", times=2)
        assert events == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "T"


class TestUnobservableCells:
    """A tracker told which cells a game UI panel covers: the reset debounce
    must not fire on a piece hiding under it, and the committed stack must
    not claim those cells are empty."""

    ROWS = 12
    # The ROAS Stacker corner: the NEXT preview covers rows 0-1, cols 8-9.
    CORNER = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})
    # Near top-out on the right, so a piece can come to rest at rows 1-2 of
    # cols 8-9 — half under the preview box.
    STACK = rows_of(
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

    def _tracker(self) -> GameStateTracker:
        tracker = GameStateTracker(confirm_frames=2, rows=self.ROWS, unobservable_cells=self.CORNER)
        attach(tracker, self.STACK)
        return tracker

    def test_covered_cells_become_row_bitmasks(self) -> None:
        tracker = GameStateTracker(rows=self.ROWS, unobservable_cells=self.CORNER)
        assert tracker.unknown_rows == (0b1100000000, 0b1100000000) + (0,) * 10
        assert GameStateTracker(rows=self.ROWS).unknown_rows == (0,) * self.ROWS

    def test_preview_pixels_never_reach_the_committed_stack(self) -> None:
        # Whatever the NEXT preview draws in the corner is discarded.
        tracker = self._tracker()
        assert feed(tracker, merge(self.STACK, [(0, 8), (1, 9)]), "T", times=6) == []
        assert tracker.committed.stack_rows == self.STACK

    def test_resting_piece_in_the_corner_never_resets_the_board(self) -> None:
        # The regression: an O resting at rows 1-2 / cols 8-9 shows only 2
        # cells. Forced empty that was a broken tetromino, and 4 identical
        # frames of it wiped the board.
        tracker = self._tracker()
        observed = merge(self.STACK, piece_cells("O", 0, 1, 8))
        for _ in range(10):
            assert feed(tracker, observed, "T") == []
        assert tracker.committed.stack_rows == self.STACK

    def test_lock_half_under_the_corner_is_still_a_verified_lock(self) -> None:
        # The same O locks and a T spawns. The reveal carries 6 added cells,
        # not 8, and the locked piece's visible half is not a tetromino:
        # without the partial-lock rule the frame would be UNEXPLAINED and
        # the next resting piece would reset the board all over again.
        tracker = self._tracker()
        resting = merge(self.STACK, piece_cells("O", 0, 1, 8))
        feed(tracker, resting, "T", times=4)
        revealed = merge(resting, piece_cells("T", 0, 0, 3))
        events = feed(tracker, revealed, "S", times=2)
        assert events == [GameEvent.PIECE_LOCKED, GameEvent.PIECE_SPAWNED]
        # Committed: the cells actually seen. O, J and L all fit the visible
        # half, so the covered cells stay a belief rather than a guess —
        # the solver-side policy (CoachEngine._solver_board) is what keeps
        # a hint out of them.
        assert tracker.committed.stack_rows == merge(self.STACK, [(2, 8), (2, 9)])
        assert tracker.committed.falling_piece == "T"

    def test_a_genuinely_new_board_still_resets(self) -> None:
        tracker = self._tracker()
        new_world = rows_of(bottom_lines("#.#.#.#.#.", "##.##.##.#", height=12), height=12)
        for _ in range(3):
            assert feed(tracker, new_world, None) == []
        assert feed(tracker, new_world, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == new_world

    def test_reset_seeds_the_hidden_belief_empty(self) -> None:
        # The observed value in a covered cell is meaningless, so a resync
        # commits nothing there whatever the preview happened to be drawing.
        tracker = self._tracker()
        board = rows_of(bottom_lines("#.#.#.#.#.", "##.##.##.#", height=12), height=12)
        feed(tracker, merge(board, list(self.CORNER)), None, times=4)
        assert tracker.committed.stack_rows == board


class TestTheEnteringPieceHint:
    """The tracker watches the preview so it can name what the top edge cuts.

    A game deals the previewed piece and shows the one after it, so a
    preview that changes from X to Y says X is the piece now entering the
    board. Until it does, a fragment two cells wide is nameless (an O, an
    S, a Z, a J and an L all fit) — and in a game that parks spawns at the
    top edge, nameless means no hint for as long as the piece sits there.
    """

    STACK = rows_of(bottom_lines("###....###"))
    FRAGMENT = ((0, 4), (0, 5))
    # A piece entering EDGE-ON: two cells in one column, which a vertical
    # I fits — and so do a J, an L, an S and a Z, so shape alone still
    # names nothing.
    COLUMN = ((0, 4), (1, 4))

    def test_the_departing_preview_names_the_entering_piece(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")  # the preview holds the O
        entering = merge(self.STACK, self.FRAGMENT)
        assert feed(tracker, entering, "I") == []  # the O is dealt: preview -> I
        assert feed(tracker, entering, "I") == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "O"
        assert tracker.committed.stack_rows == self.STACK  # still not stack

    def test_the_named_piece_is_the_one_that_descends(self) -> None:
        # The proof the name was right rather than merely early: the same
        # piece drops into full view, where the ordinary shape rule names
        # it from four cells — and it is the O, with no second spawn.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        feed(tracker, merge(self.STACK, self.FRAGMENT), "I", times=2)
        descended = merge(self.STACK, piece_cells("O", 0, 2, 4))
        assert feed(tracker, descended, "I", times=2) == []
        assert tracker.committed.falling_piece == "O"

    def test_a_preview_that_was_never_known_names_nothing(self) -> None:
        # None -> X is not a deal: the preview box is unreadable on plenty
        # of frames, and nothing about the piece entering follows from it.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        entering = merge(self.STACK, self.FRAGMENT)
        assert feed(tracker, entering, "I", times=3) == []
        assert tracker.committed.falling_piece is None
        assert tracker.committed.stack_rows == self.STACK

    def test_a_steady_preview_names_nothing(self) -> None:
        # And neither does a preview that never changes: no deal, no name.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        assert feed(tracker, merge(self.STACK, self.FRAGMENT), "O", times=3) == []
        assert tracker.committed.falling_piece is None

    def test_a_hinted_name_is_a_hint_and_not_an_observation(self) -> None:
        # The hint may be wrong (the preview is read by the same vision as
        # everything else), and a wrong name must cost no more than a wrong
        # hint. explain_grid refuses a lock whose piece does not match the
        # last OBSERVED one, so if a hint-named fragment became the last
        # observation, a misnamed piece would block its own lock: four
        # identical unexplainable frames, a spurious BOARD_RESET, and a
        # blank overlay. Here the preview says O and the piece is an S.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        feed(tracker, merge(self.STACK, self.FRAGMENT), "I", times=2)
        assert tracker.committed.falling_piece == "O"  # hinted, and shown
        # The S hard-drops to the floor and the next piece enters behind it.
        locked = piece_cells("S", 0, HEIGHT - 2, 3)
        dropped = merge(self.STACK, locked, self.FRAGMENT)
        assert feed(tracker, dropped, "I") == []
        assert feed(tracker, dropped, "I") == [GameEvent.PIECE_LOCKED]
        assert tracker.committed.stack_rows == merge(self.STACK, locked)
        # ...and the frame never reaches the reset debounce.
        assert feed(tracker, dropped, "I", times=4) == []
        assert tracker.committed.stack_rows == merge(self.STACK, locked)
        # The mechanism: the hinted name was shown, never remembered.
        assert tracker.falling is None or tracker.falling.piece != "O"

    def test_a_hint_does_not_outlive_the_deal_it_names(self) -> None:
        # The preview is unreadable in bursts, and it only ever reports a
        # deal by CHANGING: if a burst covers one previewed piece's whole
        # tenure, the next change is read a deal late and names the piece
        # dealt one turn earlier. Left standing, that hint goes on naming
        # every fragment the top edge cuts for the rest of the session, so
        # it is evidence about ONE deal and expires with it: a lock is the
        # game dealing again, and a hint that reaches a second one without
        # the preview having changed is stale.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        fragment = merge(self.STACK, self.FRAGMENT)
        feed(tracker, fragment, "I", times=2)  # the O is dealt: preview -> I
        assert tracker.committed.falling_piece == "O"
        # The O locks and the next piece enters. The preview never changes
        # again (unreadable, so the tracker is told nothing).
        first = merge(self.STACK, piece_cells("O", 0, 18, 4), self.FRAGMENT)
        assert feed(tracker, first, "I", times=2) == [GameEvent.PIECE_LOCKED]
        # That piece locks too: the second deal since the last thing the
        # preview said, and the hint does not survive it.
        second = merge(self.STACK, piece_cells("O", 0, 18, 4), piece_cells("O", 0, 16, 4))
        assert feed(tracker, merge(second, self.FRAGMENT), "I", times=2) == [GameEvent.PIECE_LOCKED]
        assert tracker.committed.stack_rows == second
        # ...so the fragment sitting at the top edge is nameless again,
        # instead of being called an O for the rest of the session.
        assert feed(tracker, merge(second, self.FRAGMENT), "I", times=3) == []
        assert tracker.committed.falling_piece is None

    def test_a_flip_the_box_is_readable_on_both_sides_of_names_the_piece(self) -> None:
        # The flip the accelerator runs on, in its plainest form: the box
        # is read on two consecutive frames and the value changes between
        # them. Nothing can have been dealt in between except the piece
        # that left, so the column at the top edge — which a vertical I, a
        # J, an L, an S and a Z all fit — is the I.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        column = merge(self.STACK, self.COLUMN)
        assert feed(tracker, column, "I") == []  # None -> I reports no deal
        assert feed(tracker, column, "Z") == []  # I -> Z: the I is entering
        assert feed(tracker, column, "Z") == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "I"

    def test_a_flip_read_after_the_box_went_dark_names_nothing(self) -> None:
        # The same flip, with the box dark across the frames it happened
        # in. A flip only ever says "the piece that was here has been
        # dealt" — it does not say WHEN, and across a dark burst long
        # enough to cover a whole tenure the value has moved on twice, so
        # the piece it names entered a deal ago and is already on the
        # board. That is the "confused two pieces" failure. What rules it
        # out is the LENGTH of the gap the flip is read across: a piece
        # that came and went inside the gap held the box for the whole of
        # it, so a gap shorter than a tenure cannot hide a deal. Here the
        # box shows the I and goes dark for longer than the budget:
        # nothing says whether the I or something after it is the column
        # at the top edge, so nothing is named.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        column = merge(self.STACK, self.COLUMN)
        assert feed(tracker, column, "I") == []
        assert feed(tracker, column, None, times=MAX_PREVIEW_GAP + 1) == []
        assert feed(tracker, column, "Z", times=3) == []
        assert tracker.committed.falling_piece is None

    def test_a_flip_read_across_a_short_dark_gap_still_names_the_piece(self) -> None:
        # ...and the other side of that budget, which is the case the
        # user's own window turns on: the box is unreadable for a few
        # frames (the game wipes the field, and the wipe blanks the box
        # along with the board), and comes back changed. No previewed
        # piece can have come and gone in six frames — a tenure is an
        # order longer — so the flip still names the piece that left, and
        # the fragment the wipe dealt is hinted instead of sitting
        # nameless for a second.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        column = merge(self.STACK, self.COLUMN)
        assert feed(tracker, column, "I") == []
        assert feed(tracker, column, None, times=6) == []
        assert feed(tracker, column, "Z", times=2) == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "I"

    def test_the_gap_is_counted_in_captures_the_board_gate_never_saw(self) -> None:
        # The gap is in CAPTURES, not in frames the board's confidence
        # gate liked — and the consumer returns before update() on every
        # rejection, so without observe_preview the clock stops for
        # exactly the events that make the box unreadable (a wipe, a
        # flash, an animation: they reject the board too). A deal hidden
        # inside such a burst would then be invisible and the flip on the
        # far side would name a piece a deal stale. Here the box is read,
        # a long burst of captures is rejected, and the flip that follows
        # is refused on its age even though the tracker saw no frames in
        # between.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        column = merge(self.STACK, self.COLUMN)
        assert feed(tracker, column, "I") == []
        for _ in range(MAX_PREVIEW_GAP + 1):
            tracker.observe_preview(None)  # rejected captures, box unreadable
        assert feed(tracker, column, "Z", times=3) == []
        assert tracker.committed.falling_piece is None

    def test_a_readable_box_dates_the_flip_even_while_the_board_is_rejected(self) -> None:
        # ...and the same burst with the box READABLE through it keeps the
        # evidence: the rejection was about the board image, and the
        # preview says the same piece throughout, so the flip on the far
        # side is one capture old and names what left.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        column = merge(self.STACK, self.COLUMN)
        assert feed(tracker, column, "I") == []
        for _ in range(30):
            tracker.observe_preview("I")  # rejected captures, box still the I
        assert feed(tracker, column, "Z", times=2) == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "I"

    def test_a_hint_whose_own_deal_ended_under_it_names_no_later_fragment(self) -> None:
        # A hint is evidence about ONE deal and expires with it. The
        # preview reports a deal by changing at the moment the new piece
        # spawns, which is the frame a lock commit is debouncing, so the
        # hint of the deal now beginning is always the young one; a hint
        # older than that debounce names the piece that has just LOCKED.
        # Here the preview goes dark the moment the O is dealt, so no
        # fresh hint arrives, and the fragment entering behind the O must
        # not inherit the O's name.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        entering = merge(self.STACK, self.FRAGMENT)
        feed(tracker, entering, "I", times=2)  # the O is dealt: preview O -> I
        assert tracker.committed.falling_piece == "O"
        # The O hard-drops and locks, with the next piece already at the
        # top edge behind it.
        dropped = merge(self.STACK, piece_cells("O", 0, 18, 4), self.FRAGMENT)
        assert feed(tracker, dropped, "I", times=2) == [GameEvent.PIECE_LOCKED]
        assert tracker.committed.stack_rows == merge(self.STACK, piece_cells("O", 0, 18, 4))
        # Counting locks let this one through: exactly one lock has
        # happened since the hint was set either way. Its AGE is what
        # separates them.
        assert feed(tracker, dropped, "I", times=3) == []
        assert tracker.committed.falling_piece is None

    def test_a_preview_that_disagrees_with_the_descending_piece_is_dropped(self) -> None:
        # A preview reading is a HYPOTHESIS, not an observation, and this
        # is what happens when it is wrong: the box is misread as an O,
        # and an L is what actually comes in. Its first two cells fit an
        # O, so for one commit the coach says O — and its next row down
        # does not fit an O at all, which falsifies the hypothesis. It is
        # dropped on that frame and the fragment names itself from shape,
        # so a wrong reading costs two frames of a wrong name rather than
        # a whole deal of one.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        two = merge(self.STACK, ((0, 4), (0, 5)))
        assert feed(tracker, two, "I", times=2) == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "O"  # the hypothesis, shown
        three = merge(self.STACK, ((0, 4), (1, 4), (1, 5)))
        assert feed(tracker, three, "I", times=2) == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "L"
        # Dropped, not merely out-voted: nothing named from it again.
        assert tracker._entering_hint is None
        # ...and the whole piece, when it arrives, agrees with the frame
        # that corrected it, with no second correction.
        whole = merge(self.STACK, piece_cells("L", 1, 0, 4))
        assert feed(tracker, whole, "I", times=2) == []
        assert tracker.committed.falling_piece == "L"

    def test_a_refuted_name_is_taken_back_and_not_merely_dropped(self) -> None:
        # Dropping the hypothesis is not the same as retracting the name it
        # already gave the piece on screen. The frame that refutes a hint
        # is usually one that names no replacement — the piece has shown a
        # second cell, which rules the hinted name out while still fitting
        # several others — so it is OCCLUDED, and OCCLUDED HOLDS the
        # committed snapshot. The refuted name used to sit there for the
        # rest of the piece's tenure at the top edge (14-17 frames in this
        # game, ~1 s), which is the user's "confused two pieces" report
        # made persistent. Here the box says T and a vertical I comes in.
        tracker = GameStateTracker(confirm_frames=2, rows=12)
        empty = (0,) * 12
        feed(tracker, empty, "T", times=2)  # the box holds the T
        one = rows_of([(0, 0)], height=12)
        assert feed(tracker, one, "I") == []  # the T is dealt: preview -> I
        assert feed(tracker, one, "I") == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "T"  # the hypothesis, shown
        # The piece descends: a second cell in the same column, which no T
        # fits and several other pieces do. The name goes back to nothing.
        two = rows_of([(0, 0), (1, 0)], height=12)
        assert feed(tracker, two, "I") == [GameEvent.PIECE_UNNAMED]
        assert tracker.last_kind is FrameKind.OCCLUDED
        assert tracker.committed.falling_piece is None
        assert tracker.committed.stack_rows == empty  # nothing structural
        # ...and it stays withdrawn while the piece is still unnameable.
        assert feed(tracker, two, "I", times=4) == []
        assert tracker.committed.falling_piece is None

    def test_the_piece_names_itself_again_once_it_is_seen_whole(self) -> None:
        # The end of that story: a withdrawn name is not a dead end. The
        # same fragment keeps descending, the shape rule reaches it, and
        # it commits under its real name with no reset in between.
        tracker = GameStateTracker(confirm_frames=2, rows=12)
        feed(tracker, (0,) * 12, "T", times=2)
        feed(tracker, rows_of([(0, 0)], height=12), "I", times=2)
        assert feed(tracker, rows_of([(0, 0), (1, 0)], height=12), "I") == [GameEvent.PIECE_UNNAMED]
        whole = rows_of(piece_cells("I", 1, 0, 0), height=12)
        assert feed(tracker, whole, "I", times=2) == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "I"

    def test_a_hinted_name_structure_confirms_stops_being_a_hypothesis(self) -> None:
        # ...and a hinted name becomes an observation the moment the shape
        # rule agrees with it, so a LATER fragment cannot retract it. The
        # O is hinted at the top edge, descends into full view where its
        # four cells name it, locks, and the next piece's first cells —
        # which no O fits — must not take the O's name back.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        feed(tracker, merge(self.STACK, self.FRAGMENT), "I", times=2)
        assert tracker.committed.falling_piece == "O"
        descended = merge(self.STACK, piece_cells("O", 0, 2, 4))
        feed(tracker, descended, "I", times=2)
        behind = merge(descended, ((0, 0), (1, 0)))
        assert feed(tracker, behind, "I", times=3) == []
        assert tracker.committed.falling_piece == "O"

    def test_a_wrong_preview_authorises_nothing_structural(self) -> None:
        # The cost ceiling on a wrong reading. A hinted name is never an
        # observation, so it can vouch for no lock; and a frame it cannot
        # name is OCCLUDED, which never counts toward the reset debounce.
        # The whole sequence above, held long enough to trip both, moves
        # neither the committed stack nor anything structural.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        events = []
        for rows in (
            merge(self.STACK, ((0, 4), (0, 5))),
            merge(self.STACK, ((0, 4), (1, 4), (1, 5))),
            merge(self.STACK, piece_cells("L", 1, 0, 4)),
        ):
            for _ in range(6):
                events += feed(tracker, rows, "I")
        assert GameEvent.PIECE_LOCKED not in events
        assert GameEvent.BOARD_RESET not in events
        assert tracker.committed.stack_rows == self.STACK
        assert tracker.falling is None or tracker.falling.piece != "O"

    def test_a_resync_drops_the_hint(self) -> None:
        # A board reset is a new world (a new game, garbage, a mid-game
        # attach). What the preview shows still holds; which piece was
        # dealt into THIS board does not.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        feed(tracker, merge(self.STACK, self.FRAGMENT), "I", times=2)
        assert tracker.committed.falling_piece == "O"
        new_world = rows_of(bottom_lines("#.#.#.#.#.", "##.##.##.#"))
        assert feed(tracker, new_world, "I", times=4) == [GameEvent.BOARD_RESET]
        assert feed(tracker, merge(new_world, self.FRAGMENT), "I", times=3) == []
        assert tracker.committed.falling_piece is None

    def test_a_piece_seen_whole_is_still_an_observation(self) -> None:
        # The other half of the rule: a name the frame's own structure
        # produced is evidence exactly as it always was.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "O")
        feed(tracker, merge(self.STACK, piece_cells("S", 0, 2, 3)), "I", times=2)
        assert tracker.committed.falling_piece == "S"
        assert tracker.falling is not None
        assert tracker.falling.piece == "S"


class TestAStackCarryingAPiece:
    """The loop cannot persist even if a piece does get into the stack.

    The self-healing half of the absorbed-piece fix, at the tracker level.
    Before it, a committed stack holding a falling piece produced one
    BOARD_RESET per row of that piece's descent — 58 frames, ~3.9 s, with
    no hint at all on the session that produced
    ``tests/fixtures/absorbed_piece``.

    The piece gets in here the way a resync can still let one in: leaning
    against a column, so it is part of a component that reaches the floor
    and nothing marks it as in flight. It has air underneath all the same,
    which is what the healing rule keys on.
    """

    STACK = rows_of(
        [(r, 0) for r in range(HEIGHT - 10, HEIGHT)],
        bottom_lines("#.########"),
    )
    CARRIED = merge(STACK, piece_cells("O", 0, HEIGHT - 10, 1))

    def test_a_carried_piece_is_taken_back_out_and_tracked(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.CARRIED, "I")
        all_events: list[GameEvent] = []
        for row in range(HEIGHT - 9, HEIGHT - 5):
            all_events += feed(tracker, merge(self.STACK, piece_cells("O", 0, row, 1)), "I")
        assert GameEvent.BOARD_RESET not in all_events, "the reset loop is back"
        assert tracker.committed.stack_rows == self.STACK
        assert tracker.committed.falling_piece == "O"

    def test_the_healed_piece_then_locks_onto_the_corrected_stack(self) -> None:
        # The end of the story: with the stack corrected, the descent ends
        # in an ordinary verified lock rather than another resync.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.CARRIED, "I")
        for row in range(HEIGHT - 9, HEIGHT - 3):
            feed(tracker, merge(self.STACK, piece_cells("O", 0, row, 1)), "I")
        landed = merge(self.STACK, piece_cells("O", 0, HEIGHT - 3, 1))
        feed(tracker, landed, "I", times=2)
        assert tracker.committed.stack_rows == self.STACK
        revealed = merge(landed, piece_cells("I", 0, 0, 3))
        assert feed(tracker, revealed, "S", times=2) == [
            GameEvent.PIECE_LOCKED,
            GameEvent.PIECE_SPAWNED,
        ]
        assert tracker.committed.stack_rows == landed


class TestResyncAndThePieceInFlight:
    """A resync must not freeze a piece that has not landed.

    A mid-game attach adopts the observed board wholesale — there is no
    memory to diff it against. What it must leave out is content that is
    demonstrably not settled: a component floating AT ROW 0, where settled
    content cannot be. Freezing a piece in flight into the stack plants
    blocks no lock ever put there, and worse, makes the piece's own descent
    read as stack cells vanishing — which nothing can explain, so the
    tracker resets again and re-absorbs it one row lower, all the way down
    (``tests/fixtures/absorbed_piece``).

    Floating alone is not enough, and the tests below pin both halves:
    naive gravity leaves real locked cells hanging over holes after a line
    clear, so content that floats LOWER DOWN is adopted like the rest of
    the board. A piece absorbed there is not stranded — its next
    descending frame takes it back out (``_carried_piece``) — while
    content deleted here is gone.
    """

    STACK = rows_of(bottom_lines("#.########", "##.#######", "#########."))

    def test_a_resync_leaves_out_a_piece_entering_from_above(self) -> None:
        tracker = GameStateTracker()
        entering = merge(self.STACK, [(0, 4), (0, 5)])
        for _ in range(3):
            assert feed(tracker, entering, "T") == []
        assert feed(tracker, entering, "T") == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == self.STACK

    def test_the_entering_piece_is_then_tracked_normally(self) -> None:
        # And the proof it was the right call: the very same piece descends
        # into full view and is named against the stack it was left out of.
        tracker = GameStateTracker()
        feed(tracker, merge(self.STACK, [(0, 4), (0, 5)]), "T", times=4)
        descended = merge(self.STACK, piece_cells("O", 0, 2, 4))
        assert feed(tracker, descended, "T", times=2) == [GameEvent.PIECE_SPAWNED]
        assert tracker.committed.falling_piece == "O"
        assert tracker.committed.stack_rows == self.STACK

    def test_a_resync_keeps_a_column_stacked_to_the_top(self) -> None:
        # The counter-case: cells at row 0 that REST on the stack are board
        # content (side columns stacked to the top, garbage pushed up).
        # Stripping those would delete a real, and very load-bearing, block.
        topped_out = merge(self.STACK, [(r, 4) for r in range(17)])
        tracker = GameStateTracker()
        for _ in range(3):
            assert feed(tracker, topped_out, None) == []
        assert feed(tracker, topped_out, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == topped_out

    def test_a_resync_leaves_out_a_fully_visible_piece_in_flight(self) -> None:
        # The absorbed-piece bug, at its root. A whole tetromino floating
        # clear of the stack is the falling piece — stack cells do not
        # float — so the resync adopts the board WITHOUT it.
        tracker = GameStateTracker()
        in_flight = merge(self.STACK, piece_cells("O", 0, 0, 4))
        for _ in range(3):
            assert feed(tracker, in_flight, None) == []
        assert feed(tracker, in_flight, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == self.STACK

    def test_the_piece_the_resync_left_out_is_then_tracked_falling(self) -> None:
        # ...and the proof it was the right call: the next frame names it
        # against the stack it was left out of, and it keeps its name all
        # the way down instead of resetting the board once a row.
        tracker = GameStateTracker(confirm_frames=2)
        feed(tracker, merge(self.STACK, piece_cells("O", 0, 0, 4)), None, times=4)
        for row in range(1, 5):
            feed(tracker, merge(self.STACK, piece_cells("O", 0, row, 4)), None, times=2)
            assert tracker.committed.falling_piece == "O"
            assert tracker.committed.stack_rows == self.STACK

    def test_a_resync_absorbs_a_piece_shaped_component_at_rest(self) -> None:
        # The guard rail, and the direction this errs in: an O sitting ON
        # the stack is board content — a piece in lock delay, or one that
        # locked while nothing was watching — and deleting it would cost
        # the solver a real block. Resting beats piece-shaped.
        tracker = GameStateTracker()
        resting = merge(self.STACK, piece_cells("O", 0, HEIGHT - 5, 4))
        for _ in range(3):
            assert feed(tracker, resting, None) == []
        assert feed(tracker, resting, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == resting

    def test_a_resync_absorbs_more_than_one_tetrominos_worth_in_the_air(self) -> None:
        # The budget. One floating component of at most four cells is a
        # piece; two floating components name no single piece, and a
        # five-cell blob is no piece at all. Both are kept as observed.
        tracker = GameStateTracker()
        two = merge(self.STACK, piece_cells("O", 0, 0, 0), piece_cells("O", 0, 0, 6))
        for _ in range(3):
            assert feed(tracker, two, None) == []
        assert feed(tracker, two, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == two

        blob = merge(self.STACK, piece_cells("O", 0, 0, 4), [(0, 6)])
        tracker = GameStateTracker()
        for _ in range(3):
            assert feed(tracker, blob, None) == []
        assert feed(tracker, blob, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == blob

    def test_a_resync_keeps_a_post_clear_island_gravity_left_floating(self) -> None:
        # "Floating" does NOT mean "in flight". Naive gravity makes SETTLED
        # cells float: clear a row and everything above descends onto a row
        # with holes in it. This is the board the repo's own Board.drop
        # produces from an ordinary position (stack at cols 0-1 over covered
        # holes, a horizontal I completing the row between them), so the
        # island is four REAL locked cells. Holding it back handed the
        # solver a board four cells short and then announced an O the player
        # does not have, as the same cells read back as added.
        post_clear = rows_of(
            [(HEIGHT - 3, 0), (HEIGHT - 3, 1), (HEIGHT - 2, 0), (HEIGHT - 2, 1)],
            [(HEIGHT - 1, c) for c in range(2, WIDTH)],
        )
        tracker = GameStateTracker(confirm_frames=2)
        for _ in range(3):
            assert feed(tracker, post_clear, None) == []
        assert feed(tracker, post_clear, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == post_clear
        # ...and the frames after it are QUIET, not a phantom O spawning.
        assert feed(tracker, post_clear, None, times=2) == []
        assert tracker.committed.falling_piece is None

    def test_a_resync_keeps_a_piece_the_covered_panel_cuts_off(self) -> None:
        # Components are maximal only on the board the capture can SEE. In
        # this session's own geometry (the NEXT preview over cols 8-9 of
        # rows 0-1) a column stacked to the top with an O locked beside it
        # is ONE grounded component in the true board and two here, the
        # inner one floating and sitting at row 0 — so every other test
        # above would strip it. It is four real locked cells, in the top
        # rows, where a phantom hole costs the most.
        covered = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})
        board = rows_of(
            [(r, c) for r in range(HEIGHT) for c in (8, 9)],
            [(0, 6), (0, 7), (1, 6), (1, 7)],
        )
        visible = tuple(
            row & ~sum(1 << c for (r, c) in covered if r == i) for i, row in enumerate(board)
        )
        tracker = GameStateTracker(confirm_frames=2, unobservable_cells=covered)
        for _ in range(3):
            assert feed(tracker, board, None) == []
        assert feed(tracker, board, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == visible

    def test_a_floating_remnant_lower_down_is_adopted_not_held_forever(self) -> None:
        # The 1-3 cell version of the same thing, and worse than one-shot: a
        # clear can leave a fragment hanging mid-board, which no panel and
        # no top edge explains, so the frame is unexplainable. Held back, it
        # made the resync land on the stack already believed — which fired
        # no event and cleared the counter, so the tracker never adopted the
        # real content at all and those cells stayed missing for good.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, None)
        remnant = merge(self.STACK, [(6, 2), (6, 3), (6, 4)])
        for _ in range(3):
            assert feed(tracker, remnant, None) == []
        assert feed(tracker, remnant, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == remnant
        assert feed(tracker, remnant, None, times=2) == []
        assert tracker.last_kind is FrameKind.QUIET

    def test_garbage_rising_still_resets(self) -> None:
        # The other guard rail: a world that really did change must still
        # re-anchor. Garbage pushes the stack up a row and adds a holed row
        # at the floor — nothing floats, so nothing is held back, and the
        # whole observed board is adopted.
        tracker = GameStateTracker()
        attach(tracker, self.STACK, None)
        risen = tuple(self.STACK[1:]) + (0b1111111011,)
        for _ in range(3):
            assert feed(tracker, risen, None) == []
        assert feed(tracker, risen, None) == [GameEvent.BOARD_RESET]
        assert tracker.committed.stack_rows == risen
