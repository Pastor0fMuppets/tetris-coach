"""GameStateTracker tests: debounce, lock/spawn/reset events, resyncs."""

import pytest

from tetris_coach.core.board import HEIGHT
from tetris_coach.vision.state import GameEvent, GameStateTracker, Snapshot

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


class TestResyncAndThePieceInFlight:
    """A resync must not freeze a piece that has not landed.

    A mid-game attach adopts the observed board wholesale — there is no
    memory to diff it against. What it must leave out is content that is
    demonstrably not settled: stack cells rest on something, so a component
    with air under all of it is the falling piece. Freezing it in plants
    blocks no lock ever put there, and worse, makes the piece's own descent
    read as stack cells vanishing — which nothing can explain, so the
    tracker resets again and re-absorbs it one row lower, all the way down
    (``tests/fixtures/absorbed_piece``).
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

    def test_a_resync_onto_the_stack_already_believed_fires_no_event(self) -> None:
        # Re-anchoring on the board the tracker already holds changes
        # nothing, and the event is not free: the consumer drops the hint on
        # a reset. The frame stays unexplainable (a pale piece read in part,
        # a torn capture) and the tracker holds, hint and all, instead of
        # blanking the overlay every fourth frame for nothing.
        tracker = GameStateTracker(confirm_frames=2)
        attach(tracker, self.STACK, "T")
        feed(tracker, merge(self.STACK, piece_cells("O", 0, 5, 4)), "T", times=2)
        assert tracker.committed.falling_piece == "O"
        # Three cells of the O read, mid-board: no panel and no top edge
        # explains the missing one, so the frame really is unexplainable.
        broken = merge(self.STACK, [(5, 4), (6, 4), (6, 5)])
        for _ in range(8):
            assert feed(tracker, broken, "T") == []
        assert tracker.committed.stack_rows == self.STACK
        assert tracker.committed.falling_piece == "O"

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
