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
