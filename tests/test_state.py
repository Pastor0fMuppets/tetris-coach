import numpy as np
import pytest

from tetris_coach.vision.pieces_vision import FallingPiece
from tetris_coach.vision.state import GameEvent, GameStateTracker


def empty_stack() -> np.ndarray:
    return np.zeros((20, 10), dtype=bool)


def stack_from(*lines: str) -> np.ndarray:
    grid = empty_stack()
    for i, line in enumerate(lines):
        r = 20 - len(lines) + i
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = True
    return grid


def falling(piece: str, row: int = 0, col: int = 4) -> FallingPiece:
    return FallingPiece(piece=piece, rotation_index=0, row=row, col=col)


class TestDebounce:
    def test_single_frame_is_not_committed(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        events = tracker.update(empty_stack(), falling("T"), "I")
        assert events == []
        assert tracker.committed is None

    def test_two_consistent_frames_commit(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        tracker.update(empty_stack(), falling("T"), "I")
        events = tracker.update(empty_stack(), falling("T", row=1), "I")
        assert GameEvent.PIECE_SPAWNED in events
        assert tracker.committed is not None
        assert tracker.committed.falling_piece == "T"
        assert tracker.committed.next_piece == "I"

    def test_flicker_frame_is_ignored(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        stack = stack_from("####......")
        tracker.update(stack, falling("T"), "I")
        tracker.update(stack, falling("T"), "I")
        committed = tracker.committed
        # One-frame glitch (mid-clear animation: stack appears different).
        glitch = stack_from("##........")
        assert tracker.update(glitch, falling("T"), "I") == []
        # Back to the real state: no state change committed at any point.
        assert tracker.update(stack, falling("T"), "I") == []
        assert tracker.committed == committed

    def test_falling_piece_position_does_not_block_commit(self) -> None:
        # The piece moves every frame; identity must ignore its position.
        tracker = GameStateTracker(confirm_frames=2)
        for row in range(3):
            tracker.update(empty_stack(), falling("S", row=row), "Z")
        assert tracker.committed is not None
        assert tracker.committed.falling_piece == "S"

    def test_confirm_frames_validation(self) -> None:
        with pytest.raises(ValueError):
            GameStateTracker(confirm_frames=0)

    def test_raw_falling_is_always_current(self) -> None:
        tracker = GameStateTracker(confirm_frames=2)
        piece = falling("L", row=7)
        tracker.update(empty_stack(), piece, None)
        assert tracker.falling == piece


def commit(tracker: GameStateTracker, stack, piece, nxt):  # type: ignore[no-untyped-def]
    """Feed the same observation twice; return the second call's events."""
    tracker.update(stack, piece, nxt)
    return tracker.update(stack, piece, nxt)


class TestEvents:
    def test_lock_detected(self) -> None:
        tracker = GameStateTracker()
        commit(tracker, empty_stack(), falling("O"), "T")
        # The O locks into the stack; a T spawns.
        stack = stack_from(
            "....##....",
            "....##....",
        )
        events = commit(tracker, stack, falling("T"), "S")
        assert GameEvent.PIECE_LOCKED in events
        assert GameEvent.PIECE_SPAWNED in events
        assert GameEvent.BOARD_RESET not in events

    def test_lock_with_line_clear(self) -> None:
        tracker = GameStateTracker()
        before = stack_from("######....")
        commit(tracker, before, falling("I"), "J")
        # The I fills the row; it clears: stack cell count 6 + 4 - 10 = 0.
        events = commit(tracker, empty_stack(), falling("J"), "L")
        assert GameEvent.PIECE_LOCKED in events
        assert GameEvent.BOARD_RESET not in events

    def test_spawn_without_lock(self) -> None:
        tracker = GameStateTracker()
        commit(tracker, empty_stack(), None, "T")
        events = commit(tracker, empty_stack(), falling("T"), "Z")
        assert events == [GameEvent.PIECE_SPAWNED]

    def test_board_reset_detected(self) -> None:
        tracker = GameStateTracker()
        stack = stack_from(
            "#####.####",
            "####.#####",
            "#########.",
        )
        commit(tracker, stack, falling("T"), "I")
        events = commit(tracker, empty_stack(), None, None)
        assert GameEvent.BOARD_RESET in events
        assert GameEvent.PIECE_LOCKED not in events

    def test_reset_then_new_game_spawn(self) -> None:
        tracker = GameStateTracker()
        stack = stack_from("##########".replace("##########", "#########."))
        commit(tracker, stack, falling("L"), "O")
        events = commit(tracker, empty_stack(), falling("I"), "O")
        assert GameEvent.BOARD_RESET in events
        assert GameEvent.PIECE_SPAWNED in events

    def test_same_piece_kind_twice_after_lock_still_spawns(self) -> None:
        # T locks and the next falling piece is also a T: the stack change
        # plus lock implies a fresh spawn even though the name is equal.
        tracker = GameStateTracker()
        commit(tracker, empty_stack(), falling("T"), "T")
        stack = stack_from(
            "....#.....",
            "...###....",
        )
        events = commit(tracker, stack, falling("T"), "I")
        assert GameEvent.PIECE_LOCKED in events
        assert GameEvent.PIECE_SPAWNED in events

    def test_lock_detected_without_observed_falling_piece(self) -> None:
        # The piece spawned and locked entirely between committed frames:
        # the stack simply grew by 4 cells on top of the old stack.
        tracker = GameStateTracker()
        before = stack_from("####......")
        commit(tracker, before, None, "T")
        after = stack_from(
            "....##....",
            "######....",
        )
        events = commit(tracker, after, None, "S")
        assert GameEvent.PIECE_LOCKED in events
        assert GameEvent.BOARD_RESET not in events

    def test_no_events_when_nothing_changes(self) -> None:
        tracker = GameStateTracker()
        stack = stack_from("#.........")
        commit(tracker, stack, falling("Z"), "S")
        assert tracker.update(stack, falling("Z", row=5), "S") == []
        assert tracker.update(stack, falling("Z", row=6), "S") == []

    def test_next_piece_change_alone_commits_quietly(self) -> None:
        tracker = GameStateTracker()
        commit(tracker, empty_stack(), falling("I"), None)
        events = commit(tracker, empty_stack(), falling("I"), "T")
        assert events == []
        assert tracker.committed is not None
        assert tracker.committed.next_piece == "T"
