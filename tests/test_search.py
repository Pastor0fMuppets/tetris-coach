import random

from tetris_coach.core.board import FULL_ROW, WIDTH, Board
from tetris_coach.core.pieces import PIECES
from tetris_coach.solver.search import Move, best_move, enumerate_drops

from .test_board import board_from_strings


def test_enumerate_drops_counts_on_empty_board() -> None:
    counts = {}
    for piece in PIECES:
        counts[piece] = sum(1 for _ in enumerate_drops(Board(), piece))
    # placements = sum over rotations of (11 - width)
    assert counts == {"I": 17, "O": 9, "T": 34, "S": 17, "Z": 17, "J": 34, "L": 34}


def test_best_move_returns_none_only_when_board_is_full() -> None:
    full = Board([FULL_ROW] * 20)
    assert best_move(full, "O") is None
    assert best_move(Board(), "O") is not None


def test_best_move_completes_a_line() -> None:
    b = board_from_strings("######....")
    move = best_move(b, "I")
    assert move is not None
    assert move.lines_cleared == 1
    assert move.board == Board()


def test_best_move_uses_lookahead() -> None:
    # Bottom row missing columns 8 and 9. With an S now and a vertical-I-
    # friendly future, 1-ply and 2-ply may differ; at minimum the 2-ply
    # search must not be worse at clearing when the next piece completes it.
    b = board_from_strings(
        "########..",
        "########..",
    )
    move = best_move(b, "O", next_piece="I")
    assert move is not None
    # The O fills the 2x2 notch and clears both rows immediately.
    assert move.col == 8
    assert move.lines_cleared == 2


def test_two_ply_avoids_greedy_trap() -> None:
    # A column-9 well, 4 deep. Greedy Dellacherie already keeps wells tidy;
    # here we check the 2-ply score really adds the next-piece ply.
    b = board_from_strings(
        "#########.",
        "#########.",
        "#########.",
        "#########.",
    )
    move_with_next = best_move(b, "I", next_piece="O")
    move_alone = best_move(b, "I")
    assert move_with_next is not None and move_alone is not None
    # Both should drop the I vertically into the well for a tetris.
    assert move_alone.lines_cleared == 4
    assert move_with_next.lines_cleared == 4
    # 2-ply score includes the O's placement score and thus differs.
    assert move_with_next.score != move_alone.score


def test_move_cells_are_absolute_board_cells() -> None:
    move = best_move(Board(), "O")
    assert move is not None
    assert len(move.cells) == 4
    for r, c in move.cells:
        assert 0 <= r < 20
        assert 0 <= c < WIDTH
    # O on an empty board must rest on the bottom two rows.
    assert {r for r, _ in move.cells} == {18, 19}


def test_deterministic_tie_break() -> None:
    a = best_move(Board(), "T", next_piece="T")
    b = best_move(Board(), "T", next_piece="T")
    assert a is not None and b is not None
    assert (a.rotation.index, a.col) == (b.rotation.index, b.col)


def play_game(seed: int, max_pieces: int) -> tuple[int, int]:
    """Self-play with 2-ply lookahead; returns (pieces placed, lines cleared)."""
    rng = random.Random(seed)
    board = Board()
    current = rng.choice(PIECES)
    upcoming = rng.choice(PIECES)
    placed = 0
    lines = 0
    while placed < max_pieces:
        move = best_move(board, current, next_piece=upcoming)
        if move is None:
            break
        board = move.board
        lines += move.lines_cleared
        placed += 1
        current = upcoming
        upcoming = rng.choice(PIECES)
    return placed, lines


def test_self_play_survives_1000_pieces() -> None:
    placed, lines = play_game(seed=20260913, max_pieces=1000)
    assert placed == 1000, f"topped out after {placed} pieces ({lines} lines)"
    # 1000 pieces = 4000 cells = at most 400 lines; a competent player
    # clears most of them.
    assert lines > 300


def test_self_play_various_seeds() -> None:
    for seed in (1, 2, 3):
        placed, _ = play_game(seed=seed, max_pieces=200)
        assert placed == 200, f"seed {seed} topped out after {placed} pieces"


def test_best_move_on_12_row_board() -> None:
    b = Board([0] * 11 + [FULL_ROW ^ 0b1111000000])
    move = best_move(b, "I")
    assert move is not None
    assert move.lines_cleared == 1
    assert move.board == Board([0] * 12)
    # Placement cells stay inside the 12-row board.
    for r, c in move.cells:
        assert 0 <= r < 12
        assert 0 <= c < WIDTH


def test_self_play_survives_on_12_row_board() -> None:
    rng = random.Random(20260913)
    board = Board([0] * 12)
    current = rng.choice(PIECES)
    upcoming = rng.choice(PIECES)
    placed = 0
    while placed < 300:
        move = best_move(board, current, next_piece=upcoming)
        if move is None:
            break
        assert len(move.board.rows) == 12
        board = move.board
        placed += 1
        current = upcoming
        upcoming = rng.choice(PIECES)
    assert placed == 300, f"topped out after {placed} pieces on 12 rows"


class TestBenchmark:
    @staticmethod
    def _midgame_board() -> Board:
        """A representative mid-game stack (uneven, one deep well)."""
        return board_from_strings(
            "..........",
            "##...#....",
            "##.###...#",
            "#######..#",
            "########.#",
            "########.#",
            "#######..#",
            "########.#",
        )

    def test_two_ply_search_under_50ms(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        board = self._midgame_board()
        # T x T is the largest search space (34 x 34 placements).
        result = benchmark(best_move, board, "T", "T")
        assert isinstance(result, Move)
        if benchmark.stats is not None:  # absent under --benchmark-disable
            assert benchmark.stats.stats.mean < 0.050
            assert benchmark.stats.stats.max < 0.100
