"""Placement search: 1-ply or 2-ply (current piece x next piece).

The search enumerates every (rotation, column) drop of the current piece;
when the next piece is known, each candidate is additionally scored by the
best achievable placement of the next piece on the resulting board (full
2-ply, no pruning: at most ~34 x 34 evaluations, fast enough on bitboards).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ..core.board import WIDTH, Board, DropResult
from ..core.pieces import ROTATIONS, Rotation
from .evaluate import DELLACHERIE, Weights, evaluate_drop

# Score assigned to a ply whose piece cannot be placed at all (top-out):
# worse than any reachable evaluation so survivable moves always win.
TOP_OUT_SCORE = -1.0e9


@dataclass(frozen=True)
class Move:
    """A chosen placement of the current piece."""

    piece: str
    rotation: Rotation
    col: int  # leftmost column of the rotation's bounding box
    row: int  # top row index of the bounding box after the drop
    score: float  # total search score (current ply + best next ply)
    lines_cleared: int
    board: Board  # board after the drop and any line clears

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        """Absolute (row, col) board cells occupied by the placed piece."""
        return tuple((self.row + r, self.col + c) for r, c in self.rotation.cells)


def enumerate_drops(board: Board, piece: str) -> Iterator[tuple[Rotation, int, DropResult]]:
    """Yield every legal (rotation, column, drop result) for ``piece``."""
    for rotation in ROTATIONS[piece]:
        for col in range(WIDTH - rotation.width + 1):
            result = board.drop(rotation, col)
            if result is not None:
                yield rotation, col, result


def _best_single_score(board: Board, piece: str, weights: Weights) -> float:
    """Best 1-ply evaluation of ``piece`` on ``board`` (TOP_OUT_SCORE if none).

    Reuses :func:`enumerate_drops` so ply-1 and ply-2 can never disagree
    about which placements are legal.
    """
    return max(
        (
            evaluate_drop(result, rotation, weights)
            for rotation, _, result in enumerate_drops(board, piece)
        ),
        default=TOP_OUT_SCORE,
    )


def best_move(
    board: Board,
    piece: str,
    next_piece: str | None = None,
    weights: Weights = DELLACHERIE,
) -> Move | None:
    """Find the best placement of ``piece`` on ``board``.

    With ``next_piece`` given, performs a full 2-ply search. Returns ``None``
    only when the current piece has no legal placement (game over). Ties are
    broken deterministically (lowest rotation index, then leftmost column).
    """
    best: Move | None = None
    for rotation, col, result in enumerate_drops(board, piece):
        score = evaluate_drop(result, rotation, weights)
        if next_piece is not None:
            score += _best_single_score(result.board, next_piece, weights)
        if best is None or score > best.score:
            best = Move(
                piece=piece,
                rotation=rotation,
                col=col,
                row=result.landing_row,
                score=score,
                lines_cleared=result.lines_cleared,
                board=result.board,
            )
    return best
