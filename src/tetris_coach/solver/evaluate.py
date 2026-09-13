"""Dellacherie move evaluation.

Six features, scored on the board *after* the drop plus two move-specific
terms (landing height and eroded piece cells). Default weights are Pierre
Dellacherie's classic hand-tuned integers (as documented by Colin Fahey):

    score = -landing_height + eroded_piece_cells
            - row_transitions - column_transitions
            - 4 * holes - cumulative_wells

Higher scores are better.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.board import DropResult
from ..core.pieces import Rotation


@dataclass(frozen=True)
class Weights:
    """Feature weights; positive contributions increase the score."""

    landing_height: float = -1.0
    eroded_cells: float = 1.0
    row_transitions: float = -1.0
    column_transitions: float = -1.0
    holes: float = -4.0
    cumulative_wells: float = -1.0


DELLACHERIE = Weights()


def landing_height(result: DropResult, rotation: Rotation) -> float:
    """Altitude of the piece's vertical midpoint, pre-clear (floor row = 0).

    The board height comes from the post-drop board's own row count, which
    is valid because :meth:`Board.drop` (line clears included) preserves the
    row count of the board it was called on.
    """
    return len(result.board.rows) - 1 - result.landing_row - (rotation.height - 1) / 2.0


def evaluate_drop(result: DropResult, rotation: Rotation, weights: Weights = DELLACHERIE) -> float:
    """Score a single placement (drop already applied)."""
    board = result.board
    return (
        weights.landing_height * landing_height(result, rotation)
        + weights.eroded_cells * (result.lines_cleared * result.eroded_cells)
        + weights.row_transitions * board.row_transitions()
        + weights.column_transitions * board.column_transitions()
        + weights.holes * board.hole_count()
        + weights.cumulative_wells * board.cumulative_wells()
    )
