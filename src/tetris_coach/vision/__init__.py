"""Vision: occupancy classification, shape recognition, state tracking."""

from .grid import cell_scores, classify_grid
from .pieces_vision import (
    Explanation,
    FallingPiece,
    FrameKind,
    explain_grid,
    identify_next,
)
from .state import GameEvent, GameStateTracker, Snapshot

__all__ = [
    "Explanation",
    "FallingPiece",
    "FrameKind",
    "GameEvent",
    "GameStateTracker",
    "Snapshot",
    "cell_scores",
    "classify_grid",
    "explain_grid",
    "identify_next",
]
