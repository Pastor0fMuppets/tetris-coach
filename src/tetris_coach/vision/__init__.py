"""Vision: reading a captured frame as a board, a piece and a NEXT box.

Two readers answer that, and :mod:`.readers` is where they meet: the
default one names each piece by the colour it is drawn in and re-derives
the board from every frame (:mod:`.colour_tracker`, :mod:`.colour_palette`,
:mod:`.colour_preview`); the other thresholds each cell to one bit and
explains the grid as a diff against a committed stack memory
(:mod:`.grid`, :mod:`.pieces_vision`, :mod:`.state`), and is what
``--tracker shape`` selects.
"""

from .grid import GridClassifier, OwnPaint, cell_scores, classify_grid
from .pieces_vision import (
    Explanation,
    FallingPiece,
    FrameKind,
    explain_grid,
    identify_next,
)
from .readers import ColourVision, FrameReading, FrameVision, ShapeVision, VisionEvent
from .state import GameEvent, GameStateTracker, Snapshot

__all__ = [
    "ColourVision",
    "Explanation",
    "FallingPiece",
    "FrameKind",
    "FrameReading",
    "FrameVision",
    "GameEvent",
    "GameStateTracker",
    "GridClassifier",
    "OwnPaint",
    "ShapeVision",
    "Snapshot",
    "VisionEvent",
    "cell_scores",
    "classify_grid",
    "explain_grid",
    "identify_next",
]
