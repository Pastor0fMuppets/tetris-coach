"""A race between the two falling-piece trackers, over the same frames.

The shipped tracker (``vision.pieces_vision`` + ``vision.state``, driven
through ``app.CoachEngine`` exactly as ``app.run`` drives it) and the
parallel prototype (``vision.colour_tracker``) answer the same question
from different representations. This package replays every committed
CONSECUTIVE capture window through both, scores each against the
independent answer sheet in ``tetris_coach.truth``, and prints a
head-to-head table.

Nothing here is imported by the shipped pipeline; it only reads it.

    python -m tetris_coach.race
"""

from tetris_coach.race.measures import (
    Episode,
    Identity,
    Score,
    StuckRun,
    episodes,
    score,
)
from tetris_coach.race.report import race, table
from tetris_coach.race.runners import FrameOutput, run_prototype, run_shipped

__all__ = [
    "Episode",
    "FrameOutput",
    "Identity",
    "Score",
    "StuckRun",
    "episodes",
    "race",
    "run_prototype",
    "run_shipped",
    "score",
    "table",
]
