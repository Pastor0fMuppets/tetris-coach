"""A race between the two falling-piece trackers, over the same frames.

The shipped tracker (``vision.pieces_vision`` + ``vision.state``, driven
through ``app.CoachEngine`` exactly as ``app.run`` drives it) and the
parallel prototype (``vision.colour_tracker``) answer the same question
from different representations. This package replays every committed
CONSECUTIVE capture window through both, scores each against the
independent answer sheet in ``tetris_coach.truth``, and prints a
head-to-head table.

Then it replays the same windows through ``app.CoachEngine`` itself
(:mod:`tetris_coach.race.engine`) and reports what the OVERLAY shows under
each tracker choice. That second table is the one the adoption was decided
on: a tracker that is right internally but late, blank or unsteady on
screen is not an improvement to anybody.

Nothing here is imported by the shipped pipeline; it only reads it.

    python -m tetris_coach.race
"""

from tetris_coach.race.engine import Shown, ShownFrame
from tetris_coach.race.engine import overall as engine_overall
from tetris_coach.race.engine import replay as engine_replay
from tetris_coach.race.engine import run as engine_race
from tetris_coach.race.engine import score as engine_score
from tetris_coach.race.engine import table as engine_table
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
    "Shown",
    "ShownFrame",
    "StuckRun",
    "engine_overall",
    "engine_race",
    "engine_replay",
    "engine_score",
    "engine_table",
    "episodes",
    "race",
    "run_prototype",
    "run_shipped",
    "score",
    "table",
]
