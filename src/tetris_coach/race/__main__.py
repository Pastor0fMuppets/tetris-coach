"""``python -m tetris_coach.race``.

Two tables, because the two questions are different. The first judges the
trackers at their own boundary -- which piece, on what board -- and is what
the choice of representation was decided on. The second replays the same
windows through the real ``CoachEngine`` and reports what the OVERLAY
shows, which is the only thing the user has ever been able to see.
"""

from tetris_coach.race.engine import main as engine_main
from tetris_coach.race.report import main as tracker_main

if __name__ == "__main__":  # pragma: no cover
    tracker_main()
    print()
    print()
    engine_main()
