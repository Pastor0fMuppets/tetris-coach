"""Run the race and print the head-to-head table.

    python -m tetris_coach.race

One block per window plus an overall block, two rows each. Every number
is a count or a percentage of a stated denominator -- nothing here is a
score out of ten, because the point of the race is to be able to say WHICH
frames a tracker got wrong, not how it feels.

``frames`` is the SCORED frames of a window, not its length: both trackers
are replayed from the window's first frame, and the first
``measures.WARMUP`` of them are scored for neither. See that module for why
-- briefly, every window is a mid-session excerpt, so scoring the head
charges a design that carries memory across frames for a bootstrap the live
session paid once before the capture began.
"""

from __future__ import annotations

from pathlib import Path

from tetris_coach.race.measures import (
    WARMUP,
    Identity,
    Score,
    load_truth,
)
from tetris_coach.race.measures import (
    score as score_window,
)
from tetris_coach.race.runners import RUNNERS, FrameOutput
from tetris_coach.truth.windows import CONSECUTIVE, WindowSpec

DEFAULT_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"

#: Width of the window-name column. Wide enough for ``ghost_beside_stack``.
NAME = len("ghost_beside_stack") + 2

HEADER = (
    f"{'window':<{NAME}}{'tracker':<11}{'frames':>7}{'judged':>7}"
    f"{'right':>7}{'WRONG':>7}{'silent':>7}{'board':>7}"
    f"{'latmed':>7}{'latmax':>7}{'never':>7}"
    f"{'flips':>7}{'hintchg':>8}{'stuck':>7}{'noHint':>7}"
)


def replays(
    fixtures: Path = DEFAULT_FIXTURES,
    windows: tuple[WindowSpec, ...] = CONSECUTIVE,
) -> dict[str, dict[str, list[FrameOutput]]]:
    """Every window through both trackers, unscored.

    Split out from :func:`race` so the same replay can be scored under more
    than one warm-up without paying for the replay again -- which is how
    the verdict's insensitivity to that choice is checked.
    """
    return {
        spec.name: {name: runner(fixtures, spec) for name, runner in RUNNERS.items()}
        for spec in windows
    }


def score_replays(
    outputs: dict[str, dict[str, list[FrameOutput]]],
    warmup: int = WARMUP,
) -> dict[str, dict[str, Score]]:
    """Score an existing set of replays under one warm-up."""
    truth = load_truth()
    return {
        window: {
            tracker: score_window(tracker, frames, truth[window], warmup)
            for tracker, frames in trackers.items()
        }
        for window, trackers in outputs.items()
    }


def race(
    fixtures: Path = DEFAULT_FIXTURES,
    windows: tuple[WindowSpec, ...] = CONSECUTIVE,
    warmup: int = WARMUP,
) -> dict[str, dict[str, Score]]:
    """Replay every window through both trackers and score them."""
    return score_replays(replays(fixtures, windows), warmup)


def overall(scores: dict[str, dict[str, Score]], tracker: str) -> Score:
    """One tracker's numbers summed over every window."""
    parts = [window[tracker] for window in scores.values()]
    total = Score(
        window="ALL",
        tracker=tracker,
        frames=sum(part.frames for part in parts),
        identity=Identity(
            judged=sum(part.identity.judged for part in parts),
            correct=sum(part.identity.correct for part in parts),
            wrong=sum(part.identity.wrong for part in parts),
            silent=sum(part.identity.silent for part in parts),
        ),
    )
    total.warmup = sum(part.warmup for part in parts)
    total.board_right = sum(part.board_right for part in parts)
    total.refused_frames = sum(part.refused_frames for part in parts)
    total.worst_wrong_run = max(part.worst_wrong_run for part in parts)
    total.worst_silent_run = max(part.worst_silent_run for part in parts)
    for part in parts:
        total.measurable += part.measurable
        total.latencies.extend(part.latencies)
        total.stuck.extend(part.stuck)
        total.episodes += part.episodes
        total.unnamed_episodes += part.unnamed_episodes
        total.name_flips += part.name_flips
        total.hint_flips += part.hint_flips
        total.silent_frames += part.silent_frames
        total.hintless_frames += part.hintless_frames
    return total


def row(score: Score) -> str:
    identity = score.identity
    median = score.median_latency
    longest = score.max_latency
    return (
        f"{score.window:<{NAME}}{score.tracker:<11}{score.frames:>7}{identity.judged:>7}"
        f"{identity.correct:>7}{identity.wrong:>7}{identity.silent:>7}"
        f"{score.board_right:>7}"
        f"{'-' if median is None else f'{median:.1f}':>7}"
        f"{'-' if longest is None else longest:>7}"
        f"{f'{score.unnamed_episodes}/{score.episodes}':>7}"
        f"{score.name_flips:>7}{score.hint_flips:>8}{score.worst_stuck:>7}"
        f"{score.hintless_frames:>7}"
    )


def table(scores: dict[str, dict[str, Score]]) -> str:
    """The head-to-head table, per window and overall."""
    warmed = {part.warmup for window in scores.values() for part in window.values()}
    head = (
        f"scored from frame {min(warmed)} of each window; "
        f"the head is replayed into both trackers and scored for neither"
    )
    lines = [head, "", HEADER, "-" * len(HEADER)]
    for window in scores.values():
        for tracker in RUNNERS:
            lines.append(row(window[tracker]))
        lines.append("")
    lines.append("-" * len(HEADER))
    for tracker in RUNNERS:
        lines.append(row(overall(scores, tracker)))
    return "\n".join(lines)


def streak_detail(scores: dict[str, dict[str, Score]]) -> str:
    """The longest unbroken stretch of each kind of failure."""
    lines = ["longest unbroken stretch, over the frames the oracle answers:"]
    for name, window in scores.items():
        for tracker in RUNNERS:
            part = window[tracker]
            lines.append(
                f"  {name:<{NAME}}{tracker:<11}wrong name {part.worst_wrong_run:>3} frames, "
                f"no name {part.worst_silent_run:>3} frames, "
                f"wrong board {part.identity.judged - part.board_right:>3} frames total, "
                f"{part.refused_frames:>3} frames refused unseen"
            )
    return "\n".join(lines)


def stuck_detail(scores: dict[str, dict[str, Score]]) -> str:
    """Every frozen run the race found, with the frames it spans.

    ``misplaced`` is what makes these readable: a freeze can hold the right
    LETTER across a piece change and still be a freeze, because the state
    that is frozen is the piece AND the board under it. The one run in this
    corpus is of exactly that kind -- an O gives way to another O over a row
    that has just cleared, so the name never looks wrong and the board is
    eight cells wrong for nearly two seconds.
    """
    lines = ["frozen past the piece change (every frame, coach state unchanged):"]
    found = False
    for name, window in scores.items():
        for tracker in RUNNERS:
            for run in window[tracker].stuck:
                found = True
                lines.append(
                    f"  {name:<{NAME}}{tracker:<11}{run.first}-{run.last}  "
                    f"froze {run.frames:>3} frames ({run.refused} of them refused unseen); "
                    f"the piece became {run.truth} at {run.changed} and it held "
                    f"{run.held or 'nothing'} for {run.stale} more, on a board "
                    f"{run.misplaced} cells wrong"
                )
    if not found:
        lines.append("  none")
    return "\n".join(lines)


def main(fixtures: Path = DEFAULT_FIXTURES) -> None:  # pragma: no cover
    scores = race(fixtures)
    print(table(scores))
    print()
    print(streak_detail(scores))
    print()
    print(stuck_detail(scores))


__all__ = [
    "main",
    "overall",
    "race",
    "replays",
    "row",
    "score_replays",
    "streak_detail",
    "stuck_detail",
    "table",
]

if __name__ == "__main__":  # pragma: no cover
    main()
