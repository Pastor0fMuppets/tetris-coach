"""Run the race and print the head-to-head table.

    python -m tetris_coach.race

One block per window plus an overall block, two rows each. Every number
is a count or a percentage of a stated denominator -- nothing here is a
score out of ten, because the point of the race is to be able to say WHICH
frames a tracker got wrong, not how it feels.
"""

from __future__ import annotations

from pathlib import Path

from tetris_coach.race.measures import (
    Identity,
    Score,
    load_truth,
)
from tetris_coach.race.measures import (
    score as score_window,
)
from tetris_coach.race.runners import RUNNERS
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


def race(
    fixtures: Path = DEFAULT_FIXTURES,
    windows: tuple[WindowSpec, ...] = CONSECUTIVE,
) -> dict[str, dict[str, Score]]:
    """Replay every window through both trackers and score them."""
    truth = load_truth()
    out: dict[str, dict[str, Score]] = {}
    for spec in windows:
        window = truth[spec.name]
        out[spec.name] = {
            name: score_window(name, runner(fixtures, spec), window)
            for name, runner in RUNNERS.items()
        }
    return out


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
    lines = [HEADER, "-" * len(HEADER)]
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
    """Every frozen run the race found, with the frames it spans."""
    lines = ["frozen past the piece change (accepted frames, coach state unchanged):"]
    found = False
    for name, window in scores.items():
        for tracker in RUNNERS:
            for run in window[tracker].stuck:
                found = True
                lines.append(
                    f"  {name:<{NAME}}{tracker:<11}{run.first}-{run.last}  "
                    f"froze {run.frames:>3} frames; the piece became "
                    f"{run.truth} at {run.changed} and it held "
                    f"{run.held or 'nothing'} for {run.stale} more"
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


__all__ = ["main", "overall", "race", "row", "streak_detail", "stuck_detail", "table"]

if __name__ == "__main__":  # pragma: no cover
    main()
