"""The race re-run where the user sits: through the real ``CoachEngine``.

:mod:`tetris_coach.race.runners` judges the two trackers at their own
boundary -- the piece each one names, the stack each one would hand the
solver. That is the right place to compare REPRESENTATIONS, and the wrong
place to decide what to ship, because nothing the user sees comes straight
off a tracker. Between the tracker and the screen sits the engine's hint
policy, and every rule in it was added in answer to a report from this
tool's own user: a target that would not stay still, a hint left painted
over a game-over screen for 21.7 s, a coach reading its own overlay back as
the board, a preview panel covering four board cells.

So this module replays the same committed windows through
:class:`~tetris_coach.app.CoachEngine` itself -- the same class ``app.run``
drives, wired the way ``app.run`` wires it -- once per tracker choice, and
measures what the user actually experiences:

MISNAMED   Frames where the placement on screen is drawn for a piece that
           is not the piece falling. The worst failure there is: it walks
           the player into a hole and they cannot tell it is doing so.
INVENTED   Frames the oracle ABSTAINS on where the hint names a piece that
           neither the answer before the abstention nor the answer after it
           names. MISNAMED can only be counted where the oracle answers, and
           the stretches it abstains on are not idle time for the user --
           they are line clears and covered boards, which is where a reader
           that hallucinates does it. Without this, a tracker could invent a
           piece on every clear in the corpus and score a clean sheet.
STALE      Frames showing a hint on a frame vision did not accept. A hold,
           not a lie -- the coach is riding out a glitch -- but a hold is
           also how a hint ends up parked over a game-over screen, so the
           count is the price of the patience and is reported as such.
LATENCY    Frames from a piece appearing (the oracle's first frame of its
           episode) to a hint for THAT piece being on screen. Silence the
           player has to wait through.
MOVES      Hint-target changes inside one piece's flight. Churn the user
           reads as flicker, and the reason HINT_SWITCH_MARGIN exists.
HINTLESS   Frames with nothing on screen at all.

The window head is unscored for the same reason it is in
:mod:`.measures`: every window is a mid-session excerpt, and a design that
carries memory across frames would otherwise be charged six times for a
bootstrap the live session pays once.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from tetris_coach.app import CoachConfig, CoachEngine
from tetris_coach.race.measures import (
    WARMUP,
    Episode,
    TruthFrame,
    TruthWindow,
    episodes,
    load_truth,
    measurable,
)
from tetris_coach.race.runners import HintTarget, next_crops
from tetris_coach.truth.windows import (
    CAPTURED_FILL_OPACITY,
    CONSECUTIVE,
    WindowSpec,
    load_window,
)

#: The tracker choices ``--tracker`` offers, in the order the table prints
#: them: the default first, the escape hatch second.
TRACKERS = ("colour", "shape")

DEFAULT_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"

#: Width of the window-name column. Wide enough for ``ghost_beside_stack``.
NAME = len("ghost_beside_stack") + 2

HEADER = (
    f"{'window':<{NAME}}{'tracker':<9}{'frames':>7}{'hints':>7}{'MISNAMED':>9}"
    f"{'INVENTED':>9}{'stale':>7}{'latmed':>7}{'latmax':>7}{'never':>7}"
    f"{'moves':>7}{'worst':>7}{'hintless':>9}{'refused':>8}"
)


@dataclass(frozen=True)
class ShownFrame:
    """What the overlay would be showing on one frame, and on what evidence."""

    frame: str
    accepted: bool  # vision could read this frame at all
    piece: str | None  # the piece the engine's reading names as falling
    hint: HintTarget | None  # the placement on screen: (piece, cells)


def replay(root: Path, spec: WindowSpec, tracker: str) -> list[ShownFrame]:
    """One window through the real engine, with ``tracker`` doing the reading."""
    config = CoachConfig(rows=spec.rows, tracker=tracker, hint_fill_opacity=CAPTURED_FILL_OPACITY)
    engine = CoachEngine(config, unobservable_cells=spec.geometry().unobservable)
    names, boards = load_window(root, spec)
    crops = next_crops(root, spec, names)
    out: list[ShownFrame] = []
    for name, board, crop in zip(names, boards, crops, strict=True):
        hint = engine.process_frame(board, crop)
        reading = engine.last_reading
        out.append(
            ShownFrame(
                frame=name,
                accepted=reading is not None and reading.accepted,
                piece=None if reading is None else reading.falling_piece,
                hint=None if hint is None else (hint.piece, tuple(sorted(hint.cells))),
            )
        )
    return out


@dataclass
class Shown:
    """What one tracker choice showed over one window."""

    window: str
    tracker: str
    frames: int
    warmup: int
    hinted: int = 0  # frames with something on screen
    hintless: int = 0  # frames with nothing on screen
    misnamed: int = 0  # frames whose hint names the wrong piece
    invented: int = 0  # hints on abstained frames naming a piece nothing else does
    stale: int = 0  # frames holding a hint over a frame vision refused
    refused: int = 0  # frames vision could not read
    judged: int = 0  # frames the oracle answers, and the hint could be judged
    latencies: list[int] = field(default_factory=list)
    measurable: int = 0  # episodes whose latency can be read at all
    never_hinted: int = 0  # episodes that never got a hint for their piece
    episodes: int = 0
    moves: int = 0  # hint-target changes inside a flight, summed
    worst_moves: int = 0  # ...and the most any one piece suffered

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def max_latency(self) -> int | None:
        return max(self.latencies) if self.latencies else None


def score(
    tracker: str,
    shown: list[ShownFrame],
    truth: TruthWindow,
    warmup: int = WARMUP,
) -> Shown:
    """Score one window's replay against the oracle's answer sheet."""
    frames = truth.frames
    plan = [episode for episode in episodes(frames) if episode.start >= warmup]
    scored = shown[warmup:]
    result = Shown(
        window=truth.window,
        tracker=tracker,
        frames=len(scored),
        warmup=warmup,
        hinted=sum(1 for frame in scored if frame.hint is not None),
        hintless=sum(1 for frame in scored if frame.hint is None),
        stale=sum(1 for frame in scored if frame.hint is not None and not frame.accepted),
        refused=sum(1 for frame in scored if not frame.accepted),
        episodes=len(plan),
    )
    for frame, answer in zip(scored, frames[warmup:], strict=True):
        if frame.hint is None or not answer.answered:
            continue
        result.judged += 1
        if frame.hint[0] != answer.piece:
            result.misnamed += 1
    result.invented = _invented(shown, frames, warmup)
    timed = set(measurable(episodes(frames)))
    for episode in plan:
        moved = _moves(shown, episode)
        result.moves += moved
        result.worst_moves = max(result.worst_moves, moved)
        shows = _first_hint(shown, episode)
        if shows is None:
            result.never_hinted += 1
        if episode in timed:
            result.measurable += 1
            if shows is not None:
                result.latencies.append(shows - episode.start)
    return result


def _invented(shown: list[ShownFrame], frames: list[TruthFrame], warmup: int) -> int:
    """Hints on abstained frames naming a piece neither neighbour names.

    The hole this closes: MISNAMED is only counted where the oracle
    answers, and ``answered`` is False for every frame of a line clear or
    a covered board -- exactly the stretches where a reader that invents a
    piece will invent one. Measured before the completed-row premise
    existed, the colour reader drew an ``I`` over four frames of
    spawn_latency's clear while the player held an O, and scored a clean
    sheet on every number in this table.

    An abstention is not an accusation, so the bar is deliberately high: a
    hint is only counted when the oracle's last word before the stretch
    AND its first word after it both name some other piece. A hint that
    matches either end is the coach holding through a gap or reading the
    next piece early, which is what it is supposed to do.
    """
    answered = [index for index, frame in enumerate(frames) if frame.answered]
    if not answered:
        return 0
    count = 0
    for index in range(warmup, len(frames)):
        hint = shown[index].hint
        if hint is None or frames[index].answered:
            continue
        before = [i for i in answered if i < index]
        after = [i for i in answered if i > index]
        neighbours = {frames[i].piece for i in (before[-1:] + after[:1])}
        if neighbours and hint[0] not in neighbours:
            count += 1
    return count


def _first_hint(shown: list[ShownFrame], episode: Episode) -> int | None:
    """Index of the first frame of ``episode`` showing a hint for its piece."""
    for index in range(episode.start, episode.end + 1):
        hint = shown[index].hint
        if hint is not None and hint[0] == episode.piece:
            return index
    return None


def _moves(shown: list[ShownFrame], episode: Episode) -> int:
    """Times the target changed inside one flight, once a hint was up.

    Counted from the frame the piece is first hinted, for the reason
    :func:`.measures.score` counts churn from the frame it is first named:
    before that the coach is late, which latency already charges it for,
    and charging the same frames twice would make a slow tracker look
    unstable as well. Frames with nothing on screen are skipped rather than
    counted as a change -- a hint going down and coming back on the same
    target is not the flicker this measures.
    """
    start = _first_hint(shown, episode)
    if start is None:
        return 0
    targets = [frame.hint for frame in shown[start : episode.end + 1] if frame.hint is not None]
    return sum(1 for a, b in pairwise(targets) if a != b)


def run(
    fixtures: Path = DEFAULT_FIXTURES,
    windows: tuple[WindowSpec, ...] = CONSECUTIVE,
    warmup: int = WARMUP,
) -> dict[str, dict[str, Shown]]:
    """Every window through the engine on both tracker choices, scored."""
    truth = load_truth()
    return {
        spec.name: {
            tracker: score(tracker, replay(fixtures, spec, tracker), truth[spec.name], warmup)
            for tracker in TRACKERS
        }
        for spec in windows
    }


def overall(scores: dict[str, dict[str, Shown]], tracker: str) -> Shown:
    """One tracker choice's numbers summed over every window."""
    parts = [window[tracker] for window in scores.values()]
    total = Shown(
        window="ALL",
        tracker=tracker,
        frames=sum(part.frames for part in parts),
        warmup=sum(part.warmup for part in parts),
    )
    for part in parts:
        total.hinted += part.hinted
        total.hintless += part.hintless
        total.misnamed += part.misnamed
        total.invented += part.invented
        total.stale += part.stale
        total.refused += part.refused
        total.judged += part.judged
        total.latencies.extend(part.latencies)
        total.measurable += part.measurable
        total.never_hinted += part.never_hinted
        total.episodes += part.episodes
        total.moves += part.moves
        total.worst_moves = max(total.worst_moves, part.worst_moves)
    return total


def row(shown: Shown) -> str:
    median = shown.median_latency
    longest = shown.max_latency
    return (
        f"{shown.window:<{NAME}}{shown.tracker:<9}{shown.frames:>7}{shown.hinted:>7}"
        f"{shown.misnamed:>9}{shown.invented:>9}{shown.stale:>7}"
        f"{'-' if median is None else f'{median:.1f}':>7}"
        f"{'-' if longest is None else longest:>7}"
        f"{f'{shown.never_hinted}/{shown.episodes}':>7}"
        f"{shown.moves:>7}{shown.worst_moves:>7}{shown.hintless:>9}{shown.refused:>8}"
    )


def table(scores: dict[str, dict[str, Shown]]) -> str:
    """The head-to-head table of what each tracker choice puts on screen."""
    warmed = {part.warmup for window in scores.values() for part in window.values()}
    head = (
        "what the overlay shows, through the real CoachEngine; "
        f"scored from frame {min(warmed)} of each window"
    )
    lines = [head, "", HEADER, "-" * len(HEADER)]
    for window in scores.values():
        for tracker in TRACKERS:
            lines.append(row(window[tracker]))
        lines.append("")
    lines.append("-" * len(HEADER))
    for tracker in TRACKERS:
        lines.append(row(overall(scores, tracker)))
    return "\n".join(lines)


def main(fixtures: Path = DEFAULT_FIXTURES) -> None:  # pragma: no cover
    print(table(run(fixtures)))


__all__ = [
    "TRACKERS",
    "Shown",
    "ShownFrame",
    "main",
    "overall",
    "replay",
    "row",
    "run",
    "score",
    "table",
]

if __name__ == "__main__":  # pragma: no cover
    main()
