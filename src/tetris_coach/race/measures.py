"""Score a tracker's replay against the oracle, on what the user feels.

Five measures, each one a complaint the user actually made or a way of
failing they would notice:

IDENTITY  Of the frames the oracle answers, how often is the named piece
          right, how often is it WRONG ("it often guesses the piece
          wrong"), and how often is there no name at all? Wrong and silent
          are counted apart on purpose: a coach that says nothing wastes
          the frame, a coach that says "T" when an S is falling walks the
          user into a hole.
LATENCY   Frames from a piece first becoming visible -- the first frame of
          the oracle's episode -- to the tracker naming it correctly.
STABILITY Within one piece's flight, how many times the named piece or the
          hint's target square changes. Churn the user sees as flicker.
STUCK     A run of ACCEPTED frames where the coach's state is frozen ACROSS
          a piece change -- the piece in play became a different piece and
          the coach went on holding the old one ("it got stuck and the
          piece didn't update for several turns"). A frozen state is not by
          itself a fault: this game has no gravity, a piece can sit at the
          top edge for forty frames, and a coach that keeps saying the same
          true thing is doing its job. What is measured is how long the
          freeze outlives the truth.
BOARD     Is the stack it would hand the solver the stack that is really
          there? A right name on a wrong board still draws the wrong
          square.
COVERAGE  How often the tracker says nothing at all.

THE WINDOW HEAD IS NOT SCORED. Every committed window is a mid-session
excerpt -- the first file in ``spawn_latency`` is ``board_00080.png`` -- but
a tracker is constructed fresh at its first frame, so a design with
cross-frame memory pays a once-per-session bootstrap SIX times here while a
memoryless one pays nothing. Measured on the shipped tracker: of its 98
frames with no piece named, 45 fall in a window's first ten frames, 28 in
the second ten, 5 in the third and 0 thereafter; wrong boards go 32, 14, 0.
:data:`WARMUP` frames at the head of each window are therefore replayed
into both trackers and scored for neither, which is what a mid-session
excerpt actually is. The choice is not delicate: the verdict is the same
anywhere from 15 to 30 (see ``tests/test_race.py``).

Episodes are derived from the oracle, not from either tracker, so both are
judged against the same piece boundaries. A piece's flight ends when the
oracle sees it become part of the settled stack, when the colour under it
changes, when the settled stack gains a tetromino's worth of cells at once
(a hard drop and the next spawn caught in the same frame), when it loses a
row's worth (a lock that cleared a line), or when the oracle abstains --
an abstention ends the episode rather than being guessed through.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from tetris_coach.race.runners import FrameOutput

Cell = tuple[int, int]

DEFAULT_TRUTH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "oracle_truth.json"

#: Frames a coach may go on holding a state the board has already left
#: before it is called stuck. A frame or two is the ordinary lag of any
#: tracker that confirms before it commits; five is a third of a second of
#: a hint for a piece the player no longer has.
STUCK_FLOOR = 5

#: Cells the settled stack must gain at once for a hard drop and the next
#: spawn to be read as two flights rather than one. A tetromino's worth.
MOVED_CELLS = 4

#: The bucket the head effect was measured in, in frames.
HEAD_BUCKET = 10

#: Frames at the head of each window that are replayed into both trackers
#: and scored for neither. Every window is a mid-session excerpt, so the
#: bootstrap a stateful design owes once per SESSION would otherwise be
#: charged once per WINDOW. Two buckets, which is where the head effect is
#: spent: the shipped tracker's silent frames per bucket run 45, 28, 5, 0
#: and its wrong boards 32, 14, 0.
WARMUP = 2 * HEAD_BUCKET


@dataclass(frozen=True)
class TruthFrame:
    """One frame of the answer sheet, as the race needs it."""

    frame: str
    verdict: str
    piece: str | None
    cells: frozenset[Cell]
    settled: frozenset[Cell]
    basis: str | None

    @property
    def answered(self) -> bool:
        """Is the falling-piece question answered on this frame?"""
        return self.verdict != "abstain" and self.piece is not None

    @property
    def content(self) -> frozenset[Cell]:
        """Everything the oracle can see on the board."""
        return self.settled | self.cells


@dataclass(frozen=True)
class TruthWindow:
    """One window's answer sheet, with the geometry it was read under."""

    window: str
    rows: int
    cols: int
    unobservable: frozenset[Cell]
    frames: list[TruthFrame]

    def settled_rows(self, frame: TruthFrame) -> tuple[int, ...]:
        """The oracle's settled stack as row bitmasks, covered cells dropped."""
        rows = [0] * self.rows
        for cell in frame.settled:
            if cell not in self.unobservable:
                rows[cell[0]] |= 1 << cell[1]
        return tuple(rows)

    def observable_rows(self, stack_rows: tuple[int, ...]) -> tuple[int, ...]:
        """A tracker's stack with its belief about covered cells dropped.

        The shipped tracker carries a belief for the cells under the NEXT
        panel and the prototype leaves them empty. Neither is observable,
        so neither is scored.
        """
        blind = [0] * self.rows
        for row, col in self.unobservable:
            blind[row] |= 1 << col
        return tuple(row & ~mask for row, mask in zip(stack_rows, blind, strict=True))


@dataclass(frozen=True)
class Episode:
    """One piece, from the frame it becomes visible to the frame it rests."""

    piece: str
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Identity:
    """How the names came out over the frames the oracle answers."""

    judged: int
    correct: int
    wrong: int
    silent: int

    def rate(self, count: int) -> float:
        return 100.0 * count / self.judged if self.judged else 0.0


@dataclass(frozen=True)
class StuckRun:
    """A stretch where the coach froze through a change of piece."""

    first: str  # frame the freeze began
    changed: str  # frame the piece in play became a different one
    last: str  # frame the freeze finally broke
    frames: int  # length of the whole frozen run
    stale: int  # frames it stayed frozen AFTER the piece changed
    held: str | None  # what the coach was still saying
    truth: str  # what was actually falling by then


@dataclass
class Score:
    """One tracker's result over one window."""

    window: str
    tracker: str
    frames: int
    identity: Identity
    latencies: list[int] = field(default_factory=list)  # frames, per measurable episode
    measurable: int = 0  # episodes whose naming latency can be read at all
    unnamed_episodes: int = 0  # episodes never named correctly at all
    name_flips: int = 0  # name changed mid-flight
    hint_flips: int = 0  # hint target changed mid-flight
    silent_frames: int = 0  # no falling piece named
    hintless_frames: int = 0  # nothing drawn on screen
    refused_frames: int = 0  # frames the tracker never got to see
    board_right: int = 0  # judged frames whose stack is the real stack
    worst_wrong_run: int = 0  # longest unbroken stretch of a wrong name
    worst_silent_run: int = 0  # longest unbroken stretch with no name
    stuck: list[StuckRun] = field(default_factory=list)
    episodes: int = 0
    warmup: int = 0  # head frames replayed into the tracker but scored for neither

    @property
    def worst_stuck(self) -> int:
        """Longest stretch spent holding a piece the board had moved past."""
        return max((run.stale for run in self.stuck), default=0)

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def max_latency(self) -> int | None:
        return max(self.latencies) if self.latencies else None

    @property
    def silent_rate(self) -> float:
        return 100.0 * self.silent_frames / self.frames if self.frames else 0.0


def load_truth(path: Path = DEFAULT_TRUTH) -> dict[str, TruthWindow]:
    """The committed answer sheet, one entry per window."""
    payload: dict[str, Any] = json.loads(path.read_text())
    out: dict[str, TruthWindow] = {}
    for window in payload["windows"]:
        frames = []
        for frame in window["frames"]:
            falling = frame.get("falling")
            frames.append(
                TruthFrame(
                    frame=frame["frame"],
                    verdict=frame["verdict"],
                    piece=None if falling is None else falling["piece"],
                    cells=frozenset()
                    if falling is None
                    else frozenset(tuple(cell) for cell in falling["cells"]),
                    settled=frozenset(tuple(cell) for cell in frame["settled"]),
                    basis=None if falling is None else falling["basis"],
                )
            )
        out[window["window"]] = TruthWindow(
            window=window["window"],
            rows=window["rows"],
            cols=window["cols"],
            unobservable=frozenset(tuple(cell) for cell in window["unobservable"]),
            frames=frames,
        )
    return out


def episodes(truth: list[TruthFrame]) -> list[Episode]:
    """Split a window into one episode per piece in flight."""
    out: list[Episode] = []
    open_at: int | None = None
    piece: str | None = None
    previous: TruthFrame | None = None
    for index, frame in enumerate(truth):
        if not frame.answered:
            if open_at is not None:
                out.append(Episode(str(piece), open_at, index - 1))
            open_at, piece, previous = None, None, None
            continue
        fresh = previous is None
        if previous is not None:
            locked = previous.cells <= frame.settled and bool(previous.cells)
            grew = len(frame.settled) - len(previous.settled) >= MOVED_CELLS
            cleared = len(previous.settled) - len(frame.settled) >= 10
            fresh = frame.piece != previous.piece or locked or grew or cleared
        if fresh and open_at is not None:
            out.append(Episode(str(piece), open_at, index - 1))
        if fresh:
            open_at, piece = index, frame.piece
        previous = frame
    if open_at is not None:
        out.append(Episode(str(piece), open_at, len(truth) - 1))
    return out


def _identity(outputs: list[FrameOutput], truth: list[TruthFrame], warmup: int = 0) -> Identity:
    judged = correct = wrong = silent = 0
    for index, (output, frame) in enumerate(zip(outputs, truth, strict=True)):
        if index < warmup or not frame.answered:
            continue
        judged += 1
        if output.piece is None:
            silent += 1
        elif output.piece == frame.piece:
            correct += 1
        else:
            wrong += 1
    return Identity(judged=judged, correct=correct, wrong=wrong, silent=silent)


def _longest(flags: list[bool]) -> int:
    """Length of the longest unbroken run of ``True``."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _stuck(
    outputs: list[FrameOutput], truth: list[TruthFrame], plan: list[Episode], warmup: int = 0
) -> list[StuckRun]:
    """Frozen runs of accepted frames that outlive the piece they hold.

    A run is maximal: consecutive accepted frames whose coach state -- the
    piece it names and the stack it would hand the solver -- never changes.
    It is only a fault if a new piece came into play inside it, and only
    reported once the coach has gone on holding the old state for
    ``STUCK_FLOOR`` frames after that.
    """
    starts = {episode.start: episode for episode in plan[1:]}
    runs: list[StuckRun] = []
    index = 0
    while index < len(outputs):
        if not outputs[index].accepted:
            index += 1
            continue
        end = index
        while (
            end + 1 < len(outputs)
            and outputs[end + 1].accepted
            and outputs[end + 1].state == outputs[index].state
        ):
            end += 1
        inside = [at for at in range(index + 1, end + 1) if at in starts and at >= warmup]
        if inside:
            changed = inside[0]
            stale = end - changed + 1
            if stale >= STUCK_FLOOR:
                runs.append(
                    StuckRun(
                        first=outputs[index].frame,
                        changed=outputs[changed].frame,
                        last=outputs[end].frame,
                        frames=end - index + 1,
                        stale=stale,
                        held=outputs[index].piece,
                        truth=starts[changed].piece,
                    )
                )
        index = end + 1
    return runs


def measurable(plan: list[Episode]) -> list[Episode]:
    """Episodes whose naming latency means anything.

    A tracker holding the last piece's name gets the next one free when the
    two are the same letter -- this game deals three I pieces in a row in
    two of these windows -- so an episode whose predecessor was the same
    piece cannot be used to time anything. The first episode of a window
    counts: the coach is starting cold there, which is a real wait.
    """
    out = []
    for position, episode in enumerate(plan):
        if position == 0 or plan[position - 1].piece != episode.piece:
            out.append(episode)
    return out


def score(
    tracker: str,
    outputs: list[FrameOutput],
    truth: TruthWindow,
    warmup: int = WARMUP,
) -> Score:
    """Everything the race measures, for one tracker over one window.

    ``warmup`` head frames are replayed into the tracker by the runner and
    then scored for NEITHER side: a mid-session excerpt should not charge a
    stateful design for a bootstrap the live session paid once, long before
    the first committed frame. Episodes that open inside the warm-up are
    excluded whole, because a tracker may have named such a piece during
    the unscored frames and its latency would read as zero for free.
    """
    frames = truth.frames
    plan = episodes(frames)
    scored = outputs[warmup:]
    judged = [
        (output, frame)
        for output, frame in zip(scored, frames[warmup:], strict=True)
        if frame.answered
    ]
    result = Score(
        window=truth.window,
        tracker=tracker,
        frames=len(scored),
        identity=_identity(outputs, frames, warmup),
        silent_frames=sum(1 for output in scored if output.piece is None),
        hintless_frames=sum(1 for output in scored if output.hint is None),
        refused_frames=sum(1 for output in scored if not output.accepted),
        board_right=sum(
            1
            for output, frame in judged
            if truth.observable_rows(output.stack_rows) == truth.settled_rows(frame)
        ),
        worst_wrong_run=_longest(
            [output.piece is not None and output.piece != frame.piece for output, frame in judged]
        ),
        worst_silent_run=_longest([output.piece is None for output, _ in judged]),
        stuck=_stuck(outputs, frames, plan, warmup),
        episodes=sum(1 for episode in plan if episode.start >= warmup),
        warmup=warmup,
    )
    timed = set(measurable(plan))
    for episode in plan:
        if episode.start < warmup:
            continue
        span = range(episode.start, episode.end + 1)
        named = [index for index in span if outputs[index].piece == episode.piece]
        if episode in timed:
            result.measurable += 1
            if named:
                result.latencies.append(named[0] - episode.start)
        if not named:
            result.unnamed_episodes += 1
            continue
        # Churn is counted from the frame the piece is first named
        # correctly: before that the coach is late, which latency already
        # charges it for, and charging the same frames twice would make a
        # slow tracker look unstable as well.
        settled = outputs[named[0] : episode.end + 1]
        names = [output.piece for output in settled if output.piece is not None]
        result.name_flips += sum(1 for a, b in pairwise(names) if a != b)
        hints = [output.hint for output in settled if output.hint is not None]
        result.hint_flips += sum(1 for a, b in pairwise(hints) if a != b)
    return result
