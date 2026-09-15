"""What the coach shows while a piece is still entering, measured.

``tests/fixtures/spawn_latency/`` holds 161 CONSECUTIVE ticks (00080-00240)
of the session the user was watching when he said: "The hint doesn't show
until the piece has fully dropped both rows onto the board. Ideally we
would store the piece from the next piece viewer to speed things up so it
can show when the new piece is only partially visible as it is falling."
See the fixtures' README for the geometry and for the cost measured over
the whole 705-frame session it came from.

The wait is deliberate, not a bug: this game parks a spawn at the top edge
of the board region, where 1-3 of its cells are on the grid and the rest
is above the capture, and two cells side by side fit an O, an S, a Z, a J
and an L alike. ``explain_grid`` holds rather than guess, the frame reads
OCCLUDED, and nothing is on screen until the piece's second row descends.

This module replays the window through ``CoachEngine`` and MEASURES that
wait, per entering piece, so the accelerator is judged against it rather
than against a description of it. The three episodes here, what each has
to work with, and what each costs now:

    frames 00086-00102  the window OPENS on a piece already at the top
                        edge. No preview CHANGE has been seen (the box
                        holds an O from the first frame, and None -> X
                        says nothing about what was dealt), so there is
                        no evidence and the rule holds: 17 frames, and
                        17 frames still.
    frames 00161-00174  the preview flips O -> I on 00158, which is the
                        game saying the O is what it just dealt — the
                        very fragment sitting at the top edge. This is
                        the one the preview CAN name, and it now does:
                        14 frames -> 2, which is the commit debounce,
                        with the O still showing two of its four cells.
                        What used to throw the evidence away was the
                        board reset four frames later: the game had
                        wiped the field, and the resync dropped the hint
                        that named the piece it was resyncing ONTO.
    frames 00217-00232  the flip here is T -> O, and it took the band
                        pass to have it at all: the T that was dealt sat
                        in the box from 00198 to 00215 drawn in a pale
                        blue that scores 0.319, under the 0.35 floor the
                        threshold is anchored at, so the box read EMPTY
                        for 18 captures and the flip on the far side was
                        dated across the whole deal and refused as a
                        straddle. Read on every capture, the flip lands
                        on the frame the T is dealt and names the
                        fragment at the top edge: 16 frames -> 2, and the
                        shape rule confirms the same T on 00233.

So the accelerator pays where there is evidence and nowhere else, which
is the point — it is an accelerator on a signal that is often present,
not a new dependency. What the band pass changed here is how often it is
present: the box is read on 150 of the window's 161 captures, against
132 before, and the 18 it gained are the whole of the third episode's
evidence.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from PIL import Image

from tetris_coach.app import CoachConfig, CoachEngine, compute_overlap_mask
from tetris_coach.capture.screen import Rect
from tetris_coach.solver.search import Move
from tetris_coach.vision.pieces_vision import FrameKind, identify_next
from tetris_coach.vision.state import MAX_HINT_AGE, MAX_PREVIEW_GAP, GameEvent

FIXTURES = Path(__file__).parent / "fixtures" / "spawn_latency"
ROWS = 12
GATE = CoachConfig().min_confidence

SESSION_BOARD = Rect(left=204, top=311, width=480, height=577)
SESSION_NEXT = Rect(left=587, top=309, width=99, height=102)
COVERED = frozenset({(0, 8), (0, 9), (1, 8), (1, 9)})

# The frames the confidence gate never lets through: the window opens
# mid-animation, and 00127-00157 is the game's end-of-round wipe.
REJECTED = [*(f"000{n}" for n in range(80, 86)), *(f"00{n}" for n in range(127, 158))]


def load(name: str) -> np.ndarray:
    """One captured frame as the capture pipeline would hand it over (BGR)."""
    return np.asarray(Image.open(FIXTURES / name))[:, :, ::-1]


def frame_numbers() -> list[str]:
    return [p.stem.split("_")[1] for p in sorted(FIXTURES.glob("board_*.png"))]


class Tick:
    """What one replayed frame did."""

    def __init__(
        self,
        number: str,
        confidence: float,
        kind: FrameKind | None,
        events: list[GameEvent],
        observed: np.ndarray,
        stack_rows: tuple[int, ...],
        falling: str | None,
        next_piece: str | None,
        read_next: str | None,
        hint: Move | None,
    ) -> None:
        self.number = number
        self.confidence = confidence
        self.kind = kind
        self.events = events
        self.observed = observed  # the occupancy the tracker was handed
        self.stack_rows = stack_rows
        self.falling = falling
        self.next_piece = next_piece  # committed
        self.read_next = read_next  # what the preview reader said this capture
        self.hint = hint

    @property
    def accepted(self) -> bool:
        return self.confidence >= GATE

    @property
    def entering_cells(self) -> list[int]:
        """Columns of row 0 the committed stack does NOT hold — a piece in."""
        return [c for c in range(10) if self.observed[0, c] and not self.stack_rows[0] >> c & 1]


@lru_cache(maxsize=1)
def replay() -> tuple[Tick, ...]:
    """Feed the whole window through CoachEngine, wired as app.run wires it."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS, tracker="shape"), unobservable_cells=covered)
    update = engine.tracker.update
    seen: list[GameEvent] = []
    handed: list[np.ndarray] = []
    read: list[str | None] = []

    def spy(occupancy, next_piece):  # type: ignore[no-untyped-def]
        handed.append(np.array(occupancy, copy=True))
        read.append(next_piece)
        events = update(occupancy, next_piece)
        seen.extend(events)
        return events

    observe = engine.tracker.observe_preview

    def preview_spy(next_piece):  # type: ignore[no-untyped-def]
        # The rejected captures: no board frame, but the box is still read
        # and the tracker still dates its evidence by them.
        read.append(next_piece)
        observe(next_piece)

    engine.tracker.update = spy  # type: ignore[method-assign]
    engine.tracker.observe_preview = preview_spy  # type: ignore[method-assign]
    ticks: list[Tick] = []
    for number in frame_numbers():
        board = load(f"board_{number}.png")
        preview = FIXTURES / f"next_{number}.png"
        seen.clear()
        handed.clear()
        read.clear()
        # The classifier is stateful, so this must read the same frame the
        # engine is about to digest — classify() is a pure function of the
        # image plus the memory, and the engine re-runs it identically.
        _occupancy, confidence = engine.classifier.classify(board)
        hint = engine.process_frame(board, load(preview.name) if preview.exists() else None)
        committed = engine.tracker.committed
        ticks.append(
            Tick(
                number=number,
                confidence=confidence,
                kind=engine.tracker.last_kind,
                events=list(seen),
                observed=handed[0] if handed else np.zeros((ROWS, 10), dtype=bool),
                stack_rows=committed.stack_rows,
                falling=committed.falling_piece,
                next_piece=committed.next_piece,
                read_next=read[0] if read else None,
                hint=hint,
            )
        )
    return tuple(ticks)


def tick(number: str) -> Tick:
    return next(t for t in replay() if t.number == number)


def entering_episodes() -> list[tuple[str, str | None, int | None]]:
    """``(first sighting, first hint, frames between)`` per entering piece.

    A SIGHTING is the first accepted frame showing cells at row 0 that the
    committed stack does not hold and that nothing on screen names yet — a
    piece entering, with no hint for it. The episode closes on the first
    accepted frame carrying a hint for a named falling piece. This is the
    user's own measure: what he sees is the gap between the piece
    appearing and something appearing on it.
    """
    episodes: list[tuple[str, str | None, int | None]] = []
    opened: tuple[int, str] | None = None
    for index, t in enumerate(replay()):
        if not t.accepted:
            continue
        named = t.falling is not None and t.hint is not None
        if named:
            if opened is not None:
                episodes.append((opened[1], t.number, index - opened[0]))
                opened = None
        elif t.entering_cells and opened is None:
            opened = (index, t.number)
        elif opened is not None and not t.entering_cells:
            episodes.append((opened[1], None, None))  # gone without ever being named
            opened = None
    if opened is not None:
        episodes.append((opened[1], None, None))
    return episodes


class Replay(NamedTuple):
    """One pass of the window: what was hinted, and what it committed."""

    hints: list[tuple[str, str | None]]  # (frame, "piece@col") per capture
    events: Counter[GameEvent]
    stack: tuple[int, ...]


def replay_window() -> Replay:
    """The window again, on a fresh engine, reading the preview as patched."""
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS, tracker="shape"), unobservable_cells=covered)
    events: Counter[GameEvent] = Counter()
    update = engine.tracker.update

    def spy(occupancy, next_piece):  # type: ignore[no-untyped-def]
        seen = update(occupancy, next_piece)
        events.update(seen)
        return seen

    engine.tracker.update = spy  # type: ignore[method-assign]
    hints: list[tuple[str, str | None]] = []
    for number in frame_numbers():
        preview = FIXTURES / f"next_{number}.png"
        hint = engine.process_frame(
            load(f"board_{number}.png"), load(preview.name) if preview.exists() else None
        )
        hints.append((number, f"{hint.piece}@{hint.col}" if hint is not None else None))
    return Replay(hints, events, engine.tracker.committed.stack_rows)


class HintLife(NamedTuple):
    """The tracker's own hint bookkeeping after one capture."""

    number: str
    age: int  # captures since the deal this hint reports
    holding: str | None  # the committed name this hint is holding up
    alive: str | None  # the hint itself, still able to name a fragment


def hint_life() -> list[HintLife]:
    """One pass of the window, recording the tracker's hint clock.

    So the budget that expires a hint is set against what a hint
    MEASURABLY does here rather than against a description of it.
    """
    covered = compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS)
    engine = CoachEngine(CoachConfig(rows=ROWS, tracker="shape"), unobservable_cells=covered)
    out: list[HintLife] = []
    for number in frame_numbers():
        preview = FIXTURES / f"next_{number}.png"
        engine.process_frame(
            load(f"board_{number}.png"), load(preview.name) if preview.exists() else None
        )
        tracker = engine.tracker
        out.append(
            HintLife(number, tracker._hint_age, tracker._hinted_commit, tracker._entering_hint)
        )
    return out


def test_no_hint_here_is_still_doing_its_job_when_the_budget_runs_out() -> None:
    # The lower anchor on MAX_HINT_AGE, measured on the frames rather than
    # argued. A hint expires on its own clock now, because every other
    # rule that ends one waits for an event (a lock, a resync, a frame
    # that contradicts it) and a HOLD swap or a restart produces none of
    # them — so a name nothing can contradict used to stand for the rest
    # of the session. The clock's cost is a hint cut short, and the budget
    # has to clear the longest one that is still doing its job.
    #
    # Here that is 15 captures, and TWICE: the flip on 00158 names the O,
    # the name is on screen from 00163, and the shape rule takes over on
    # 00174 at age 16; the flip on 00217 names the T, on screen from
    # 00219, and the shape rule reaches it on 00232, again at age 16. The
    # budget clears both by nine.
    holding = [t for t in hint_life() if t.holding is not None]
    assert [t.number for t in holding] == [
        *(f"00{n}" for n in range(163, 174)),
        *(f"00{n}" for n in range(219, 232)),
    ]
    assert max(t.age for t in holding) == 15
    assert MAX_HINT_AGE > 15
    # ...and why the clock is worth having, on these same frames: a hint
    # goes on LIVING long after it stops holding a name up. This one is
    # spent at age 16, when the shape rule names the O itself, and the
    # only thing that would ever have ended it is the O's lock on 00198,
    # at age 40 — 24 further captures of a live hint able to name
    # whatever turned up at the top edge. The clock ends it on 00182
    # instead. Every other rule needs an event like that lock, and a hold
    # swap or a restart provides none at all.
    alive = [t for t in hint_life() if t.alive is not None]
    assert max(t.age for t in alive) == MAX_HINT_AGE
    first = [t for t in alive if t.alive == "O"]
    assert first[-1].number == "00182"  # was 00198, the lock
    # Costing nothing here: the name was the shape rule's own eight
    # captures before the budget ran out, so nothing is withdrawn.
    assert GameEvent.PIECE_UNNAMED not in replay_window().events


def test_a_preview_that_lies_costs_the_ambiguous_window_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The price of the accelerator, measured rather than asserted away.
    # Naming a clipped fragment from the preview means a preview that is
    # WRONG names it wrong, and no rule can do better than the frame: two
    # cells side by side fit an O and an S alike, so while the piece sits
    # at the top edge showing two cells there is nothing to contradict the
    # name. Here every O the box shows is read as an S — a consistent lie,
    # which is the worst case, since a wobbling one is refused by the
    # preview debounce before it can flip anything.
    #
    # What that costs is 12 frames (0.80 s) of a wrong hint on the ONE
    # piece whose name came from the preview, ending the moment the O's
    # second row descends and rules the S out. What it does NOT cost is
    # anything structural: the same committed stack at the end of the
    # window, the same locks, the same resets. A hinted name is never
    # evidence, and this is the ceiling on what a wrong one can do.
    import tetris_coach.vision.readers as readers_module

    honest = replay_window()
    real = readers_module.identify_next
    monkeypatch.setattr(
        readers_module,
        "identify_next",
        lambda image, **kwargs: "S" if real(image, **kwargs) == "O" else real(image, **kwargs),
    )
    lying = replay_window()

    differing = [
        (n, h, li) for (n, h), (_, li) in zip(honest.hints, lying.hints, strict=True) if h != li
    ]
    assert [n for n, _, _ in differing] == [f"00{n}" for n in range(163, 175)]
    assert {li for _, _, li in differing} == {"S@0"}  # the phantom, 12 frames
    assert dict(honest.hints)["00175"] == "O@0"  # the frame the shape corrects it
    assert dict(lying.hints)["00175"] == "O@0"
    assert lying.stack == honest.stack
    assert lying.events[GameEvent.PIECE_LOCKED] == honest.events[GameEvent.PIECE_LOCKED] == 2
    assert lying.events[GameEvent.BOARD_RESET] == honest.events[GameEvent.BOARD_RESET] == 2


def test_the_window_is_the_readme_geometry() -> None:
    assert compute_overlap_mask(SESSION_BOARD, SESSION_NEXT, rows=ROWS) == COVERED
    assert len(frame_numbers()) == 161
    assert frame_numbers()[0] == "00080"
    assert frame_numbers()[-1] == "00240"


def test_the_window_replays_without_an_unexplainable_surprise() -> None:
    # The baseline this measurement stands on. Two resets, both real: the
    # window opens mid-game on a board no memory explains, and the game
    # wipes the field at 00127-00157 and deals a new one.
    events = Counter(e for t in replay() for e in t.events)
    assert events[GameEvent.BOARD_RESET] == 2
    assert events[GameEvent.PIECE_LOCKED] == 2
    assert events[GameEvent.PIECE_SPAWNED] == 4
    assert [t.number for t in replay() if not t.accepted] == REJECTED
    kinds = Counter(t.kind for t in replay() if t.accepted and t.kind is not None)
    assert kinds[FrameKind.UNEXPLAINED] == 8  # the two resets' confirmation runs


def test_the_piece_the_preview_can_name_is_hinted_while_still_half_off_screen() -> None:
    # The headline, and the user's own measure: the gap between a piece
    # appearing at the top edge and something appearing on it.
    #
    #     sighting -> hint     before   after
    #     00086                    17      17   no flip has been seen
    #     00161                    14       2   the flip names the O
    #     00217                    16       2   the flip names the T
    #
    assert entering_episodes() == [
        ("00086", "00103", 17),
        ("00161", "00163", 2),
        ("00217", "00219", 2),
    ]
    # Two frames is the commit debounce, not a wait: the name is there on
    # 00162, the frame after the resync, and the tracker commits any
    # candidate on its second identical frame.
    assert tick("00162").kind is FrameKind.FALLING
    assert tick("00163").falling == "O" and tick("00163").hint is not None
    # ...and what it is naming is a piece the capture can only half see:
    # two cells on the grid, of a tetromino that has four. This is the
    # thing the user asked for — a hint while the piece is still falling
    # into view, instead of after it has finished arriving.
    assert tick("00162").entering_cells == [4, 5]
    assert tick("00163").entering_cells == [4, 5]
    # The proof the name was right rather than merely early: the same
    # piece keeps descending, the shape rule reaches it at 00174, and it
    # is the same O — no second spawn, no hint changing target.
    assert tick("00174").kind is FrameKind.FALLING
    assert all(tick(f"00{n}").falling == "O" for n in range(163, 199))


def test_the_episode_with_no_evidence_still_holds_rather_than_guesses() -> None:
    # The other half of the headline. Nothing about the first episode
    # changed, and nothing about it should: the window opens on a piece
    # already at the top edge, no preview CHANGE has been seen, and a box
    # that has always held an O says nothing about what was dealt. A hint
    # withheld beats a hint confidently wrong.
    waiting = [tick(f"{n:05d}") for n in range(90, 102)]
    assert all(t.kind is FrameKind.OCCLUDED for t in waiting)
    assert all(t.falling is None and t.hint is None for t in waiting)
    assert [t.entering_cells for t in waiting] == [[4, 5]] * 12
    assert {t.read_next for t in waiting} == {"O"}  # read, and still no flip


def test_the_third_episode_is_named_by_a_flip_the_band_pass_made_readable() -> None:
    # The episode this window was committed to measure and the band pass
    # is what moved. Before, the box read empty from 00198 to 00215 while
    # a pale T sat in it, the I -> O flip on 00216 was dated 19 captures
    # back across the whole deal, the tracker refused it as a straddle,
    # and these fourteen frames were OCCLUDED with nothing on them.
    named = [tick(f"00{n}") for n in range(219, 232)]
    assert all(t.kind is FrameKind.FALLING for t in named)
    assert all(t.falling == "T" and t.hint is not None for t in named)
    # Three cells of a T sliding across the top edge — the fragment the
    # flip names, and a fragment an S, a Z, a J and an L fit as well.
    assert [t.entering_cells for t in named] == [[3, 4, 5]] * 13
    # The proof the name was right rather than merely early: the shape
    # rule reaches the same T on 00233 and it holds to the end of the
    # window — no second spawn, no hint changing target.
    assert tick("00233").falling == "T"
    assert all(tick(f"00{n}").falling == "T" for n in range(219, 241))
    assert {t.hint.piece for t in named if t.hint is not None} == {"T"}


def test_the_wait_is_a_quarter_of_the_frames_the_gate_accepts() -> None:
    # The aggregate the README reports over the whole session (164 of 705
    # OCCLUDED, 23%), on the committed window. Every OCCLUDED frame is one
    # where the coach knows a piece is there and cannot name it; every
    # frame it stops being one is a frame with a hint on it.
    accepted = [t for t in replay() if t.accepted]
    kinds = Counter(t.kind for t in accepted if t.kind is not None)
    assert len(accepted) == 124
    assert kinds[FrameKind.OCCLUDED] == 17  # was 43, then 31
    assert sum(1 for t in replay() if t.hint is not None) == 134  # was 108, then 120


def test_the_flip_that_names_the_O_is_dated_across_the_wipe_that_dealt_it() -> None:
    # The dating rule, on the capture sequence it has to survive. The gate
    # rejects the whole end-of-round wipe (00127-00157), and the tracker
    # used to see nothing at all through it: the flip on 00158 was dated
    # against 00126, 32 captures and a whole deal earlier, which is the
    # straddle the rule exists to refuse. The box is now read on every
    # capture, so the same flip is dated against 00151 — the wipe blanks
    # the box for its last six frames only, and six is far short of a
    # tenure, so no previewed piece can have come and gone inside it.
    assert [tick(f"00{n}").read_next for n in range(149, 160)] == [
        *("O", "O", "O"),  # 00149-00151: rejected captures, box readable
        *(None,) * 6,  # 00152-00157: the wipe blanks the box too
        *("I", "I"),  # 00158: the flip, six captures after the last O
    ]
    assert 6 <= MAX_PREVIEW_GAP
    # ...and the flip is believed: the O it names is the fragment the wipe
    # dealt, hinted while two of its four cells are still off screen.
    assert tick("00163").falling == "O"


def test_the_preview_is_read_on_most_frames_including_the_pale_T() -> None:
    # The signal's availability. The box is read on 150 of the window's
    # 161 captures — 119 of the 124 the board gate accepts, plus 31 it
    # rejects, which count for the dating rule just the same.
    #
    # 18 of those 150 are the gap this window was committed to show: the
    # T at 00198-00215 is drawn in a pale blue scoring 0.319 against the
    # box's background, under the 0.35 uniformity floor the board reader
    # and the preview reader share, so the threshold's mask kept nothing
    # and the box read EMPTY. The band pass reads them, and the third
    # episode's flip stops arriving a deal late.
    assert sum(1 for t in replay() if t.read_next is not None) == 150  # was 132
    assert sum(1 for t in replay() if t.accepted and t.read_next is not None) == 119
    assert all(identify_next(load(f"next_00{n}.png")) == "T" for n in range(198, 216))
    assert identify_next(load("next_00197.png")) == "I"
    assert identify_next(load("next_00216.png")) == "O"
    # The 11 still unread are the end-of-round wipe, which blanks the box
    # along with the field: nothing is in it to read.
    unread = [t.number for t in replay() if t.read_next is None]
    assert unread == [f"00{n}" for n in (*range(121, 126), *range(152, 158))]


def test_the_flip_that_names_the_second_piece_is_on_the_frame_it_enters() -> None:
    # The evidence the accelerator has to work with, in this window. The
    # game wipes the field, deals an O, and the preview flips O -> I on
    # 00158 — the same frame the O's first two cells appear at the top
    # edge. The shape rule agrees on 00174, when its second row descends.
    assert tick("00157").next_piece == "O"  # committed before the wipe
    assert [tick(f"00{n}").entering_cells for n in range(158, 162)] == [[4, 5]] * 4
    assert tick("00174").kind is FrameKind.FALLING
    assert tick("00175").falling == "O"
