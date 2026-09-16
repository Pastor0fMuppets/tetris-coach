"""What the overlay actually shows, pinned, for both tracker choices.

The tracker-level race lives in ``test_race.py``; this is the same eight
windows replayed through :class:`~tetris_coach.app.CoachEngine` itself, so
what is asserted here is what a user would have seen on screen. The
numbers, scored from frame ``WARMUP`` of each window:

    tracker  hints  MISNAMED  INVENTED  stale  latmed  latmax  moves  hintless
    colour     470         0         0     21     0.0       0      0         4
    shape      442         7         0     49     1.0       3      3        32

``hint_stutter`` and ``stray_after_clear`` are the two windows committed
for the flashing hint the user reported, and this is the table that
measures what they were reported FOR: with the colour tracker both now
score 0 moves, 0 misnamed and 0 hintless, on 31 and 21 scored frames. On
the reading that shipped, hint_stutter alone put the hint on a different
target and back 31 times in 51 frames.

MISNAMED is the one that matters: a placement drawn for a piece the player
does not have walks them into a hole and looks exactly like a placement
that is right. The colour tracker draws none in this corpus; the shipped
tracker draws six, each on the frame a new piece arrives while its
committed state still holds the piece that just locked.

INVENTED covers the stretches MISNAMED cannot reach, because the oracle
abstains on them: line clears and covered boards. It is not a measure this
race always had, and it was added because the colour reader was drawing an
``I`` over four frames of spawn_latency's line-clear flash and scoring a
clean sheet on every other number here.

``stale`` is not a fault by itself -- it is the deliberate hold that rides
out a glitch (``CoachConfig.max_stale_frames``), and every run of it here
is far inside that patience. It is counted because the same hold is what
left a hint parked over a game-over screen for 21.7 s before the
withdrawal rule existed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from tetris_coach.app import CoachConfig
from tetris_coach.race.engine import (
    TRACKERS,
    Shown,
    ShownFrame,
    overall,
    replay,
    run,
    score,
    table,
)
from tetris_coach.race.measures import WARMUP, TruthFrame, TruthWindow, load_truth
from tetris_coach.truth.windows import CONSECUTIVE, by_name

FIXTURES = Path(__file__).parent / "fixtures"
WINDOWS = tuple(spec.name for spec in CONSECUTIVE)


@lru_cache(maxsize=1)
def scores() -> dict[str, dict[str, Shown]]:
    return run(FIXTURES)


@lru_cache(maxsize=2)
def totals(tracker: str) -> Shown:
    return overall(scores(), tracker)


def test_the_default_tracker_is_the_colour_one() -> None:
    # The whole point of the adoption: what a user gets without a flag.
    assert CoachConfig().tracker == "colour"
    assert TRACKERS[0] == "colour"


@pytest.mark.parametrize(
    ("tracker", "hints", "misnamed", "invented", "stale", "moves", "hintless"),
    [
        ("colour", 470, 0, 0, 21, 0, 4),
        ("shape", 442, 7, 0, 49, 3, 32),
    ],
)
def test_what_each_tracker_puts_on_screen(
    tracker: str,
    hints: int,
    misnamed: int,
    invented: int,
    stale: int,
    moves: int,
    hintless: int,
) -> None:
    total = totals(tracker)
    assert (
        total.hinted,
        total.misnamed,
        total.invented,
        total.stale,
        total.moves,
        total.hintless,
    ) == (hints, misnamed, invented, stale, moves, hintless)
    assert total.frames == 474


def test_no_frame_of_this_corpus_shows_a_hint_for_the_wrong_piece() -> None:
    # The headline. A hint can only be judged on a frame the oracle
    # answers AND the coach was drawing something: with the colour tracker
    # that is all 449 of the answered frames -- the overlay is never blank
    # when the oracle can say what is falling -- and none of them names the
    # wrong piece.
    total = totals("colour")
    assert total.misnamed == 0
    assert total.judged == 449
    shape = totals("shape")
    assert shape.misnamed == 7
    assert shape.judged == 417  # and blank on 32 answered frames besides


def test_a_hint_arrives_the_frame_the_piece_does() -> None:
    colour, shape = totals("colour"), totals("shape")
    assert colour.median_latency == 0.0
    assert colour.max_latency == 0
    assert colour.never_hinted == 0
    # The shipped tracker is a frame or three late and, in one window,
    # never hints a piece at all.
    assert shape.median_latency == 1.0
    assert shape.max_latency == 3
    assert shape.never_hinted == 2
    # Seven of the twelve scored episodes can time anything at all: an
    # episode whose predecessor was the same letter gets its name free.
    assert colour.measurable == shape.measurable == 7
    assert len(colour.latencies) == 7
    assert len(shape.latencies) == 5  # two were never hinted at all


def test_a_target_never_moves_while_its_piece_is_in_flight() -> None:
    # HINT_SWITCH_MARGIN and the hold-while-in-flight rule, measured where
    # the user sees them rather than at the solver.
    assert totals("colour").moves == 0
    assert totals("colour").worst_moves == 0


def test_every_stale_hold_is_inside_the_patience_it_is_allowed() -> None:
    # A held hint is only defensible while it is short. Both trackers hold
    # over the same fourteen pale_piece frames, where the capture is
    # showing a web page and not a board at all; nothing in this corpus
    # comes near the 45-frame withdrawal.
    patience = CoachConfig().max_stale_frames
    for tracker in TRACKERS:
        for window in scores().values():
            assert window[tracker].stale <= patience


@pytest.mark.parametrize("window", WINDOWS)
def test_the_colour_tracker_regresses_nothing_window_by_window(window: str) -> None:
    # The adoption's condition: not better on average, better or equal on
    # every window and every measure the user feels.
    colour, shape = scores()[window]["colour"], scores()[window]["shape"]
    assert colour.misnamed <= shape.misnamed
    assert colour.stale <= shape.stale
    assert colour.moves <= shape.moves
    assert colour.hintless <= shape.hintless
    assert colour.never_hinted <= shape.never_hinted
    assert colour.hinted >= shape.hinted
    if colour.latencies and shape.latencies:
        assert max(colour.latencies) <= max(shape.latencies)


def test_the_replay_is_the_engine_the_app_runs() -> None:
    # Not a re-implementation: the harness drives CoachEngine.process_frame
    # and reads the hint it returns, frame by frame, with the window's own
    # covered cells -- the same wiring app.run does.
    spec = by_name("live_session")
    shown = replay(FIXTURES, spec, "colour")
    assert len(shown) == len(load_truth()["live_session"].frames)
    assert all(frame.hint is None or frame.hint[0] == frame.piece for frame in shown)


def test_the_score_is_read_off_the_scored_frames_only() -> None:
    # The warm-up is replayed into the engine and scored for neither
    # tracker, exactly as in the tracker-level race.
    spec = by_name("live_session")
    truth = load_truth()["live_session"]
    shown = replay(FIXTURES, spec, "colour")
    assert score("colour", shown, truth, warmup=0).frames == len(shown)
    assert score("colour", shown, truth).frames == len(shown) - WARMUP


def test_the_table_renders() -> None:
    text = table(scores())
    assert "MISNAMED" in text
    assert "INVENTED" in text
    assert "colour" in text and "shape" in text
    assert text.count("ALL") == 2


def _answer(name: str, piece: str | None) -> TruthFrame:
    """One frame of an answer sheet: a named piece, or an abstention."""
    verdict = "abstain" if piece is None else "confident"
    cells = frozenset() if piece is None else frozenset({(0, 0)})
    return TruthFrame(name, verdict, piece, cells, frozenset(), "shape")


def _drawing(piece: str | None) -> ShownFrame:
    hint = None if piece is None else (piece, ((11, 0),))
    return ShownFrame(frame="x", accepted=True, piece=piece, hint=hint)


def test_a_hint_invented_across_an_abstention_is_counted() -> None:
    # The hole INVENTED closes. The oracle abstains for three frames --
    # a line clear, say -- and says O on either side of them. A hint for
    # an I in the middle is a piece the player never had, and no other
    # measure in this table can see it: MISNAMED skips unanswered frames,
    # and the coach was not blank, so hintless does not count it either.
    frames = [_answer("1", "O"), *(_answer(str(n), None) for n in (2, 3, 4)), _answer("5", "O")]
    truth = TruthWindow("synthetic", 12, 10, frozenset(), frames)
    invented = [_drawing("O"), _drawing("I"), _drawing("I"), _drawing(None), _drawing("O")]
    assert score("colour", invented, truth, warmup=0).invented == 2
    assert score("colour", invented, truth, warmup=0).misnamed == 0

    # Holding the hint the piece before and after the gap justifies is the
    # patience the engine is designed to have, and is not counted.
    held = [_drawing("O")] * 5
    assert score("colour", held, truth, warmup=0).invented == 0

    # Neither is reading the NEXT piece early: an abstention that ends in
    # a T excuses a hint for a T inside it.
    frames[-1] = _answer("5", "T")
    early = TruthWindow("synthetic", 12, 10, frozenset(), frames)
    assert score("colour", [_drawing("O"), *([_drawing("T")] * 4)], early, warmup=0).invented == 0
