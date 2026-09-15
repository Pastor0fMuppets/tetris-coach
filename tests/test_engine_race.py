"""What the overlay actually shows, pinned, for both tracker choices.

The tracker-level race lives in ``test_race.py``; this is the same six
windows replayed through :class:`~tetris_coach.app.CoachEngine` itself, so
what is asserted here is what a user would have seen on screen. The
numbers, scored from frame ``WARMUP`` of each window:

    tracker  hints  MISNAMED  stale  latmed  latmax  moves  hintless
    colour     415         0     14     0.0       0      0         7
    shape      397         6     49     1.0       3      3        25

MISNAMED is the one that matters: a placement drawn for a piece the player
does not have walks them into a hole and looks exactly like a placement
that is right. The colour tracker draws none in this corpus; the shipped
tracker draws six, each on the frame a new piece arrives while its
committed state still holds the piece that just locked.

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
from tetris_coach.race.engine import TRACKERS, Shown, overall, replay, run, score, table
from tetris_coach.race.measures import WARMUP, load_truth
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
    ("tracker", "hints", "misnamed", "stale", "moves", "hintless"),
    [
        ("colour", 415, 0, 14, 0, 7),
        ("shape", 397, 6, 49, 3, 25),
    ],
)
def test_what_each_tracker_puts_on_screen(
    tracker: str, hints: int, misnamed: int, stale: int, moves: int, hintless: int
) -> None:
    total = totals(tracker)
    assert (total.hinted, total.misnamed, total.stale, total.moves, total.hintless) == (
        hints,
        misnamed,
        stale,
        moves,
        hintless,
    )
    assert total.frames == 422


def test_no_frame_of_this_corpus_shows_a_hint_for_the_wrong_piece() -> None:
    # The headline. A hint can only be judged on a frame the oracle
    # answers AND the coach was drawing something: with the colour tracker
    # that is all 397 of the answered frames -- the overlay is never blank
    # when the oracle can say what is falling -- and none of them names the
    # wrong piece.
    total = totals("colour")
    assert total.misnamed == 0
    assert total.judged == 397
    shape = totals("shape")
    assert shape.misnamed == 6
    assert shape.judged == 372  # and blank on 25 answered frames besides


def test_a_hint_arrives_the_frame_the_piece_does() -> None:
    colour, shape = totals("colour"), totals("shape")
    assert colour.median_latency == 0.0
    assert colour.max_latency == 0
    assert colour.never_hinted == 0
    # The shipped tracker is a frame or three late and, in one window,
    # never hints a piece at all.
    assert shape.median_latency == 1.0
    assert shape.max_latency == 3
    assert shape.never_hinted == 1
    # Six of the eleven scored episodes can time anything at all: an
    # episode whose predecessor was the same letter gets its name free.
    assert colour.measurable == shape.measurable == 6
    assert len(colour.latencies) == 6
    assert len(shape.latencies) == 5  # the sixth was never hinted at all


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
    assert "colour" in text and "shape" in text
    assert text.count("ALL") == 2
