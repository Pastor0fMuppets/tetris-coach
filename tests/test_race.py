"""The race itself, pinned: both trackers over every consecutive window.

These are the numbers the head-to-head verdict rests on, so they are
asserted rather than printed. Run ``python -m tetris_coach.race`` for the
table; this file is what stops it drifting.

Read the numbers as: of the 517 frames the oracle will be quoted on, how
many does each tracker name right, name WRONG, or not name at all.

    shipped     412 right    7 wrong   98 silent   board right 439/517
    prototype   502 right    0 wrong   15 silent   board right 517/517

The shipped tracker's seven wrong names are each exactly one frame long,
and each is the frame a new piece arrives on, where its committed state
still holds the piece that just locked. It is a lag, not a guess -- worth
saying plainly, because the user's report was that it "often guesses the
piece wrong" and on THESE windows it does not guess wrong at all. What it
does instead is go quiet: 98 frames with no piece named, in runs of up to
23.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from tetris_coach.race.measures import (
    STUCK_FLOOR,
    Episode,
    TruthFrame,
    TruthWindow,
    episodes,
    load_truth,
    measurable,
    score,
)
from tetris_coach.race.report import overall, race, table
from tetris_coach.race.runners import FrameOutput

WINDOWS = (
    "spawn_latency",
    "live_session",
    "ghost_session",
    "absorbed_piece",
    "pale_piece",
    "ghost_beside_stack",
)


@lru_cache(maxsize=1)
def scores() -> dict[str, dict[str, object]]:
    return race()  # type: ignore[return-value]


def part(window: str, tracker: str):  # type: ignore[no-untyped-def]
    return scores()[window][tracker]


def test_both_trackers_see_every_frame_of_every_window() -> None:
    """Same frames, same order, both runners -- or the race means nothing."""
    truth = load_truth()
    for window in WINDOWS:
        shipped = part(window, "shipped")
        prototype = part(window, "prototype")
        assert shipped.frames == prototype.frames == len(truth[window].frames)
    assert sum(part(w, "shipped").frames for w in WINDOWS) == 542


def test_the_episodes_are_the_pieces_the_oracle_saw() -> None:
    """19 flights over the six windows, and which piece each one is.

    17 pieces, plus 2 more because the oracle abstains through
    ``spawn_latency``'s line-clear animation and through the four frames
    where a piece slides behind the NEXT panel: an abstention ends the
    episode rather than being guessed through, so the O on either side of
    each gap is counted as two flights.
    """
    truth = load_truth()
    seen = {window: [e.piece for e in episodes(truth[window].frames)] for window in WINDOWS}
    assert seen == {
        "spawn_latency": ["I", "O", "O", "O", "I", "T"],
        "live_session": ["I", "O", "I", "I", "O"],
        "ghost_session": ["O", "I", "I", "I"],
        "absorbed_piece": ["T", "I"],
        "pale_piece": ["T"],
        "ghost_beside_stack": ["I"],
    }
    assert sum(len(names) for names in seen.values()) == 19


def test_only_a_piece_change_can_time_a_tracker() -> None:
    """Three I pieces in a row cannot measure how fast anything is.

    A tracker holding the last name gets the next one free when the two
    are the same letter, so those episodes are excluded from latency; 14
    of the 19 remain.
    """
    truth = load_truth()
    timed = {window: len(measurable(episodes(truth[window].frames))) for window in WINDOWS}
    assert timed == {
        "spawn_latency": 4,
        "live_session": 4,
        "ghost_session": 2,
        "absorbed_piece": 2,
        "pale_piece": 1,
        "ghost_beside_stack": 1,
    }
    assert sum(timed.values()) == 14


@pytest.mark.parametrize(
    ("tracker", "right", "wrong", "silent"),
    [("shipped", 412, 7, 98), ("prototype", 502, 0, 15)],
)
def test_identity_over_every_window(tracker: str, right: int, wrong: int, silent: int) -> None:
    """The headline: who names the falling piece, and who names it wrong."""
    total = overall(scores(), tracker)  # type: ignore[arg-type]
    assert total.identity.judged == 517
    assert (total.identity.correct, total.identity.wrong, total.identity.silent) == (
        right,
        wrong,
        silent,
    )


def test_no_wrong_name_survives_a_single_frame_on_either_tracker() -> None:
    """The shipped tracker's seven wrong frames are seven separate lags.

    Every one is the frame a new piece arrives on. A wrong name that
    persisted would be the failure the user described; a wrong name that
    is gone by the next frame is the cost of confirming before committing.
    """
    for window in WINDOWS:
        assert part(window, "shipped").worst_wrong_run <= 1
        assert part(window, "prototype").worst_wrong_run == 0


def test_silence_is_the_shipped_failure_mode_and_it_is_long() -> None:
    """98 frames with nothing named, in runs up to 23 (1.5 s at 15 fps)."""
    shipped = overall(scores(), "shipped")  # type: ignore[arg-type]
    prototype = overall(scores(), "prototype")  # type: ignore[arg-type]
    assert shipped.worst_silent_run == 23
    assert prototype.worst_silent_run == 14
    # The prototype's one long silence is absorbed_piece's cold open: the
    # window starts on a piece whose colour no NEXT box in it ever shows,
    # so nothing can name it until it descends far enough to be a shape.
    assert part("absorbed_piece", "prototype").worst_silent_run == 14
    elsewhere = [part(w, "prototype").worst_silent_run for w in WINDOWS if w != "absorbed_piece"]
    assert max(elsewhere) == 1


@pytest.mark.parametrize(
    ("tracker", "latencies"),
    [
        ("shipped", [17, 1, 3, 3, 3, 1, 2, 1, 15, 1, 16, 17]),
        ("prototype", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 14, 0, 0, 0]),
    ],
)
def test_latency_from_a_piece_appearing_to_it_being_named(
    tracker: str, latencies: list[int]
) -> None:
    """Frames each tracker waits before it can name the piece in play.

    The prototype names 13 of its 14 on the frame the piece appears. The
    shipped tracker's two misses (14 of 14 measurable episodes named, but
    only 12 timed) are episodes it never names at all.
    """
    total = overall(scores(), tracker)  # type: ignore[arg-type]
    assert total.measurable == 14
    assert sorted(total.latencies) == sorted(latencies)


def test_neither_tracker_freezes_through_a_piece_change_on_these_windows() -> None:
    """The reported 25-frame freeze does not reproduce here.

    Both trackers update within one frame of the piece changing on every
    one of the 19 flights, so the user's "it got stuck" is not in these
    captures. Reported as a null result, not as a pass.
    """
    for window in WINDOWS:
        assert part(window, "shipped").stuck == []
        assert part(window, "prototype").stuck == []


def test_the_freeze_detector_is_not_vacuous() -> None:
    """A frozen coach over a real piece change IS caught.

    Without this the null result above could mean the detector never
    fires. The frames are the oracle's own ``live_session`` answer sheet;
    only the tracker's side is invented, and it is invented frozen.
    """
    truth = load_truth()["live_session"]
    plan = episodes(truth.frames)
    frozen = [
        FrameOutput(
            frame=frame.frame,
            accepted=True,
            piece="I",
            stack_rows=(0,) * truth.rows,
            next_piece=None,
            hint=None,
        )
        for frame in truth.frames
    ]
    result = score("frozen", frozen, truth)
    assert result.stuck, "a coach frozen for a whole window was not caught"
    first = result.stuck[0]
    # live_session's second flight is the O that enters at 00059; a coach
    # still saying "I" from there on is stale for every frame that follows.
    assert (first.changed, first.held, first.truth) == ("00059", "I", "O")
    assert first.stale >= STUCK_FLOOR
    assert result.worst_stuck == max(run.stale for run in result.stuck)
    assert plan[1].piece == "O"


def test_the_board_handed_to_the_solver() -> None:
    """A right name on a wrong board still draws the wrong square."""
    shipped = overall(scores(), "shipped")  # type: ignore[arg-type]
    prototype = overall(scores(), "prototype")  # type: ignore[arg-type]
    assert prototype.board_right == 517, "the prototype's stack is the oracle's stack"
    assert shipped.board_right == 439
    # Where the shipped board is wrong, frame by frame: 68 of the 78 are
    # runs of re-anchoring -- it begins every window believing the board
    # is empty and has to re-derive the stack from what moves, and does it
    # again after the oracle's own blind spell in spawn_latency (00134-
    # 00153). The other 10 are single frames of commit lag, one per lock.
    # Not corruption -- but 68 frames is 4.5 seconds of coaching a board
    # that is not there, and the prototype has no such state to rebuild.
    assert (
        part("spawn_latency", "shipped").identity.judged
        - part("spawn_latency", "shipped").board_right
        == 34
    )


def test_stability_within_a_flight() -> None:
    """Once a piece is named, does the coach keep saying the same thing?"""
    shipped = overall(scores(), "shipped")  # type: ignore[arg-type]
    prototype = overall(scores(), "prototype")  # type: ignore[arg-type]
    assert shipped.name_flips == 0
    assert prototype.name_flips == 0
    # The shipped hint moves three times mid-flight; the prototype's never
    # does, even though it re-solves from scratch on every single frame
    # rather than holding a hint between events.
    assert shipped.hint_flips == 3
    assert prototype.hint_flips == 0


def test_coverage_and_what_the_gate_costs() -> None:
    """How often each coach has nothing on screen at all."""
    shipped = overall(scores(), "shipped")  # type: ignore[arg-type]
    prototype = overall(scores(), "prototype")  # type: ignore[arg-type]
    assert (shipped.hintless_frames, prototype.hintless_frames) == (98, 36)
    # 79 of the shipped tracker's 542 frames never reach it: the
    # confidence gate refuses them and the previous hint stays up. The
    # prototype has no gate, so it reads all 542.
    assert shipped.refused_frames == 79
    assert prototype.refused_frames == 0


def test_the_table_renders() -> None:
    text = table(scores())  # type: ignore[arg-type]
    assert "prototype" in text and "shipped" in text
    assert text.count("ALL") == 2


def test_an_episode_ends_where_the_oracle_stops_answering() -> None:
    """An abstention closes the flight instead of being guessed through."""
    frames = [
        TruthFrame("1", "confident", "T", frozenset({(0, 0)}), frozenset(), "shape"),
        TruthFrame("2", "abstain", None, frozenset(), frozenset(), None),
        TruthFrame("3", "confident", "T", frozenset({(0, 0)}), frozenset(), "shape"),
    ]
    window = TruthWindow("synthetic", 12, 10, frozenset(), frames)
    assert episodes(window.frames) == [Episode("T", 0, 0), Episode("T", 2, 2)]
