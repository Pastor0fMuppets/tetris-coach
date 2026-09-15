"""The race itself, pinned: both trackers over every consecutive window.

These are the numbers the head-to-head verdict rests on, so they are
asserted rather than printed. Run ``python -m tetris_coach.race`` for the
table; this file is what stops it drifting.

Read the numbers as: of the 397 frames the oracle will be quoted on, how
many does each tracker name right, name WRONG, or not name at all.

    shipped     366 right    6 wrong   25 silent   board right 365/397
    prototype   397 right    0 wrong    0 silent   board right 397/397

Scored from frame :data:`WARMUP` of each window, because every window is a
mid-session excerpt and both trackers are built fresh at its first frame.
The uncorrected race -- scoring from frame 0 -- reads 412/7/98 and 439/517
against 502/0/15 and 517/517, and roughly HALF that gap is the harness
rather than the representation: 73 of the shipped tracker's 98 silences and
46 of its 78 wrong boards are in a window's first twenty frames.
``test_the_window_head_is_a_bootstrap_tax_not_a_result`` pins both readings
side by side so the correction cannot be quietly undone in either
direction.

The shipped tracker's six wrong names are each exactly one frame long, and
each is the frame a new piece arrives on, where its committed state still
holds the piece that just locked. It is a lag, not a guess -- worth saying
plainly, because the user's report was that it "often guesses the piece
wrong" and on THESE windows it does not guess wrong at all. What it does
instead is go quiet.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from tetris_coach.race.measures import (
    STUCK_FLOOR,
    WARMUP,
    Episode,
    TruthFrame,
    TruthWindow,
    episodes,
    load_truth,
    measurable,
    score,
)
from tetris_coach.race.report import overall, replays, score_replays, table
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
def raw():  # type: ignore[no-untyped-def]
    """Both trackers over every window, replayed once and scored never.

    Cached because the replay is the expensive half and several tests want
    it scored under a different warm-up.
    """
    return replays()


@lru_cache(maxsize=8)
def at(warmup: int):  # type: ignore[no-untyped-def]
    return score_replays(raw(), warmup)


def scores() -> dict[str, dict[str, object]]:
    return at(WARMUP)  # type: ignore[no-any-return]


def part(window: str, tracker: str):  # type: ignore[no-untyped-def]
    return scores()[window][tracker]


def test_both_trackers_see_every_frame_and_are_scored_on_the_same_ones() -> None:
    """Same frames, same order, same warm-up -- or the race means nothing."""
    truth = load_truth()
    outputs = raw()
    for window in WINDOWS:
        shipped = part(window, "shipped")
        prototype = part(window, "prototype")
        # Replayed whole...
        assert len(outputs[window]["shipped"]) == len(truth[window].frames)
        assert len(outputs[window]["prototype"]) == len(truth[window].frames)
        # ...and scored from the same frame of it.
        assert shipped.warmup == prototype.warmup == WARMUP
        assert shipped.frames == prototype.frames == len(truth[window].frames) - WARMUP
    assert sum(len(outputs[w]["shipped"]) for w in WINDOWS) == 542
    assert sum(part(w, "shipped").frames for w in WINDOWS) == 422


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
    [("shipped", 366, 6, 25), ("prototype", 397, 0, 0)],
)
def test_identity_over_every_window(tracker: str, right: int, wrong: int, silent: int) -> None:
    """The headline: who names the falling piece, and who names it wrong."""
    total = overall(scores(), tracker)  # type: ignore[arg-type]
    assert total.identity.judged == 397
    assert (total.identity.correct, total.identity.wrong, total.identity.silent) == (
        right,
        wrong,
        silent,
    )


def test_the_window_head_is_a_bootstrap_tax_not_a_result() -> None:
    """Both readings, side by side, so the correction cannot be undone quietly.

    Scored from frame 0 the shipped tracker reads 412/517 right and 439/517
    board. Scored from frame 20 it reads 366/397 and 365/397. The prototype
    barely moves. The difference is not a fix to either tracker: it is six
    fresh constructions of a design that carries state across frames, in
    six excerpts that each begin in the middle of a session.

    That the head is where the shipped tracker's trouble lives, rather than
    trouble being uniform and the head merely shorter, is the whole claim,
    so it is asserted directly: 73 of its 98 silences are in the first
    twenty frames of a window.
    """
    cold, warm = at(0), at(WARMUP)
    cold_shipped, warm_shipped = overall(cold, "shipped"), overall(warm, "shipped")
    cold_proto, warm_proto = overall(cold, "prototype"), overall(warm, "prototype")

    assert (cold_shipped.identity.correct, cold_shipped.identity.judged) == (412, 517)
    assert cold_shipped.board_right == 439
    assert (cold_proto.identity.correct, cold_proto.identity.judged) == (502, 517)
    assert cold_proto.board_right == 517

    # The head holds 73 of 98 silent frames and 46 of 78 wrong boards.
    assert cold_shipped.identity.silent - warm_shipped.identity.silent == 73
    cold_wrong_board = cold_shipped.identity.judged - cold_shipped.board_right
    warm_wrong_board = warm_shipped.identity.judged - warm_shipped.board_right
    assert (cold_wrong_board, warm_wrong_board) == (78, 32)

    # Roughly half the published identity gap was the harness. The prototype
    # still wins on the corrected measure, by about half as much.
    cold_gap = cold_proto.identity.rate(cold_proto.identity.correct) - cold_shipped.identity.rate(
        cold_shipped.identity.correct
    )
    warm_gap = warm_proto.identity.rate(warm_proto.identity.correct) - warm_shipped.identity.rate(
        warm_shipped.identity.correct
    )
    assert round(cold_gap, 1) == 17.4
    assert round(warm_gap, 1) == 7.8
    assert warm_gap > 0, "the prototype still wins once the harness is corrected"


@pytest.mark.parametrize("warmup", [15, 20, 25, 30])
def test_the_warm_up_length_does_not_decide_the_verdict(warmup: int) -> None:
    """Anywhere the head effect is spent, the same thing is true.

    A correction chosen to produce an answer is not a correction. The
    shipped tracker sits between 90% and 93% and the prototype at 100% for
    every warm-up from a second to two seconds of capture, so the number 20
    is doing no work beyond naming where the bootstrap ends.
    """
    shipped = overall(at(warmup), "shipped")  # type: ignore[arg-type]
    prototype = overall(at(warmup), "prototype")  # type: ignore[arg-type]
    right = shipped.identity.rate(shipped.identity.correct)
    board = 100.0 * shipped.board_right / shipped.identity.judged
    assert 90.0 <= right <= 93.0, right
    assert 90.0 <= board <= 92.0, board
    assert prototype.identity.correct == prototype.identity.judged
    assert prototype.board_right == prototype.identity.judged


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
    """25 frames with nothing named, in a run of 16 (1.1 s at 15 fps).

    Once the window head is not scored the shipped tracker's silence is
    smaller but it does not go away, and it is still the whole of its
    remaining gap: 25 silent against 6 wrong. The prototype is never silent
    on a scored frame of any window.
    """
    shipped = overall(scores(), "shipped")  # type: ignore[arg-type]
    prototype = overall(scores(), "prototype")  # type: ignore[arg-type]
    assert shipped.worst_silent_run == 16
    assert shipped.identity.silent == 25
    assert prototype.worst_silent_run == 0
    # The 16 is live_session 00119 onward, and it is NOT a cold start: it
    # is 51 frames into the window, well past anything a warm-up excuses.
    assert part("live_session", "shipped").worst_silent_run == 16
    elsewhere = [part(w, "shipped").worst_silent_run for w in WINDOWS if w != "live_session"]
    assert max(elsewhere) == 3


@pytest.mark.parametrize(
    ("tracker", "latencies"),
    [("shipped", [1, 1, 1, 1, 3]), ("prototype", [0, 0, 0, 0, 0, 0])],
)
def test_latency_from_a_piece_appearing_to_it_being_named(
    tracker: str, latencies: list[int]
) -> None:
    """Frames each tracker waits before it can name the piece in play.

    Only 6 of the 19 flights can be timed once the window head is unscored:
    an episode that opens inside the warm-up is excluded whole, because the
    tracker may have named it during the frames nobody is scoring and its
    latency would then read as zero for free. That cost is real and it
    falls on the prototype's headline as much as the shipped tracker's --
    it is what an honest correction costs.

    The prototype names all 6 on the frame the piece appears. The shipped
    tracker times 5: one measurable episode it never names correctly at all
    (live_session's last O, where it is silent for 16 frames).
    """
    total = overall(scores(), tracker)  # type: ignore[arg-type]
    assert total.measurable == 6
    assert total.episodes == 11
    assert sorted(total.latencies) == sorted(latencies)


def test_the_reported_freeze_is_in_this_corpus_and_it_is_the_shipped_tracker() -> None:
    """ "It got stuck and the piece didn't update for several turns" -- here it is.

    spawn_latency 00103-00160: the shipped coach's state does not move for
    58 frames, 31 of them frames its confidence gate refused outright. At
    00134 the bottom row clears -- I read the pixels of board_00134.png: row
    11 columns 0-7 are (252, 251, 250), the board's own background, and only
    columns 8 and 9 hold anything. The coach goes on showing that row full
    for 27 more frames, 1.8 s at 15 fps, on a board 10 cells away from the
    one in front of the player.

    The name never looks wrong -- an O gives way to another O -- which is
    exactly why a measure that only watched the letter reported nothing.

    The prototype has no freeze on any window. That is not a subtle result
    and it should not be oversold either: it has no cross-frame state that
    COULD freeze, so this measure can only ever be a null for it.
    """
    runs = [(w, run) for w in WINDOWS for run in part(w, "shipped").stuck]
    assert [w for w, _ in runs] == ["spawn_latency"]
    ((_, freeze),) = runs
    assert (freeze.first, freeze.changed, freeze.last) == ("00103", "00134", "00160")
    assert (freeze.frames, freeze.stale, freeze.refused) == (58, 27, 31)
    assert (freeze.held, freeze.truth) == ("O", "O"), "the letter is right throughout"
    assert freeze.misplaced == 10, "the board under it is not"
    assert overall(scores(), "shipped").worst_stuck == 27  # type: ignore[arg-type]

    for window in WINDOWS:
        assert part(window, "prototype").stuck == []


def test_refusal_is_a_freeze_and_a_measure_that_skips_it_is_blind() -> None:
    """Why the detector counts refused frames: without them it sees nothing.

    The shipped engine's gate does not blank the screen when it refuses a
    frame -- it leaves the last hint up. So refusal IS how this tracker
    freezes, and a freeze detector that steps over unaccepted frames can
    never observe the thing it exists to observe. Re-running the identical
    rule with only that filter restored reports no freeze at all, on the
    very window that holds one.
    """
    truth = load_truth()["spawn_latency"]
    outputs = raw()["spawn_latency"]["shipped"]
    assert score("shipped", outputs, truth).stuck, "the freeze is there"

    blinded = [
        FrameOutput(
            frame=output.frame,
            accepted=output.accepted,
            # Every refused frame given a state nothing else can equal, which
            # is what skipping it amounts to: the run is cut at every refusal.
            piece=output.piece if output.accepted else f"break-{index}",
            stack_rows=output.stack_rows,
            next_piece=output.next_piece,
            hint=output.hint,
        )
        for index, output in enumerate(outputs)
    ]
    assert score("blinded", blinded, truth).stuck == []


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
    # Scored from frame 0: this coach has no bootstrap to excuse, it was
    # invented frozen.
    result = score("frozen", frozen, truth, warmup=0)
    assert result.stuck, "a coach frozen for a whole window was not caught"
    first = result.stuck[0]
    # live_session's second flight is the O that enters at 00059; a coach
    # still saying "I" from there on is stale for every frame that follows.
    assert (first.changed, first.held, first.truth) == ("00059", "I", "O")
    assert first.stale >= STUCK_FLOOR
    assert result.worst_stuck == max(run.stale for run in result.stuck)
    assert plan[1].piece == "O"
    # And the warm-up hides a freeze only for as long as the warm-up: the
    # same frozen coach is still caught, on the next piece change after it.
    warmed = score("frozen", frozen, truth, warmup=WARMUP)
    assert warmed.stuck and warmed.stuck[0].changed == "00099"


def test_the_board_handed_to_the_solver() -> None:
    """A right name on a wrong board still draws the wrong square."""
    shipped = overall(scores(), "shipped")  # type: ignore[arg-type]
    prototype = overall(scores(), "prototype")  # type: ignore[arg-type]
    assert prototype.board_right == 397, "the prototype's stack is the oracle's stack"
    assert shipped.board_right == 365
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
        == 25
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
    # The prototype's 22nd blank frame is spawn_latency's line-clear
    # animation, which it now reads as holding no piece: removing the
    # translucency rule (see test_colour_tracker.py) stopped it calling a
    # flash-tinted cell a dim copy of the I. The oracle abstains there too.
    assert (shipped.hintless_frames, prototype.hintless_frames) == (25, 22)
    # 64 of the shipped tracker's 422 scored frames never reach it: the
    # confidence gate refuses them and the previous hint stays up.
    assert shipped.refused_frames == 64
    # The prototype refuses 14, all of them pale_piece's web page, where
    # its premise -- cells drawn as flat rectangles -- does not hold. It
    # used to accept those and report a stack read off the page. Where the
    # two differ is what a refusal LOOKS like: the shipped engine leaves
    # its last hint on the screen, the prototype draws nothing.
    assert prototype.refused_frames == 14
    assert part("pale_piece", "prototype").refused_frames == 14
    assert sum(part(w, "prototype").refused_frames for w in WINDOWS if w != "pale_piece") == 0


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
