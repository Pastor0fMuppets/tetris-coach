"""Debounced game-state tracking anchored on a committed stack memory.

The tracker keeps ONE authoritative piece of memory: the committed stack,
as one bitmask per board row (the row count is set at construction,
default 20) — never ``None``. Every frame the falling piece is
DERIVED by :func:`~.pieces_vision.explain_grid` as a set difference
against that memory, never guessed geometrically from a single frame.

The committed stack changes only through an explicit, debounced,
structurally verified transition (a FALLING/QUIET/LOCKED commit, or a
board-reset resync). A frame the classifier cannot explain produces no
candidate, touches nothing, and the last committed state is held — there
is no best-effort commit path.

An observation's identity is (stack rows, falling piece name, next piece
name) — the falling piece's *position* is deliberately excluded, since it
changes every frame while the piece drops.

Events:

- ``PIECE_SPAWNED``: a (new) falling piece appeared.
- ``PIECE_LOCKED``: a lock was structurally verified — the piece's cells
  became immutable-and-explained (rows cleared, or the next spawn
  appeared). NOTE the deliberate semantics: vision cannot distinguish
  lock delay from lock without the game's timer, so PIECE_LOCKED fires at
  the verification point, not at touchdown. A piece resting on the stack
  stays FALLING (and the hint stays up) until the lock is revealed.
- ``PIECE_UNNAMED``: the committed falling piece's NAME came from the
  preview hint and this frame ruled it out. The name is withdrawn (the
  committed falling piece goes back to ``None``) and the consumer takes
  its hint off the screen. The stack is untouched: a hinted name was
  never evidence, so nothing structural was ever decided on it.
- ``BOARD_RESET``: ``reset_confirm_frames`` consecutive IDENTICAL
  unexplainable frames — a stable new world memory cannot explain (new
  game, garbage rising, mid-game attach). The tracker re-anchors on the
  observed board. Clear animations never trip it (their frames morph, and
  the settled board is explained as a lock); a one-frame glitch never
  does (the counter resets on the next coherent frame).

A resync adopts the observed board as the stack, minus the piece in
flight — a component that rests on nothing AND touches row 0, where
settled content cannot be. Freezing a piece in flight into the stack is
the corruption that made every hint after the first one wrong: the
piece's own descent then reads as stack cells vanishing, which nothing
can explain, so the tracker resets again and re-absorbs it one row lower,
the whole way down. Floating LOWER DOWN is not evidence of anything —
naive gravity leaves settled cells hanging over holes after a clear — so
it is adopted like the rest of the board (see ``strip_piece_in_flight``).
Everything the resync does not hold back is adopted unconditionally: an
earlier "re-anchoring on the board already believed fires no event"
short-circuit existed only because the strip removed real content, and
with the strip narrowed no unexplainable frame can reach it (measured:
0 of 82249 frames whose strip lands exactly on the committed stack are
UNEXPLAINED; every one is FALLING or OCCLUDED, neither of which resets).
Left in, it was a livelock: a real floating remnant the strip deleted
made the resync a no-op for good, and the tracker never adopted it.

``unobservable_cells`` names board cells the capture can never read (a
game UI panel floating over the playfield). The observed value there is
meaningless and is discarded; what the committed stack holds for those
cells is a BELIEF, seeded empty at bootstrap/resync and thereafter moved
only by an explained transition (a lock merging its hidden half in, a
clear shifting rows through). An OCCLUDED frame — a piece the covered
region is hiding — is coherent: it holds the committed state and, unlike
an unexplainable frame, never counts toward a board reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from ..core.board import DEFAULT_HEIGHT, WIDTH
from .pieces_vision import FallingPiece, FrameKind, explain_grid, strip_piece_in_flight

# The longest gap, in CAPTURES, a preview flip may be dated across. A flip
# read after a gap can only be a deal LATE if a whole previewed piece came
# and went inside that gap — which needs the gap to be at least one tenure
# long. The shortest tenure in the committed evidence is 18 captures (the
# pale T, unread in the box on spawn_latency 00198-00215), and the longest
# gap a flip has to survive to keep the win the window measures is 6 (the
# end-of-round wipe hides the box on 00152-00157, and the flip on 00158
# names the O that wipe dealt). Twelve sits between them with margin on
# both sides: nothing shorter can straddle a deal, nothing longer than the
# wipe's own blind spot is believed.
MAX_PREVIEW_GAP = 12

# The longest a hint may go on naming things, in CAPTURES since the deal it
# reports. A hint is evidence about ONE deal, and every other rule that says
# so is attached to an EVENT — a lock, a resync, a frame that contradicts it.
# When no such event arrives the hint has no expiry at all, and the cases
# where none arrives are exactly the ones nothing else can see: a HOLD swap
# (a new piece at the top edge with no lock and no flip), a restart whose
# first piece the old game's preview named, a piece parked at the top edge
# showing two cells that fit the name whatever it really is. Measured, that
# was not a long wrong name but an unbounded one: 300 frames — twenty
# seconds — of a swapped-in piece wearing the previous piece's name, and it
# would have held for the rest of the session.
#
# So the hint expires on its own clock too, and the budget is one tenure with
# margin: longer than the longest a hint has ever legitimately had a name on
# screen (15 captures, the O at spawn_latency 00159-00173, structure taking
# over on the 16th) and than the longest tenure in the committed evidence (18
# captures, the pale T unread in the box on 00198-00215), so no hint is ever
# cut short of the window it exists to cover; and short enough that a name
# nothing can contradict stands for 1.6 s rather than forever — twice the
# 0.80 s a consistently lying preview costs on a name the frames CAN
# contradict (test_spawn_latency). This is a ceiling, not a detector: the
# capture cannot see a hold swap, so the choice is where to stop believing a
# hint nothing has confirmed, not whether to notice the swap.
MAX_HINT_AGE = 24


class GameEvent(Enum):
    PIECE_SPAWNED = auto()
    PIECE_LOCKED = auto()
    PIECE_UNNAMED = auto()
    BOARD_RESET = auto()


@dataclass(frozen=True)
class Snapshot:
    """Committed, debounced game state."""

    stack_rows: tuple[int, ...]  # bitmask per row, row 0 = top
    falling_piece: str | None
    next_piece: str | None


def _rows_from_grid(grid: NDArray[np.bool_]) -> tuple[int, ...]:
    rows, cols = grid.shape
    return tuple(int(sum(1 << c for c in range(cols) if grid[r, c])) for r in range(rows))


def _rows_from_cells(cells: frozenset[tuple[int, int]] | None, rows: int) -> tuple[int, ...]:
    """``(row, col)`` cells as one bitmask per row; cells off the board drop."""
    out = [0] * rows
    for r, c in cells or ():
        if 0 <= r < rows and 0 <= c < WIDTH:
            out[r] |= 1 << c
    return tuple(out)


class GameStateTracker:
    """Tracks committed game state across debounced frames."""

    def __init__(
        self,
        confirm_frames: int = 2,
        reset_confirm_frames: int = 4,
        max_missing_cells: int = 2,
        rows: int = DEFAULT_HEIGHT,
        unobservable_cells: frozenset[tuple[int, int]] | None = None,
    ) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        if reset_confirm_frames < 1:
            raise ValueError("reset_confirm_frames must be >= 1")
        if rows < 1:
            raise ValueError("rows must be >= 1")
        self._confirm_frames = confirm_frames
        # Board cells the capture can never read, as one bitmask per row.
        # Public so the consumer that hands a board to the solver can apply
        # its own policy to the same cells (see CoachEngine._solver_board).
        self.unknown_rows: tuple[int, ...] = _rows_from_cells(unobservable_cells, rows)
        self._reset_confirm_frames = reset_confirm_frames
        self._max_missing_cells = max_missing_cells
        # Bootstrap IS the ordinary rule set: a fresh game diffs cleanly
        # from the empty snapshot; a mid-game attach resyncs via the reset
        # rule. The committed snapshot is never None. ``rows`` exists ONLY
        # for this bootstrap: the empty committed stack must exist before
        # any frame, so its length cannot be derived from data; every later
        # commit takes its length from the observed frame.
        self._committed = Snapshot((0,) * rows, None, None)
        self._pending: Snapshot | None = None
        self._pending_kind: FrameKind | None = None
        self._pending_count = 0
        self._last_falling: FallingPiece | None = None
        # How explain_grid classified the most recent update() frame; for
        # debug/status display only, never for control flow.
        self.last_kind: FrameKind | None = None
        self._unexplained_rows: tuple[int, ...] | None = None
        self._unexplained_count = 0
        # Preview memory, for naming a piece the top edge has cut in half.
        # The last CONFIRMED preview reading, and the one it replaced.
        self._preview_piece: str | None = None
        # A reading seen once and not yet confirmed by a second one, with
        # how many captures ago it was first seen and what gap preceded
        # it: a flip is dated from the first sighting, not from the
        # capture that confirms it (see _observe_preview).
        self._preview_pending: str | None = None
        self._pending_age = 0
        self._pending_gap = 0
        self._entering_hint: str | None = None
        # How many CAPTURES have gone by since the box was last readable.
        # A flip is only news of the deal happening NOW when the gap it is
        # dated across is too short to hide a whole tenure (see update()).
        self._preview_gap = 0
        # The name the committed falling piece was given by a hint and that
        # nothing structural has confirmed since. It is the subject of the
        # retraction rule in update(): a hypothesis that has been SHOWN has
        # to be takeable back, not merely droppable.
        self._hinted_commit: str | None = None
        # Frames since the hint was set. A hint is evidence about ONE deal,
        # so it must not outlive it, and its age is how the rules below
        # tell the flip reporting the deal now committing from a flip left
        # over from a deal ago (see update()).
        self._hint_age = 0

    @property
    def committed(self) -> Snapshot:
        return self._committed

    @property
    def falling(self) -> FallingPiece | None:
        """Last *coherent* raw falling-piece observation (position included).

        Not necessarily from the current frame: unexplained frames and
        lock transitions do not overwrite it.
        """
        return self._last_falling

    def observe_preview(self, next_piece: str | None) -> None:
        """Take one capture's preview reading with no board frame attached.

        The confidence gate judges the BOARD image, and the consumer
        returns before :meth:`update` whenever it rejects one — but the
        preview box is a different region of the screen with its own
        readability, and every rule below is about how many CAPTURES old
        the evidence is, not about how many of them the board gate liked.
        Feeding those captures here is what makes the dating rules mean
        what they say: without it the tracker's clock stops for the whole
        of a wipe or a flash, which is exactly the event that makes the
        box unreadable AND the board unreadable, so the guard against a
        flip straddling a deal is bypassed by the only thing that can
        cause one. Measured on tests/fixtures/spawn_latency: the gate
        rejects 00127-00157 (31 captures, ~2.1 s) and the headline flip on
        00158 used to be dated against 00126, 32 captures earlier.

        Separate from the private ``_observe_preview`` :meth:`update`
        calls, so that overriding this on an instance (as the fixture
        replays do, to log what the box said on a rejected capture)
        cannot double-count the captures that carry a board frame too.
        """
        self._observe_preview(next_piece)

    def _observe_preview(self, next_piece: str | None) -> None:
        # The piece that LEAVES the preview is the piece entering the
        # board: a game deals the previewed piece and shows the one after
        # it. That is the only evidence anything has about the name of a
        # piece the board region's top edge has cut in half — 1-3 cells at
        # row 0 fit several tetrominoes, and this session's game parks
        # them there for seconds. Only a change of a KNOWN preview counts:
        # the preview reading is None whenever the box is mid-animation or
        # unreadable, and None -> X says nothing about what was dealt.
        # ...and only a CHANGE the box holds. The preview is read by the
        # same vision as everything else, and a single frame of it can be
        # wrong (a fade, a flash, a piece drawn against a colour close to
        # its own). One misread capture is not one bad flip but TWO — X ->
        # W and then W -> X — and the second is the dangerous one: it
        # names W, a piece the game never dealt, and applies it to
        # whatever fragment happens to be parked at the top edge, which
        # the structural rules had correctly refused to name. So a reading
        # becomes the box's content only when a second consecutive
        # readable capture agrees with it, exactly as every other
        # observation here is debounced before it is believed; a value
        # that comes and goes inside one capture flips nothing, in either
        # direction. The cost is one capture of latency on a real flip,
        # which the hint's age then carries honestly.
        # ...and only a flip the box is in a position to be REPORTING. A
        # flip says "the piece that was here has been dealt"; it does not
        # say WHEN. The box goes unreadable in bursts (mid-animation,
        # mid-flash, a pale piece against a pale ground), and across a
        # burst long enough to cover a whole tenure the box's value has
        # moved on TWICE: X -> [Y never read] -> Z reads as a flip naming
        # X, a whole deal after X entered — the "confused two pieces"
        # failure, and it is refused. Two readings cannot straddle two
        # deals when the gap between them is shorter than a tenure: the
        # piece in between would have had to hold the box for the whole
        # gap. So the flip is dated by the GAP it is read across, in
        # captures, and MAX_PREVIEW_GAP is the budget (see its comment).
        # Measured on tests/fixtures/spawn_latency: the T dealt at 00198
        # sat unread in the box for 18 captures, and the flip at 00216
        # carried the I before it — refused; the wipe at 00152-00157
        # hides the box for 6, and the flip at 00158 names the O that the
        # wipe dealt — accepted.
        self._hint_age += 1
        if self._preview_pending is not None:
            self._pending_age += 1
        if next_piece is None:
            self._preview_gap += 1
            return
        gap, self._preview_gap = self._preview_gap, 0
        if next_piece == self._preview_piece:
            self._preview_pending = None  # a one-capture wobble, and back
            return
        if next_piece != self._preview_pending:
            # First sighting of a new value: remember WHEN it was first
            # seen and what gap preceded it, since that — not this frame —
            # is when the deal it reports happened.
            self._preview_pending = next_piece
            self._pending_age = 0
            self._pending_gap = gap
            return
        if self._preview_piece is not None and self._pending_gap <= MAX_PREVIEW_GAP:
            self._entering_hint = self._preview_piece
            self._hint_age = self._pending_age
        self._preview_piece = next_piece
        self._preview_pending = None

    def update(self, occupancy: NDArray[np.bool_], next_piece: str | None) -> list[GameEvent]:
        """Feed one frame's full occupancy grid; returns committed events."""
        # Whatever the capture read in an unobservable cell is not board
        # content: discard it here so no rule can mistake it for one. The
        # frame is then all-zero there, which explain_grid reads as "no
        # evidence" (not "empty") because it is told which cells those are.
        rows = tuple(
            o & ~u for o, u in zip(_rows_from_grid(occupancy), self.unknown_rows, strict=True)
        )
        self._observe_preview(next_piece)
        # The hint's own expiry, before the frame is read: a hint older than
        # a tenure cannot still be naming the piece "entering from above",
        # because the piece its deal dealt entered a tenure ago and has
        # either been named by its own shape or gone. Dropped HERE so this
        # frame is read from shape alone, and the name it already committed
        # is taken back below — dropping the hypothesis without withdrawing
        # the name leaves the name on screen, which is the lesson the
        # refutation rule already paid for.
        hint_expired = self._entering_hint is not None and self._hint_age > MAX_HINT_AGE
        if hint_expired:
            self._entering_hint = None
        explanation = explain_grid(
            rows,
            self._committed.stack_rows,
            self._last_falling,
            max_missing_cells=self._max_missing_cells,
            unknown_rows=self.unknown_rows,
            entering_hint=self._entering_hint,
        )
        self.last_kind = explanation.kind
        if explanation.hint_refuted:
            # A preview reading is a HYPOTHESIS, and this frame falsified
            # it: the piece coming in from above showed cells no placement
            # of the hinted piece fits. Dropped here, before anything is
            # decided on it, so the rest of this frame — and every frame
            # until the next flip — reads from shape alone, and the name
            # the hypothesis already committed is taken back below.
            #
            # What this does NOT do is make a wrong reading harmless. A
            # frame can only refute what it contradicts, and while the
            # piece is clipped to two cells at the top edge it fits an O,
            # an S, a Z, a J and an L alike — so a preview that says S
            # over an O buys a confidently wrong hint for exactly as long
            # as the fragment is ambiguous. Measured on the committed
            # window with every O misread as an S: 12 frames, 0.80 s, on
            # the one piece the preview named, ending when its second row
            # descends; the committed stack, the locks and the resets are
            # untouched (test_spawn_latency). That is the accelerator's
            # price, and it is the same trade as naming the fragment at
            # all: the alternative measured here is 13 frames of no hint
            # on EVERY piece.
            self._entering_hint = None

        if explanation.falling is not None and not explanation.hinted_name:
            # Structure has named the piece in flight. Whatever the hint
            # called it, the name on screen is no longer a guess, so it is
            # no longer the retraction rule's business (and a later,
            # unrelated fragment must not be read as contradicting it).
            # Deliberately NOT also dropping the hint here. Ending its life
            # at the confirmation looks tidier — the piece it names has
            # finished entering — and measured, it buys nothing and costs
            # something: the frame after a hold swap then reads OCCLUDED
            # instead of FALLING, which HOLDS the same stale name on screen
            # (same name, same correcting frame), while leaving nothing in
            # _hinted_commit for the retraction rule or the clock to take
            # back. A hint that has done its job is harmless where it is;
            # what needed bounding is a hint nothing ever confirmed.
            self._hinted_commit = None
        elif self._hinted_commit is not None and (
            hint_expired
            or (
                explanation.entering_names and self._hinted_commit not in explanation.entering_names
            )
        ):
            # RETRACTION, on either of the two things that end a hypothesis:
            # a frame that CONTRADICTS it, or the hint behind it running out
            # of time (above). The second exists because the first can only
            # fire on a frame that says something: while a piece sits at the
            # top edge showing two cells that fit the name, no frame
            # contradicts anything, and a name a hold swap or a restart put
            # on the wrong piece is never taken back at all. Both ends are
            # the same act, and it is the WITHDRAWAL, not the drop, that
            # the user sees.
            #
            # Dropping the hypothesis is not enough on its own:
            # by the time a frame contradicts it, the name it picked has
            # usually been committed and drawn — and the contradicting
            # frame is normally an OCCLUDED one (the piece has shown a
            # second cell, which rules the hinted name out without naming
            # a replacement), which HOLDS the committed snapshot. The
            # refuted name then stayed on screen for the rest of the
            # piece's tenure at the top edge (measured in this game:
            # 14-17 frames, ~1 s), which is the user's "confused two
            # pieces" report made persistent. So the name is taken back on
            # the frame that refutes it, and the coach goes back to showing
            # nothing — what it shows for any piece it cannot name.
            #
            # Undebounced on purpose: the debounce exists to stop a torn
            # frame COMMITTING something, and this commits nothing. It
            # withdraws a claim, which is the safe direction — the cost of
            # a retraction a glitch frame caused is a hint missing for a
            # frame or two, against a confidently wrong hint for a second.
            self._hinted_commit = None
            self._entering_hint = None
            self._committed = Snapshot(self._committed.stack_rows, None, self._committed.next_piece)
            self._pending = None
            self._pending_kind = None
            self._pending_count = 0
            self._unexplained_rows = None
            self._unexplained_count = 0
            return [GameEvent.PIECE_UNNAMED]

        if explanation.kind is FrameKind.OCCLUDED:
            # Coherent: the covered region is hiding a piece. Hold the
            # committed state (and any pending transition), and — the whole
            # point — do NOT let a stationary hidden piece reach the reset
            # debounce, which would wipe the board it is resting on.
            self._unexplained_rows = None
            self._unexplained_count = 0
            return []

        if explanation.kind is FrameKind.UNEXPLAINED:
            if rows == self._unexplained_rows:
                self._unexplained_count += 1
            else:
                self._unexplained_rows = rows
                self._unexplained_count = 1
            if self._unexplained_count >= self._reset_confirm_frames:
                # How far back the new world this resync adopts goes, kept
                # before the counter is cleared below (see the hint rule).
                run_length = self._unexplained_count
                # Re-anchor on the observed board. Unobservable cells were
                # blanked above, so the belief is seeded EMPTY: a resync is
                # usually a fresh game, and there is no evidence for
                # anything else. (Seeding them filled instead would be a
                # different lie, and a permanent one — a board whose top
                # rows are filled has columns nothing can be dropped into.)
                # A piece at the top edge — entering, or just spawned, and
                # floating — is the one thing on the frame that is
                # demonstrably not settled, and a resync is exactly when
                # one is likely to be sitting there (an attach mid-game, in
                # a game that parks spawns at the top edge). It is left out
                # rather than frozen into the stack. Everything else the
                # frame shows is adopted, floating or not: a piece absorbed
                # lower down is taken back out by _carried_piece on its next
                # descending frame, while content deleted here is gone.
                resynced = strip_piece_in_flight(rows, self.unknown_rows)
                self._committed = Snapshot(resynced, None, next_piece)
                self._pending = None
                self._pending_kind = None
                self._pending_count = 0
                self._unexplained_rows = None
                self._unexplained_count = 0
                self._last_falling = None
                self._hinted_commit = None
                # A resync is a new world (a new game, garbage, a mid-game
                # attach). What the preview shows still holds, but which
                # piece was dealt into THIS board does not — UNLESS the
                # flip that said so happened inside the run of frames this
                # resync is adopting, which is the new world already on
                # screen. Then the deal it reports is this board's own,
                # and the piece it names is the fragment the strip above
                # just held back: dropping it there is what made the
                # user's own case slow. Measured on the spawn_latency
                # window: the game wipes the field and deals an O, the box
                # flips O -> I on 00158 as the O's first cells appear at
                # the top edge, and the reset four frames later threw that
                # away and left the O nameless until 00175.
                #
                # What that ASSUMES is that the new world continues the
                # same piece sequence — true of a field wipe, of rising
                # garbage and of a mid-game attach, and NOT true of a new
                # game or a restart, where the departing name is the old
                # game's preview content and what is dealt first has
                # nothing to do with it (right about one time in seven).
                # Nothing in the frames separates the two: both show a
                # field replaced wholesale and a box that changed. So it
                # is bounded rather than prevented — the kept hint is a
                # hypothesis like any other, the first frame that
                # contradicts it takes the name back (the retraction
                # above), and it can decide nothing structural. Pinned
                # both ways in test_state.py.
                if self._hint_age >= run_length:
                    self._entering_hint = None
                return [GameEvent.BOARD_RESET]
            # Pending is untouched: a torn frame between the confirmations
            # of a real transition must not restart its count.
            return []

        self._unexplained_rows = None
        self._unexplained_count = 0
        if explanation.kind is FrameKind.FALLING and explanation.falling is not None:
            # LOCKED frames must NOT overwrite this until they commit: the
            # L1/C1 anchors need the pre-lock position to re-derive the
            # same candidate on the confirmation frame.
            #
            # A name the entering hint supplied is NOT an observation: the
            # frame showed 1-3 cells that several tetrominoes fit, and the
            # preview picked one. Good enough to hint on, and not good
            # enough to become evidence — explain_grid uses this piece's
            # NAME to refuse a lock ("a piece cannot change identity
            # between flight and lock"), so a misnamed entering piece would
            # block its own lock and, four identical frames later, reset
            # the board. In this game a piece goes straight from the top
            # edge to a hard drop, so the correction the ordinary rules
            # apply on the way down never gets to run.
            self._last_falling = None if explanation.hinted_name else explanation.falling

        candidate = Snapshot(
            stack_rows=explanation.stack_rows,
            falling_piece=(explanation.falling.piece if explanation.falling is not None else None),
            next_piece=next_piece,
        )

        if candidate == self._committed:
            self._pending = None
            self._pending_kind = None
            self._pending_count = 0
            return []

        if candidate == self._pending:
            self._pending_count += 1
            self._pending_kind = explanation.kind
        else:
            self._pending = candidate
            self._pending_kind = explanation.kind
            self._pending_count = 1

        if self._pending_count < self._confirm_frames:
            return []

        previous = self._committed
        kind = self._pending_kind
        self._committed = candidate
        self._pending = None
        self._pending_kind = None
        self._pending_count = 0
        # Remember whether this name is a hypothesis, so the frame that
        # contradicts it can take it back (see the retraction rule above).
        self._hinted_commit = candidate.falling_piece if explanation.hinted_name else None
        if kind is FrameKind.LOCKED:
            # The pre-lock piece is absorbed into the stack; from now on
            # the anchor is the residual spawn (or nothing).
            self._last_falling = explanation.falling
            # ...and the deal the hint describes is over unless the hint
            # IS this deal's. A lock is the game dealing again, and the
            # preview reports that deal by changing at the moment the new
            # piece spawns — which is the frame this commit is debouncing,
            # so the hint of the deal now beginning is already set and its
            # age is at most the debounce (plus the one frame the flip may
            # lead the lock becoming visible by). An OLDER hint is one
            # whose own deal has just ended under it: it names the piece
            # that locked, not the fragment now at the top edge, and left
            # standing it would name every later fragment too — the
            # preview is unreadable in bursts, and one burst spanning a
            # whole tenure would poison every top-edge naming after it.
            # Counting locks instead of frames could not tell the two
            # apart: both show exactly one lock since the hint was set.
            if self._hint_age > self._confirm_frames:
                self._entering_hint = None
        return self._events(previous, candidate, kind)

    @staticmethod
    def _events(old: Snapshot, new: Snapshot, kind: FrameKind | None) -> list[GameEvent]:
        events: list[GameEvent] = []
        if kind is FrameKind.LOCKED:
            events.append(GameEvent.PIECE_LOCKED)
            if new.falling_piece is not None:
                # The common lock+spawn commit: a fresh spawn even when the
                # name equals the locked piece's.
                events.append(GameEvent.PIECE_SPAWNED)
        elif new.falling_piece is not None and new.falling_piece != old.falling_piece:
            # None -> name, or a hold swap changing the name. Same-name
            # hold swaps are invisible and correctly quiet; next-piece-only
            # changes commit quietly with no events.
            events.append(GameEvent.PIECE_SPAWNED)
        return events
