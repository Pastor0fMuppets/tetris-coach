"""Derive per-frame ground truth for a capture window, from the pixels.

The problem this exists to solve: two falling-piece trackers are about to
be raced against each other on the committed fixture windows, and a race
needs an answer sheet neither runner wrote. So nothing here reads a
tracker's opinion, and nothing here is imported by the shipped pipeline.

What it produces, per frame: whether the board is visible at all, which
cells are settled stack, which cells are the falling piece and what that
piece is called, which cells are a game-drawn landing preview (a
"ghost"), and which cells the oracle refuses to judge. A frame the oracle
cannot read confidently is reported as an ABSTENTION with its reasons,
never as a guess -- an answer sheet with blanks on it is still an answer
sheet; one with guesses on it is not.

How it reads a frame
--------------------

1. **Cells are measured, not thresholded.** Each cell contributes the
   MEDIAN colour of its central patch plus the fraction of that patch
   within a tight tolerance of the median. This game paints cells as flat
   rectangles, so a cell that is one thing is uniform: 63 389 of the
   65 040 cells in the six consecutive windows are uniform to the last
   unit, and every non-uniform one is something drawn OVER the board
   (this tool's own rotation badge, the game's "ROW CLEARED" card, the
   end-of-round panel, drifting confetti).

2. **The palette is discovered, not assumed.** Colours are clustered out
   of the window's own cells; the biggest cluster is the background and
   the rest are content. No table of piece colours is compiled in, and
   the mapping from colour to PIECE NAME is never assumed at all -- see
   below.

3. **The coach's own paint is un-composited, not merely detected.** This
   tool's overlay is on screen when the next frame is captured. Every
   hint cell carries ``HINT_PEN_BGR`` at full opacity around its border
   (``overlay.renderer.HintStyle``: a 3 px stroke inset 1.5 px, which
   lands outside the central patch) and the same colour at
   ``HINT_FILL_OPACITY`` over whatever is underneath. So a cell's paint
   is read off the BORDER and the interior is then inverted back through
   the alpha blend, which recovers what the board really shows there --
   empty board under a hint, or a piece under a hint. Detecting the paint
   without inverting it would delete real pieces: on ``live_session``
   00059 an I hard-drops onto the square the hint was marking.

4. **Pieces are named by SHAPE; colour only segments.** A falling piece
   is followed across frames as one episode, and the episode is named
   from a frame where all four of its cells are visible and form exactly
   one tetromino. Colour is used to separate touching cells into
   components and to check continuity -- never to name. The colour ->
   name map that falls out is therefore a RESULT, reported per window, to
   be compared against any tracker's assumed palette rather than
   substituted for it. An episode that never shows four cells is named
   from that derived map and marked ``basis="colour"`` so a race can
   exclude those frames when judging a colour-first tracker.

5. **Falling vs settled is decided by support and by time.** A component
   4-connected to the bottom row is resting; anything airborne is in
   flight. A resting piece is settled UNLESS a later frame shows it
   moving, which the oracle checks for explicitly and abstains on if it
   ever happens.

What it is not
--------------

It is not a replacement for eyes. Sections of this module name the exact
frames a human opened to check it, and ``tests/test_truth_oracle.py``
pins those hand-read frames as literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# This tool's own paint. Not a fact about the game: a fact about us.
# ---------------------------------------------------------------------------

#: ``overlay.renderer.HintStyle.color`` (#00e5ff) in the BGR the capture
#: pipeline produces. Hard-coded rather than imported because that module
#: pulls in Qt, and the oracle must stay importable with no GUI. The
#: constant is pinned against the renderer in ``tests/test_truth_oracle.py``.
HINT_PEN_BGR = (255, 229, 0)

#: ``overlay.renderer.HintStyle.fill_opacity``. The hint's interior is this
#: much pen over whatever the board shows there.
HINT_FILL_OPACITY = 0.18

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

#: Fraction of a cell inset on every side before its colour is measured.
#: Wide enough to clear the game's cell borders, the gap between cells and
#: the hint's own 3 px pen stroke at inset 1.5 (measured: the pen never
#: reaches the central patch -- 748 of the 769 painted cells in the corpus
#: carry exactly zero pen pixels inside it, and the 21 that do are the
#: rotation badge, which is a filled disc in the middle of a cell).
PATCH_MARGIN = 0.30

#: A cell's measured colour may sit this far (Chebyshev, uint8 units) from
#: a palette entry and still be called that entry. This game renders flat
#: colour: the spread inside a piece is zero, so the tolerance only has to
#: absorb PNG round-trip noise. Kept tight on purpose -- the game's
#: "ROW CLEARED" card sits 7 units off the background and must not pass as
#: background.
COLOUR_TOLERANCE = 6

#: Fraction of a cell's central patch that must actually BE the palette
#: colour it was matched to. Below this the cell has something drawn over
#: it (a badge, a popup card, a panel) and its content is unknown.
COVERAGE_FLOOR = 0.90

#: Chebyshev radius used to cluster cell colours into palette entries.
#: Comfortably below the separation between this game's colours (the two
#: closest, the pale periwinkle T and the hint fill over bare board, are
#: 24 units apart) and comfortably above the noise inside one.
CLUSTER_RADIUS = 14

#: A colour cluster must cover at least this many cell-frames in a window
#: before it is treated as a palette entry rather than an artifact.
MIN_CLUSTER_CELLS = 8

#: Share of a frame's observable cells that may be unreadable before the
#: oracle stops believing it is looking at a board at all.
UNREADABLE_SHARE = 0.2

#: The alpha band the game's landing-preview outline is drawn at, over the
#: board's own background. Measured per piece colour across the corpus:
#: 0.262 (I), 0.279 (O), 0.267 (T).
GHOST_ALPHA = (0.2, 0.34)

#: How far a pixel may sit off the background-to-piece line and still be
#: called part of that outline. Two units: the board's hairline gridline is
#: 4.2 off the pale periwinkle T's outline, and must not be mistaken for it.
GHOST_RESIDUAL = 2.0

#: Pixels of that outline a cell needs before it counts as previewed.
#: Measured: a cell of the preview carries 114-161, a cell next to one at
#: most 5.
GHOST_PIXEL_FLOOR = 40

#: Pixels trimmed off each side of a cell before its outline pixels are
#: counted, so a stroke on a shared border is not credited to both cells.
GHOST_EDGE_INSET = 2

#: How far a piece's colour must sit from the background before its
#: preview outline can be told from the board at all. The outline is a
#: quarter of that distance, and under this the two are indistinguishable.
GHOST_MIN_CONTRAST = 24.0

Cell = tuple[int, int]
CellSet = frozenset[Cell]
Verdict = Literal["confident", "partial", "abstain"]
NameBasis = Literal["shape", "ghost", "colour"]

# ---------------------------------------------------------------------------
# Tetromino shapes, written out rather than imported
# ---------------------------------------------------------------------------


def _normalise(cells: frozenset[Cell]) -> frozenset[Cell]:
    """Translate ``cells`` so its top-left bounding corner is the origin."""
    if not cells:
        return frozenset()
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return frozenset((r - top, c - left) for r, c in cells)


def _rotations(cells: frozenset[Cell]) -> list[frozenset[Cell]]:
    """All four rotations of ``cells``, normalised."""
    out = []
    current = cells
    for _ in range(4):
        current = _normalise(frozenset((c, -r) for r, c in current))
        out.append(current)
    return out


_BASE_SHAPES: dict[str, frozenset[Cell]] = {
    "I": frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    "O": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "T": frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    "S": frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    "Z": frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    "J": frozenset({(0, 0), (1, 0), (1, 1), (1, 2)}),
    "L": frozenset({(0, 2), (1, 0), (1, 1), (1, 2)}),
}

_SHAPE_NAMES: dict[frozenset[Cell], str] = {
    rotation: name for name, base in _BASE_SHAPES.items() for rotation in _rotations(base)
}


def name_of_shape(cells: frozenset[Cell]) -> str | None:
    """The tetromino ``cells`` is, or ``None`` if it is not a whole one.

    Four cells name at most one tetromino, so this is unambiguous where it
    answers at all. Fewer than four cells -- a piece still entering from
    above the capture -- deliberately get no answer here; they are named
    from the rest of their episode instead.
    """
    if len(cells) != 4:
        return None
    return _SHAPE_NAMES.get(_normalise(cells))


# ---------------------------------------------------------------------------
# Geometry and measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Board grid and the cells the capture cannot observe.

    ``unobservable`` is the set some games force on the capture: a NEXT
    preview floating over the board's top corner shows the next piece, not
    the board (``app.compute_overlap_mask`` names those cells from the two
    selected rectangles). The oracle reports them as unknown, never as
    empty -- it has no evidence there either.
    """

    rows: int
    cols: int = 10
    unobservable: CellSet = frozenset()

    def observable(self) -> list[Cell]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.unobservable
        ]


class FrameMeasurement(NamedTuple):
    """Raw per-cell measurements of one frame. No interpretation yet."""

    median: NDArray[np.float32]  # (rows, cols, 3) central-patch median colour
    uniformity: NDArray[np.float32]  # (rows, cols) share of the patch equal to it
    pen_border: NDArray[np.float32]  # (rows, cols) share of the WHOLE cell that is pen
    pen_inside: NDArray[np.float32]  # (rows, cols) share of the central patch that is pen
    patch: list[list[NDArray[np.int16]]]  # the central patches themselves


def measure_frame(
    image: NDArray[np.uint8], geometry: Geometry, margin: float = PATCH_MARGIN
) -> FrameMeasurement:
    """Measure every cell of a BGR board crop.

    The grid is the board rectangle split evenly, which is the same split
    every part of this project uses; a cell's colour is the median of its
    central patch, so a few confetti pixels or a thin gridline cannot move
    it, and ``uniformity`` records how much of the patch that median
    actually accounts for.
    """
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected a 3-channel BGR image, got shape {img.shape}")
    height, width = int(img.shape[0]), int(img.shape[1])
    rows, cols = geometry.rows, geometry.cols
    if height < rows or width < cols:
        raise ValueError(f"image {width}x{height} too small for a {cols}x{rows} grid")

    ys = np.linspace(0, height, rows + 1)
    xs = np.linspace(0, width, cols + 1)
    pen = np.array(HINT_PEN_BGR, dtype=np.int16)

    median = np.zeros((rows, cols, 3), dtype=np.float32)
    uniformity = np.zeros((rows, cols), dtype=np.float32)
    pen_border = np.zeros((rows, cols), dtype=np.float32)
    pen_inside = np.zeros((rows, cols), dtype=np.float32)
    patches: list[list[NDArray[np.int16]]] = []

    for r in range(rows):
        cell_h = float(ys[r + 1] - ys[r])
        fy0 = round(float(ys[r]))
        fy1 = max(fy0 + 1, round(float(ys[r + 1])))
        iy0 = round(float(ys[r]) + cell_h * margin)
        iy1 = max(iy0 + 1, round(float(ys[r + 1]) - cell_h * margin))
        row_patches: list[NDArray[np.int16]] = []
        for c in range(cols):
            cell_w = float(xs[c + 1] - xs[c])
            fx0 = round(float(xs[c]))
            fx1 = max(fx0 + 1, round(float(xs[c + 1])))
            ix0 = round(float(xs[c]) + cell_w * margin)
            ix1 = max(ix0 + 1, round(float(xs[c + 1]) - cell_w * margin))

            patch = img[iy0:iy1, ix0:ix1].reshape(-1, 3).astype(np.int16)
            centre = np.median(patch, axis=0)
            median[r, c] = centre
            uniformity[r, c] = float((np.abs(patch - centre).max(axis=1) <= 10).mean())
            pen_inside[r, c] = float((np.abs(patch - pen).max(axis=1) <= 12).mean())

            whole = img[fy0:fy1, fx0:fx1].reshape(-1, 3).astype(np.int16)
            pen_border[r, c] = float((np.abs(whole - pen).max(axis=1) <= 12).mean())
            row_patches.append(patch)
        patches.append(row_patches)

    return FrameMeasurement(median, uniformity, pen_border, pen_inside, patches)


# ---------------------------------------------------------------------------
# Palette discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """The colours a window is painted in, clustered out of the window.

    ``background`` is the single most common cell colour; ``content`` is
    every other colour that recurs. Nothing here knows which content
    colour is which piece -- that is derived later, from shape.
    """

    background: tuple[int, int, int]
    content: tuple[tuple[int, int, int], ...]

    def entries(self) -> tuple[tuple[int, int, int], ...]:
        return (self.background, *self.content)


def discover_palette(
    measurements: list[FrameMeasurement],
    geometry: Geometry,
    radius: int = CLUSTER_RADIUS,
    minimum: int = MIN_CLUSTER_CELLS,
) -> Palette:
    """Cluster a window's UNPAINTED, uniform cells into a palette.

    Painted cells are left out: the hint's translucent fill would other-
    wise contribute one spurious palette entry per colour it is drawn
    over. They are folded back in at labelling time by inverting the
    blend, which lands them on an entry discovered here.
    """
    samples: list[NDArray[np.float32]] = []
    for m in measurements:
        for r, c in geometry.observable():
            if m.pen_border[r, c] > 0.02 or m.uniformity[r, c] < COVERAGE_FLOOR:
                continue
            samples.append(m.median[r, c])
    if not samples:
        raise ValueError("no readable cells to build a palette from")

    clusters: list[tuple[NDArray[np.float32], list[NDArray[np.float32]]]] = []
    for colour in samples:
        for centre, members in clusters:
            if float(np.abs(centre - colour).max()) <= radius:
                members.append(colour)
                break
        else:
            clusters.append((colour, [colour]))

    kept: list[tuple[tuple[int, int, int], int]] = []
    for _, members in clusters:
        if len(members) < minimum:
            continue
        centre = np.median(np.array(members), axis=0)
        kept.append(((int(centre[0]), int(centre[1]), int(centre[2])), len(members)))
    kept.sort(key=lambda item: -item[1])
    return Palette(kept[0][0], tuple(colour for colour, _ in kept[1:]))


# ---------------------------------------------------------------------------
# Per-cell labelling
# ---------------------------------------------------------------------------

EMPTY = "empty"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrameLabels:
    """What each cell of one frame is, before any reasoning across time."""

    content: dict[Cell, str]  # EMPTY, UNKNOWN, or "b,g,r" of a content colour
    painted: CellSet  # cells carrying the coach's hint fill
    badges: CellSet  # cells carrying the coach's opaque rotation badge
    unreadable: CellSet  # observable cells the oracle cannot name


def label_frame(m: FrameMeasurement, palette: Palette, geometry: Geometry) -> FrameLabels:
    """Name every cell of a frame, or refuse to.

    A cell ringed by the hint's pen has its interior inverted back through
    the alpha blend first, so what is reported is what the BOARD shows
    there and not what the overlay made of it. A cell wearing the opaque
    rotation badge is refused outright: the badge hides what is under it,
    and no amount of colour reasoning gets it back.
    """
    content: dict[Cell, str] = {}
    painted: set[Cell] = set()
    badges: set[Cell] = set()
    unreadable: set[Cell] = set()

    pen = np.array(HINT_PEN_BGR, dtype=np.float32)
    for r, c in geometry.observable():
        if m.pen_inside[r, c] > 0.05:
            badges.add((r, c))
            content[(r, c)] = UNKNOWN
            unreadable.add((r, c))
            continue

        is_painted = bool(m.pen_border[r, c] > 0.02)
        if is_painted:
            painted.add((r, c))

        best: tuple[int, int, int] | None = None
        best_distance = float("inf")
        for entry in palette.entries():
            shown = (
                HINT_FILL_OPACITY * pen + (1.0 - HINT_FILL_OPACITY) * np.array(entry, np.float32)
                if is_painted
                else np.array(entry, np.float32)
            )
            distance = float(np.abs(m.median[r, c] - shown).max())
            if distance < best_distance:
                best_distance, best = distance, entry

        if best is None or best_distance > COLOUR_TOLERANCE:
            content[(r, c)] = UNKNOWN
            unreadable.add((r, c))
            continue

        shown = (
            HINT_FILL_OPACITY * pen + (1.0 - HINT_FILL_OPACITY) * np.array(best, np.float32)
            if is_painted
            else np.array(best, np.float32)
        )
        coverage = float((np.abs(m.patch[r][c] - shown).max(axis=1) <= 10).mean())
        if coverage < COVERAGE_FLOOR:
            content[(r, c)] = UNKNOWN
            unreadable.add((r, c))
            continue

        content[(r, c)] = EMPTY if best == palette.background else ",".join(str(v) for v in best)

    for cell in geometry.unobservable:
        content[cell] = UNKNOWN
    return FrameLabels(content, frozenset(painted), frozenset(badges), frozenset(unreadable))


# ---------------------------------------------------------------------------
# Structure within one frame
# ---------------------------------------------------------------------------


def _components(cells: dict[Cell, str], geometry: Geometry) -> list[tuple[str, CellSet]]:
    """4-connected groups of occupied cells that share a colour."""
    occupied = {cell: colour for cell, colour in cells.items() if colour not in (EMPTY, UNKNOWN)}
    seen: set[Cell] = set()
    out: list[tuple[str, CellSet]] = []
    for start, colour in sorted(occupied.items()):
        if start in seen:
            continue
        stack = [start]
        group: set[Cell] = set()
        seen.add(start)
        while stack:
            r, c = stack.pop()
            group.add((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in occupied and (nr, nc) not in seen and occupied[(nr, nc)] == colour:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        out.append((colour, frozenset(group)))
    return out


def _support(cells: dict[Cell, str], geometry: Geometry, unknown_conducts: bool) -> CellSet:
    """Cells 4-connected to the floor through things that could hold a piece.

    Colour is irrelevant: a piece resting on a stack of another colour is
    resting. What is NOT irrelevant is a cell the oracle cannot read. Run
    with ``unknown_conducts`` both ways, this answers two different
    questions -- "is this piece resting on something, assuming every cell
    I cannot read is solid?" and "... assuming every one is empty?" -- and
    a piece the two answers disagree about is a piece whose support is
    undecided. That is not a corner case: on ``spawn_latency`` 00127 the
    two cells under the landed O are lost to the game's own confetti, and
    the strict answer alone would report a two-cell O hanging in mid-air.
    """
    solid = {
        cell
        for cell, colour in cells.items()
        if colour not in (EMPTY, UNKNOWN) or (unknown_conducts and colour == UNKNOWN)
    }
    floor = [(geometry.rows - 1, c) for c in range(geometry.cols)]
    stack = [cell for cell in floor if cell in solid]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in solid and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return frozenset(seen)


def _full_rows(cells: dict[Cell, str], geometry: Geometry) -> list[int]:
    """Rows whose every cell reads occupied -- a line clear about to play."""
    out = []
    for r in range(geometry.rows):
        row = [cells.get((r, c), UNKNOWN) for c in range(geometry.cols)]
        if all(value not in (EMPTY, UNKNOWN) for value in row):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# The per-frame answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTruth:
    """What the oracle says about one frame.

    ``verdict`` is deliberately three-valued. ``confident`` means every
    observable cell is accounted for and the falling-piece question is
    answered. ``partial`` means the falling piece is answered but some
    cells are hidden by something drawn over the board (this tool's own
    rotation badge, the game's reward card) -- usable for refereeing the
    piece, not for refereeing the stack. ``abstain`` means the oracle will
    not be quoted on this frame at all, and ``reasons`` says why.
    """

    frame: str
    verdict: Verdict
    reasons: tuple[str, ...]
    board_visible: bool
    falling_piece: str | None
    falling_cells: tuple[Cell, ...]
    falling_basis: NameBasis | None
    falling_clipped: bool  # more of the piece may lie outside what is observed
    settled_cells: tuple[Cell, ...]
    ghost_cells: tuple[Cell, ...]
    ghost_complete: bool  # the visible preview is a whole tetromino
    ghost_obscured: bool  # the coach's own paint covers part of the landing square
    own_paint_cells: tuple[Cell, ...]
    badge_cells: tuple[Cell, ...]
    unreadable_cells: tuple[Cell, ...]
    colour_of_falling: str | None
    colour_free_cells: bool  # the falling cells follow from occupancy alone


@dataclass
class WindowTruth:
    """Every frame of one window, plus what deriving it turned up."""

    window: str
    geometry: Geometry
    palette: Palette
    colour_names: dict[str, str]
    frames: list[FrameTruth] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def count(self, verdict: Verdict) -> int:
        return sum(1 for frame in self.frames if frame.verdict == verdict)


# ---------------------------------------------------------------------------
# Deriving a window
# ---------------------------------------------------------------------------


@dataclass
class _Episode:
    """One piece, followed from the frame it appears to the frame it rests."""

    colour: str
    first: int
    indices: list[int] = field(default_factory=list)
    shape_names: set[str] = field(default_factory=set)
    ghost_names: set[str] = field(default_factory=set)


class _FrameState(NamedTuple):
    reasons: list[str]
    grounded: CellSet
    falling: CellSet
    colour: str | None
    episode: int | None


class _Structure(NamedTuple):
    airborne: list[tuple[str, CellSet]]
    grounded: CellSet
    undecided: bool


def _structure(labels: FrameLabels, geometry: Geometry) -> _Structure:
    """Which groups are in flight, which cells are resting, and how sure."""
    generous = _support(labels.content, geometry, unknown_conducts=True)
    strict = _support(labels.content, geometry, unknown_conducts=False)
    airborne: list[tuple[str, CellSet]] = []
    undecided = False
    for colour, cells in _components(labels.content, geometry):
        loose = not (cells & generous)
        if loose != (not (cells & strict)):
            undecided = True
        if loose:
            airborne.append((colour, cells))
    occupied = {cell for cell, colour in labels.content.items() if colour not in (EMPTY, UNKNOWN)}
    return _Structure(airborne, frozenset(occupied) & generous, undecided)


def _occupancy_components(cells: dict[Cell, str], geometry: Geometry) -> list[CellSet]:
    """Connected groups of occupied cells, blind to colour.

    Used only for the audit field ``colour_free_cells``: where a frame's
    airborne group is the same set with and without colour, the oracle's
    answer owes nothing to colour having segmented it, and a shape-only
    tracker is being judged against something it could in principle have
    computed.
    """
    flat = {cell: "x" for cell, colour in cells.items() if colour not in (EMPTY, UNKNOWN)}
    return [group for _, group in _components(flat, geometry)]


def ghost_cells(
    image: NDArray[np.uint8],
    piece_colour: tuple[int, int, int],
    background: tuple[int, int, int],
    geometry: Geometry,
) -> CellSet:
    """Cells holding the game's landing preview for the piece in flight.

    ROAS Stacker does draw a ghost, and it is invisible to any reading
    that samples cell centres -- which is every reading in this project,
    and why two fixture READMEs conclude the game draws none. It is an
    OUTLINE: a thin stroke just inside the perimeter of where the piece
    will land, in the piece's own colour at roughly 26% over the board.
    Measured across the corpus that alpha runs 0.23-0.30, so the test
    here is not a colour constant but a LINE: a pixel counts when it lies
    on the segment between the background and the piece's colour, at an
    alpha in ``GHOST_ALPHA`` and within ``GHOST_RESIDUAL`` of it. That
    matters, because the board's own hairline gridline sits four units
    off the pale periwinkle T's ghost and a fixed tolerance would swallow
    it.

    The stroke lands just inside the cell, so counting pixels in each
    cell rect inset by a couple of pixels attributes it cleanly: across
    the corpus a ghost cell carries 114-161 such pixels and its
    neighbours carry at most 5.

    What comes back is what is VISIBLE. Where the coach's own hint is
    painted on the landing square -- which happens often, the solver and
    the game frequently agreeing about where a piece belongs -- the paint
    covers the stroke and those cells are simply absent; the caller is
    told so rather than being handed a guess.
    """
    v = np.asarray(image).astype(np.float32)
    piece = np.array(piece_colour, dtype=np.float32)
    ground = np.array(background, dtype=np.float32)
    delta = piece - ground
    channel = int(np.argmax(np.abs(delta)))
    if abs(float(delta[channel])) < GHOST_MIN_CONTRAST:
        return frozenset()
    alpha = (v[:, :, channel] - ground[channel]) / float(delta[channel])
    predicted = ground[None, None, :] + alpha[:, :, None] * delta[None, None, :]
    hit = (
        (np.abs(v - predicted).max(axis=2) <= GHOST_RESIDUAL)
        & (alpha >= GHOST_ALPHA[0])
        & (alpha <= GHOST_ALPHA[1])
    )

    height, width = int(v.shape[0]), int(v.shape[1])
    ys = np.linspace(0, height, geometry.rows + 1)
    xs = np.linspace(0, width, geometry.cols + 1)
    found: set[Cell] = set()
    for r in range(geometry.rows):
        y0 = round(float(ys[r])) + GHOST_EDGE_INSET
        y1 = round(float(ys[r + 1])) - GHOST_EDGE_INSET
        for c in range(geometry.cols):
            x0 = round(float(xs[c])) + GHOST_EDGE_INSET
            x1 = round(float(xs[c + 1])) - GHOST_EDGE_INSET
            if int(hit[y0:y1, x0:x1].sum()) >= GHOST_PIXEL_FLOOR:
                found.add((r, c))
    return frozenset(found)


def _completions(seen: CellSet, allowed: CellSet, geometry: Geometry) -> set[str]:
    """Names of the tetrominoes ``seen`` could be, once ``allowed`` is added.

    Used on a ghost the coach's own hint is painted over: the stroke
    survives on some of its cells and not others, and this asks which
    whole pieces the surviving cells belong to if the missing ones are
    exactly the painted ones. Where the answer is a single name the ghost
    still names the piece; where it is several, it names nothing.
    """
    names: set[str] = set()
    if not seen:
        return names
    for name, base in _BASE_SHAPES.items():
        for rotation in _rotations(base):
            for dr in range(geometry.rows):
                for dc in range(geometry.cols):
                    placed = frozenset((r + dr, c + dc) for r, c in rotation)
                    if any(r >= geometry.rows or c >= geometry.cols for r, c in placed):
                        continue
                    if seen <= placed and (placed - seen) <= allowed:
                        names.add(name)
    return names


def derive_window(
    window: str,
    names: list[str],
    images: list[NDArray[np.uint8]],
    geometry: Geometry,
) -> WindowTruth:
    """Derive per-frame truth for one window of CONSECUTIVE captures.

    Consecutive matters twice over: a piece is named from whichever frame
    of its own episode shows it whole, and a resting piece is only called
    settled because the frames after it show it never moving again.
    """
    measurements = [measure_frame(image, geometry) for image in images]
    palette = discover_palette(measurements, geometry)
    labels = [label_frame(m, palette, geometry) for m in measurements]

    notes: list[str] = []
    observable = len(geometry.observable())

    # Pass 1 -- structure within each frame, and episodes across them.
    episodes: list[_Episode] = []
    states: list[_FrameState] = []
    open_episode: int | None = None
    unaccounted = False

    for index, label in enumerate(labels):
        reasons: list[str] = []
        unreadable = len(label.unreadable)
        if unreadable >= UNREADABLE_SHARE * observable:
            reasons.append("board-not-readable")
            states.append(_FrameState(reasons, frozenset(), frozenset(), None, None))
            open_episode = None
            unaccounted = False
            continue
        if unreadable:
            reasons.append(f"cells-hidden:{unreadable}")

        loose, grounded, undecided = _structure(label, geometry)
        if undecided:
            reasons.append("support-undecided")
        full = _full_rows(label.content, geometry)
        if full:
            # A completed row is still on screen while the clear plays, and
            # this game recolours it: on spawn_latency 00127 the O that
            # completed row 11 reads as the stack's blue, and two frames
            # later the whole row flashes white. Nothing about the piece
            # that landed survives that, so the oracle says nothing.
            reasons.append("line-clear-animation:" + ",".join(str(r) for r in full))

        previous = states[index - 1] if index else None
        if previous is not None and previous.grounded and not previous.reasons:
            vanished = previous.grounded - grounded
            if vanished:
                reasons.append(f"settled-cells-vanished:{len(vanished)}")

        newly_grounded = bool(previous is not None and (grounded - previous.grounded))
        # A piece that stops being visible without landing has not stopped
        # existing. It has gone somewhere the capture cannot see -- on this
        # game, behind the NEXT panel floating over the board's top corner
        # (spawn_latency 00154: the O slides into cols 8-9 of rows 0-1 and
        # the whole piece is inside the unobservable mask). Saying "no
        # falling piece" there would be a lie, so the oracle says nothing
        # until something explains where the piece went.
        if previous is not None and previous.falling and not loose and not newly_grounded:
            unaccounted = True
        if unaccounted and (newly_grounded or loose):
            unaccounted = False
        if unaccounted:
            reasons.append("piece-unaccounted-for")

        if any(len(cells) > 4 for _, cells in loose):
            reasons.append("airborne-group-larger-than-a-tetromino")
            states.append(_FrameState(reasons, grounded, frozenset(), None, None))
            open_episode = None
            continue
        if len(loose) > 1:
            reasons.append(f"several-airborne-groups:{len(loose)}")
            states.append(_FrameState(reasons, grounded, frozenset(), None, None))
            open_episode = None
            continue
        if not loose:
            states.append(_FrameState(reasons, grounded, frozenset(), None, None))
            open_episode = None
            continue

        colour, cells = loose[0]
        continues = (
            open_episode is not None
            and previous is not None
            and previous.episode == open_episode
            and episodes[open_episode].colour == colour
            and not newly_grounded
        )
        if not continues:
            episodes.append(_Episode(colour=colour, first=index))
            open_episode = len(episodes) - 1
        assert open_episode is not None
        episodes[open_episode].indices.append(index)
        named = name_of_shape(cells)
        if named is not None:
            episodes[open_episode].shape_names.add(named)
        states.append(_FrameState(reasons, grounded, cells, colour, open_episode))

    # Pass 1b -- the game's own landing preview, which is a copy of the
    # piece in flight and so an INDEPENDENT reading of its shape: it shows
    # all four cells even while the piece itself is still half above the
    # capture. Where the coach's hint is painted over the landing square
    # the outline is gone from those cells, and the piece is named from
    # the preview only when exactly one whole tetromino fits what is left.
    ghosts: list[CellSet] = []
    ghost_hidden: list[bool] = []
    for index, state in enumerate(states):
        if not state.falling or state.colour is None:
            ghosts.append(frozenset())
            ghost_hidden.append(False)
            continue
        colour = tuple(int(part) for part in state.colour.split(","))
        seen = ghost_cells(
            images[index], (colour[0], colour[1], colour[2]), palette.background, geometry
        )
        ghosts.append(seen)
        whole = name_of_shape(seen)
        episode = episodes[state.episode] if state.episode is not None else None
        if whole is not None:
            if episode is not None:
                episode.ghost_names.add(whole)
            ghost_hidden.append(False)
            continue
        painted = labels[index].painted
        fits = _completions(seen, painted, geometry)
        ghost_hidden.append(bool(painted))
        if len(fits) == 1 and episode is not None:
            episode.ghost_names.add(next(iter(fits)))

    # Pass 2 -- name each episode from the frames that show it whole, then
    # let the colour -> name map those namings produce cover the rest.
    colour_names: dict[str, str] = {}
    episode_name: list[str | None] = []
    episode_basis: list[NameBasis | None] = []
    for episode in episodes:
        if (
            episode.shape_names
            and episode.ghost_names
            and episode.shape_names != episode.ghost_names
        ):
            notes.append(
                f"episode at {names[episode.first]}: the piece reads "
                + ",".join(sorted(episode.shape_names))
                + " and its landing preview reads "
                + ",".join(sorted(episode.ghost_names))
            )
        for source, pool in (("shape", episode.shape_names), ("ghost", episode.ghost_names)):
            if len(pool) == 1:
                name = next(iter(pool))
                known = colour_names.setdefault(episode.colour, name)
                if known != name:
                    notes.append(
                        f"colour {episode.colour} was named both {known} and {name} by shape"
                    )
                episode_name.append(name)
                episode_basis.append(source)  # type: ignore[arg-type]
                break
            if len(pool) > 1:
                notes.append(
                    f"episode at {names[episode.first]} showed several whole shapes: "
                    + ",".join(sorted(pool))
                )
        else:
            episode_name.append(None)
            episode_basis.append(None)
    for position, episode in enumerate(episodes):
        if episode_name[position] is not None:
            continue
        if episode.colour in colour_names:
            episode_name[position] = colour_names[episode.colour]
            episode_basis[position] = "colour"
        else:
            notes.append(f"episode at {names[episode.first]} could not be named at all")

    # Pass 3 -- write the answer.
    frames: list[FrameTruth] = []
    for index, (name, label, state) in enumerate(zip(names, labels, states, strict=True)):
        reasons = list(state.reasons)
        board_visible = "board-not-readable" not in reasons
        piece = episode_name[state.episode] if state.episode is not None else None
        basis = episode_basis[state.episode] if state.episode is not None else None
        if state.episode is not None and piece is None:
            reasons.append("piece-not-named")

        occupied = {
            cell
            for cell, colour in label.content.items()
            if colour not in (EMPTY, UNKNOWN) and cell not in geometry.unobservable
        }
        ghost = ghosts[index]
        settled = occupied - state.falling - ghost
        if not board_visible:
            # Nothing on this frame is the board: another window is over
            # it, or the game has replaced it. Reporting cells here would
            # be reporting someone else's pixels as Tetris.
            occupied, settled, ghost = set(), set(), frozenset()
        # The preview is drawn where the piece will land, and the coach
        # very often wants the piece in the same place, so its own hint is
        # painted over the preview and erases the stroke. That makes the
        # preview answer a lower bound; it says nothing about the piece or
        # the stack, so it is recorded here and not in the verdict.
        ghost_complete = name_of_shape(ghost) is not None
        ghost_obscured = bool(state.falling) and not ghost_complete and bool(label.painted)

        # The reported cells are what can be SEEN of the piece. A piece
        # touching the top edge has the rest of itself above the capture,
        # and one touching the preview box has the rest of itself behind
        # the box: both are lower bounds, and a race must not score a
        # tracker down for naming four cells where the oracle saw two.
        touches_an_edge = any(
            r == 0 or (nr, nc) in geometry.unobservable
            for r, c in state.falling
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        )
        clipped = touches_an_edge and name_of_shape(state.falling) is None

        colour_free = bool(state.falling) and any(
            group == state.falling for group in _occupancy_components(label.content, geometry)
        )

        blocking = [r for r in reasons if not r.startswith("cells-hidden:")]
        verdict: Verdict = (
            "confident" if not reasons else ("partial" if not blocking else "abstain")
        )
        frames.append(
            FrameTruth(
                frame=name,
                verdict=verdict,
                reasons=tuple(reasons),
                board_visible=board_visible,
                falling_piece=piece,
                falling_cells=tuple(sorted(state.falling)),
                falling_basis=basis,
                falling_clipped=clipped,
                settled_cells=tuple(sorted(settled)),
                ghost_cells=tuple(sorted(ghost)),
                ghost_complete=ghost_complete,
                ghost_obscured=ghost_obscured,
                own_paint_cells=tuple(sorted(label.painted)),
                badge_cells=tuple(sorted(label.badges)),
                unreadable_cells=tuple(sorted(label.unreadable)),
                colour_of_falling=state.colour,
                colour_free_cells=colour_free,
            )
        )

    return WindowTruth(
        window=window,
        geometry=geometry,
        palette=palette,
        colour_names=colour_names,
        frames=frames,
        notes=notes,
    )
