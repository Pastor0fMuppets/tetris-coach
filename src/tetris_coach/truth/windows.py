"""The committed capture windows the oracle is derived over.

Each entry is the geometry its README records, plus whether the window is
a genuine run of CONSECUTIVE captures. The oracle reads a window as a
film -- a piece is named from the frame of its own episode that shows it
whole, and a resting piece is called settled because later frames never
move it -- so a SAMPLED window is not something it can be run over at
all, and ``pale_preview`` is marked as such rather than quietly included.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tetris_coach.truth.oracle import Cell, Geometry

Rect = tuple[int, int, int, int]  # left, top, width, height


@dataclass(frozen=True)
class WindowSpec:
    """One fixture directory and how to read it."""

    name: str
    rows: int
    board_rect: Rect
    next_rect: Rect
    consecutive: bool = True

    def geometry(self) -> Geometry:
        return Geometry(rows=self.rows, cols=10, unobservable=overlap_mask(self))


#: Fraction of a cell inset on every side before a reader samples it. The
#: same margin both readers use, written out here rather than imported for
#: the reason the mask below is written out here.
SAMPLED_MARGIN = 0.25


def overlap_mask(spec: WindowSpec, cols: int = 10) -> frozenset[Cell]:
    """Board cells whose sampled patch lies under the NEXT preview box.

    The same board-relative fraction test ``app.compute_overlap_mask``
    does, written out here so the oracle owes the engine nothing;
    ``tests/test_truth_oracle.py`` checks the two agree on every window.
    """
    left, top, width, height = spec.board_rect
    nleft, ntop, nwidth, nheight = spec.next_rect
    if width <= 0 or height <= 0:
        return frozenset()
    fx0 = (nleft - left) / width
    fx1 = (nleft + nwidth - left) / width
    fy0 = (ntop - top) / height
    fy1 = (ntop + nheight - top) / height
    masked = {
        (r, c)
        for r in range(spec.rows)
        for c in range(cols)
        if fx0 < (c + 1 - SAMPLED_MARGIN) / cols
        and fx1 > (c + SAMPLED_MARGIN) / cols
        and fy0 < (r + 1 - SAMPLED_MARGIN) / spec.rows
        and fy1 > (r + SAMPLED_MARGIN) / spec.rows
    }
    return frozenset(masked)


WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("spawn_latency", 12, (204, 311, 480, 577), (587, 309, 99, 102)),
    WindowSpec("live_session", 12, (303, 313, 477, 579), (686, 316, 94, 94)),
    WindowSpec("ghost_session", 12, (238, 317, 476, 574), (620, 314, 97, 96)),
    WindowSpec("absorbed_piece", 12, (238, 317, 476, 574), (620, 314, 97, 96)),
    WindowSpec("pale_piece", 12, (276, 314, 477, 578), (659, 317, 98, 95)),
    WindowSpec("ghost_beside_stack", 12, (276, 314, 477, 578), (659, 317, 98, 95)),
    WindowSpec("pale_preview", 12, (204, 311, 480, 577), (587, 309, 99, 102), consecutive=False),
    WindowSpec("hint_stutter", 12, (198, 316, 480, 573), (581, 315, 98, 93)),
    WindowSpec("stray_after_clear", 12, (198, 316, 480, 573), (581, 315, 98, 93)),
)

CONSECUTIVE = tuple(spec for spec in WINDOWS if spec.consecutive)


def by_name(name: str) -> WindowSpec:
    for spec in WINDOWS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def load_window(root: Path, spec: WindowSpec) -> tuple[list[str], list[NDArray[np.uint8]]]:
    """Read a window's board crops in frame order, as BGR arrays.

    The PNGs are ordinary RGB files and the capture pipeline produces BGR,
    so the channel order is reversed on the way in -- the same conversion
    every replay test in this project does.
    """
    from PIL import Image

    directory = root / spec.name
    paths = sorted(directory.glob("board_*.png"))
    if not paths:
        raise FileNotFoundError(f"no board frames in {directory}")
    names = [path.stem.split("_", 1)[1] for path in paths]
    images = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)[:, :, ::-1] for path in paths
    ]
    return names, images
