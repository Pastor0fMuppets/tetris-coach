"""Drive each tracker over a capture window and record what the user sees.

Both runners produce the same :class:`FrameOutput` per frame, so the
scoring in :mod:`tetris_coach.race.measures` never has to know which
tracker it is judging. What is recorded is deliberately the OUTPUT side,
not the internals: the piece the coach would currently name, the stack it
would hand the solver, and the placement it would be drawing on screen.
A tracker that is internally confused but externally right costs the user
nothing, and a tracker that is internally serene while showing the wrong
square costs the user everything.

Two honest asymmetries, stated here rather than buried:

* The shipped engine has a CONFIDENCE GATE. A refused frame never reaches
  its tracker and the previous hint stays on screen, so its state is
  recorded on every frame -- including refused ones -- because that is
  what the user is looking at. ``accepted`` says which frames it actually
  digested. The prototype has no gate: it reads every frame, so every
  frame is accepted.
* The shipped engine solves on EVENTS and holds the hint in between. The
  prototype has no hint policy at all, so its hint is re-solved from each
  frame's own reading. That is the natural policy for a design with no
  memory, and it is the harsher one for stability -- see the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tetris_coach.app import CoachConfig, CoachEngine
from tetris_coach.core.board import Board
from tetris_coach.solver.search import Move, best_move
from tetris_coach.truth.windows import WindowSpec, load_window
from tetris_coach.vision.colour_tracker import ColourTracker

Cell = tuple[int, int]
HintTarget = tuple[str, tuple[Cell, ...]]


@dataclass(frozen=True)
class FrameOutput:
    """What one tracker would be showing the user on one frame."""

    frame: str
    accepted: bool  # the frame reached the tracker rather than being refused
    piece: str | None  # the falling piece it would name
    stack_rows: tuple[int, ...]  # the board it would hand the solver
    next_piece: str | None
    hint: HintTarget | None  # the placement it would be drawing

    @property
    def state(self) -> tuple[str | None, tuple[int, ...]]:
        """The coach's state: what it believes is falling, onto what."""
        return (self.piece, self.stack_rows)


def _target(move: Move | None) -> HintTarget | None:
    return None if move is None else (move.piece, tuple(sorted(move.cells)))


@lru_cache(maxsize=4096)
def _solve(stack_rows: tuple[int, ...], piece: str, next_piece: str | None) -> HintTarget | None:
    """The placement a coach with this board and this piece would show.

    Cached because a window holds the same board for runs of frames and
    the 2-ply search is the expensive thing in this harness.
    """
    return _target(best_move(Board(stack_rows), piece, next_piece))


def _next_crops(root: Path, spec: WindowSpec, names: list[str]) -> list[NDArray[np.uint8] | None]:
    from PIL import Image

    crops: list[NDArray[np.uint8] | None] = []
    for name in names:
        path = root / spec.name / f"next_{name}.png"
        if not path.exists():
            crops.append(None)
            continue
        crops.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)[:, :, ::-1])
    return crops


def run_shipped(root: Path, spec: WindowSpec) -> list[FrameOutput]:
    """Replay a window through ``CoachEngine``, wired as ``app.run`` wires it."""
    config = CoachConfig(rows=spec.rows)
    engine = CoachEngine(config, unobservable_cells=spec.geometry().unobservable)
    names, boards = load_window(root, spec)
    crops = _next_crops(root, spec, names)
    out: list[FrameOutput] = []
    for name, board, crop in zip(names, boards, crops, strict=True):
        # classify() is a pure function of the image plus the classifier's
        # memory and process_frame re-runs it identically, so reading the
        # confidence here does not perturb the replay -- the same thing
        # tests/test_live_session.py relies on.
        _occupancy, confidence = engine.classifier.classify(board)
        hint = engine.process_frame(board, crop)
        committed = engine.tracker.committed
        out.append(
            FrameOutput(
                frame=name,
                accepted=confidence >= config.min_confidence,
                piece=committed.falling_piece,
                stack_rows=committed.stack_rows,
                next_piece=committed.next_piece,
                hint=_target(hint),
            )
        )
    return out


def run_prototype(root: Path, spec: WindowSpec) -> list[FrameOutput]:
    """Replay a window through ``ColourTracker``, solving on every frame."""
    tracker = ColourTracker(
        rows=spec.rows,
        cols=10,
        unobservable_cells=spec.geometry().unobservable,
    )
    names, boards = load_window(root, spec)
    crops = _next_crops(root, spec, names)
    out: list[FrameOutput] = []
    for name, board, crop in zip(names, boards, crops, strict=True):
        report = tracker.update(board, crop)
        piece = None if report.falling is None else report.falling.piece
        hint = None if piece is None else _solve(report.stack_rows, piece, report.next_piece)
        out.append(
            FrameOutput(
                frame=name,
                accepted=True,
                piece=piece,
                stack_rows=report.stack_rows,
                next_piece=report.next_piece,
                hint=hint,
            )
        )
    return out


RUNNERS = {"shipped": run_shipped, "prototype": run_prototype}
