# Tetris Coach — working notes for Claude

Read SPEC.md first; it is the source of truth for architecture and scope. Key design
decisions there are settled — implement, don't redesign.

- Pure-visual tool: never add code that sends input to the game.
- `core/` and `solver/` must stay importable and testable with no GUI/display and no
  macOS-only dependencies. macOS-only code lives in `capture/` and `overlay/` behind
  import guards.
- Performance is a feature: keep the 2-ply search under 50 ms (benchmark enforced).
- Run `pytest`, `ruff check`, and `mypy` before committing.
- Commit in small increments with clear messages.
