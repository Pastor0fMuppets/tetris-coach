#!/usr/bin/env bash
# Set up tetris-coach on a fresh Mac. Idempotent: safe to re-run.
#
#   git clone https://github.com/Pastor0fMuppets/tetris-coach.git
#   cd tetris-coach && ./scripts/setup-mac.sh
#
# What it does NOT do: grant Screen Recording. macOS only lets you do that
# by hand, and the app will read a blank screen until you have (see below).
set -euo pipefail

cd "$(dirname "$0")/.."
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

[ "$(uname -s)" = "Darwin" ] || { echo "This tool is macOS-only (it captures the screen and draws a Qt overlay)." >&2; exit 1; }

say "1/3  Python 3.11+"
if command -v uv >/dev/null 2>&1; then
  echo "uv already installed: $(uv --version)"
else
  echo "Installing uv (manages the Python version so the system one is left alone)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv in ~/.local/bin, which may not be on PATH yet.
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo "uv installed but not on PATH. Add ~/.local/bin to PATH and re-run." >&2; exit 1; }
fi

say "2/3  Virtual environment and dependencies"
uv venv --python 3.12
uv pip install -e '.[dev]'

say "3/3  Checking it imports"
.venv/bin/python -c "import tetris_coach, PySide6, mss; print('ok:', tetris_coach.__file__)"

cat <<'NOTE'

Setup done. One manual step remains, and the tool cannot do it for you.

SCREEN RECORDING PERMISSION
  macOS will not let any program see other windows until you allow it, and
  it grants the permission to the TERMINAL APP, not to this project.

  1. Run the command below in Terminal.app.
  2. macOS should prompt "Terminal would like to record this computer's
     screen" -> allow it. If no prompt appears, open
     System Settings > Privacy & Security > Screen & System Audio Recording
     and switch Terminal on (add it with + if it is not listed).
  3. QUIT TERMINAL COMPLETELY (Cmd-Q, not just closing the window) and
     reopen it. macOS only applies the permission at launch.

  Until that is done the capture returns the desktop wallpaper and the
  coach will sit there saying nothing. That is the single most common way
  this looks broken.

TO RUN (single hint, the version to start from)
  git checkout v0.2-playable
  PYTHONUNBUFFERED=1 .venv/bin/tetris-coach --debug --rows 12

  --rows 12 suits ROAS Stacker. Use --rows 20 for a standard Tetris board.
  On the current main branch instead, --no-next-hint gives the same single
  hint plus later fixes.

NOTE
