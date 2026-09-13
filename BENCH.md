# Search-time benchmark

Full 2-ply search (`solver.search.best_move` with a known next piece): every
(rotation, column) placement of the current piece, each scored by the best
next-piece placement on the resulting board. Pure CPython on bitboards, no
pruning. Budget: **< 50 ms** (enforced by
`tests/test_search.py::TestBenchmark::test_two_ply_search_under_50ms`).

## Apple M2 (macOS 14.5, 8 GB, CPython 3.12.14)

300 runs per scenario, wall-clock (`time.perf_counter`):

| Scenario                                   | p50     | p95     | max      |
| ------------------------------------------ | ------- | ------- | -------- |
| Empty board, T then T (worst-case ~34x34)  | 6.4 ms  | 6.5 ms  | 6.6 ms   |
| Mid-game board, T then T                   | 7.5 ms  | 7.7 ms  | 11.0 ms  |
| Mid-game board, I then I                   | 1.9 ms  | 2.0 ms  | 2.1 ms   |
| Self-play, 500 moves, random pieces        | 3.3 ms  | 7.0 ms  | 7.5 ms   |

pytest-benchmark (same machine, T x T on the mid-game board): mean 7.5 ms,
median 7.5 ms over 134 rounds.

Headroom vs. the 50 ms budget is ~6x at p95, so the precompute-ahead loop in
`app.py` (solve during the fall of the previous piece) comfortably hides the
search latency entirely.

## CI hardware

Not yet measured on CI. Re-run and append results with:

```sh
python -m pytest tests/test_search.py::TestBenchmark -q  # asserts the budget
python -m tetris_coach.cli --demo --pieces 500           # prints p50/p95
```
