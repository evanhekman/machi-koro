## status: DEPRECATED
All active strategy work lives in the Rust solver (`solver/`). The Python layer below is
a reference/prototyping layer that has been superseded and should not be used for benchmarking
or performance analysis.

## files (deprecated)
- `analysis.py` — Python expectimax reference implementation (superseded by `solver/src/solver.rs`)
- `lookahead.py` — Python income-maximising lookahead (superseded by `solver/src/income_strategy.rs`)
- `simulate.py` — Python game runner (superseded by `solver/src/bin/simulate.rs`)
- `strategies.py` — heuristic baselines (buy_cheapest, rush_landmarks, random)

## active strategies (Rust solver)
- `--winprob`   — maximise P(win within depth turns)
- `--minturns`  — minimise expected turns to win
- `--coast`     — minimise coast-time heuristic
- `--maxincome` — maximise expected income per turn
