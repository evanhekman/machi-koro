## project overview
expectimax analysis engine for solitaire Machi Koro. maximises P(win within N turns) via
depth-limited search with memoisation. to compensate, only BLUE and GREEN cards are in the supply, with 3 of each instead of 6.

## key files
- `engine.py` — game rules and state transitions
- `analysis.py` — expectimax solver and strategy wrapper
- `simulate.py` — run simulated games
- `strategies.py` — potential strategies to use

## things that didn't work
- pareto frontier dominance pruning
  - each card introduces a dimension -> too many aspects to make this reasonable
  - took longer than before
- coin capping
  - 52+ coins is very rare, most of the time just an extra thing to check
  - took longer than before

## notes
- never create markdown files unless specifically requested
- use shebang to reference the `.venv/bin/python` interpreter
