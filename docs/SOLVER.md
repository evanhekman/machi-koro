## modules
- `state.rs` — AState, card/landmark constants, pack_key() for cache keying
- `dice.rs` — DIST_1 (1 die) and DIST_2 (2 dice) probability tables
- `income.rs` — calc_income(state, roll): deterministic income per roll outcome
- `build.rs` — enumerate and apply purchase options each turn
- `solver.rs` — core expectimax: analyze(), compute_ev(), best_build(), roll_ev()
- `cache.rs` — zstd delta file save/load
- `main.rs` — CLI, thread pool, first-turn recommendation table

## cache semantics
each depth pass writes only new entries as a delta (cache/d{N:02}.bin.zst).
at runtime only one delta is loaded — the depth just below the target.
this works because analyze(state, d) only ever recurses to depth d-1,
so frozen only needs depth-(d-1) entries to avoid all recomputation.
after each pass, frozen is trimmed to retain only that depth's entries,
keeping memory at O(reachable states) instead of O(reachable states × depth).

## key constants
- MAX_COINS=52: equals total landmark cost; coins above this are unreachable in practice
- WRITE_SHARDS=512: DashMap shard count to minimise lock contention during parallel writes
- PAR_THRESHOLD=4: depth below which dice outcomes run serially (parallelism overhead > benefit)

## winprob strategy behaviour by depth

`choose_build` passes `depth` (not `depth-1`) to `search` on the post-purchase state,
so the effective lookahead from the pre-purchase state is `depth+1` turns.

**depth horizon:** the earliest possible win is 7 turns (verified exhaustively via trace_win).
this requires: roll 1/2/3 on turn 1 (+1 coin), then roll 4 every turn to chain CS income
into landmarks in order: CS→CS→CS→SM→TS→AP→RT (or permutations of the last four).
no amusement park extra turns are needed — it's a straight 7-turn sequence.

**why depth 6 shows a large performance jump over depth 5:**
`choose_build` at depth 6 searches depth 6 ahead from the post-purchase state = 7 turns
from the start. this is exactly the minimum win horizon, so depth 6 has real signal on
turn 1 and immediately commits to the optimal CS purchase. depth 5 is blind on turn 1
and must wait until mid-game before it can see a win path.

**how low depths (e.g. depth 3) still win:**
when blind, the strategy always skips (all options return P=0, `None` is first and wins
the argmax). coins accumulate passively from wheat/bakery. once coins are high enough
that a depth-3 search from the current state can see a win, the strategy kicks in —
typically buying landmarks and interleaving cheap income cards to bridge coin gaps.
depth 3 typically stays blind until ~40-50 coins, then finishes in ~6 turns.
average game length at depth 3 ≈ 92 turns vs depth 6 ≈ 20 turns.

**amusement park approximation:**
extra turns from rolling doubles with amusement park are modelled as a geometric series
(best / (1 - 1/6) = best × 6/5) rather than real recursive branching. this means:
- it correctly accounts for infinite chains of potential doubles
- it does NOT correctly handle cases where a win strictly requires the extra turn
  (scaling zero is still zero — phantom wins cannot be manufactured)
- verified: P(win)=0.000000 at depth 6 from initial state is exact, not a rounding error

## active work
feasibility pruning: early return 0.0 for states where expected income × remaining depth
is insufficient to cover remaining landmark cost. see conversation context for tradeoffs.
