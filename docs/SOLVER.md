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

## active work
feasibility pruning: early return 0.0 for states where expected income × remaining depth
is insufficient to cover remaining landmark cost. see conversation context for tradeoffs.
