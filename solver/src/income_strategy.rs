use dashmap::DashMap;

use crate::build::{apply_build, build_options_slice, Opt};
use crate::coast::{earn_income, expected_income};
use crate::state::{AState, WIN_LMS};

const WIN_VALUE: f64 = 16.0;

/// Best expected income achievable after `depth` more (buy → earn income) cycles.
///
/// Mirrors coast_lookahead but maximises expected_income rather than minimising
/// coast_time. Won states return WIN_VALUE (just above the theoretical income
/// ceiling of ~15), so any path that wins always beats a non-winning path.
pub fn income_lookahead(state: AState, depth: usize, cache: &DashMap<u64, f64>) -> f64 {
    if state.is_won() { return WIN_VALUE; }

    let key = state.pack_key(depth as u8);
    if let Some(v) = cache.get(&key) { return *v; }

    if depth == 0 {
        let v = expected_income(&state);
        cache.insert(key, v);
        return v;
    }

    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    // No-buy baseline: earn income this turn, then recurse.
    let mut best = income_lookahead(earn_income(state), depth - 1, cache);

    for &opt in &opts[1..n] {
        let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
        let val = if nl == WIN_LMS {
            WIN_VALUE
        } else {
            income_lookahead(earn_income(AState { coins: nc, cards: nk, landmarks: nl }), depth - 1, cache)
        };
        if val > best { best = val; }
    }

    cache.insert(key, best);
    best
}

/// Best currently-affordable build for game simulation using the max-income strategy.
pub fn choose_build_income(state: AState, depth: usize, cache: &DashMap<u64, f64>) -> Opt {
    if state.is_won() { return None; }
    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    let eval = |s: AState| -> f64 {
        if depth == 0 { expected_income(&s) } else { income_lookahead(earn_income(s), depth - 1, cache) }
    };

    let mut best_opt: Opt = None;
    let mut best_val = eval(state);

    for &opt in &opts[1..n] {
        let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
        if nl == WIN_LMS { return opt; }
        let val = eval(AState { coins: nc, cards: nk, landmarks: nl });
        if val > best_val { best_val = val; best_opt = opt; }
    }

    best_opt
}
