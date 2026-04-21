use dashmap::DashMap;

use crate::build::{apply_build, build_options_slice, Opt};
use crate::dice::{Outcome, DIST_1, DIST_2};
use crate::income::calc_income;
use crate::solver::add_coins;
use crate::state::{AState, WIN_LMS};

pub const INF: f64 = 1e9;

/// Expected turns to win from `state` looking `depth` turns ahead.
/// Returns 0.0 if already won, INF if no winning path found within horizon.
pub fn lookahead(state: AState, depth: usize, cache: &DashMap<u64, f64>) -> f64 {
    if state.is_won() { return 0.0; }
    if depth == 0 { return INF; }

    let key = state.pack_key(depth as u8);
    if let Some(v) = cache.get(&key) { return *v; }

    let dist: &[Outcome] = if state.has_train() { DIST_2 } else { DIST_1 };

    let ev = if state.has_tower() {
        // Radio Tower: for each outcome, choose min(keep, reroll).
        // reroll_inner = weighted average of best_build_ev over a fresh roll.
        let reroll_inner: f64 = dist.iter()
            .map(|o| {
                let new_coins = add_coins(state.coins, calc_income(&state, o.roll));
                o.prob * best_build_ev(
                    AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks },
                    depth,
                    o.is_dbl && state.has_park(),
                    cache,
                )
            })
            .sum();
        dist.iter()
            .map(|o| {
                let new_coins = add_coins(state.coins, calc_income(&state, o.roll));
                let keep_inner = best_build_ev(
                    AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks },
                    depth,
                    o.is_dbl && state.has_park(),
                    cache,
                );
                o.prob * (1.0 + keep_inner.min(reroll_inner))
            })
            .sum()
    } else {
        roll_ev(state, dist, depth, cache)
    };

    cache.insert(key, ev);
    ev
}

fn roll_ev(state: AState, dist: &[Outcome], depth: usize, cache: &DashMap<u64, f64>) -> f64 {
    dist.iter()
        .map(|o| {
            let new_coins = add_coins(state.coins, calc_income(&state, o.roll));
            let inner = best_build_ev(
                AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks },
                depth,
                o.is_dbl && state.has_park(),
                cache,
            );
            o.prob * (1.0 + inner)
        })
        .sum()
}

fn best_build_ev(state: AState, depth: usize, extra_turn: bool, cache: &DashMap<u64, f64>) -> f64 {
    if state.is_won() { return 0.0; }
    let next_depth = depth - 1;
    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);
    let best = opts[..n]
        .iter()
        .map(|&opt| {
            let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
            if nl == WIN_LMS { return 0.0; }
            lookahead(AState { coins: nc, cards: nk, landmarks: nl }, next_depth, cache)
        })
        .fold(f64::INFINITY, f64::min);

    let _ = extra_turn;
    best
}

/// Best build option for game simulation — picks the build that minimizes expected turns to win.
pub fn choose_build(
    state: AState,
    depth: usize,
    extra_turn: bool,
    cache: &DashMap<u64, f64>,
) -> Opt {
    if state.is_won() { return None; }
    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    let mut best_opt: Opt = opts[0];
    let mut best_val: f64 = f64::INFINITY;

    for &opt in &opts[..n] {
        let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
        if nl == WIN_LMS { return opt; }
        let val = lookahead(AState { coins: nc, cards: nk, landmarks: nl }, depth, cache);
        if val < best_val {
            best_val = val;
            best_opt = opt;
        }
    }

    let _ = extra_turn;
    best_opt
}

/// Expected turns for a fresh reroll — used to decide whether to reroll with Radio Tower.
/// Returns the inner cost (without the +1 for the current turn).
pub fn reroll_ev(state: AState, depth: usize, cache: &DashMap<u64, f64>) -> f64 {
    let dist: &[Outcome] = if state.has_train() { DIST_2 } else { DIST_1 };
    dist.iter()
        .map(|o| {
            let new_coins = add_coins(state.coins, calc_income(&state, o.roll));
            o.prob * best_build_ev(
                AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks },
                depth,
                o.is_dbl && state.has_park(),
                cache,
            )
        })
        .sum()
}

/// Expected turns for keeping a specific roll (inner cost, without the +1 for this turn).
pub fn keep_ev(state: AState, roll: u8, is_dbl: bool, depth: usize, cache: &DashMap<u64, f64>) -> f64 {
    let new_coins = add_coins(state.coins, calc_income(&state, roll));
    let after = AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks };
    best_build_ev(after, depth, is_dbl && state.has_park(), cache)
}
