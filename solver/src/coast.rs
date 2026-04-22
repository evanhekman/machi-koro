use dashmap::DashMap;

use crate::build::{apply_build, build_options_slice, Opt};
use crate::dice::{DIST_1, DIST_2};
use crate::income::calc_income;
use crate::state::{AState, LANDMARK_COSTS, MAX_COINS, WIN_LMS};

const INF: f64 = 1e9;

/// Expected coins per turn given current state.
///
/// Accounts for optimal dice count (1 vs 2), Amusement Park (extra turn on
/// doubles via geometric multiplier), and Radio Tower (one reroll at threshold
/// = base E[income] since the reroll result must be kept).
///
/// Combined tower + park formula (closed-form, no iteration):
///   E_full = E_tower / (1 − p_dbl × (1 + p_rr))
/// where p_rr = P(non-doubles roll that triggers a reroll).
pub fn expected_income(state: &AState) -> f64 {
    let e1: f64 = DIST_1.iter().map(|o| o.prob * calc_income(state, o.roll) as f64).sum();
    // Tower with 1 die: reroll threshold = e1. Park has no effect (no doubles).
    let val1 = if state.has_tower() {
        DIST_1.iter().map(|o| o.prob * (calc_income(state, o.roll) as f64).max(e1)).sum()
    } else {
        e1
    };
    if !state.has_train() { return val1; }
    val1.max(two_dice_income(state))
}

fn two_dice_income(state: &AState) -> f64 {
    let e2: f64 = DIST_2.iter().map(|o| o.prob * calc_income(state, o.roll) as f64).sum();
    // Tower: reroll threshold = e2 (fresh mandatory-keep roll → expected = e2).
    let e_tower: f64 = if state.has_tower() {
        DIST_2.iter().map(|o| o.prob * (calc_income(state, o.roll) as f64).max(e2)).sum()
    } else {
        e2
    };
    if !state.has_park() { return e_tower; }
    // Park: extra turn on doubles. With tower, rerolling non-doubles also has a
    // chance of landing on doubles, so p_kept_dbl = p_dbl × (1 + p_rr).
    let p_dbl: f64 = DIST_2.iter().filter(|o| o.is_dbl).map(|o| o.prob).sum();
    let p_rr: f64 = if state.has_tower() {
        DIST_2.iter()
            .filter(|o| !o.is_dbl && (calc_income(state, o.roll) as f64) < e2)
            .map(|o| o.prob)
            .sum()
    } else {
        0.0
    };
    e_tower / (1.0 - p_dbl * (1.0 + p_rr))
}

/// Whether rolling 2 dice yields higher E[income] than 1 die for this state.
/// Accounts for park and tower. Only meaningful when state.has_train().
pub fn prefers_two_dice(state: &AState) -> bool {
    let e1: f64 = DIST_1.iter().map(|o| o.prob * calc_income(state, o.roll) as f64).sum();
    let val1 = if state.has_tower() {
        DIST_1.iter().map(|o| o.prob * (calc_income(state, o.roll) as f64).max(e1)).sum()
    } else {
        e1
    };
    two_dice_income(state) >= val1
}

/// Turns needed to win at current E[income] without any further purchases.
///
/// When coins already cover remaining landmark costs, returns the number of
/// remaining landmarks (each still requires one buy action). This ensures
/// coast_time is never 0 unless the game is actually won, so the buy-vs-coast
/// comparison always has a positive baseline to beat.
pub fn coast_time(state: &AState) -> f64 {
    if state.is_won() { return 0.0; }
    let remaining_lms = (0..4).filter(|&i| state.landmarks & (1 << i) == 0).count();
    let remaining_cost: u32 = (0..4usize)
        .filter(|&i| state.landmarks & (1 << i) == 0)
        .map(|i| LANDMARK_COSTS[i] as u32)
        .sum();
    let buy_actions = remaining_lms as f64;
    if state.coins as u32 >= remaining_cost {
        return buy_actions;
    }
    let needed = (remaining_cost - state.coins as u32) as f64;
    let e_inc = expected_income(state);
    if e_inc <= 0.0 { return INF; }
    needed / e_inc + buy_actions
}

fn remaining_landmarks(state: AState) -> usize {
    (0..4).filter(|&i| state.landmarks & (1 << i) == 0).count()
}

pub fn earn_income(state: AState) -> AState {
    let e_inc = expected_income(&state);
    let coins = ((state.coins as f64 + e_inc).round() as u32).min(MAX_COINS as u32) as u8;
    AState { coins, cards: state.cards, landmarks: state.landmarks }
}

/// Minimum coast_time achievable after `depth` more (buy → earn income) cycles.
///
/// `state` is a post-income, pre-buy state, matching the point where
/// `choose_build_coast` is called in the game loop. Each depth step makes an
/// optional purchase with the current coins, then earns E[income], and recurses
/// with depth-1. Both the no-buy and buy branches earn income in the same step,
/// so the comparison is symmetric at all depths.
///
/// depth=0: coast_time(state) — no more planning.
/// depth=N: one buy decision now, then N-1 more cycles.
pub fn coast_lookahead(state: AState, depth: usize, cache: &DashMap<u64, f64>) -> f64 {
    if state.is_won() { return 0.0; }

    let key = state.pack_key(depth as u8);
    if let Some(v) = cache.get(&key) { return *v; }

    if depth == 0 {
        let v = coast_time(&state);
        cache.insert(key, v);
        return v;
    }

    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    // No-buy: earn income this turn, then recurse.
    let mut best = coast_lookahead(earn_income(state), depth - 1, cache);

    // Buy one thing, then earn income, then recurse.
    for &opt in &opts[1..n] {
        let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
        if nl == WIN_LMS { best = 0.0_f64.min(best); continue; }
        let post_buy = AState { coins: nc, cards: nk, landmarks: nl };
        // Short-circuit: if we can already afford all remaining landmarks, no
        // further income cycles are needed — just count the buy actions.
        let val = if coast_time(&post_buy) <= remaining_landmarks(post_buy) as f64 {
            coast_time(&post_buy)
        } else {
            coast_lookahead(earn_income(post_buy), depth - 1, cache)
        };
        if val < best { best = val; }
    }

    cache.insert(key, best);
    best
}

/// Best currently-affordable build for game simulation using the coast strategy.
///
/// Argmin of coast_lookahead over all options. The no-buy and buy paths both
/// earn income as the first lookahead step, so the comparison is symmetric.
pub fn choose_build_coast(state: AState, depth: usize, cache: &DashMap<u64, f64>) -> Opt {
    if state.is_won() { return None; }
    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    let eval = |s: AState| -> f64 {
        let ct = coast_time(&s);
        if depth == 0 || ct <= remaining_landmarks(s) as f64 { ct }
        else { coast_lookahead(earn_income(s), depth - 1, cache) }
    };

    let mut best_opt: Opt = None;
    let mut best_val = eval(state);

    for &opt in &opts[1..n] {
        let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
        if nl == WIN_LMS { return opt; }
        let val = eval(AState { coins: nc, cards: nk, landmarks: nl });
        if val <= best_val { best_val = val; best_opt = opt; }
    }

    best_opt
}
