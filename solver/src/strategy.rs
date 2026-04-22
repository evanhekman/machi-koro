use dashmap::DashMap;

use crate::build::{apply_build, build_options_slice, Opt};
use crate::coast::{expected_income, prefers_two_dice};
use crate::dice::{Outcome, DIST_1, DIST_2};
use crate::income::calc_income;
use crate::solver::add_coins;
use crate::state::{AState, WIN_LMS};

pub const INF: f64 = 1e9;
const P_DBL: f64 = 1.0 / 6.0;
const WIN_INCOME: f64 = 16.0; // just above the ~15 theoretical income ceiling

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strategy {
    WinProb,   // maximise P(win within depth turns); leaf = 0.0
    MinTurns,  // minimise expected turns to win;     leaf = INF
    MaxIncome, // maximise E[income/turn];             leaf = expected_income(state)
}

impl Strategy {
    #[inline] pub fn win_val(self) -> f64 {
        match self { Strategy::WinProb => 1.0, Strategy::MinTurns => 0.0, Strategy::MaxIncome => WIN_INCOME }
    }
    #[inline] pub fn leaf_val(self, state: &AState) -> f64 {
        match self { Strategy::WinProb => 0.0, Strategy::MinTurns => INF, Strategy::MaxIncome => expected_income(state) }
    }
    #[inline] pub fn turn_cost(self) -> f64 {
        match self { Strategy::WinProb | Strategy::MaxIncome => 0.0, Strategy::MinTurns => 1.0 }
    }
    #[inline] pub fn worst(self) -> f64 {
        match self { Strategy::WinProb | Strategy::MaxIncome => f64::NEG_INFINITY, Strategy::MinTurns => f64::INFINITY }
    }

    #[inline]
    pub fn best_of(self, a: f64, b: f64) -> f64 {
        match self { Strategy::WinProb | Strategy::MaxIncome => a.max(b), Strategy::MinTurns => a.min(b) }
    }

    #[inline]
    pub fn is_better(self, candidate: f64, current_best: f64) -> bool {
        match self { Strategy::WinProb | Strategy::MaxIncome => candidate > current_best, Strategy::MinTurns => candidate < current_best }
    }

    fn adjust_extra_turn(self, val: f64) -> f64 {
        match self {
            Strategy::WinProb => (val / (1.0 - P_DBL)).min(1.0),
            Strategy::MinTurns => val, // TODO: geometric series for MinTurns
            Strategy::MaxIncome => val, // park already modelled in expected_income leaf
        }
    }

    /// Which dice distribution to use for this strategy and state.
    #[inline]
    pub fn dist(self, state: &AState) -> &'static [Outcome] {
        if !state.has_train() { return DIST_1; }
        match self {
            Strategy::MaxIncome => if prefers_two_dice(state) { DIST_2 } else { DIST_1 },
            _ => DIST_2,
        }
    }
}

/// Main search function. Returns P(win) for WinProb, expected turns for MinTurns.
pub fn search(state: AState, depth: usize, s: Strategy, cache: &DashMap<u64, f64>) -> f64 {
    if state.is_won() { return s.win_val(); }
    if depth == 0 { return s.leaf_val(&state); }

    let key = state.pack_key(depth as u8);
    if let Some(v) = cache.get(&key) { return *v; }

    let dist: &[Outcome] = s.dist(&state);

    let ev = if state.has_tower() {
        // Precompute inner value of a fresh reroll (shared across all outcomes).
        let reroll_inner: f64 = dist.iter()
            .map(|o| {
                let nc = add_coins(state.coins, calc_income(&state, o.roll));
                o.prob * best_build_inner(
                    AState { coins: nc, cards: state.cards, landmarks: state.landmarks },
                    depth, o.is_dbl && state.has_park(), s, cache,
                )
            })
            .sum();

        // For each outcome, choose best(keep, reroll) then add turn cost.
        s.turn_cost() + dist.iter()
            .map(|o| {
                let nc = add_coins(state.coins, calc_income(&state, o.roll));
                let keep_inner = best_build_inner(
                    AState { coins: nc, cards: state.cards, landmarks: state.landmarks },
                    depth, o.is_dbl && state.has_park(), s, cache,
                );
                o.prob * s.best_of(keep_inner, reroll_inner)
            })
            .sum::<f64>()
    } else {
        s.turn_cost() + dist.iter()
            .map(|o| {
                let nc = add_coins(state.coins, calc_income(&state, o.roll));
                o.prob * best_build_inner(
                    AState { coins: nc, cards: state.cards, landmarks: state.landmarks },
                    depth, o.is_dbl && state.has_park(), s, cache,
                )
            })
            .sum::<f64>()
    };

    cache.insert(key, ev);
    ev
}

/// Value after the build decision, before adding turn cost.
fn best_build_inner(
    state: AState,
    depth: usize,
    extra_turn: bool,
    s: Strategy,
    cache: &DashMap<u64, f64>,
) -> f64 {
    if state.is_won() { return s.win_val(); }
    let next_depth = depth - 1;
    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    let best = opts[..n].iter()
        .map(|&opt| {
            let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
            if nl == WIN_LMS { return s.win_val(); }
            let next = AState { coins: nc, cards: nk, landmarks: nl };
            if next_depth == 0 { return s.leaf_val(&next); }
            search(next, next_depth, s, cache)
        })
        .fold(s.worst(), |a, b| s.best_of(a, b));

    if extra_turn { s.adjust_extra_turn(best) } else { best }
}

/// Choose the best build option for game simulation.
pub fn choose_build(
    state: AState,
    depth: usize,
    extra_turn: bool,
    s: Strategy,
    cache: &DashMap<u64, f64>,
) -> Opt {
    if state.is_won() { return None; }
    let (opts, n) = build_options_slice(state.coins, &state.cards, state.landmarks);

    let mut best_opt: Opt = opts[0];
    let mut best_val: f64 = s.worst();

    for &opt in &opts[..n] {
        let (nc, nk, nl) = apply_build(state.coins, &state.cards, state.landmarks, opt);
        if nl == WIN_LMS { return opt; }
        let val = search(AState { coins: nc, cards: nk, landmarks: nl }, depth, s, cache);
        if s.is_better(val, best_val) {
            best_val = val;
            best_opt = opt;
        }
    }

    let _ = extra_turn;
    best_opt
}

/// Expected inner value of a fresh reroll (without turn cost). Used for Radio Tower decisions.
pub fn reroll_ev(state: AState, depth: usize, s: Strategy, cache: &DashMap<u64, f64>) -> f64 {
    let dist: &[Outcome] = s.dist(&state);
    dist.iter()
        .map(|o| {
            let nc = add_coins(state.coins, calc_income(&state, o.roll));
            o.prob * best_build_inner(
                AState { coins: nc, cards: state.cards, landmarks: state.landmarks },
                depth, o.is_dbl && state.has_park(), s, cache,
            )
        })
        .sum()
}

/// Inner value of keeping a specific roll. Used for Radio Tower decisions.
pub fn keep_ev(
    state: AState,
    roll: u8,
    is_dbl: bool,
    depth: usize,
    s: Strategy,
    cache: &DashMap<u64, f64>,
) -> f64 {
    let nc = add_coins(state.coins, calc_income(&state, roll));
    best_build_inner(
        AState { coins: nc, cards: state.cards, landmarks: state.landmarks },
        depth, is_dbl && state.has_park(), s, cache,
    )
}
