use dashmap::DashMap;
use rand::Rng;
use rayon::prelude::*;

use crate::build::apply_build;
use crate::coast::{choose_build_coast, coast_lookahead};
use crate::income::calc_income;
use crate::solver::add_coins;
use crate::state::AState;
use crate::strategy::{self, Strategy};

const MAX_TURNS: u32 = 1000;
const WRITE_SHARDS: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimStrategy {
    WinProb,
    MinTurns,
    Coast,
    MaxIncome,
}

pub struct DepthResult {
    pub depth: usize,
    pub turns: Vec<u32>,
    pub elapsed_secs: f64,
}

pub fn run_all(sim_strategy: SimStrategy, depths: &[usize], n_games: usize) -> Vec<DepthResult> {
    depths
        .iter()
        .map(|&depth| {
            let cache: DashMap<u64, f64> = DashMap::with_shard_amount(WRITE_SHARDS);
            let t0 = std::time::Instant::now();

            // Warm the cache analytically before parallel simulation.
            match sim_strategy {
                SimStrategy::WinProb   => { strategy::search(AState::initial(), depth, Strategy::WinProb,  &cache); }
                SimStrategy::MinTurns  => { strategy::search(AState::initial(), depth, Strategy::MinTurns, &cache); }
                SimStrategy::Coast     => { coast_lookahead(AState::initial(), depth, &cache); }
                SimStrategy::MaxIncome => { strategy::search(AState::initial(), depth, Strategy::MaxIncome, &cache); }
            }

            let turns: Vec<u32> = (0..n_games)
                .into_par_iter()
                .map(|_| run_game(sim_strategy, depth, &mut rand::thread_rng(), &cache))
                .collect();

            let elapsed_secs = t0.elapsed().as_secs_f64();
            let mean = turns.iter().map(|&t| t as f64).sum::<f64>() / turns.len() as f64;
            println!(
                "  depth {:2}  games={:>6}  mean={:.1}  cache={:>9}  ({:.2}s)",
                depth, n_games, mean, cache.len(), elapsed_secs
            );

            DepthResult { depth, turns, elapsed_secs }
        })
        .collect()
}

fn run_game(
    sim_strategy: SimStrategy,
    depth: usize,
    rng: &mut impl Rng,
    cache: &DashMap<u64, f64>,
) -> u32 {
    let mut state = AState::initial();
    let mut turns = 0u32;

    while !state.is_won() && turns < MAX_TURNS {
        turns += 1;

        let (mut roll, mut is_dbl) = sample_roll(&state, rng);

        // Radio Tower reroll decision (not applicable for Coast).
        if state.has_tower() && depth > 0 {
            if let Some(s) = to_search_strategy(sim_strategy) {
                let kev = strategy::keep_ev(state, roll, is_dbl, depth, s, cache);
                let rev = strategy::reroll_ev(state, depth, s, cache);
                if s.is_better(rev, kev) {
                    let (r2, d2) = sample_roll(&state, rng);
                    roll = r2;
                    is_dbl = d2;
                }
            }
        }

        let extra_turn = is_dbl && state.has_park();
        let income = calc_income(&state, roll);
        let new_coins = add_coins(state.coins, income);
        let mid = AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks };

        let opt = match sim_strategy {
            SimStrategy::WinProb   => strategy::choose_build(mid, depth, extra_turn, Strategy::WinProb,  cache),
            SimStrategy::MinTurns  => strategy::choose_build(mid, depth, extra_turn, Strategy::MinTurns, cache),
            SimStrategy::Coast     => choose_build_coast(mid, depth, cache),
            SimStrategy::MaxIncome => strategy::choose_build(mid, depth, extra_turn, Strategy::MaxIncome, cache),
        };

        let (nc, nk, nl) = apply_build(new_coins, &state.cards, state.landmarks, opt);
        state = AState { coins: nc, cards: nk, landmarks: nl };
    }

    turns
}

fn to_search_strategy(s: SimStrategy) -> Option<Strategy> {
    match s {
        SimStrategy::WinProb   => Some(Strategy::WinProb),
        SimStrategy::MinTurns  => Some(Strategy::MinTurns),
        SimStrategy::Coast     => None,
        SimStrategy::MaxIncome => Some(Strategy::MaxIncome),
    }
}

fn sample_roll(state: &AState, rng: &mut impl Rng) -> (u8, bool) {
    if state.has_train() && use_two_dice(state) {
        let d1: u8 = rng.gen_range(1..=6);
        let d2: u8 = rng.gen_range(1..=6);
        (d1 + d2, d1 == d2)
    } else {
        (rng.gen_range(1..=6), false)
    }
}

/// Roll 2 dice only if it yields higher E[income] than 1 die (accounting for park/tower).
fn use_two_dice(state: &AState) -> bool {
    crate::coast::prefers_two_dice(state)
}
