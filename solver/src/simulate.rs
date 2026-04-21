use dashmap::DashMap;
use rand::Rng;
use rayon::prelude::*;

use crate::build::apply_build;
use crate::income::calc_income;
use crate::lookahead::{choose_build, keep_ev, lookahead, reroll_ev};
use crate::solver::add_coins;
use crate::state::AState;

const MAX_TURNS: u32 = 1000;
const WRITE_SHARDS: usize = 512;

pub struct DepthResult {
    pub depth: usize,
    pub turns: Vec<u32>,
    pub elapsed_secs: f64,
}

pub fn run_all(depths: &[usize], n_games: usize) -> Vec<DepthResult> {
    depths
        .iter()
        .map(|&depth| {
            // Fresh income-EV cache per depth; shared (and warmed) across all N games.
            let cache: DashMap<u64, f64> = DashMap::with_shard_amount(WRITE_SHARDS);
            let t0 = std::time::Instant::now();

            // Warm the cache analytically before parallel simulation so that
            // all games share a pre-built lookup table.
            lookahead(AState::initial(), depth, &cache);

            let turns: Vec<u32> = (0..n_games)
                .into_par_iter()
                .map(|_| run_game(depth, &mut rand::thread_rng(), &cache))
                .collect();

            let elapsed_secs = t0.elapsed().as_secs_f64();
            let mean = turns.iter().map(|&t| t as f64).sum::<f64>() / turns.len() as f64;
            println!(
                "  depth {:2}  games={:>6}  mean={:.1}  cache={:>9}  ({:.2}s)",
                depth,
                n_games,
                mean,
                cache.len(),
                elapsed_secs
            );

            DepthResult { depth, turns, elapsed_secs }
        })
        .collect()
}

fn run_game(depth: usize, rng: &mut impl Rng, cache: &DashMap<u64, f64>) -> u32 {
    let mut state = AState::initial();
    let mut turns = 0u32;

    while !state.is_won() && turns < MAX_TURNS {
        turns += 1;

        let (mut roll, mut is_dbl) = sample_roll(&state, rng);

        // Radio Tower: reroll if analytical income EV of a fresh roll beats keeping.
        if state.has_tower() && depth > 0 {
            let kev = keep_ev(state, roll, is_dbl, depth, cache);
            let rev = reroll_ev(state, depth, cache);
            if rev < kev {
                let (r2, d2) = sample_roll(&state, rng);
                roll = r2;
                is_dbl = d2;
            }
        }

        let extra_turn = is_dbl && state.has_park();
        let income = calc_income(&state, roll);
        let new_coins = add_coins(state.coins, income);
        let new_state = AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks };

        let opt = choose_build(new_state, depth, extra_turn, cache);
        let (nc, nk, nl) = apply_build(new_coins, &state.cards, state.landmarks, opt);
        state = AState { coins: nc, cards: nk, landmarks: nl };
        // Amusement Park extra turn: if extra_turn, the outer while loop provides
        // the bonus roll automatically on the next iteration.
    }

    turns
}

fn sample_roll(state: &AState, rng: &mut impl Rng) -> (u8, bool) {
    if state.has_train() {
        let d1: u8 = rng.gen_range(1..=6);
        let d2: u8 = rng.gen_range(1..=6);
        (d1 + d2, d1 == d2)
    } else {
        (rng.gen_range(1..=6), false)
    }
}
