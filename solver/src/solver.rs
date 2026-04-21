use std::collections::HashMap;
use std::path::Path;
use dashmap::DashMap;
use rayon::prelude::*;

use crate::state::{AState, NUM_CARDS, MAX_COINS, WIN_LMS};
use crate::income::calc_income;
use crate::build::{build_options_slice, apply_build};
use crate::dice::{Outcome, DIST_1, DIST_2};
use crate::cache;

const P_DBL: f64 = 1.0 / 6.0;

/// Minimum recursion depth at which dice outcomes run in parallel.
const PAR_THRESHOLD: usize = 4;

/// DashMap shard count. More shards → lower lock-contention probability
/// when many workers write concurrently. Must be a power of two.
/// With 8 workers and 512 shards, P(collision per op) ≈ 1.6 % vs 12.5 % at 64.
const WRITE_SHARDS: usize = 512;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn run(
    max_depth: usize,
    pool: &rayon::ThreadPool,
    start_depth: usize,
    mut frozen: HashMap<u64, f64>,
    cache_dir: &Path,
) -> (Vec<(usize, f64, f64)>, HashMap<u64, f64>) {
    let initial = AState::initial();
    let mut results = Vec::with_capacity(max_depth);

    for d in (start_depth + 1)..=max_depth {
        let write: DashMap<u64, f64> = DashMap::with_shard_amount(WRITE_SHARDS);

        let t0 = std::time::Instant::now();
        let val = pool.install(|| analyze(initial, d, &frozen, &write));
        let elapsed = t0.elapsed().as_secs_f64();

        let new_entries = write.len();
        frozen.reserve(new_entries);
        let mut delta = HashMap::with_capacity(new_entries);
        for (k, v) in write {
            if frozen.insert(k, v).is_none() {
                delta.insert(k, v);
            }
        }

        // Discard all entries except depth d: to compute depth d+1 we only ever
        // need depth-d lookups (those are immediate hits; we never recurse deeper).
        // This keeps frozen at O(states) instead of O(states × depth).
        let d_bits = (d as u64) << 32;
        let mask: u64 = 0xff_u64 << 32;
        frozen.retain(|&k, _| k & mask == d_bits);
        frozen.shrink_to_fit();

        println!(
            "  depth {:2}  P(win)={:.6}  cache={:>9}  new={:>8}  ({:.3}s)",
            d, val, frozen.len(), new_entries, elapsed
        );

        if let Err(e) = cache::save_delta(cache_dir, d, &delta) {
            eprintln!("  warning: failed to save cache delta for depth {d}: {e}");
        }

        results.push((d, val, elapsed));
    }

    (results, frozen)
}

// ---------------------------------------------------------------------------
// Core expectimax
// ---------------------------------------------------------------------------

pub fn analyze(
    state: AState,
    depth: usize,
    frozen: &HashMap<u64, f64>,
    write: &DashMap<u64, f64>,
) -> f64 {
    if state.is_won() { return 1.0; }
    if depth == 0 { return 0.0; }

    let key = state.pack_key(depth as u8);

    if let Some(&v) = frozen.get(&key) { return v; }
    if let Some(v) = write.get(&key) { return *v; }

    let ev = compute_ev(state, depth, frozen, write);
    write.insert(key, ev);
    ev
}

fn compute_ev(
    state: AState,
    depth: usize,
    frozen: &HashMap<u64, f64>,
    write: &DashMap<u64, f64>,
) -> f64 {
    let dist: &[Outcome] = if state.has_train() { DIST_2 } else { DIST_1 };

    if state.has_tower() {
        // Compute val_reroll first so sub-caches are warm for the keep loop.
        let val_reroll = roll_ev(state, dist, depth, frozen, write);
        dist.iter()
            .map(|o| {
                let new_coins = add_coins(state.coins, calc_income(&state, o.roll));
                let val_keep = best_build(
                    new_coins, state.cards, state.landmarks, depth,
                    o.is_dbl && state.has_park(), frozen, write,
                );
                o.prob * val_keep.max(val_reroll)
            })
            .sum()
    } else {
        roll_ev(state, dist, depth, frozen, write)
    }
}

pub fn roll_ev(
    state: AState,
    dist: &[Outcome],
    depth: usize,
    frozen: &HashMap<u64, f64>,
    write: &DashMap<u64, f64>,
) -> f64 {
    let eval = |o: &Outcome| {
        let new_coins = add_coins(state.coins, calc_income(&state, o.roll));
        o.prob * best_build(
            new_coins, state.cards, state.landmarks, depth,
            o.is_dbl && state.has_park(), frozen, write,
        )
    };

    if depth >= PAR_THRESHOLD {
        dist.par_iter().map(eval).sum()
    } else {
        dist.iter().map(eval).sum()
    }
}

pub fn best_build(
    coins: u8,
    cards: [u8; NUM_CARDS],
    landmarks: u8,
    depth: usize,
    extra_turn: bool,
    frozen: &HashMap<u64, f64>,
    write: &DashMap<u64, f64>,
) -> f64 {
    if landmarks == WIN_LMS { return 1.0; }
    let next_depth = depth - 1;

    let (opts, n) = build_options_slice(coins, &cards, landmarks);

    let best = opts[..n].iter().map(|&opt| {
        let (nc, nk, nl) = apply_build(coins, &cards, landmarks, opt);
        if nl == WIN_LMS { return 1.0_f64; }
        analyze(AState { coins: nc, cards: nk, landmarks: nl }, next_depth, frozen, write)
    }).fold(0.0_f64, f64::max);

    if extra_turn {
        (best / (1.0 - P_DBL)).min(1.0)
    } else {
        best
    }
}

/// Like `best_build` but also returns which option was chosen.
pub fn choose_build(
    coins: u8,
    cards: [u8; NUM_CARDS],
    landmarks: u8,
    depth: usize,
    extra_turn: bool,
    frozen: &HashMap<u64, f64>,
    write: &DashMap<u64, f64>,
) -> (crate::build::Opt, f64) {
    use crate::build::{apply_build as ab, build_options_slice, Opt};

    if landmarks == WIN_LMS { return (None, 1.0); }
    let next_depth = depth.saturating_sub(1);

    let (opts, n) = build_options_slice(coins, &cards, landmarks);

    let mut best_opt: Opt = None;
    let mut best_val: f64 = -1.0;

    for &opt in &opts[..n] {
        let (nc, nk, nl) = ab(coins, &cards, landmarks, opt);
        let val = if nl == WIN_LMS {
            1.0
        } else if next_depth == 0 {
            0.0
        } else {
            analyze(AState { coins: nc, cards: nk, landmarks: nl }, next_depth, frozen, write)
        };
        if val > best_val {
            best_val = val;
            best_opt = opt;
        }
    }

    let best_val = if extra_turn {
        (best_val / (1.0 - P_DBL)).min(1.0)
    } else {
        best_val
    };

    (best_opt, best_val)
}

#[inline]
pub fn add_coins(base: u8, income: u8) -> u8 {
    (base as u16 + income as u16).min(MAX_COINS as u16) as u8
}
