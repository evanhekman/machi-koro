/// Trace a single game turn-by-turn to see exactly what the WinProb
/// strategy buys at a given depth, including when blind (P=0).
use dashmap::DashMap;
use machi_koro_solver::build::apply_build;
use machi_koro_solver::income::calc_income;
use machi_koro_solver::solver::add_coins;
use machi_koro_solver::state::{AState, NUM_CARDS, CARD_KEYS, LANDMARK_KEYS};
use machi_koro_solver::strategy::{self, Strategy};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn opt_label(opt: machi_koro_solver::build::Opt) -> String {
    match opt {
        None => "skip".to_string(),
        Some(i) if (i as usize) < NUM_CARDS => CARD_KEYS[i as usize].to_string(),
        Some(i) => LANDMARK_KEYS[i as usize - NUM_CARDS].to_string(),
    }
}

fn state_label(state: &AState) -> String {
    let cards: Vec<String> = CARD_KEYS.iter().enumerate()
        .filter(|&(i, _)| state.cards[i] > 0)
        .map(|(i, k)| format!("{}×{}", state.cards[i], k))
        .collect();
    let lms: Vec<&str> = LANDMARK_KEYS.iter().enumerate()
        .filter(|&(i, _)| state.landmarks & (1 << i) != 0)
        .map(|(_, k)| *k)
        .collect();
    format!("[{}] lms=[{}]", cards.join(", "), lms.join(", "))
}

fn run_game(depth: usize, cache: &DashMap<u64, f64>, rng: &mut StdRng) -> u32 {
    let mut state = AState::initial();
    let mut turns = 0u32;

    while !state.is_won() && turns < 200 {
        turns += 1;

        let roll: u8 = if state.has_train() {
            rng.gen_range(1u8..=6) + rng.gen_range(1u8..=6)
        } else {
            rng.gen_range(1u8..=6)
        };

        let income = calc_income(&state, roll);
        let new_coins = add_coins(state.coins, income);
        let mid = AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks };

        let opt = strategy::choose_build(mid, depth, false, Strategy::WinProb, cache);
        let (nc, nk, nl) = apply_build(new_coins, &state.cards, state.landmarks, opt);
        let next = AState { coins: nc, cards: nk, landmarks: nl };
        let val = strategy::search(next, depth, Strategy::WinProb, cache);

        println!(
            "  t{turns:>3}  roll={roll:>2}  income={income:>2}  coins={new_coins:>2}->{nc}  buy={:<22}  P(win)={val:.6}  {sl}",
            opt_label(opt), sl=state_label(&next)
        );

        state = next;
    }
    turns
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depth: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let seed: u64   = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("WinProb trace  depth={depth}  seed={seed}\n");
    let cache: DashMap<u64, f64> = DashMap::with_shard_amount(512);
    strategy::search(AState::initial(), depth, Strategy::WinProb, &cache);
    println!("cache warmed ({} entries)\n", cache.len());

    let mut rng = StdRng::seed_from_u64(seed);
    let turns = run_game(depth, &cache, &mut rng);
    println!("\nFinished in {turns} turns.");
}
