use dashmap::DashMap;
use machi_koro_solver::build::apply_build;
use machi_koro_solver::coast::{expected_income, prefers_two_dice};
use machi_koro_solver::income::calc_income;
use machi_koro_solver::solver::add_coins;
use machi_koro_solver::state::{AState, CARD_KEYS, LANDMARK_KEYS};
use machi_koro_solver::strategy::{self, Strategy};
use rand::Rng;

const N_GAMES: usize = 5;

fn opt_label(opt: machi_koro_solver::build::Opt) -> String {
    match opt {
        None => "pass".to_string(),
        Some(i) if (i as usize) < 10 => format!("buy {}", CARD_KEYS[i as usize]),
        Some(i) => format!("buy {}", LANDMARK_KEYS[i as usize - 10]),
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
    format!("coins={}  ei={:.2}  [{}]  lms=[{}]",
        state.coins, expected_income(state), cards.join(", "), lms.join(", "))
}

fn run_traced_game(depth: usize, cache: &DashMap<u64, f64>, rng: &mut impl Rng) -> u32 {
    let mut state = AState::initial();
    let mut turns = 0u32;

    while !state.is_won() && turns < 200 {
        turns += 1;

        let (roll, is_dbl) = if state.has_train() && prefers_two_dice(&state) {
            let d1: u8 = rng.gen_range(1..=6);
            let d2: u8 = rng.gen_range(1..=6);
            (d1 + d2, d1 == d2)
        } else {
            (rng.gen_range(1..=6), false)
        };

        let income = calc_income(&state, roll);
        let new_coins = add_coins(state.coins, income);
        let mid = AState { coins: new_coins, cards: state.cards, landmarks: state.landmarks };
        let extra_turn = is_dbl && state.has_park();

        let opt = strategy::choose_build(mid, depth, extra_turn, Strategy::MaxIncome, cache);
        let (nc, nk, nl) = apply_build(new_coins, &state.cards, state.landmarks, opt);
        let next = AState { coins: nc, cards: nk, landmarks: nl };

        println!("  t{turns:>2} roll={roll:>2} income={income:>2}  {}  => {}",
            state_label(&mid), opt_label(opt));

        state = next;
    }
    turns
}

fn main() {
    for depth in [4, 5] {
        println!("\n{}", "=".repeat(60));
        println!("DEPTH {depth}");
        println!("{}", "=".repeat(60));

        let cache: DashMap<u64, f64> = DashMap::with_shard_amount(512);
        strategy::search(AState::initial(), depth, Strategy::MaxIncome, &cache);

        let mut rng = rand::thread_rng();
        // Use a fixed seed sequence so depth 4 and 5 see identical rolls
        for game in 1..=N_GAMES {
            println!("\n--- Game {game} (depth={depth}) ---");
            let turns = run_traced_game(depth, &cache, &mut rng);
            println!("  => finished in {turns} turns");
        }
    }
}
