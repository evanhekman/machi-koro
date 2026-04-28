/// Exhaustively finds all win paths reachable within N turns.
/// Enumerates every (roll, purchase) sequence — no probability weighting,
/// no caching, no approximations. Prints the exact state at each step.
///
/// Pruning: if remaining landmarks > remaining turns, we can't possibly buy
/// one per turn, so cut the branch. (Ignores amusement park extra turns.)
use machi_koro_solver::build::{apply_build, build_options_slice};
use machi_koro_solver::income::calc_income;
use machi_koro_solver::state::{AState, NUM_CARDS, CARD_KEYS, LANDMARK_KEYS};

const MAX_DEPTH: usize = 7;
const MAX_WINS: usize = 3;

struct Finder {
    wins_found: usize,
}

#[derive(Clone)]
struct Step {
    roll: u8,
    buy: String,
    state: AState,
}

fn landmarks_remaining(lms: u8) -> usize {
    (0..4).filter(|&i| (lms & (1 << i)) == 0).count()
}

impl Finder {
    fn search(&mut self, state: AState, depth: usize, path: &mut Vec<Step>) {
        if self.wins_found >= MAX_WINS { return; }
        if state.is_won() {
            self.wins_found += 1;
            println!("=== WIN PATH {} ({} turns) ===", self.wins_found, path.len());
            for (i, step) in path.iter().enumerate() {
                println!(
                    "  turn {:2}: roll={:2}  buy={:<22}  coins={:2}  lms={:04b}",
                    i + 1, step.roll, step.buy, step.state.coins, step.state.landmarks
                );
            }
            println!();
            return;
        }
        if depth == 0 { return; }

        // Prune: can't buy one landmark per remaining turn
        if landmarks_remaining(state.landmarks) > depth { return; }

        let dice_range: &[u8] = if state.has_train() {
            &[2,3,4,5,6,7,8,9,10,11,12]
        } else {
            &[1,2,3,4,5,6]
        };

        for &roll in dice_range {
            let income = calc_income(&state, roll);
            let coins_after = (state.coins as u16 + income as u16).min(52) as u8;
            let mid = AState { coins: coins_after, cards: state.cards, landmarks: state.landmarks };

            let (opts, n) = build_options_slice(mid.coins, &mid.cards, mid.landmarks);
            for &opt in &opts[..n] {
                let (nc, nk, nl) = apply_build(mid.coins, &mid.cards, mid.landmarks, opt);
                let next = AState { coins: nc, cards: nk, landmarks: nl };
                path.push(Step { roll, buy: opt_name(opt), state: next });
                self.search(next, depth - 1, path);
                path.pop();
                if self.wins_found >= MAX_WINS { return; }
            }
        }
    }
}

fn opt_name(opt: machi_koro_solver::build::Opt) -> String {
    match opt {
        None => "skip".to_string(),
        Some(i) if (i as usize) < NUM_CARDS => CARD_KEYS[i as usize].to_string(),
        Some(i) => LANDMARK_KEYS[i as usize - NUM_CARDS].to_string(),
    }
}

fn main() {
    let depth: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_DEPTH);

    println!("Searching for win paths within {} turns...\n", depth);
    let mut finder = Finder { wins_found: 0 };
    let mut path = Vec::new();
    finder.search(AState::initial(), depth, &mut path);

    if finder.wins_found == 0 {
        println!("No win paths found within {} turns.", depth);
    }
}
