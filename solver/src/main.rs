use machi_koro_solver::{
    build::{apply_build, build_options_slice},
    cache,
    dice::DIST_1,
    income::calc_income,
    solver::analyze,
    state::{AState, CARD_KEYS, LANDMARK_KEYS, MAX_COINS, NUM_CARDS, WIN_LMS},
};
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let mut depth = 10usize;
    let mut workers = num_cpus();

    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--depth" if i + 1 < raw.len() => {
                depth = raw[i + 1].parse().expect("--depth requires an integer");
                i += 2;
            }
            s if s.starts_with("--depth=") => {
                depth = s[8..].parse().expect("--depth= requires an integer");
                i += 1;
            }
            "--workers" if i + 1 < raw.len() => {
                workers = raw[i + 1].parse().expect("--workers requires an integer");
                i += 2;
            }
            s if s.starts_with("--workers=") => {
                workers = s[10..].parse().expect("--workers= requires an integer");
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Usage: solver [--depth N] [--workers N]");
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("failed to build thread pool");

    let cache_dir = PathBuf::from("../cache");
    let (frozen, start_depth) = match cache::load_deltas(&cache_dir, depth) {
        Ok((f, d)) if d > 0 => {
            println!("Cache on disk covers depth {d}, loaded single delta ({} entries)", f.len());
            (f, d)
        }
        Ok(_) => (HashMap::new(), 0),
        Err(e) => {
            eprintln!("warning: could not load cache: {e}");
            (HashMap::new(), 0)
        }
    };

    if start_depth >= depth {
        println!("Cache already covers depth {depth}. Nothing to compute.");
        print_first_turn(depth, &frozen);
        return;
    }

    println!("Expectimax solver  depth={depth}  workers={workers}");
    println!("{}", "=".repeat(60));

    let t_total = std::time::Instant::now();
    let (results, frozen) = machi_koro_solver::solver::run(depth, &pool, start_depth, frozen, &cache_dir);

    let (_, last_val, _) = results.last().unwrap();
    println!();
    println!("Win probability in {depth} turns : {last_val:.8}");
    println!("Cache entries                    : {}", frozen.len());
    println!("Total time                       : {:.3}s", t_total.elapsed().as_secs_f64());
    println!();

    print_first_turn(depth, &frozen);
}

fn print_first_turn(depth: usize, frozen: &HashMap<u64, f64>) {
    let initial = AState::initial();
    let write: DashMap<u64, f64> = DashMap::new();

    println!("Recommended first turn (initial: 3 coins, wheat+bakery):");
    println!("{:>5}  {:>6}  {:>5}  {:<22}  P(win|buy)", "Roll", "Income", "Coins", "Best buy");
    println!("{}", "-".repeat(65));

    for o in DIST_1.iter() {
        let income = calc_income(&initial, o.roll);
        let new_coins = (initial.coins as u16 + income as u16).min(MAX_COINS as u16) as u8;

        let (opts, n) = build_options_slice(new_coins, &initial.cards, initial.landmarks);

        let mut best_opt = None;
        let mut best_val: f64 = -1.0;

        for &opt in &opts[..n] {
            let (nc, nk, nl) = apply_build(new_coins, &initial.cards, initial.landmarks, opt);
            let val = if nl == WIN_LMS {
                1.0
            } else if depth <= 1 {
                0.0
            } else {
                analyze(
                    AState { coins: nc, cards: nk, landmarks: nl },
                    depth - 1,
                    frozen,
                    &write,
                )
            };
            if val > best_val {
                best_val = val;
                best_opt = opt;
            }
        }

        let label: &str = match best_opt {
            None => "skip",
            Some(idx) if (idx as usize) < NUM_CARDS => CARD_KEYS[idx as usize],
            Some(idx) => LANDMARK_KEYS[idx as usize - NUM_CARDS],
        };

        println!(
            "  {:>3}  {:>6}  {:>5}  {:<22}  {:.8}",
            o.roll, income, new_coins, label, best_val
        );
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
