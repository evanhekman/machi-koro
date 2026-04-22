use machi_koro_solver::simulate::{run_all, SimStrategy};
use std::path::PathBuf;

fn main() {
    let mut strategy = SimStrategy::MinTurns;
    let mut depths: Vec<usize> = vec![1, 3, 5, 7];
    let mut n_games: usize = 500;
    let mut out = PathBuf::from("results.json");

    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--winprob"   => { strategy = SimStrategy::WinProb;   i += 1; }
            "--minturns"  => { strategy = SimStrategy::MinTurns;  i += 1; }
            "--coast"     => { strategy = SimStrategy::Coast;     i += 1; }
            "--maxincome" => { strategy = SimStrategy::MaxIncome; i += 1; }
            "--depths" => {
                depths.clear();
                i += 1;
                while i < raw.len() && !raw[i].starts_with('-') {
                    depths.push(raw[i].parse().expect("--depths: expected integers"));
                    i += 1;
                }
            }
            "--n" if i + 1 < raw.len() => {
                n_games = raw[i + 1].parse().expect("--n requires an integer");
                i += 2;
            }
            s if s.starts_with("--n=") => {
                n_games = s[4..].parse().expect("--n= requires an integer");
                i += 1;
            }
            "--out" if i + 1 < raw.len() => {
                out = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            s if s.starts_with("--out=") => {
                out = PathBuf::from(&s[6..]);
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Usage: simulate [--winprob|--minturns|--coast|--maxincome] [--depths D...] [--n N] [--out FILE]");
                eprintln!("  --winprob    maximise P(win within depth turns)");
                eprintln!("  --minturns   minimise expected turns to win (default)");
                eprintln!("  --coast      minimise coast-time heuristic");
                eprintln!("  --maxincome  maximise expected income per turn");
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }

    if depths.is_empty() {
        eprintln!("error: no depths specified");
        std::process::exit(1);
    }

    println!(
        "Simulation  strategy={:?}  depths={:?}  games={}",
        strategy, depths, n_games
    );
    println!("{}", "=".repeat(60));

    let results = run_all(strategy, &depths, n_games);

    let mut json = String::from("{\n");
    for (idx, r) in results.iter().enumerate() {
        json.push_str(&format!("  \"{}\": [", r.depth));
        for (j, &t) in r.turns.iter().enumerate() {
            if j > 0 { json.push(','); }
            json.push_str(&t.to_string());
        }
        json.push(']');
        if idx + 1 < results.len() { json.push(','); }
        json.push('\n');
    }
    json.push('}');

    std::fs::write(&out, &json)
        .unwrap_or_else(|e| eprintln!("warning: could not write output: {e}"));
    println!("Results written to {}", out.display());
}
