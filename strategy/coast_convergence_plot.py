#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Convergence plot for any strategy: mean turns to win vs depth, with optional asymptote.

Usage:
    ./solver/target/release/simulate --coast --depths 1 2 3 4 5 6 7 --n 1000 --out /tmp/coast.json
    python strategy/coast_convergence_plot.py --data /tmp/coast.json --strategy coast --neural-mean 18.3 --out /tmp/coast_convergence.png

    ./solver/target/release/simulate --maxincome --depths 1 2 3 4 5 6 7 --n 1000 --out /tmp/maxincome.json
    python strategy/coast_convergence_plot.py --data /tmp/maxincome.json --strategy maxincome --out /tmp/maxincome_convergence.png

Legacy alias: --coast FILE is equivalent to --data FILE --strategy coast
"""

from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path: str) -> dict[int, list[int]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_neural_mean(path: str) -> float:
    with open(path) as f:
        data = json.load(f)
    turns = data["turns"]
    return sum(turns) / len(turns)


def stderr(turns: list[int]) -> float:
    n = len(turns)
    mean = sum(turns) / n
    var = sum((t - mean) ** 2 for t in turns) / (n - 1)
    return (var / n) ** 0.5


def plot(data: dict[int, list[int]], strategy: str, neural_mean: float | None, out: str) -> None:
    depths = sorted(data)
    means = [sum(data[d]) / len(data[d]) for d in depths]
    n_games = len(data[depths[0]])

    title = f"{strategy} strategy convergence  (n={n_games} games per depth)"

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(depths, means, "o-", color="#5b9bd5",
            linewidth=2, markersize=6, label=f"{strategy} strategy (mean)")

    if neural_mean is not None:
        ax.axhline(neural_mean, color="coral", linestyle="--", linewidth=1.5,
                   label=f"Neural net mean = {neural_mean:.1f}")

    ax.set_xlabel("Depth", fontsize=12)
    ax.set_ylabel("Mean turns to win", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(depths)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="JSON from simulate (depth -> [turns])")
    parser.add_argument("--coast", default=None,
                        help="Alias for --data with --strategy coast")
    parser.add_argument("--strategy", default=None,
                        help="Strategy name for title/colour (coast, maxincome, minturns, winprob)")
    parser.add_argument("--neural", default=None,
                        help="JSON from neural/histogram.py ({turns: [...]})")
    parser.add_argument("--neural-mean", type=float, default=None,
                        help="Neural net mean turns (optional asymptote line)")
    parser.add_argument("--out", default="convergence.png")
    args = parser.parse_args()

    # Resolve --coast legacy alias
    data_path = args.data or args.coast
    strategy = args.strategy or ("coast" if args.coast else None)
    if not data_path:
        print("Error: provide --data FILE (or legacy --coast FILE)", file=sys.stderr)
        sys.exit(1)
    if not strategy:
        print("Error: provide --strategy NAME", file=sys.stderr)
        sys.exit(1)

    neural_mean = None
    if args.neural_mean is not None:
        neural_mean = args.neural_mean
    elif args.neural is not None:
        neural_mean = load_neural_mean(args.neural)
        print(f"Neural net mean: {neural_mean:.2f} turns")

    data = load_data(data_path)
    print(f"Depths: {sorted(data)}  games each: {len(next(iter(data.values())))}")

    plot(data, strategy, neural_mean, args.out)


if __name__ == "__main__":
    main()
