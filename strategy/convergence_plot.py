#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Convergence plot: mean turns to win vs lookahead depth, with neural net asymptote.

Usage:
    # Generate data first:
    ./solver/target/release/simulate --depths 3 4 5 6 7 8 9 10 --n 200 --out /tmp/lookahead.json
    python neural/histogram.py --n 500 --out /tmp/neural.png  # also writes /tmp/neural.json

    # Then plot:
    python strategy/convergence_plot.py --lookahead /tmp/lookahead.json --neural /tmp/neural.json --out convergence.png

    # Or supply the neural mean directly:
    python strategy/convergence_plot.py --lookahead /tmp/lookahead.json --neural-mean 18.3 --out convergence.png
"""

from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_lookahead(path: str) -> dict[int, list[int]]:
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


def plot(lookahead: dict[int, list[int]], neural_mean: float, out: str) -> None:
    depths = sorted(lookahead)
    means = [sum(lookahead[d]) / len(lookahead[d]) for d in depths]
    errs = [stderr(lookahead[d]) for d in depths]
    n_games = len(lookahead[depths[0]])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(depths, means, yerr=errs, fmt="o-", color="#4c9be8",
                capsize=4, linewidth=2, markersize=6, label="Lookahead (mean ± SE)")

    ax.axhline(neural_mean, color="coral", linestyle="--", linewidth=1.5,
               label=f"Neural net mean = {neural_mean:.1f}")

    ax.set_xlabel("Lookahead depth", fontsize=12)
    ax.set_ylabel("Mean turns to win", fontsize=12)
    ax.set_title(f"Lookahead convergence  (n={n_games} games per depth)", fontsize=13)
    ax.set_xticks(depths)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookahead", required=True,
                        help="JSON from the Rust simulator (depth -> [turns])")
    parser.add_argument("--neural", default=None,
                        help="JSON from neural/histogram.py ({turns: [...]})")
    parser.add_argument("--neural-mean", type=float, default=None,
                        help="Neural net mean turns (alternative to --neural)")
    parser.add_argument("--out", default="convergence.png")
    args = parser.parse_args()

    if args.neural_mean is not None:
        neural_mean = args.neural_mean
    elif args.neural is not None:
        neural_mean = load_neural_mean(args.neural)
        print(f"Neural net mean: {neural_mean:.2f} turns")
    else:
        print("Error: provide --neural or --neural-mean", file=sys.stderr)
        sys.exit(1)

    lookahead = load_lookahead(args.lookahead)
    print(f"Depths: {sorted(lookahead)}  games each: {len(next(iter(lookahead.values())))}")

    plot(lookahead, neural_mean, args.out)


if __name__ == "__main__":
    main()
