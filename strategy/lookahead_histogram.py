#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Win-time distribution for lookahead strategies at varying depths.

Usage:
    python strategy/lookahead_histogram.py --depths 1 3 5 7 --n 500
    python strategy/lookahead_histogram.py --depths 2 4 --n 1000 --out my_plot.png

Columns = depths; each subplot shows the turn distribution for that lookahead.
"""

from __future__ import annotations
import argparse, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine as E
from strategy.lookahead import make_lookahead_strategy


MAX_TURNS = 500


def run_games(n: int, depth: int) -> list[int]:
    import time
    strategy = make_lookahead_strategy(depth)
    turns_list: list[int] = []
    t0 = time.time()
    for i in range(n):
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n - i) / rate if rate > 0 else float("inf")
        print(f"  depth={depth}: {i}/{n}  {rate:.1f} games/s  eta {eta:.0f}s   ", end="\r", flush=True)
        state = E.create_game_solitaire()
        while state.phase != "end" and state.players[0].turns < MAX_TURNS:
            action = strategy(state, 0)
            _apply(state, action)
        turns_list.append(state.players[0].turns)
    print(f"  depth={depth}: {n}/{n} done in {time.time()-t0:.1f}s")
    return turns_list


def _apply(state: E.GameState, action: dict) -> None:
    t = action["type"]
    if t == "roll":
        E.action_roll(state, action.get("n_dice", 1))
    elif t == "reroll":
        E.action_reroll(state, action["do_reroll"])
    elif t == "choose_purple":
        E.action_choose_purple(state, action["card"])
    elif t == "tv_station":
        E.action_tv_station(state, action["target"])
    elif t == "business_center":
        E.action_business_center(
            state, action["target"], action["give_card"], action["take_card"]
        )
    elif t == "buy":
        E.action_buy(state, action.get("card"))


def plot(results: dict[int, list[int]], out: str) -> None:
    depths = sorted(results)
    n_plots = len(depths)
    all_turns = [t for turns in results.values() for t in turns]
    bins = range(min(all_turns), max(all_turns) + 2)

    cols = min(n_plots, 4)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharey=True, sharex=True)
    axes_flat = [axes] if n_plots == 1 else (axes.flat if rows > 1 else axes)

    colors = plt.cm.viridis([i / max(len(depths) - 1, 1) for i in range(len(depths))])

    for ax, depth, color in zip(axes_flat, depths, colors):
        turns = results[depth]
        avg = sum(turns) / len(turns)
        mn, mx = min(turns), max(turns)
        p90 = sorted(turns)[int(0.9 * len(turns))]
        ax.hist(turns, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(avg, color="white", linestyle="--", linewidth=1.2)
        ax.set_title(f"lookahead n={depth}\nmean={avg:.1f}  p90={p90}  min={mn}  max={mx}")
        ax.set_xlabel("Turns to win")
        ax.set_ylabel("Games")

    # Hide unused subplots
    for ax in list(axes_flat)[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("Lookahead Strategy — Win-Time Distribution", fontsize=13)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5, 7],
                        help="Lookahead depths to compare (ignored when --json is set)")
    parser.add_argument("--n", type=int, default=500, help="Games per depth (ignored when --json is set)")
    parser.add_argument("--json", default=None, help="Path to JSON file produced by the Rust simulator")
    parser.add_argument("--out", default="lookahead_histogram.png")
    args = parser.parse_args()

    if args.json:
        import json
        with open(args.json) as f:
            raw = json.load(f)
        results: dict[int, list[int]] = {int(k): v for k, v in raw.items()}
    else:
        print(f"Running {args.n} games × {len(args.depths)} depths: {args.depths}")
        results = {}
        for d in sorted(args.depths):
            results[d] = run_games(args.n, d)

    plot(results, args.out)


if __name__ == "__main__":
    main()
