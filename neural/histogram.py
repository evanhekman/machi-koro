#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Visualize win-time distribution using top-3 weighted sampling.

Usage:
    python neural/histogram.py [--n N] [--model MODEL_DIR]

Defaults: 1000 games, neural/model_1
"""
from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from engine import create_game_solitaire, action_roll, action_reroll, action_buy
from model import MachiKoroNet
from encode import encode_state, buy_mask, dice_mask, BUY_KEYS


def _load_net(model_dir: str) -> MachiKoroNet:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    net = MachiKoroNet(cfg)
    ckpt = torch.load(os.path.join(model_dir, "latest.pt"), weights_only=True)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def _top3_weighted(logits: torch.Tensor) -> int:
    """Sample from the top-3 actions by probability weight."""
    probs = torch.softmax(logits, dim=-1)
    top_vals, top_idx = torch.topk(probs, min(3, probs.shape[-1]))
    sampled = torch.multinomial(top_vals, 1)
    return int(top_idx[sampled])


def run_game(net: MachiKoroNet) -> int:
    state = create_game_solitaire()
    while state.phase != "end":
        x = encode_state(state, 0)
        if state.phase == "roll":
            logits = net.dice_logits(x).masked_fill(~dice_mask(state, 0), float("-inf"))
            action_roll(state, n_dice=_top3_weighted(logits) + 1)
        elif state.phase == "reroll":
            logits = net.reroll_logits(x)
            action_reroll(state, do_reroll=bool(_top3_weighted(logits)))
        elif state.phase == "build":
            logits = net.buy_logits(x).masked_fill(~buy_mask(state, 0), float("-inf"))
            action_buy(state, BUY_KEYS[_top3_weighted(logits)])
    return state.players[0].turns


def simulate(net: MachiKoroNet, n: int) -> list[int]:
    results = []
    for i in range(n):
        if i % 100 == 0:
            print(f"  top-3 weighted: {i}/{n}", end="\r")
        results.append(run_game(net))
    print(f"  top-3 weighted: {n}/{n} done")
    return results


def plot(turns: list[int], model_name: str) -> None:
    bins = range(min(turns), max(turns) + 2)
    n = len(turns)
    avg = sum(turns) / n
    mn, mx = min(turns), max(turns)
    p90 = sorted(turns)[int(0.9 * n)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(turns, bins=bins, color="coral", edgecolor="white", linewidth=0.4)
    ax.axvline(avg, color="white", linestyle="--", linewidth=1.2, label=f"mean={avg:.1f}")
    ax.set_title(f"{model_name} — Top-3 Weighted Sampling\nmean={avg:.1f}  p90={p90}  min={mn}  max={mx}")
    ax.set_xlabel("Turns to win")
    ax.set_ylabel("Games")
    ax.legend()
    fig.suptitle("Neural Net Win-Time Distribution", fontsize=14)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of games per strategy")
    parser.add_argument("--model", default=os.path.join(_HERE, "model_1"), help="Model directory")
    parser.add_argument("--out", default="histogram.png", help="Output PNG path")
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    print(f"Loading model from {args.model}")
    net = _load_net(args.model)

    print(f"Running {args.n} games (top-3 weighted)...")
    with torch.no_grad():
        turns = simulate(net, args.n)

    fig = plot(turns, model_name)
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

    data_out = args.out.rsplit(".", 1)[0] + ".json"
    with open(data_out, "w") as f:
        json.dump({"model": model_name, "n": len(turns), "turns": turns}, f)
    print(f"Data saved to {data_out}")


if __name__ == "__main__":
    main()
