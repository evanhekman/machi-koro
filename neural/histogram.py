#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Visualize win-time distributions for greedy vs. top-3 weighted sampling.

Usage:
    python neural/histogram.py [--n N] [--model MODEL_DIR]

Defaults: 1000 games, neural/model_0
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


def run_game(net: MachiKoroNet, greedy: bool) -> int:
    state = create_game_solitaire()
    while state.phase != "end":
        x = encode_state(state, 0)
        if state.phase == "roll":
            logits = net.dice_logits(x).masked_fill(~dice_mask(state, 0), float("-inf"))
            action = int(logits.argmax()) if greedy else _top3_weighted(logits)
            action_roll(state, n_dice=action + 1)
        elif state.phase == "reroll":
            logits = net.reroll_logits(x)
            action = int(logits.argmax()) if greedy else _top3_weighted(logits)
            action_reroll(state, do_reroll=bool(action))
        elif state.phase == "build":
            logits = net.buy_logits(x).masked_fill(~buy_mask(state, 0), float("-inf"))
            action = int(logits.argmax()) if greedy else _top3_weighted(logits)
            action_buy(state, BUY_KEYS[action])
    return state.players[0].turns


def simulate(net: MachiKoroNet, n: int, greedy: bool) -> list[int]:
    label = "greedy" if greedy else "top-3 weighted"
    results = []
    for i in range(n):
        if i % 100 == 0:
            print(f"  {label}: {i}/{n}", end="\r")
        results.append(run_game(net, greedy=greedy))
    print(f"  {label}: {n}/{n} done")
    return results


def plot(greedy_turns: list[int], sampled_turns: list[int]) -> None:
    all_turns = greedy_turns + sampled_turns
    bins = range(min(all_turns), max(all_turns) + 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Neural Net Win-Time Distribution", fontsize=14)

    for ax, turns, label, color in [
        (axes[0], greedy_turns, "Greedy (argmax)", "steelblue"),
        (axes[1], sampled_turns, "Top-3 Weighted Sample", "coral"),
    ]:
        n = len(turns)
        avg = sum(turns) / n
        mn, mx = min(turns), max(turns)
        ax.hist(turns, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(avg, color="white", linestyle="--", linewidth=1.2, label=f"mean={avg:.1f}")
        ax.set_title(f"{label}\nmean={avg:.1f}  min={mn}  max={mx}")
        ax.set_xlabel("Turns to win")
        ax.set_ylabel("Games")
        ax.legend()

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of games per strategy")
    parser.add_argument("--model", default=os.path.join(_HERE, "model_0"), help="Model directory")
    parser.add_argument("--out", default="histogram.png", help="Output PNG path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    net = _load_net(args.model)

    print(f"Running {args.n} games per strategy...")
    with torch.no_grad():
        greedy_turns = simulate(net, args.n, greedy=True)
        sampled_turns = simulate(net, args.n, greedy=False)

    fig = plot(greedy_turns, sampled_turns)
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
