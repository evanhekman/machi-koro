#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Train a MachiKoroNet using REINFORCE.

Usage:
    python neural/run.py <experiment_dir>

The experiment directory must contain config.json. All outputs
(log.jsonl, latest.pt, optional per-episode checkpoints) are written there.
"""

from __future__ import annotations
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import torch
from torch.distributions import Categorical

from engine import create_game_solitaire, action_roll, action_reroll, action_buy
from model import MachiKoroNet
from encode import encode_state, buy_mask, dice_mask, BUY_KEYS


def run_episode(net: MachiKoroNet, greedy: bool = False):
    state = create_game_solitaire()
    log_probs: list[torch.Tensor] = []

    while state.phase != "end":
        x = encode_state(state, 0)

        if state.phase == "roll":
            logits = net.dice_logits(x).masked_fill(~dice_mask(state, 0), float("-inf"))
            if greedy:
                action = int(logits.argmax())
            else:
                dist = Categorical(logits=logits)
                t = dist.sample()
                log_probs.append(dist.log_prob(t))
                action = int(t)
            action_roll(state, n_dice=action + 1)

        elif state.phase == "reroll":
            logits = net.reroll_logits(x)
            if greedy:
                action = int(logits.argmax())
            else:
                dist = Categorical(logits=logits)
                t = dist.sample()
                log_probs.append(dist.log_prob(t))
                action = int(t)
            action_reroll(state, do_reroll=bool(action))

        elif state.phase == "build":
            logits = net.buy_logits(x).masked_fill(~buy_mask(state, 0), float("-inf"))
            if greedy:
                action = int(logits.argmax())
            else:
                dist = Categorical(logits=logits)
                t = dist.sample()
                log_probs.append(dist.log_prob(t))
                action = int(t)
            action_buy(state, BUY_KEYS[action])

    return log_probs, state.players[0].turns


def evaluate(net: MachiKoroNet, n: int) -> dict:
    turns_list = [run_episode(net, greedy=True)[1] for _ in range(n)]
    return {
        "avg_turns": round(sum(turns_list) / n, 2),
        "min_turns": min(turns_list),
        "max_turns": max(turns_list),
    }


def train(cfg: dict, out_dir: str) -> None:
    net = MachiKoroNet(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 3e-4))

    baseline = 0.0
    baseline_decay = cfg.get("baseline_decay", 0.99)
    n_episodes = cfg.get("n_episodes", 100_000)
    eval_every = cfg.get("eval_every", 1_000)
    eval_n = cfg.get("eval_n", 200)
    checkpoint_every = cfg.get("checkpoint_every", 0)  # 0 = only keep latest

    log_path = os.path.join(out_dir, "log.jsonl")
    latest = os.path.join(out_dir, "latest.pt")

    # Resume from checkpoint if one exists
    start_ep = 1
    if os.path.exists(latest):
        ckpt = torch.load(latest, weights_only=True)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        baseline = ckpt["baseline"]
        start_ep = ckpt["episode"] + 1
        print(f"Resumed from episode {start_ep - 1}")

    for ep in range(start_ep, n_episodes + 1):
        log_probs, turns = run_episode(net)
        reward = float(-turns)
        baseline = baseline_decay * baseline + (1 - baseline_decay) * reward
        advantage = reward - baseline

        loss = -torch.stack(log_probs).sum() * advantage
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % eval_every == 0:
            stats = evaluate(net, eval_n)
            entry = {"episode": ep, **stats}
            print(
                f"ep {ep:7d}  avg={stats['avg_turns']:.1f}  "
                f"min={stats['min_turns']}  max={stats['max_turns']}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            ckpt = {
                "episode": ep,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "baseline": baseline,
            }
            torch.save(ckpt, latest)
            if checkpoint_every and ep % checkpoint_every == 0:
                torch.save(ckpt, os.path.join(out_dir, f"ep_{ep:07d}.pt"))

    print("Done.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python neural/run.py <experiment_dir>")
        sys.exit(1)

    out_dir = sys.argv[1]
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)

    print(f"Config: {cfg}")
    train(cfg, out_dir)


if __name__ == "__main__":
    main()
