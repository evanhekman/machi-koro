#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Train a MachiKoroNet using REINFORCE.

Usage:
    python neural/run.py <experiment_dir>

The experiment directory must contain config.json. All outputs
(log.jsonl, latest.pt, optional per-epoch checkpoints) are written there.
"""
from __future__ import annotations
import gc, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import matplotlib.pyplot as plt
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


def _moving_avg(data: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(data)):
        chunk = data[max(0, i - window + 1): i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def train(cfg: dict, out_dir: str) -> None:
    net       = MachiKoroNet(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 3e-4))

    baseline        = 0.0
    baseline_decay  = cfg.get("baseline_decay", 0.99)
    n_epochs        = cfg.get("n_epochs", 500)
    games_per_epoch = cfg.get("games_per_epoch", 100)
    save_every      = cfg.get("save_every", 50)

    log_path    = os.path.join(out_dir, "log.jsonl")
    latest_path = os.path.join(out_dir, "latest.pt")

    # Resume from checkpoint if one exists
    start_epoch = 1
    epoch_avgs: list[float] = []
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, weights_only=True)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        baseline    = ckpt["baseline"]
        start_epoch = ckpt["epoch"] + 1
        epoch_avgs  = ckpt.get("epoch_avgs", [])
        print(f"Resumed from epoch {start_epoch - 1}")

    # Live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg turns to win  (lower is better)")
    ax.set_title("Machi Koro — Neural Training")
    line_raw,    = ax.plot([], [], alpha=0.3, color="steelblue")
    line_smooth, = ax.plot([], [], color="steelblue", linewidth=2, label="20-epoch avg")
    ax.legend()
    plt.tight_layout()
    plt.pause(0.001)

    def _update_plot() -> None:
        xs = list(range(1, len(epoch_avgs) + 1))
        line_raw.set_data(xs, epoch_avgs)
        line_smooth.set_data(xs, _moving_avg(epoch_avgs, 20))
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    for epoch in range(start_epoch, n_epochs + 1):
        gc.collect()
        epoch_turns: list[int] = []

        for _ in range(games_per_epoch):
            log_probs, turns = run_episode(net)
            reward    = float(-turns)
            baseline  = baseline_decay * baseline + (1 - baseline_decay) * reward
            advantage = reward - baseline

            loss = -torch.stack(log_probs).sum() * advantage
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del log_probs, loss  # free computation graph immediately

            epoch_turns.append(turns)

        avg = sum(epoch_turns) / len(epoch_turns)
        epoch_avgs.append(avg)

        print(f"epoch {epoch:4d}  avg={avg:.1f}  "
              f"min={min(epoch_turns)}  max={max(epoch_turns)}")

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "avg_turns": round(avg, 2),
                "min_turns": min(epoch_turns),
                "max_turns": max(epoch_turns),
            }) + "\n")

        if epoch % 5 == 0 or epoch == n_epochs:
            _update_plot()

        smooth = _moving_avg(epoch_avgs, 20)
        converged = len(smooth) > 20 and abs(smooth[-1] - smooth[-21]) < 1

        if converged:
            print(f"Converged at epoch {epoch} (20-epoch avg stable within 1 turn)")

        if epoch % save_every == 0 or epoch == n_epochs or converged:
            torch.save({
                "epoch": epoch,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "baseline": baseline,
                "epoch_avgs": epoch_avgs,
            }, latest_path)

        if converged:
            break

    plt.ioff()
    plt.show()
    print("Done.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python neural/run.py <experiment_dir>")
        sys.exit(1)

    out_dir  = sys.argv[1]
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
