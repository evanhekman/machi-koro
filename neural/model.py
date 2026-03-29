#!/Users/evanhekman/machi-koro/.venv/bin/python
from __future__ import annotations
import torch.nn as nn

STATE_DIM = 26  # 10 supply + 10 owned + 4 landmarks + coins + roll
DICE_ACTIONS = 2  # 0 = one die, 1 = two dice
REROLL_ACTIONS = 2  # 0 = keep,    1 = reroll
BUY_ACTIONS = 15  # 0-9 cards, 10-13 landmarks, 14 = pass


def _trunk(in_dim: int, layers: list[int]) -> tuple[nn.Module, int]:
    if not layers:
        return nn.Identity(), in_dim
    sizes = [in_dim] + layers
    mods = []
    for i in range(len(sizes) - 1):
        mods += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*mods), layers[-1]


def _head(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    sizes = [in_dim] + hidden + [out_dim]
    mods = []
    for i in range(len(sizes) - 1):
        mods.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class MachiKoroNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.trunk_net, trunk_out = _trunk(STATE_DIM, cfg.get("trunk", [64, 64]))
        self.dice_head = _head(trunk_out, cfg.get("dice_head", []), DICE_ACTIONS)
        self.reroll_head = _head(trunk_out, cfg.get("reroll_head", []), REROLL_ACTIONS)
        self.buy_head = _head(trunk_out, cfg.get("buy_head", [32]), BUY_ACTIONS)

    def _h(self, x):
        return self.trunk_net(x)

    def dice_logits(self, x):
        return self.dice_head(self._h(x))

    def reroll_logits(self, x):
        return self.reroll_head(self._h(x))

    def buy_logits(self, x):
        return self.buy_head(self._h(x))
