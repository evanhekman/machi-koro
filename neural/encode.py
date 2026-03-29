#!/Users/evanhekman/machi-koro/.venv/bin/python
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from strategy.analysis import CARD_KEYS, LANDMARK_KEYS
from engine import CARDS, LANDMARKS, GameState

# Buy action index → card/landmark key or None (pass)
BUY_KEYS: list[str | None] = list(CARD_KEYS) + list(LANDMARK_KEYS) + [None]


def encode_state(state: GameState, pid: int) -> torch.Tensor:
    p = state.players[pid]
    roll = sum(state.last_dice) if state.last_dice else 0
    vec = (
        [state.supply.get(k, 0) for k in CARD_KEYS]  # 10: market supply
        + [p.cards.get(k, 0) for k in CARD_KEYS]  # 10: owned cards
        + [int(p.landmarks[k]) for k in LANDMARK_KEYS]  #  4: landmarks (0/1)
        + [p.coins, roll]  #  2: coins, last roll
    )
    return torch.tensor(vec, dtype=torch.float32)


def buy_mask(state: GameState, pid: int) -> torch.Tensor:
    p = state.players[pid]
    mask = torch.zeros(15, dtype=torch.bool)
    for i, k in enumerate(CARD_KEYS):
        if state.supply.get(k, 0) > 0 and p.coins >= CARDS[k].cost:
            mask[i] = True
    for i, k in enumerate(LANDMARK_KEYS):
        if not p.landmarks[k] and p.coins >= LANDMARKS[k]["cost"]:
            mask[10 + i] = True
    mask[14] = True  # pass is always valid
    return mask


def dice_mask(state: GameState, pid: int) -> torch.Tensor:
    mask = torch.ones(2, dtype=torch.bool)
    if not state.players[pid].landmarks["train_station"]:
        mask[1] = False
    return mask
