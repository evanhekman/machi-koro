from __future__ import annotations
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from engine import GameState

import engine as E

# A Strategy is a callable: (state: GameState, player_id: int) -> action dict
# Action dicts:
#   {"type": "roll", "n_dice": 1|2}
#   {"type": "reroll", "do_reroll": bool}
#   {"type": "tv_station", "target": int}
#   {"type": "business_center", "target": int, "give_card": str, "take_card": str}
#   {"type": "buy", "card": str | None}


def _default_roll(state: GameState, pid: int) -> dict:
    n = 2 if state.players[pid].landmarks["train_station"] else 1
    return {"type": "roll", "n_dice": n}


def _take_from_richest(state: GameState, pid: int) -> dict:
    target = max(
        (i for i in range(state.n_players) if i != pid),
        key=lambda i: state.players[i].coins,
    )
    return {"type": "tv_station", "target": target}


def _business_center_greedy(state: GameState, pid: int) -> dict:
    """Give our cheapest non-purple card, take their most expensive non-purple card."""
    ap = state.players[pid]
    our = [
        (c, E.CARDS[c].cost)
        for c, n in ap.cards.items()
        if n > 0 and c in E.CARDS and E.CARDS[c].color != E.Color.PURPLE
    ]
    if not our:
        # Fallback: nothing tradeable, just pick anything (shouldn't happen)
        give = next(iter(ap.cards))
        return {
            "type": "business_center",
            "target": (pid + 1) % state.n_players,
            "give_card": give,
            "take_card": give,
        }
    give = min(our, key=lambda x: x[1])[0]
    best = (-1, None, None)  # (cost, target, card)
    for tid in range(state.n_players):
        if tid == pid:
            continue
        for c, n in state.players[tid].cards.items():
            if n > 0 and c in E.CARDS and E.CARDS[c].color != E.Color.PURPLE:
                if E.CARDS[c].cost > best[0]:
                    best = (E.CARDS[c].cost, tid, c)
    if best[1] is None:
        return {
            "type": "business_center",
            "target": (pid + 1) % state.n_players,
            "give_card": give,
            "take_card": give,
        }
    return {
        "type": "business_center",
        "target": best[1],
        "give_card": give,
        "take_card": best[2],
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def strategy_buy_cheapest(state: GameState, pid: int) -> dict:
    """Roll max dice, never reroll, buy cheapest available card (landmarks last)."""
    phase = state.phase
    if phase == "roll":
        return _default_roll(state, pid)
    if phase == "reroll":
        return {"type": "reroll", "do_reroll": False}
    if phase == "choose_purple":
        return {"type": "choose_purple", "card": state.pending_purple[0]}
    if phase == "tv_station":
        return _take_from_richest(state, pid)
    if phase == "business_center":
        return _business_center_greedy(state, pid)
    if phase == "build":
        builds = E.available_builds(state)
        if not builds:
            return {"type": "buy", "card": None}
        # Prefer landmarks, else cheapest card
        lms = [b for b in builds if b in E.LANDMARKS]
        if lms:
            return {
                "type": "buy",
                "card": min(lms, key=lambda x: E.LANDMARKS[x]["cost"]),
            }
        card = min(
            (b for b in builds if b in E.CARDS),
            key=lambda x: E.CARDS[x].cost,
            default=None,
        )
        return {"type": "buy", "card": card}
    return {"type": "buy", "card": None}


def strategy_rush_landmarks(state: GameState, pid: int) -> dict:
    """Save coins and only buy landmarks, skip regular cards."""
    phase = state.phase
    if phase == "roll":
        return _default_roll(state, pid)
    if phase == "reroll":
        return {"type": "reroll", "do_reroll": False}
    if phase == "choose_purple":
        return {"type": "choose_purple", "card": state.pending_purple[0]}
    if phase == "tv_station":
        return _take_from_richest(state, pid)
    if phase == "business_center":
        return _business_center_greedy(state, pid)
    if phase == "build":
        builds = E.available_builds(state)
        lms = [b for b in builds if b in E.LANDMARKS]
        if lms:
            return {
                "type": "buy",
                "card": min(lms, key=lambda x: E.LANDMARKS[x]["cost"]),
            }
        return {"type": "buy", "card": None}
    return {"type": "buy", "card": None}


def strategy_random(state: GameState, pid: int) -> dict:
    """Roll random dice count, random reroll decision, buy random affordable card."""
    phase = state.phase
    if phase == "roll":
        has_ts = state.players[pid].landmarks["train_station"]
        return {"type": "roll", "n_dice": random.choice([1, 2]) if has_ts else 1}
    if phase == "reroll":
        return {"type": "reroll", "do_reroll": random.random() < 0.5}
    if phase == "choose_purple":
        return {"type": "choose_purple", "card": random.choice(state.pending_purple)}
    if phase == "tv_station":
        others = [i for i in range(state.n_players) if i != pid]
        return {"type": "tv_station", "target": random.choice(others)}
    if phase == "business_center":
        return _business_center_greedy(state, pid)
    if phase == "build":
        builds = E.available_builds(state)
        return {"type": "buy", "card": random.choice(builds) if builds else None}
    return {"type": "buy", "card": None}


def strategy_analysis(state: "GameState", pid: int) -> dict:
    from strategy.analysis import strategy_analysis as _sa

    return _sa(state, pid)


def strategy_neural(state: "GameState", pid: int) -> dict:
    import os, json, torch
    from neural.model import MachiKoroNet
    from neural.encode import encode_state, buy_mask, dice_mask, BUY_KEYS

    _net = strategy_neural._net  # type: ignore[attr-defined]
    if _net is None:
        _dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neural", "model_0")
        with open(os.path.join(_dir, "config.json")) as f:
            cfg = json.load(f)
        net = MachiKoroNet(cfg)
        ckpt = torch.load(os.path.join(_dir, "latest.pt"), weights_only=True)
        net.load_state_dict(ckpt["model"])
        net.eval()
        strategy_neural._net = net
        _net = net

    phase = state.phase
    x = encode_state(state, pid)

    with torch.no_grad():
        if phase == "roll":
            logits = _net.dice_logits(x).masked_fill(~dice_mask(state, pid), float("-inf"))
            return {"type": "roll", "n_dice": int(logits.argmax()) + 1}
        if phase == "reroll":
            return {"type": "reroll", "do_reroll": bool(int(_net.reroll_logits(x).argmax()))}
        if phase == "build":
            logits = _net.buy_logits(x).masked_fill(~buy_mask(state, pid), float("-inf"))
            return {"type": "buy", "card": BUY_KEYS[int(logits.argmax())]}

    if phase == "choose_purple":
        return {"type": "choose_purple", "card": state.pending_purple[0]}
    if phase == "tv_station":
        return _take_from_richest(state, pid)
    if phase == "business_center":
        return _business_center_greedy(state, pid)
    return {"type": "buy", "card": None}


strategy_neural._net = None  # type: ignore[attr-defined]


STRATEGIES: dict[str, callable] = {
    "buy_cheapest": strategy_buy_cheapest,
    "rush_landmarks": strategy_rush_landmarks,
    "random": strategy_random,
    "analysis": strategy_analysis,
    "neural": strategy_neural,
}
