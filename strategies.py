from __future__ import annotations
from typing import TYPE_CHECKING

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
    our = [(c, E.CARDS[c].cost) for c, n in ap.cards.items()
           if n > 0 and c in E.CARDS and E.CARDS[c].color != E.Color.PURPLE]
    if not our:
        # Fallback: nothing tradeable, just pick anything (shouldn't happen)
        give = next(iter(ap.cards))
        return {"type": "business_center", "target": (pid + 1) % state.n_players,
                "give_card": give, "take_card": give}
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
        return {"type": "business_center", "target": (pid + 1) % state.n_players,
                "give_card": give, "take_card": give}
    return {"type": "business_center", "target": best[1], "give_card": give, "take_card": best[2]}


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
            return {"type": "buy", "card": min(lms, key=lambda x: E.LANDMARK_COSTS[x])}
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
    if phase == "tv_station":
        return _take_from_richest(state, pid)
    if phase == "business_center":
        return _business_center_greedy(state, pid)
    if phase == "build":
        builds = E.available_builds(state)
        lms = [b for b in builds if b in E.LANDMARKS]
        if lms:
            return {"type": "buy", "card": min(lms, key=lambda x: E.LANDMARK_COSTS[x])}
        return {"type": "buy", "card": None}
    return {"type": "buy", "card": None}


def strategy_random(state: GameState, pid: int) -> dict:
    """Roll random dice count, random reroll decision, buy random affordable card."""
    import random
    phase = state.phase
    if phase == "roll":
        has_ts = state.players[pid].landmarks["train_station"]
        return {"type": "roll", "n_dice": random.choice([1, 2]) if has_ts else 1}
    if phase == "reroll":
        return {"type": "reroll", "do_reroll": random.random() < 0.5}
    if phase == "tv_station":
        others = [i for i in range(state.n_players) if i != pid]
        return {"type": "tv_station", "target": random.choice(others)}
    if phase == "business_center":
        return _business_center_greedy(state, pid)
    if phase == "build":
        builds = E.available_builds(state)
        return {"type": "buy", "card": random.choice(builds) if builds else None}
    return {"type": "buy", "card": None}


STRATEGIES: dict[str, callable] = {
    "buy_cheapest":    strategy_buy_cheapest,
    "rush_landmarks":  strategy_rush_landmarks,
    "random":          strategy_random,
}
