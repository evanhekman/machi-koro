#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Income-maximising lookahead strategy for Machi Koro solitaire.

At each build decision, evaluates every affordable option by simulating
n turns of expectimax and picking the one with the highest expected
cumulative income. Winning (all 4 landmarks built) returns INF, so the
strategy always prefers winning to any amount of income.

Usage:
    from strategy.lookahead import make_lookahead_strategy
    strategy = make_lookahead_strategy(5)   # 5-turn lookahead
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import engine as E
from strategy.analysis import (
    AState, calc_income, build_options, apply_build,
    DIST_1, DIST_2, NUM_CARDS, CARD_KEYS, LANDMARK_KEYS,
)

INF: float = 1e9
MAX_COINS: int = 52
P_DBL: float = 1.0 / 6.0


# ---------------------------------------------------------------------------
# Lookahead value function
# ---------------------------------------------------------------------------

def _lookahead(state: AState, depth: int, cache: dict) -> float:
    """Expected cumulative income over `depth` full turns, starting before a roll.
    Returns INF if the state is already won."""
    if state.is_won():
        return INF
    if depth == 0:
        return 0.0

    key = state + (depth,)
    if key in cache:
        return cache[key]

    dist = DIST_2 if state.has_train else DIST_1

    if state.has_tower:
        reroll_ev = _roll_ev(state, dist, depth, cache)
        total = 0.0
        for (roll, is_dbl), prob in dist.items():
            income = calc_income(state.cards, state.landmarks, roll)
            after_roll = AState(min(state.coins + income, MAX_COINS), state.cards, state.landmarks)
            keep_ev = income + _best_build(after_roll, depth, is_dbl and state.has_park, cache)
            total += prob * max(keep_ev, reroll_ev)
    else:
        total = _roll_ev(state, dist, depth, cache)

    cache[key] = total
    return total


def _roll_ev(state: AState, dist: dict, depth: int, cache: dict) -> float:
    total = 0.0
    for (roll, is_dbl), prob in dist.items():
        income = calc_income(state.cards, state.landmarks, roll)
        after_roll = AState(min(state.coins + income, MAX_COINS), state.cards, state.landmarks)
        total += prob * (income + _best_build(after_roll, depth, is_dbl and state.has_park, cache))
    return total


def _best_build(state: AState, depth: int, extra_turn: bool, cache: dict) -> float:
    """Max expected income over all build options, then recurse with depth-1."""
    if state.is_won():
        return INF
    opts = build_options(state.coins, state.cards, state.landmarks)
    best = max(_lookahead(apply_build(state, opt), depth - 1, cache) for opt in opts)
    if extra_turn:
        return min(best / (1.0 - P_DBL), INF)
    return best


def _opt_to_key(opt: int | None) -> str | None:
    if opt is None:
        return None
    if opt < NUM_CARDS:
        return CARD_KEYS[opt]
    return LANDMARK_KEYS[opt - NUM_CARDS]


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def make_lookahead_strategy(n: int):
    """Return a strategy callable that looks n turns ahead when choosing a build."""
    def strategy(state: E.GameState, pid: int) -> dict:
        phase = state.phase

        if phase == "roll":
            n_dice = 2 if state.players[pid].landmarks.get("train_station") else 1
            return {"type": "roll", "n_dice": n_dice}

        if phase == "reroll":
            return {"type": "reroll", "do_reroll": False}

        if phase == "choose_purple":
            return {"type": "choose_purple", "card": state.pending_purple[0]}

        if phase == "tv_station":
            target = max(
                (i for i in range(state.n_players) if i != pid),
                key=lambda i: state.players[i].coins,
            )
            return {"type": "tv_station", "target": target}

        if phase == "business_center":
            from strategy.strategies import _business_center_greedy
            return _business_center_greedy(state, pid)

        if phase == "build":
            p = state.players[pid]
            cards = tuple(p.cards.get(k, 0) for k in CARD_KEYS)
            lms = tuple(bool(p.landmarks.get(k, False)) for k in LANDMARK_KEYS)
            astate = AState(coins=p.coins, cards=cards, landmarks=lms)

            # Use engine's available_builds as the authoritative action space,
            # then map each to an astate option index for lookahead evaluation.
            engine_opts = E.available_builds(state)
            candidates: list[int | None] = [None]
            for key in engine_opts:
                if key in CARD_KEYS:
                    candidates.append(CARD_KEYS.index(key))
                elif key in LANDMARK_KEYS:
                    candidates.append(NUM_CARDS + LANDMARK_KEYS.index(key))

            cache: dict = {}
            best_opt = max(candidates, key=lambda o: _lookahead(apply_build(astate, o), n, cache))
            return {"type": "buy", "card": _opt_to_key(best_opt)}

        return {"type": "buy", "card": None}

    strategy.__name__ = f"lookahead_{n}"
    return strategy
