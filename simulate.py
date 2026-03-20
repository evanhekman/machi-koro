"""
Headless batch simulator. Run directly:
    python simulate.py buy_cheapest rush_landmarks --n 10000
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable
import argparse

import engine as E
from strategies import STRATEGIES

Strategy = Callable


def apply_action(state: E.GameState, action: dict) -> None:
    t = action["type"]
    if t == "roll":
        E.action_roll(state, action.get("n_dice", 1))
    elif t == "reroll":
        E.action_reroll(state, action["do_reroll"])
    elif t == "tv_station":
        E.action_tv_station(state, action["target"])
    elif t == "business_center":
        E.action_business_center(state, action["target"], action["give_card"], action["take_card"])
    elif t == "buy":
        E.action_buy(state, action.get("card"))
    else:
        raise ValueError(f"Unknown action type: {t}")


def run_game(strategies: list[Strategy], max_turns: int = 500) -> dict:
    state = E.create_game(len(strategies))
    turns = 0
    while state.phase != "end" and turns < max_turns:
        pid = state.current_player
        prev_phase = state.phase
        action = strategies[pid](state, pid)
        apply_action(state, action)
        if state.phase == "roll" and prev_phase != "roll":
            turns += 1
    return {"winner": state.winner, "turns": turns}


def run_n(strategies: list[Strategy], n: int) -> dict:
    wins: dict[int, int] = defaultdict(int)
    total_turns = 0
    for _ in range(n):
        result = run_game(strategies)
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        total_turns += result["turns"]
    return {
        "n_games": n,
        "wins": dict(wins),
        "win_rates": {pid: wins[pid] / n for pid in range(len(strategies))},
        "avg_turns": total_turns / n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategies", nargs="+", choices=list(STRATEGIES.keys()),
                        help="Strategy name for each player")
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()

    strat_fns = [STRATEGIES[s] for s in args.strategies]
    print(f"Running {args.n:,} games: {args.strategies}")
    results = run_n(strat_fns, args.n)
    print(f"Win rates: { {args.strategies[pid]: f'{rate:.1%}' for pid, rate in results['win_rates'].items()} }")
    print(f"Avg turns: {results['avg_turns']:.1f}")
