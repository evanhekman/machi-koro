#!/Users/evanhekman/machi-koro/.venv/bin/python
"""
Expectimax analysis engine for Machi Koro solitaire.

Maximises the probability of winning (all 4 landmarks built) within a
depth-limited search tree. Dice outcomes are probability-weighted (expectimax);
build decisions are maximised.

Tree structure per turn:
  [chance] dice roll  →  [decision] Radio Tower reroll?
    →  [decision] what to buy  →  recurse with depth-1
                                   (or same depth on Amusement Park doubles)

Usage (standalone):
    python strategy/analysis.py [--depth N]

Usage (as strategy):
    from strategy.analysis import strategy_analysis
    # add to strategies.STRATEGIES or pass directly to simulate.run_game
"""

from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
from enum import IntEnum
from typing import NamedTuple

import engine as E

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEPTH: int = 10  # turns to look ahead
WIN_VALUE: float = 1.0  # leaf value when all landmarks are built

# ---------------------------------------------------------------------------
# Card / landmark tables
# ---------------------------------------------------------------------------


# Solitaire uses only BLUE and GREEN cards (no RED / PURPLE in supply)
class Card(IntEnum):
    wheat_field = 0  # BLUE   roll 1      cost 1
    ranch = 1  # BLUE   roll 2      cost 1
    bakery = 2  # GREEN  roll 2,3    cost 1   (+mall)
    convenience_store = 3  # GREEN  roll 4      cost 2   (+mall)
    forest = 4  # BLUE   roll 5      cost 3
    cheese_factory = 5  # GREEN  roll 7      cost 5   (3×ranch)
    furniture_factory = 6  # GREEN  roll 8      cost 3   (3×forest+mine)
    mine = 7  # BLUE   roll 9      cost 6
    apple_orchard = 8  # BLUE   roll 10     cost 3
    fruit_veg_market = 9  # GREEN  roll 11,12  cost 2   (2×wheat+apple)


NUM_CARDS: int = len(Card)


class Landmark(IntEnum):
    train_station = NUM_CARDS + 0  # cost  4  — roll 2 dice
    shopping_mall = NUM_CARDS + 1  # cost 10  — +1 to bakery/conv_store income
    amusement_park = NUM_CARDS + 2  # cost 16  — extra turn on doubles
    radio_tower = NUM_CARDS + 3  # cost 22  — optional reroll


# String keys kept for engine interop and display
CARD_KEYS: tuple[str, ...] = tuple(c.name for c in Card)
LANDMARK_KEYS: tuple[str, ...] = tuple(lm.name for lm in Landmark)

CARD_COSTS: tuple[int, ...] = tuple(E.CARDS[k].cost for k in CARD_KEYS)
LANDMARK_COSTS: tuple[int, ...] = (4, 10, 16, 22)
SUPPLY_MAX: int = 3  # copies per card in solitaire

# Pre-built plain tuples — avoids enum __iter__ overhead in the hot loop
_ALL_CARDS: tuple[Card, ...] = tuple(Card)
_ALL_LANDMARKS: tuple[Landmark, ...] = tuple(Landmark)

# ---------------------------------------------------------------------------
# Dice distributions  {(roll_total, is_doubles): probability}
# ---------------------------------------------------------------------------


def _build_dists() -> tuple[
    dict[tuple[int, bool], float], dict[tuple[int, bool], float]
]:
    one: dict[tuple[int, bool], float] = {(i, False): 1.0 / 6.0 for i in range(1, 7)}
    two: dict[tuple[int, bool], float] = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            key = (d1 + d2, d1 == d2)
            two[key] = two.get(key, 0.0) + 1.0 / 36.0
    return one, two


DIST_1, DIST_2 = _build_dists()

# ---------------------------------------------------------------------------
# Hashable analysis state
# ---------------------------------------------------------------------------


class AState(NamedTuple):
    coins: int
    cards: tuple[int, ...]  # counts in CARD_KEYS order
    landmarks: tuple[bool, ...]  # (train, mall, park, tower)

    # Convenience properties ------------------------------------------------
    @property
    def has_train(self) -> bool:
        return self.landmarks[0]

    @property
    def has_mall(self) -> bool:
        return self.landmarks[1]

    @property
    def has_park(self) -> bool:
        return self.landmarks[2]

    @property
    def has_tower(self) -> bool:
        return self.landmarks[3]

    def is_won(self) -> bool:
        return all(self.landmarks)


def initial_astate() -> AState:
    cards = [0] * NUM_CARDS
    cards[Card.wheat_field] = 1
    cards[Card.bakery] = 1
    return AState(coins=3, cards=tuple(cards), landmarks=(False,) * 4)


def gs_to_astate(gs: E.GameState) -> AState:
    """Convert a live GameState to AState for the active player."""
    p = gs.active_player
    cards = tuple(p.cards.get(k, 0) for k in CARD_KEYS)
    lms = tuple(bool(p.landmarks.get(k, False)) for k in LANDMARK_KEYS)
    return AState(coins=p.coins, cards=cards, landmarks=lms)


# ---------------------------------------------------------------------------
# Income calculation (pure function)
# ---------------------------------------------------------------------------


def calc_income(cards: tuple[int, ...], landmarks: tuple[bool, ...], roll: int) -> int:
    """Coins gained from a single roll in solitaire (only BLUE + GREEN fire)."""
    has_mall = landmarks[1]
    c = cards
    income = 0

    if roll == 1:
        income += c[0]  # wheat_field
    if roll == 2:
        income += c[1]  # ranch (BLUE)
    if roll in (2, 3):
        income += c[2] * (2 if has_mall else 1)  # bakery
    if roll == 4:
        income += c[3] * (4 if has_mall else 3)  # conv_store
    if roll == 5:
        income += c[4]  # forest
    if roll == 7:
        income += c[5] * 3 * c[1]  # cheese_factory × ranch
    if roll == 8:
        income += c[6] * 3 * (c[4] + c[7])  # furn_factory × (forest+mine)
    if roll == 9:
        income += c[7] * 5  # mine
    if roll == 10:
        income += c[8] * 3  # apple_orchard
    if roll in (11, 12):
        income += c[9] * 2 * (c[0] + c[8])  # fruit_veg × (wheat+apple)

    return income


# ---------------------------------------------------------------------------
# Build options & state transitions
# ---------------------------------------------------------------------------

_opts_cache: dict[tuple, tuple] = {}


def build_options(
    coins: int, cards: tuple[int, ...], lms: tuple[bool, ...]
) -> tuple[int | None, ...]:
    """
    All affordable cards/landmarks the player can purchase, plus None (skip).
    Cards are limited by SUPPLY_MAX; landmarks by whether already built.
    Returns Card/Landmark IntEnum members (ints) for zero-cost index lookup.
    Result is cached per (coins, cards, lms) — depth-independent.
    Key is a plain tuple; AState (a NamedTuple) hashes/compares equal to it,
    so entries written by either caller hit the same cache slot.
    """
    key = (coins, cards, lms)
    cached = _opts_cache.get(key)
    if cached is not None:
        return cached
    opts: list[int | None] = [None]
    for card in _ALL_CARDS:
        if coins >= CARD_COSTS[card] and cards[card] < SUPPLY_MAX:
            opts.append(card)
    for lm in _ALL_LANDMARKS:
        if coins >= LANDMARK_COSTS[lm - NUM_CARDS] and not lms[lm - NUM_CARDS]:
            opts.append(lm)
    result = tuple(opts)
    _opts_cache[key] = result
    return result


def apply_build(state: AState, option: int | None) -> AState:
    """Return new AState after purchasing option (Card/Landmark int, or None = skip)."""
    if option is None:
        return state
    if option < NUM_CARDS:  # Card
        cs = list(state.cards)
        cs[option] += 1
        return AState(state.coins - CARD_COSTS[option], tuple(cs), state.landmarks)
    # Landmark
    i = option - NUM_CARDS
    lms = list(state.landmarks)
    lms[i] = True
    return AState(state.coins - LANDMARK_COSTS[i], state.cards, tuple(lms))


# ---------------------------------------------------------------------------
# Core expectimax
# ---------------------------------------------------------------------------

_WIN_LMS = (True, True, True, True)

# Cache key is (coins, cards, landmarks, depth) — raw tuple to avoid constructing
# AState objects on lookups that turn out to be cache hits.
_cache: dict[tuple, float] = {}


def analyze(state: AState, depth: int) -> float:
    """
    Win probability from `state` over the next `depth` turns, assuming
    optimal play (maximise probability of having all 4 landmarks).

    Tree structure:
      - Chance node : dice roll (probability-weighted)
      - Decision    : Radio Tower keep/reroll
      - Decision    : what to build
      - Recursion   : depth-1
    """
    if state.landmarks == _WIN_LMS:
        return WIN_VALUE
    if depth <= 0:
        return 0.0

    key = state + (depth,)
    if key in _cache:
        return _cache[key]

    dist = DIST_2 if state.has_train else DIST_1

    if state.has_tower:
        val_reroll = _roll_ev(state, dist, depth)
        cards, lms = state.cards, state.landmarks
        has_park = lms[2]  # inlined to avoid property overhead
        ev = 0.0
        for (roll, is_dbl), prob in dist.items():
            income = calc_income(cards, lms, roll)
            val_keep = _best_build(
                state.coins + income, cards, lms, depth, is_dbl and has_park
            )
            ev += prob * max(val_keep, val_reroll)
    else:
        ev = _roll_ev(state, dist, depth)

    _cache[key] = ev
    return ev


def _roll_ev(state: AState, dist: dict[tuple[int, bool], float], depth: int) -> float:
    """Expected value of rolling (no reroll option)."""
    cards, lms = state.cards, state.landmarks
    # lms[2] is has_park — inlined to avoid property call overhead on each iteration
    has_park = lms[2]
    ev = 0.0
    for (roll, is_dbl), prob in dist.items():
        income = calc_income(cards, lms, roll)
        # Pass raw args to avoid NamedTuple __new__ overhead for throwaway states
        ev += prob * _best_build(
            state.coins + income, cards, lms, depth, is_dbl and has_park
        )
    return ev


_P_DBL = 1.0 / 6.0  # P(doubles) with 2 dice


def _best_build(
    coins: int,
    cards: tuple[int, ...],
    lms: tuple[bool, ...],
    depth: int,
    extra_turn: bool,
) -> float:
    """
    Maximum win probability over all build options from the post-income state.
    extra_turn=True (Amusement Park doubles) → one free extra turn whose infinite
    chain of potential further extra turns is collapsed via geometric series:
      V_extra = base + P_DBL * V_extra  →  V_extra = base / (1 - P_DBL) = base * 6/5

    Takes raw (coins, cards, lms) instead of AState to avoid NamedTuple __new__
    overhead on the ~3.3M intermediate states created in _roll_ev that are never
    stored — only used as a build_options cache key and then discarded.
    AState construction is still deferred until a real cache miss.
    """
    if lms == _WIN_LMS:
        return WIN_VALUE
    next_depth = depth - 1
    best = 0.0

    for opt in build_options(coins, cards, lms):
        # Compute new state components inline — no AState construction yet
        if opt is None:
            nc, nk, nl = coins, cards, lms
        elif opt < NUM_CARDS:  # Card — opt IS the index
            nc = coins - CARD_COSTS[opt]
            nk = cards[:opt] + (cards[opt] + 1,) + cards[opt + 1 :]
            nl = lms
        else:  # Landmark — index is opt - NUM_CARDS
            i = opt - NUM_CARDS
            nc = coins - LANDMARK_COSTS[i]
            nk = cards
            nl = lms[:i] + (True,) + lms[i + 1 :]

        if nl == _WIN_LMS:
            val = WIN_VALUE
        elif next_depth <= 0:
            val = 0.0
        else:
            key = (nc, nk, nl, next_depth)  # plain tuple — no AState yet
            if key in _cache:
                val = _cache[key]
            else:
                val = analyze(AState(nc, nk, nl), next_depth)

        if val > best:
            best = val

    if extra_turn:
        best = min(WIN_VALUE, best / (1.0 - _P_DBL))
    return best


def clear_cache() -> None:
    _cache.clear()
    _opts_cache.clear()


def cache_stats() -> dict:
    return {"entries": len(_cache)}


# ---------------------------------------------------------------------------
# Strategy wrapper — drop-in for strategies.STRATEGIES
# ---------------------------------------------------------------------------


def strategy_analysis(state: E.GameState, pid: int) -> dict:
    """
    Expectimax strategy for solitaire.
    Handles phases: roll, reroll, build.
    """
    phase = state.phase
    astate = gs_to_astate(state)

    if phase == "roll":
        return _action_roll(astate)
    if phase == "reroll":
        return _action_reroll(astate, state.last_dice)
    if phase == "build":
        return _action_build(astate)

    # Shouldn't be reached in solitaire, but handle gracefully
    return {"type": "buy", "card": None}


def _action_roll(astate: AState) -> dict:
    """Choose 1 or 2 dice (requires Train Station for 2)."""
    if not astate.has_train:
        return {"type": "roll", "n_dice": 1}

    # Compare expected win probability for each dice count
    ev1 = _roll_ev_for(astate, 1)
    ev2 = _roll_ev_for(astate, 2)
    return {"type": "roll", "n_dice": 2 if ev2 >= ev1 else 1}


def _roll_ev_for(astate: AState, n_dice: int) -> float:
    """Expected win prob when rolling n_dice dice from astate (no reroll)."""
    dist = DIST_2 if n_dice == 2 else DIST_1
    cards, lms = astate.cards, astate.landmarks
    has_park = lms[2]  # inlined to avoid property overhead
    ev = 0.0
    for (roll, is_dbl), prob in dist.items():
        income = calc_income(cards, lms, roll)
        ev += prob * _best_build(
            astate.coins + income, cards, lms, DEPTH, is_dbl and has_park
        )
    return ev


def _action_reroll(astate: AState, last_dice: list[int]) -> dict:
    """Radio Tower: decide whether to keep current roll or reroll."""
    roll = sum(last_dice)
    n_dice = len(last_dice)
    is_dbl = n_dice == 2 and last_dice[0] == last_dice[1]
    dist = DIST_2 if n_dice == 2 else DIST_1

    income = calc_income(astate.cards, astate.landmarks, roll)
    val_keep = _best_build(
        astate.coins + income,
        astate.cards,
        astate.landmarks,
        DEPTH,
        is_dbl and astate.has_park,
    )

    val_reroll = _roll_ev(astate, dist, DEPTH)  # expected value of fresh roll
    return {"type": "reroll", "do_reroll": val_reroll > val_keep}


def _action_build(astate: AState) -> dict:
    """Choose what to buy (or skip) to maximise win probability."""
    best_opt: int | None = None
    best_val: float = -1.0

    for opt in build_options(astate.coins, astate.cards, astate.landmarks):
        val = analyze(apply_build(astate, opt), DEPTH)
        if val > best_val:
            best_val = val
            best_opt = opt

    # Engine expects a string card/landmark name or None
    name = best_opt.name if best_opt is not None else None
    return {"type": "buy", "card": name}


# ---------------------------------------------------------------------------
# Standalone analysis
# ---------------------------------------------------------------------------


def _show_graph(
    depths: list[int], times: list[float], save_path: str | None = None
) -> None:
    import matplotlib

    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    plt.style.use("dark_background")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"Expectimax search time — depths 1–{depths[-1]}", fontsize=13, color="white"
    )

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Left: raw time
    axes[0].plot(depths, times, "o-", color="white")
    axes[0].set_xlabel("Depth (turns ahead)")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Time per depth")
    axes[0].set_xticks(depths)

    # Right: log scale — exponential growth shows as a straight line
    log_times = [math.log10(max(t, 1e-9)) for t in times]
    axes[1].plot(depths, log_times, "o-", color="white")
    axes[1].set_xlabel("Depth (turns ahead)")
    axes[1].set_ylabel("log₁₀(time in seconds)")
    axes[1].set_title("Same data on a log scale\n(straight line = exponential growth)")
    axes[1].set_xticks(depths)
    axes[1].axhline(0, color="#444444", linewidth=0.5)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Time graph saved to {save_path}")
    else:
        plt.show()


def _run_analysis(depth: int, save_graph: str | None = None) -> None:
    import signal

    clear_cache()
    s0 = initial_astate()
    dist = DIST_1  # no Train Station at start

    depths: list[int] = []
    probs: list[float] = []
    times: list[float] = []

    def _on_interrupt(sig, frame):
        print("\nInterrupted — saving graph of completed depths...")
        if depths:
            _show_graph(depths, times, save_graph)
        sys.exit(0)

    if save_graph is not None:
        signal.signal(signal.SIGINT, _on_interrupt)

    print(f"Expectimax analysis  depth={depth}")
    print("=" * 50)

    t0 = time.perf_counter()
    prob = 0.0
    for d in range(1, depth + 1):
        print(f"  evaluating depth {d}...", end="", flush=True)
        t1 = time.perf_counter()
        prob = analyze(s0, d)
        elapsed_d = time.perf_counter() - t1
        depths.append(d)
        probs.append(prob)
        times.append(elapsed_d)
        print(f"  P(win)={prob:.6f}  cache={len(_cache):,}  ({elapsed_d:.2f}s)")
    elapsed = time.perf_counter() - t0

    print(f"\nWin probability in {depth} turns : {prob:.8f}")
    print(f"Cache entries                    : {len(_cache):,}")
    print(f"Time                             : {elapsed:.3f}s")

    if save_graph is not None and depths:
        _show_graph(depths, times, save_graph)

    print(f"\nRecommended first turn (initial state: 3 coins, wheat+bakery):")
    print(f"{'Roll':>5}  {'Income':>6}  {'Coins':>5}  {'Best buy':<22}  P(win|buy)")
    print("-" * 65)

    for (roll, _), prob_roll in sorted(dist.items()):
        income = calc_income(s0.cards, s0.landmarks, roll)
        s_income = AState(s0.coins + income, s0.cards, s0.landmarks)
        opts = build_options(s_income.coins, s_income.cards, s_income.landmarks)

        best_opt: int | None = None
        best_val: float = -1.0
        for opt in opts:
            val = analyze(apply_build(s_income, opt), depth - 1)
            if val > best_val:
                best_val = val
                best_opt = opt

        label = best_opt.name if best_opt is not None else "skip"
        print(
            f"  {roll:>3}  {income:>6}  {s_income.coins:>5}  {label:<22}  {best_val:.8f}"
        )


if __name__ == "__main__":
    depth = DEPTH
    save_graph: str | None = None
    profile = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i].startswith("--depth="):
            depth = int(args[i].split("=")[1])
        elif args[i] == "--depth" and i + 1 < len(args):
            depth = int(args[i + 1])
            i += 1
        elif args[i] == "--show-time-graph":
            save_graph = save_graph or "time_graph.png"
        elif args[i] == "--save-time-graph" and i + 1 < len(args):
            save_graph = args[i + 1]
            i += 1
        elif args[i].startswith("--save-time-graph="):
            save_graph = args[i].split("=", 1)[1]
        elif args[i] == "--profile":
            profile = True
        i += 1

    if profile:
        import cProfile, pstats, io

        pr = cProfile.Profile()
        pr.enable()
        _run_analysis(depth, save_graph)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
        print(s.getvalue())
    else:
        _run_analysis(depth, save_graph)
