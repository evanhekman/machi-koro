from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random


class Color(str, Enum):
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    PURPLE = "purple"


@dataclass(frozen=True)
class CardDef:
    name: str
    color: Color
    activation: tuple[int, ...]
    cost: int
    description: str


CARDS: dict[str, CardDef] = {
    "wheat_field":       CardDef("Wheat Field",             Color.BLUE,   (1,),      1,  "Get 1 coin from bank. (anyone's turn)"),
    "ranch":             CardDef("Ranch",                   Color.BLUE,   (2,),      1,  "Get 1 coin from bank. (anyone's turn)"),
    "bakery":            CardDef("Bakery",                  Color.GREEN,  (2, 3),    1,  "Get 1 coin from bank. (your turn)"),
    "cafe":              CardDef("Cafe",                    Color.RED,    (3,),      2,  "Take 1 coin from active player. (their turn)"),
    "convenience_store": CardDef("Convenience Store",       Color.GREEN,  (4,),      2,  "Get 3 coins from bank. (your turn)"),
    "forest":            CardDef("Forest",                  Color.BLUE,   (5,),      3,  "Get 1 coin from bank. (anyone's turn)"),
    "stadium":           CardDef("Stadium",                 Color.PURPLE, (6,),      6,  "Take 2 coins from all players. (your turn)"),
    "tv_station":        CardDef("TV Station",              Color.PURPLE, (6,),      7,  "Take 5 coins from any one player. (your turn)"),
    "business_center":   CardDef("Business Center",         Color.PURPLE, (6,),      8,  "Trade one establishment with another player. (your turn)"),
    "cheese_factory":    CardDef("Cheese Factory",          Color.GREEN,  (7,),      5,  "Get 3 coins per Ranch you own. (your turn)"),
    "furniture_factory": CardDef("Furniture Factory",       Color.GREEN,  (8,),      3,  "Get 3 coins per Forest or Mine you own. (your turn)"),
    "mine":              CardDef("Mine",                    Color.BLUE,   (9,),      6,  "Get 5 coins from bank. (anyone's turn)"),
    "family_restaurant": CardDef("Family Restaurant",       Color.RED,    (9, 10),   3,  "Take 2 coins from active player. (their turn)"),
    "apple_orchard":     CardDef("Apple Orchard",           Color.BLUE,   (10,),     3,  "Get 3 coins from bank. (anyone's turn)"),
    "fruit_veg_market":  CardDef("Fruit and Veg Market",    Color.GREEN,  (11, 12),  2,  "Get 2 coins per Wheat Field or Apple Orchard. (your turn)"),
}

LANDMARKS: list[str] = ["train_station", "shopping_mall", "amusement_park", "radio_tower"]

LANDMARK_COSTS: dict[str, int] = {
    "train_station":  4,
    "shopping_mall":  10,
    "amusement_park": 16,
    "radio_tower":    22,
}

LANDMARK_NAMES: dict[str, str] = {
    "train_station":  "Train Station",
    "shopping_mall":  "Shopping Mall",
    "amusement_park": "Amusement Park",
    "radio_tower":    "Radio Tower",
}

SUPPLY_COUNTS: dict[str, int] = {k: 6 for k in CARDS}


@dataclass
class PlayerState:
    coins: int = 3
    cards: dict[str, int] = field(default_factory=lambda: {"wheat_field": 1, "bakery": 1})
    landmarks: dict[str, bool] = field(default_factory=lambda: {lm: False for lm in LANDMARKS})

    def card_count(self, name: str) -> int:
        return self.cards.get(name, 0)


@dataclass
class GameState:
    players: list[PlayerState]
    supply: dict[str, int] = field(default_factory=lambda: dict(SUPPLY_COUNTS))
    current_player: int = 0
    # Phases: roll | reroll | tv_station | business_center | build | end
    phase: str = "roll"
    last_roll: Optional[int] = None
    last_dice: Optional[list[int]] = None
    winner: Optional[int] = None
    pending_purple: list[str] = field(default_factory=list)

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def active_player(self) -> PlayerState:
        return self.players[self.current_player]

    def to_dict(self) -> dict:
        return {
            "players": [
                {
                    "coins": p.coins,
                    "cards": p.cards,
                    "landmarks": p.landmarks,
                }
                for p in self.players
            ],
            "supply": self.supply,
            "current_player": self.current_player,
            "phase": self.phase,
            "last_roll": self.last_roll,
            "last_dice": self.last_dice,
            "winner": self.winner,
            "pending_purple": self.pending_purple,
        }


def create_game(n_players: int) -> GameState:
    return GameState(players=[PlayerState() for _ in range(n_players)])


# ---------------------------------------------------------------------------
# Income helpers
# ---------------------------------------------------------------------------

def _base_income(card_name: str, owner: PlayerState) -> int:
    """Income for a single copy of card_name, factoring in multiplier cards."""
    match card_name:
        case "cheese_factory":    return 3 * owner.card_count("ranch")
        case "furniture_factory": return 3 * (owner.card_count("forest") + owner.card_count("mine"))
        case "fruit_veg_market":  return 2 * (owner.card_count("wheat_field") + owner.card_count("apple_orchard"))
        case "convenience_store": return 3
        case "mine":              return 5
        case "apple_orchard":     return 3
        case "family_restaurant": return 2
        case _:                   return 1


def _mall_bonus(card_name: str) -> bool:
    """Shopping mall gives +1 to these cards."""
    return card_name in ("bakery", "convenience_store", "cafe", "family_restaurant")


def resolve_income(state: GameState, roll: int) -> None:
    """Resolve red → blue → green income then queue purple effects."""
    active = state.current_player
    ap = state.players[active]

    # Red: other players take from active player
    for pid, p in enumerate(state.players):
        if pid == active:
            continue
        has_mall = p.landmarks["shopping_mall"]
        for card_name, count in p.cards.items():
            if count == 0:
                continue
            card = CARDS.get(card_name)
            if not card or card.color != Color.RED or roll not in card.activation:
                continue
            per_card = _base_income(card_name, p) + (1 if has_mall and _mall_bonus(card_name) else 0)
            taken = min(per_card * count, ap.coins)
            ap.coins -= taken
            p.coins += taken

    # Blue: everyone collects from bank
    for p in state.players:
        for card_name, count in p.cards.items():
            if count == 0:
                continue
            card = CARDS.get(card_name)
            if not card or card.color != Color.BLUE or roll not in card.activation:
                continue
            p.coins += _base_income(card_name, p) * count

    # Green: active player collects from bank
    has_mall = ap.landmarks["shopping_mall"]
    for card_name, count in ap.cards.items():
        if count == 0:
            continue
        card = CARDS.get(card_name)
        if not card or card.color != Color.GREEN or roll not in card.activation:
            continue
        per_card = _base_income(card_name, ap) + (1 if has_mall and _mall_bonus(card_name) else 0)
        ap.coins += per_card * count

    # Queue purple effects
    state.pending_purple = [
        card_name
        for card_name, count in ap.cards.items()
        if count > 0
        for card in [CARDS.get(card_name)]
        if card and card.color == Color.PURPLE and roll in card.activation
        for _ in range(count)
    ]
    _advance_purple(state)


def _advance_purple(state: GameState) -> None:
    if not state.pending_purple:
        state.phase = "build"
        return
    effect = state.pending_purple[0]
    ap = state.active_player
    if effect == "stadium":
        for pid, p in enumerate(state.players):
            if pid == state.current_player:
                continue
            taken = min(2, p.coins)
            p.coins -= taken
            ap.coins += taken
        state.pending_purple.pop(0)
        _advance_purple(state)
    elif effect == "tv_station":
        state.phase = "tv_station"
    elif effect == "business_center":
        state.phase = "business_center"
    else:
        state.pending_purple.pop(0)
        _advance_purple(state)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_roll(state: GameState, n_dice: int = 1, rng: random.Random | None = None) -> None:
    assert state.phase == "roll"
    assert n_dice in (1, 2)
    if n_dice == 2:
        assert state.active_player.landmarks["train_station"], "Train Station required to roll 2 dice"
    r = rng or random
    dice = [r.randint(1, 6) for _ in range(n_dice)]
    state.last_dice = dice
    state.last_roll = sum(dice)
    if state.active_player.landmarks["radio_tower"]:
        state.phase = "reroll"
    else:
        resolve_income(state, state.last_roll)


def action_reroll(state: GameState, do_reroll: bool, rng: random.Random | None = None) -> None:
    assert state.phase == "reroll"
    if do_reroll:
        r = rng or random
        n_dice = len(state.last_dice)
        dice = [r.randint(1, 6) for _ in range(n_dice)]
        state.last_dice = dice
        state.last_roll = sum(dice)
    resolve_income(state, state.last_roll)


def action_tv_station(state: GameState, target: int) -> None:
    assert state.phase == "tv_station"
    assert 0 <= target < state.n_players and target != state.current_player
    taken = min(5, state.players[target].coins)
    state.players[target].coins -= taken
    state.active_player.coins += taken
    state.pending_purple.pop(0)
    _advance_purple(state)


def action_business_center(state: GameState, target: int, give_card: str, take_card: str) -> None:
    assert state.phase == "business_center"
    ap = state.active_player
    tp = state.players[target]
    assert target != state.current_player
    assert ap.card_count(give_card) > 0, f"You don't own {give_card}"
    assert tp.card_count(take_card) > 0, f"Target doesn't own {take_card}"
    assert CARDS[give_card].color != Color.PURPLE, "Can't trade purple cards"
    assert CARDS[take_card].color != Color.PURPLE, "Can't trade purple cards"
    ap.cards[give_card] -= 1
    tp.cards[give_card] = tp.cards.get(give_card, 0) + 1
    tp.cards[take_card] -= 1
    ap.cards[take_card] = ap.cards.get(take_card, 0) + 1
    state.pending_purple.pop(0)
    _advance_purple(state)


def action_buy(state: GameState, card_name: str | None) -> None:
    assert state.phase == "build"
    if card_name is not None:
        ap = state.active_player
        if card_name in LANDMARKS:
            cost = LANDMARK_COSTS[card_name]
            assert not ap.landmarks[card_name], "Already built"
            assert ap.coins >= cost, "Not enough coins"
            ap.coins -= cost
            ap.landmarks[card_name] = True
        else:
            assert card_name in CARDS, f"Unknown card: {card_name}"
            assert state.supply.get(card_name, 0) > 0, "Out of supply"
            assert ap.coins >= CARDS[card_name].cost, "Not enough coins"
            if CARDS[card_name].color == Color.PURPLE:
                assert ap.cards.get(card_name, 0) == 0, "Already own this establishment"
            ap.coins -= CARDS[card_name].cost
            ap.cards[card_name] = ap.cards.get(card_name, 0) + 1
            state.supply[card_name] -= 1
        if all(ap.landmarks[lm] for lm in LANDMARKS):
            state.phase = "end"
            state.winner = state.current_player
            return
    _end_turn(state)


def _end_turn(state: GameState) -> None:
    # Amusement park: extra turn on doubles
    if (
        state.active_player.landmarks["amusement_park"]
        and state.last_dice is not None
        and len(state.last_dice) == 2
        and state.last_dice[0] == state.last_dice[1]
    ):
        state.phase = "roll"
        return
    state.current_player = (state.current_player + 1) % state.n_players
    state.phase = "roll"


def available_builds(state: GameState) -> list[str]:
    """Cards and landmarks the active player can currently afford."""
    ap = state.active_player
    result = []
    for card_name, card in CARDS.items():
        if state.supply.get(card_name, 0) > 0 and ap.coins >= card.cost:
            if card.color == Color.PURPLE and ap.cards.get(card_name, 0) > 0:
                continue
            result.append(card_name)
    for lm in LANDMARKS:
        if not ap.landmarks[lm] and ap.coins >= LANDMARK_COSTS[lm]:
            result.append(lm)
    return result
