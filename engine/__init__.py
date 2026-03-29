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
    "wheat_field": CardDef(
        "Wheat Field", Color.BLUE, (1,), 1, "Get 1 coin from bank, on anyone's turn."
    ),
    "ranch": CardDef(
        "Ranch", Color.BLUE, (2,), 1, "Get 1 coin from bank, on anyone's turn."
    ),
    "bakery": CardDef(
        "Bakery", Color.GREEN, (2, 3), 1, "Get 1 coin from bank, on your turn."
    ),
    "cafe": CardDef(
        "Cafe", Color.RED, (3,), 2, "Take 1 coin from active player, on their turn."
    ),
    "convenience_store": CardDef(
        "Convenience Store",
        Color.GREEN,
        (4,),
        2,
        "Get 3 coins from bank, on your turn.",
    ),
    "forest": CardDef(
        "Forest", Color.BLUE, (5,), 3, "Get 1 coin from bank, on anyone's turn."
    ),
    "stadium": CardDef(
        "Stadium", Color.PURPLE, (6,), 6, "Take 2 coins from all players, on your turn."
    ),
    "tv_station": CardDef(
        "TV Station",
        Color.PURPLE,
        (6,),
        7,
        "Take 5 coins from any one player, on your turn.",
    ),
    "business_center": CardDef(
        "Business Center",
        Color.PURPLE,
        (6,),
        8,
        "Trade one establishment with another player, on your turn.",
    ),
    "cheese_factory": CardDef(
        "Cheese Factory",
        Color.GREEN,
        (7,),
        5,
        "Get 3 coins per Ranch you own, on your turn.",
    ),
    "furniture_factory": CardDef(
        "Furniture Factory",
        Color.GREEN,
        (8,),
        3,
        "Get 3 coins per Forest or Mine you own, on your turn.",
    ),
    "mine": CardDef(
        "Mine", Color.BLUE, (9,), 6, "Get 5 coins from bank, on anyone's turn."
    ),
    "family_restaurant": CardDef(
        "Family Restaurant",
        Color.RED,
        (9, 10),
        3,
        "Take 2 coins from active player, on their turn.",
    ),
    "apple_orchard": CardDef(
        "Apple Orchard",
        Color.BLUE,
        (10,),
        3,
        "Get 3 coins from bank, on anyone's turn.",
    ),
    "fruit_veg_market": CardDef(
        "Fruit and Veg Market",
        Color.GREEN,
        (11, 12),
        2,
        "Get 2 coins per Wheat Field or Apple Orchard, on your turn.",
    ),
}

LANDMARKS: dict[str, dict] = {
    "train_station": {"name": "Train Station", "cost": 4},
    "shopping_mall": {"name": "Shopping Mall", "cost": 10},
    "amusement_park": {"name": "Amusement Park", "cost": 16},
    "radio_tower": {"name": "Radio Tower", "cost": 22},
}

SUPPLY_COUNTS: dict[str, int] = {k: 6 for k in CARDS}
SOLITAIRE_SUPPLY: dict[str, int] = {
    k: 3 for k, v in CARDS.items() if v.color in (Color.BLUE, Color.GREEN)
}


@dataclass
class PlayerState:
    coins: int = 3
    cards: dict[str, int] = field(
        default_factory=lambda: {"wheat_field": 1, "bakery": 1}
    )
    landmarks: dict[str, bool] = field(
        default_factory=lambda: {lm: False for lm in LANDMARKS}
    )
    turns: int = 0

    def card_count(self, name: str) -> int:
        return self.cards.get(name, 0)


@dataclass
class GameState:
    players: list[PlayerState]
    supply: dict[str, int] = field(default_factory=lambda: dict(SUPPLY_COUNTS))
    current_player: int = 0
    # Phases: roll | reroll | choose_purple | tv_station | business_center | build | end
    phase: str = "roll"
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
                    "turns": p.turns,
                }
                for p in self.players
            ],
            "supply": self.supply,
            "current_player": self.current_player,
            "phase": self.phase,
            "last_roll": sum(self.last_dice) if self.last_dice else None,
            "last_dice": self.last_dice,
            "winner": self.winner,
            "pending_purple": self.pending_purple,
        }


def create_game(n_players: int) -> GameState:
    return GameState(players=[PlayerState() for _ in range(n_players)])


def create_game_solitaire() -> GameState:
    return GameState(players=[PlayerState()], supply=dict(SOLITAIRE_SUPPLY))


def _base_income(card_name: str, owner: PlayerState) -> int:
    match card_name:
        case "cheese_factory":
            return 3 * owner.card_count("ranch")
        case "furniture_factory":
            return 3 * (owner.card_count("forest") + owner.card_count("mine"))
        case "fruit_veg_market":
            return 2 * (
                owner.card_count("wheat_field") + owner.card_count("apple_orchard")
            )
        case "convenience_store":
            return 3
        case "mine":
            return 5
        case "apple_orchard":
            return 3
        case "family_restaurant":
            return 2
        case _:
            return 1


def _mall_bonus(card_name: str) -> bool:
    return card_name in ("bakery", "convenience_store", "cafe", "family_restaurant")


def resolve_income(state: GameState, roll: int) -> None:
    active = state.current_player
    ap = state.players[active]

    n = state.n_players
    for i in range(n - 1):
        pid = (active - 1 - i) % n
        p = state.players[pid]
        has_mall = p.landmarks["shopping_mall"]
        for card_name, count in p.cards.items():
            card = CARDS.get(card_name)
            if not card or card.color != Color.RED or roll not in card.activation:
                continue
            per_card = _base_income(card_name, p) + (
                1 if has_mall and _mall_bonus(card_name) else 0
            )
            taken = min(per_card * count, ap.coins)
            ap.coins -= taken
            p.coins += taken

    for p in state.players:
        for card_name, count in p.cards.items():
            card = CARDS.get(card_name)
            if not card or card.color != Color.BLUE or roll not in card.activation:
                continue
            p.coins += _base_income(card_name, p) * count

    has_mall = ap.landmarks["shopping_mall"]
    for card_name, count in ap.cards.items():
        card = CARDS.get(card_name)
        if not card or card.color != Color.GREEN or roll not in card.activation:
            continue
        per_card = _base_income(card_name, ap) + (
            1 if has_mall and _mall_bonus(card_name) else 0
        )
        ap.coins += per_card * count

    state.pending_purple = [
        card_name
        for card_name, count in ap.cards.items()
        if count > 0
        and (card := CARDS.get(card_name)) is not None
        and card.color == Color.PURPLE
        and roll in card.activation
    ]
    _resolve_purple_choice(state)


def _resolve_purple_choice(state: GameState) -> None:
    if not state.pending_purple:
        state.phase = "build"
    elif len(state.pending_purple) == 1:
        _activate_purple(state, state.pending_purple[0])
    else:
        state.phase = "choose_purple"


def _activate_purple(state: GameState, card_name: str) -> None:
    state.pending_purple = []
    if card_name == "stadium":
        ap = state.active_player
        for pid, p in enumerate(state.players):
            if pid == state.current_player:
                continue
            taken = min(2, p.coins)
            p.coins -= taken
            ap.coins += taken
        state.phase = "build"
    elif card_name == "tv_station":
        state.phase = "tv_station"
    elif card_name == "business_center":
        state.phase = "business_center"


def _end_turn(state: GameState) -> None:
    state.active_player.turns += 1
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


def action_roll(
    state: GameState, n_dice: int = 1, rng: random.Random | None = None
) -> None:
    assert state.phase == "roll"
    assert n_dice in (1, 2)
    if n_dice == 2:
        assert state.active_player.landmarks["train_station"], (
            "Train Station required to roll 2 dice"
        )
    r = rng or random
    dice = [r.randint(1, 6) for _ in range(n_dice)]
    state.last_dice = dice
    if state.active_player.landmarks["radio_tower"]:
        state.phase = "reroll"
    else:
        resolve_income(state, sum(dice))


def action_reroll(
    state: GameState, do_reroll: bool, rng: random.Random | None = None
) -> None:
    assert state.phase == "reroll"
    if do_reroll:
        r = rng or random
        dice = [r.randint(1, 6) for _ in range(len(state.last_dice))]
        state.last_dice = dice
    resolve_income(state, sum(state.last_dice))


def action_choose_purple(state: GameState, card_name: str) -> None:
    assert state.phase == "choose_purple"
    assert card_name in state.pending_purple, f"{card_name} not available this turn"
    _activate_purple(state, card_name)


def action_tv_station(state: GameState, target: int) -> None:
    assert state.phase == "tv_station"
    assert 0 <= target < state.n_players and target != state.current_player
    taken = min(5, state.players[target].coins)
    state.players[target].coins -= taken
    state.active_player.coins += taken
    state.phase = "build"


def action_business_center(
    state: GameState, target: int, give_card: str, take_card: str
) -> None:
    assert state.phase == "business_center"
    ap = state.active_player
    tp = state.players[target]
    assert target != state.current_player
    assert ap.card_count(give_card) > 0, f"You don't own {give_card}"
    assert tp.card_count(take_card) > 0, f"Target doesn't own {take_card}"
    assert CARDS[give_card].color != Color.PURPLE, "Can't trade purple cards"
    assert CARDS[take_card].color != Color.PURPLE, "Can't trade purple cards"
    ap.cards[give_card] -= 1
    if ap.cards[give_card] == 0:
        del ap.cards[give_card]
    tp.cards[give_card] = tp.cards.get(give_card, 0) + 1
    tp.cards[take_card] -= 1
    if tp.cards[take_card] == 0:
        del tp.cards[take_card]
    ap.cards[take_card] = ap.cards.get(take_card, 0) + 1
    state.phase = "build"


def action_buy(state: GameState, card_name: str | None) -> None:
    assert state.phase == "build"
    if card_name is not None:
        ap = state.active_player
        if card_name in LANDMARKS:
            cost = LANDMARKS[card_name]["cost"]
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
            ap.turns += 1
            state.phase = "end"
            state.winner = state.current_player
            return
    _end_turn(state)


def available_builds(state: GameState) -> list[str]:
    ap = state.active_player
    result = []
    for card_name, card in CARDS.items():
        if state.supply.get(card_name, 0) > 0 and ap.coins >= card.cost:
            if card.color == Color.PURPLE and ap.cards.get(card_name, 0) > 0:
                continue
            result.append(card_name)
    for lm, lm_data in LANDMARKS.items():
        if not ap.landmarks[lm] and ap.coins >= lm_data["cost"]:
            result.append(lm)
    return result


def get_player_cards(state: GameState, pid: int) -> dict[str, int]:
    return dict(state.players[pid].cards)


def get_market(state: GameState) -> dict[str, int]:
    return dict(state.supply)


def get_coins(state: GameState, pid: int) -> int:
    return state.players[pid].coins


def get_turn(state: GameState, pid: int) -> int:
    return state.players[pid].turns


def roll(state: GameState) -> None:
    n_dice = 2 if state.active_player.landmarks["train_station"] else 1
    action_roll(state, n_dice)


def reroll(state: GameState, do_reroll: bool) -> None:
    action_reroll(state, do_reroll)


def purchase(state: GameState, card_name: str | None) -> None:
    action_buy(state, card_name)
