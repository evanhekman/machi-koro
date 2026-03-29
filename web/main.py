from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

import engine as E

app = FastAPI()

_state: Optional[E.GameState] = None


class NewGameRequest(BaseModel):
    n_players: int = 2
    solitaire: bool = False


class ActionRequest(BaseModel):
    type: str
    n_dice: Optional[int] = 1
    do_reroll: Optional[bool] = False
    target: Optional[int] = None
    give_card: Optional[str] = None
    take_card: Optional[str] = None
    card: Optional[str] = None


@app.post("/api/new_game")
def new_game(req: NewGameRequest):
    global _state
    if req.solitaire:
        _state = E.create_game_solitaire()
    else:
        if req.n_players < 2 or req.n_players > 4:
            raise HTTPException(400, "n_players must be 2–4")
        _state = E.create_game(req.n_players)
    return _state.to_dict()


@app.get("/api/state")
def get_state():
    if _state is None:
        raise HTTPException(400, "No game in progress")
    return _state.to_dict()


@app.get("/api/available_builds")
def get_available_builds():
    if _state is None:
        raise HTTPException(400, "No game in progress")
    return E.available_builds(_state)


@app.post("/api/action")
def take_action(req: ActionRequest):
    global _state
    if _state is None:
        raise HTTPException(400, "No game in progress")
    try:
        match req.type:
            case "roll":
                E.action_roll(_state, req.n_dice or 1)
            case "reroll":
                E.action_reroll(_state, req.do_reroll or False)
            case "choose_purple":
                if req.card is None:
                    raise HTTPException(400, "card required")
                E.action_choose_purple(_state, req.card)
            case "tv_station":
                if req.target is None:
                    raise HTTPException(400, "target required")
                E.action_tv_station(_state, req.target)
            case "business_center":
                if None in (req.target, req.give_card, req.take_card):
                    raise HTTPException(400, "target, give_card, take_card required")
                E.action_business_center(
                    _state, req.target, req.give_card, req.take_card
                )
            case "buy":
                E.action_buy(_state, req.card)
            case _:
                raise HTTPException(400, f"Unknown action type: {req.type}")
    except AssertionError as e:
        raise HTTPException(400, str(e))
    return _state.to_dict()


@app.get("/api/card_defs")
def card_defs():
    return {
        name: {
            "name": card.name,
            "color": card.color.value,
            "activation": list(card.activation),
            "cost": card.cost,
            "description": card.description,
        }
        for name, card in E.CARDS.items()
    }


@app.get("/api/landmark_defs")
def landmark_defs():
    return {
        lm: {"name": data["name"], "cost": data["cost"]}
        for lm, data in E.LANDMARKS.items()
    }


app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
)
