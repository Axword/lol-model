"""Stan gry niezalezny od zrodla (replay GRID, OCR, Oracle's Elixir) i cechy modelu live.

Konwencja: indeks 0 = strona niebieska, 1 = czerwona. Wszystkie cechy modelu to
roznice (niebiescy - czerwoni), wiec model jest antysymetryczny.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROLES = ("Top", "Jungle", "Mid", "ADC", "Support")
BARON_BUFF_S = 180.0
ELDER_BUFF_S = 150.0
SOUL_DRAGONS = 4


@dataclass
class GameState:
    t: float  # czas gry w sekundach
    gold: np.ndarray = field(default_factory=lambda: np.full((2, 5), 500.0))  # laczne zloto [strona, rola]
    level: np.ndarray = field(default_factory=lambda: np.ones((2, 5)))
    alive: np.ndarray = field(default_factory=lambda: np.ones((2, 5)))
    kills: np.ndarray = field(default_factory=lambda: np.zeros(2))
    towers: np.ndarray = field(default_factory=lambda: np.zeros(2))
    inhibitors: np.ndarray = field(default_factory=lambda: np.zeros(2))
    dragons: np.ndarray = field(default_factory=lambda: np.zeros(2))
    barons: np.ndarray = field(default_factory=lambda: np.zeros(2))
    grubs_heralds: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_baron_t: np.ndarray = field(default_factory=lambda: np.full(2, -1e9))
    last_elder_t: np.ndarray = field(default_factory=lambda: np.full(2, -1e9))
    prior: float = 0.5  # przedmeczowe P(wygrana niebieskich)
    core_logit: float = 0.0  # logit z modelu 'core' trenowanego na Oracle's Elixir (0 = brak)


def state_from_dict(d: dict) -> GameState:
    """Kontrakt dla adapterow zewnetrznych (OCR, GRID live, Oracle's Elixir).

    {"t": sek, "blue": {"gold": [5], "level": [5], "alive": [5]}, "red": {...},
     "kills": [b, r], "towers": [b, r], "inhibitors": [b, r], "dragons": [b, r],
     "barons": [b, r], "grubs_heralds": [b, r], "last_baron_t": [b, r],
     "last_elder_t": [b, r], "prior": 0.5}  # kolejnosc rol: Top, Jungle, Mid, ADC, Support
    Brakujace pola dostaja wartosci domyslne (np. alive=1, level=1).
    """
    gs = GameState(t=float(d["t"]), prior=float(d.get("prior", 0.5)))
    for i, side in enumerate(("blue", "red")):
        sd = d.get(side, {})
        for key, attr in (("gold", "gold"), ("level", "level"), ("alive", "alive")):
            if key in sd:
                getattr(gs, attr)[i] = np.asarray(sd[key], float)
    for key in ("kills", "towers", "inhibitors", "dragons", "barons", "grubs_heralds",
                "last_baron_t", "last_elder_t"):
        if key in d:
            setattr(gs, key, np.asarray(d[key], float))
    return gs


# cechy "podstawowe" (roznice); kazda wchodzi tez w interakcji z czasem
BASE_FEATURES = (
    *(f"gd_{r.lower()}" for r in ROLES),
    "gd_total", "carry_share", "level_diff", "kills", "towers", "inhibitors",
    "dragons", "soul", "barons", "baron_buff", "elder_buff", "grubs_heralds", "alive",
)
FEATURE_NAMES = ("prior_logit", "core_logit") + BASE_FEATURES + tuple(f"{f}*t" for f in BASE_FEATURES)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _carry_share(gold_side: np.ndarray) -> float:
    total = gold_side.sum()
    return float((gold_side[2] + gold_side[3]) / total) if total > 0 else 0.4


def base_features(gs: GameState) -> np.ndarray:
    gd = (gs.gold[0] - gs.gold[1]) / 1000.0
    d = lambda a: float(a[0] - a[1])  # noqa: E731
    active = lambda last, dur, s: float(0 <= gs.t - last[s] < dur)  # noqa: E731
    buff = lambda last, dur: active(last, dur, 0) - active(last, dur, 1)  # noqa: E731
    return np.array([
        *gd,
        gd.sum(),
        _carry_share(gs.gold[0]) - _carry_share(gs.gold[1]),
        float(gs.level[0].sum() - gs.level[1].sum()),
        d(gs.kills), d(gs.towers), d(gs.inhibitors),
        d(gs.dragons),
        float(gs.dragons[0] >= SOUL_DRAGONS) - float(gs.dragons[1] >= SOUL_DRAGONS),
        d(gs.barons),
        buff(gs.last_baron_t, BARON_BUFF_S),
        buff(gs.last_elder_t, ELDER_BUFF_S),
        d(gs.grubs_heralds),
        float(gs.alive[0].sum() - gs.alive[1].sum()),
    ])


def features(gs: GameState) -> np.ndarray:
    """Pelny wektor cech: [prior_logit, core_logit, base..., base * (t/30min)...]."""
    base = base_features(gs)
    tau = min(gs.t / 1800.0, 1.5)
    return np.concatenate([[_logit(gs.prior), gs.core_logit], base, base * tau])
