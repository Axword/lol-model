"""Kursy -> value bety: de-vig, korekta zaufania do modelu, EV i ulamkowy Kelly."""
from __future__ import annotations

from dataclasses import dataclass


def devig(odds_blue: float, odds_red: float) -> float:
    """Uczciwe P(blue) z kursow dziesietnych (proporcjonalne usuniecie marzy)."""
    ib, ir = 1 / odds_blue, 1 / odds_red
    return ib / (ib + ir)


@dataclass
class Signal:
    side: str  # "blue" | "red"
    odds: float
    p_model: float  # surowe P modelu dla tej strony
    p_adj: float  # P po sciagnieciu ku rynkowi (trust)
    p_market: float  # uczciwe P rynku (po de-vig)
    edge: float  # p_adj - 1/odds
    ev: float  # p_adj*odds - 1
    stake: float  # ulamek bankrolla (ulamkowy Kelly, z limitem)
    upset: bool  # strona, na ktora rynek daje <50%


def evaluate(p_blue: float, odds_blue: float, odds_red: float, *, trust: float = 0.5,
             min_edge: float = 0.04, kelly_mult: float = 0.25, max_stake: float = 0.03) -> Signal | None:
    """Zwraca najlepszy sygnal albo None.

    trust: waga modelu wzgledem rynku (0 = ufamy rynkowi, 1 = modelowi). Model z ~100 gier
    nie powinien dostawac 1.0; p_adj = p_rynku + trust*(p_modelu - p_rynku).
    """
    p_mkt_blue = devig(odds_blue, odds_red)
    best = None
    for side, p_m, p_mk, odds in (("blue", p_blue, p_mkt_blue, odds_blue),
                                  ("red", 1 - p_blue, 1 - p_mkt_blue, odds_red)):
        p_adj = p_mk + trust * (p_m - p_mk)
        edge, ev = p_adj - 1 / odds, p_adj * odds - 1
        if edge < min_edge or ev <= 0:
            continue
        stake = min(max_stake, kelly_mult * ev / (odds - 1))
        sig = Signal(side, odds, p_m, p_adj, p_mk, edge, ev, stake, p_mk < 0.5)
        if best is None or sig.ev > best.ev:
            best = sig
    return best
