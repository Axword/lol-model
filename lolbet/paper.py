"""Symulacja bez podgladu wyniku: wirtualny bankroll (domyslnie 100 zl), zaklady wg market.evaluate.

Brak look-ahead: decyzja w chwili t widzi tylko p_blue i kursy z chwili t; wynik gry wchodzi do
bankrolla dopiero w chwili rozliczenia (settle_t). Stawka = ulamek bankrolla pomniejszonego o
stawki jeszcze nierozliczonych zakladow. Maks. jeden zaklad na gre (pierwszy sygnal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import market


@dataclass
class Decision:
    game_id: str
    t: float  # moment decyzji (sekundy epoch lub dowolna os rosnaca)
    p_blue: float  # prawdopodobienstwo modelu (musi pochodzic z modelu nieznajacego wyniku tej gry)
    odds_blue: float
    odds_red: float
    blue_win: int  # wynik; uzywany dopiero przy rozliczeniu
    settle_t: float  # moment rozliczenia (koniec gry)


@dataclass
class Bet:
    game_id: str
    t: float
    side: str
    odds: float
    stake: float
    p_adj: float
    p_market: float
    settle_t: float
    won: bool
    pnl: float


@dataclass
class Report:
    start: float
    final: float
    bets: list[Bet] = field(default_factory=list)
    max_drawdown: float = 0.0

    @property
    def staked(self) -> float:
        return sum(b.stake for b in self.bets)

    @property
    def roi(self) -> float:
        return (self.final - self.start) / self.staked if self.staked else 0.0

    def summary(self) -> str:
        n = len(self.bets)
        hit = sum(b.won for b in self.bets) / n if n else 0.0
        return (f"zakladow={n} trafionych={hit:.1%} obrot={self.staked:.2f} "
                f"bankroll {self.start:.2f} -> {self.final:.2f} "
                f"ROI/obrot={self.roi:+.1%} max_obsuniecie={self.max_drawdown:.1%}")


def simulate(decisions: list[Decision], bankroll: float = 100.0, min_stake: float = 2.0,
             **eval_kwargs) -> Report:
    rep = Report(start=bankroll, final=bankroll)
    cash = bankroll
    open_bets: list[Bet] = []
    seen: set[str] = set()
    peak = bankroll

    def settle(upto: float):
        nonlocal cash, peak
        for b in sorted([b for b in open_bets if b.settle_t <= upto], key=lambda b: b.settle_t):
            cash += b.stake + b.pnl
            open_bets.remove(b)
            peak = max(peak, cash)
            rep.max_drawdown = max(rep.max_drawdown, (peak - cash) / peak)

    for d in sorted(decisions, key=lambda x: x.t):
        settle(d.t)
        if d.game_id in seen:
            continue
        sig = market.evaluate(d.p_blue, d.odds_blue, d.odds_red, **eval_kwargs)
        if sig is None:
            continue
        stake = round(sig.stake * cash, 2)
        if stake < min_stake or stake > cash:
            continue
        seen.add(d.game_id)
        won = (sig.side == "blue") == bool(d.blue_win)
        pnl = stake * (sig.odds - 1) if won else -stake
        cash -= stake
        b = Bet(d.game_id, d.t, sig.side, sig.odds, stake, sig.p_adj, sig.p_market, d.settle_t, won, pnl)
        open_bets.append(b)
        rep.bets.append(b)
    settle(float("inf"))
    rep.final = cash
    return rep
