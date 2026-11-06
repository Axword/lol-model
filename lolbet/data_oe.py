"""Loader Oracle's Elixir: wiersze per gracz -> gry z punktami kontrolnymi 10/15/20/25 min.

OE daje per gracz gold/xp/cs/kills w tych czterech momentach, ale NIE daje objectives w czasie
(tylko sumy koncowe), wiec stany z OE maja zerowe objectives i sluza do modelu "core".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_spy import DraftGame
from .gamestate import GameState

CHECKPOINTS = (10, 15, 20, 25)
POS = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
# skumulowane XP wymagane na poziom 1..18 (stale, nie zalezy od patcha)
XP_TO_LEVEL = np.array([0, 280, 660, 1140, 1720, 2400, 3180, 4060, 5040, 6120,
                        7300, 8580, 9960, 11440, 13020, 14700, 16480, 18360], float)


def level_from_xp(xp: np.ndarray) -> np.ndarray:
    """Poziom z lacznego XP (z interpolacja w obrebie poziomu, max 18)."""
    xp = np.asarray(xp, float)
    lvl = np.searchsorted(XP_TO_LEVEL, xp, side="right").astype(float)  # 1..18
    return np.clip(lvl, 1, 18)


class OEGame:
    """Jedna gra: metadane + tablice [strona, rola, checkpoint]."""
    __slots__ = ("gameid", "date", "league", "patch", "blue_team", "red_team", "blue_win",
                 "gold", "xp", "cs", "kills", "champs", "length")

    def to_state(self, ci: int) -> GameState | None:
        g, x, k = self.gold[:, :, ci], self.xp[:, :, ci], self.kills[:, :, ci]
        if np.isnan(g).any() or np.isnan(x).any():
            return None
        gs = GameState(t=CHECKPOINTS[ci] * 60.0)
        gs.gold = g.copy()
        gs.level = level_from_xp(x)
        gs.kills = np.nan_to_num(k).sum(axis=1)
        return gs


def load_oe(paths: list[str | Path], only_complete: bool = True) -> list[OEGame]:
    cols = ["gameid", "datacompleteness", "league", "date", "patch", "side", "position",
            "teamid", "champion", "result", "gamelength"]
    for m in CHECKPOINTS:
        cols += [f"goldat{m}", f"xpat{m}", f"csat{m}", f"killsat{m}"]
    frames = [pd.read_csv(p, usecols=cols, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.position.isin(POS)]
    if only_complete:
        df = df[df.datacompleteness == "complete"]
    games = []
    for gid, g in df.groupby("gameid", sort=False):
        if len(g) != 10 or set(g.side) != {"Blue", "Red"}:
            continue
        og = OEGame()
        og.gameid, og.date, og.league, og.patch = gid, str(g.date.iloc[0]), g.league.iloc[0], str(g.patch.iloc[0])
        og.length = float(g.gamelength.iloc[0])
        shape = (2, 5, len(CHECKPOINTS))
        og.gold, og.xp, og.cs, og.kills = (np.full(shape, np.nan) for _ in range(4))
        og.champs = {}
        for r in g.itertuples(index=False):
            s, p = (0 if r.side == "Blue" else 1), POS[r.position]
            og.champs[(s, p)] = r.champion
            for ci, m in enumerate(CHECKPOINTS):
                og.gold[s, p, ci] = getattr(r, f"goldat{m}")
                og.xp[s, p, ci] = getattr(r, f"xpat{m}")
                og.cs[s, p, ci] = getattr(r, f"csat{m}")
                og.kills[s, p, ci] = getattr(r, f"killsat{m}")
        blue = g[g.side == "Blue"]
        og.blue_team, og.red_team = str(blue.teamid.iloc[0]), str(g[g.side == "Red"].teamid.iloc[0])
        og.blue_win = int(blue.result.iloc[0])
        games.append(og)
    games.sort(key=lambda x: (x.date, x.gameid))
    return games


def to_draft_games(games: list[OEGame]) -> list[DraftGame]:
    """OEGame -> DraftGame (id calkowity, champion/team jako id liczbowe przez slownik)."""
    champ_ids: dict[str, int] = {}
    team_ids: dict[str, int] = {}
    out = []
    for i, g in enumerate(games):
        picks = {(s, p): champ_ids.setdefault(c, len(champ_ids)) for (s, p), c in g.champs.items()}
        out.append(DraftGame(i, g.date, team_ids.setdefault(g.blue_team, len(team_ids)),
                             team_ids.setdefault(g.red_team, len(team_ids)), g.blue_win, picks))
    return out


def checkpoint_dataset(games: list[OEGame], priors: dict[int, float] | None = None):
    """Wiersze (stan na checkpoincie) -> X, y, indeks gry, minuta. priors: {indeks gry: P}."""
    from .gamestate import features
    X, y, gi, ts = [], [], [], []
    for i, g in enumerate(games):
        for ci in range(len(CHECKPOINTS)):
            s = g.to_state(ci)
            if s is None:
                continue
            if priors is not None:
                s.prior = priors.get(i, 0.5)
            X.append(features(s)); y.append(g.blue_win); gi.append(i); ts.append(s.t)
    return np.array(X), np.array(y), np.array(gi), np.array(ts)
