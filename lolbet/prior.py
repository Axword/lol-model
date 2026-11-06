"""Model przedmeczowy: sila druzyn (Elo) + draft (championy, matchupy, duo).

Regresja logistyczna na rzadkich cechach. Wagi grup cech sa realizowane przez skale kolumn:
skala = odchylenie standardowe a priori efektu w log-odds (L2 z C=1), czyli im mniejsza, tym
mocniejszy shrinkage ku zeru. Rzadkie pary (duo vs duo) dostaja male skale, wiec zbliżaja sie
do 0, a frekwentne efekty (champion) moga odejsc dalej - bez recznych progow "min. N gier".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from .data_spy import DraftGame

TOP, JG, MID, ADC, SUP = range(5)
DEFAULT_SCALES = {
    "elo": 3.0, "champ": 0.30, "matchup": 0.15, "bot_duo": 0.20,
    "bot_vs_bot": 0.10, "mid_jg": 0.15, "mid_top": 0.12,
}


def compute_elo(games: list[DraftGame], k: float = 24.0, base: float = 1500.0):
    """Chronologicznie: Elo przed kazda gra {id: (blue, red)} oraz koncowe oceny."""
    rating: dict[int, float] = {}
    pre = {}
    for g in games:
        rb, rr = rating.get(g.blue_team, base), rating.get(g.red_team, base)
        pre[g.id] = (rb, rr)
        exp = 1 / (1 + 10 ** ((rr - rb) / 400))
        delta = k * (g.blue_win - exp)
        rating[g.blue_team], rating[g.red_team] = rb + delta, rr - delta
    return pre, rating


def _team_keys(g: DraftGame, side: int) -> dict[str, list]:
    p = g.picks
    c = lambda r: p.get((side, r))  # noqa: E731
    keys = {"champ": [], "bot_duo": [], "mid_jg": [], "mid_top": []}
    for r in range(5):
        if c(r) is not None:
            keys["champ"].append((r, c(r)))
    for name, a, b in (("bot_duo", ADC, SUP), ("mid_jg", MID, JG), ("mid_top", MID, TOP)):
        if c(a) is not None and c(b) is not None:
            keys[name].append((c(a), c(b)))
    # picki bez znanej roli: sam champion jako "any"
    for (s, r), champ in p.items():
        if s == side and r < 0:
            keys["champ"].append((-1, champ))
    return keys


def _signed_keys(g: DraftGame) -> list[tuple[str, tuple, float]]:
    """Lista (grupa, klucz, znak). Znak +1 gdy efekt przypisany stronie niebieskiej."""
    out = []
    kb, kr = _team_keys(g, 0), _team_keys(g, 1)
    for grp in ("champ", "bot_duo", "mid_jg", "mid_top"):
        out += [(grp, k, 1.0) for k in kb[grp]]
        out += [(grp, k, -1.0) for k in kr[grp]]
    # matchupy tej samej linii: kanonicznie (a<b), znak zalezy od tego, kto ma mniejsze id
    for r in range(5):
        a, b = g.picks.get((0, r)), g.picks.get((1, r))
        if a is not None and b is not None:
            out.append(("matchup", (r,) + (a, b) if a < b else (r, b, a), 1.0 if a < b else -1.0))
    # duo bot vs duo bot
    db_, dr_ = kb["bot_duo"], kr["bot_duo"]
    if db_ and dr_:
        a, b = db_[0], dr_[0]
        out.append(("bot_vs_bot", (a, b) if a < b else (b, a), 1.0 if a < b else -1.0))
    return out


@dataclass
class DraftPrior:
    scales: dict = field(default_factory=lambda: dict(DEFAULT_SCALES))
    vocab: dict = field(default_factory=dict)
    model: LogisticRegression | None = None
    ratings: dict = field(default_factory=dict)

    def _matrix(self, games, elo_pre, fit=False):
        rows, cols, vals = [], [], []
        if fit:
            self.vocab = {}
        n_extra = 1  # kolumna 0 = elo
        for i, g in enumerate(games):
            eb, er = elo_pre[g.id]
            rows.append(i); cols.append(0); vals.append((eb - er) / 400.0 * self.scales["elo"])
            for grp, key, sign in _signed_keys(g):
                vk = (grp, key)
                if fit and vk not in self.vocab:
                    self.vocab[vk] = n_extra + len(self.vocab)
                j = self.vocab.get(vk)
                if j is not None:
                    rows.append(i); cols.append(j); vals.append(sign * self.scales[grp])
        width = n_extra + len(self.vocab)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(games), width))

    def fit(self, games: list[DraftGame], elo_pre: dict) -> "DraftPrior":
        X = self._matrix(games, elo_pre, fit=True)
        y = np.array([g.blue_win for g in games])
        self.model = LogisticRegression(C=1.0, max_iter=500).fit(X, y)
        return self

    def predict(self, games: list[DraftGame], elo_pre: dict) -> np.ndarray:
        return self.model.predict_proba(self._matrix(games, elo_pre))[:, 1]

    def predict_draft(self, g: DraftGame, ratings: dict | None = None) -> float:
        r = ratings if ratings is not None else self.ratings
        pre = {g.id: (r.get(g.blue_team, 1500.0), r.get(g.red_team, 1500.0))}
        return float(self.predict([g], pre)[0])


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def holdout_report(games: list[DraftGame], frac: float = 0.75, scales: dict | None = None) -> dict:
    """Trening na pierwszych `frac` gier (chronologicznie), ocena na reszcie."""
    elo_pre, _ = compute_elo(games)
    cut = int(len(games) * frac)
    tr, te = games[:cut], games[cut:]
    y = np.array([g.blue_win for g in te])
    res = {"n_train": len(tr), "n_test": len(te), "baseline_0.5": logloss(np.full(len(te), .5), y)}
    for name, sc in (("elo_only", {**DEFAULT_SCALES, **{k: 1e-6 for k in DEFAULT_SCALES if k != "elo"}}),
                     ("full", scales or DEFAULT_SCALES)):
        m = DraftPrior(scales=sc).fit(tr, elo_pre)
        p = m.predict(te, elo_pre)
        res[name] = logloss(p, y)
        res[f"{name}_acc"] = float(np.mean((p > .5) == y))
    return res


def tune_scales(games: list[DraftGame], iters: int = 25, seed: int = 0) -> dict:
    """Losowe przeszukiwanie skal (priorow) na pierwszych 75% gier, walidacja na ostatniej 1/3 tej czesci."""
    rng = np.random.default_rng(seed)
    elo_pre, _ = compute_elo(games)
    cut = int(len(games) * 0.75)
    inner = int(cut * 0.67)
    tr, va = games[:inner], games[inner:cut]
    y = np.array([g.blue_win for g in va])
    best, best_ll = dict(DEFAULT_SCALES), math.inf
    cands = [dict(DEFAULT_SCALES)] + [
        {k: v * float(np.exp(rng.normal(0, 0.7))) if k != "elo" else v for k, v in DEFAULT_SCALES.items()}
        for _ in range(iters)]
    for sc in cands:
        ll = logloss(DraftPrior(scales=sc).fit(tr, elo_pre).predict(va, elo_pre), y)
        if ll < best_ll:
            best, best_ll = sc, ll
    return best


def oof_priors(games: list[DraftGame], scales: dict | None = None, blocks: int = 6) -> dict[int, float]:
    """Prior dla kazdej gry z modelu nauczonego na pozostalych blokach (bez przecieku wlasnej gry)."""
    elo_pre, _ = compute_elo(games)
    n = len(games)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    out: dict[int, float] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        te = games[a:b]
        tr = games[:a] + games[b:]
        m = DraftPrior(scales=scales or DEFAULT_SCALES).fit(tr, elo_pre)
        for g, p in zip(te, m.predict(te, elo_pre)):
            out[g.id] = float(p)
    return out


def fit_final(games: list[DraftGame], scales: dict | None = None) -> DraftPrior:
    elo_pre, ratings = compute_elo(games)
    m = DraftPrior(scales=scales or DEFAULT_SCALES).fit(games, elo_pre)
    m.ratings = ratings
    return m
