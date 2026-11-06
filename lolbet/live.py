"""Model live: P(wygrana niebieskich | stan gry, czas, prior z draftu).

Regresja logistyczna na cechach-roznicach (niebiescy - czerwoni) + przewaga strony.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .gamestate import FEATURE_NAMES, GameState, features
from .prior import logloss


# cechy niedostepne w Oracle's Elixir (brak objectives i zywych graczy w czasie)
CORE_EXCLUDE = ("towers", "inhibitors", "dragons", "soul", "barons", "baron_buff", "elder_buff",
                "grubs_heralds", "alive")


def build_dataset(states_by_game: dict[int, list[GameState]], winners: dict[int, int],
                  priors: dict[int, float]):
    X, y, groups, ts = [], [], [], []
    for gid, states in states_by_game.items():
        for s in states:
            s.prior = priors.get(gid, 0.5)
            X.append(features(s)); y.append(winners[gid]); groups.append(gid); ts.append(s.t)
    return np.array(X), np.array(y), np.array(groups), np.array(ts)


@dataclass
class LiveModel:
    """Regresja logistyczna ze skalowaniem cech i wyrazem wolnym (przewaga strony niebieskiej).

    use_prior=False domyslnie: na 118 grach (Worlds, druzyny miedzy regionami) prior z draftu
    pogarszal walidacje; wlacz po treningu na wiekszym zbiorze (Oracle's Elixir).
    """
    C: float = 0.01
    use_prior: bool = False
    use_core: bool = False  # cecha core_logit (model z OE)
    exclude: tuple = ()  # nazwy bazowych cech do pominiecia (z wariantem *t)
    scaler: StandardScaler | None = None
    clf: LogisticRegression | None = None

    @property
    def cols(self) -> list[int]:
        def keep(n):
            if n == "prior_logit":
                return self.use_prior
            if n == "core_logit":
                return self.use_core
            return n.replace("*t", "") not in self.exclude
        return [i for i, n in enumerate(FEATURE_NAMES) if keep(n)]

    def fit(self, X, y):
        Xc = X[:, self.cols]
        self.scaler = StandardScaler(with_mean=False).fit(Xc)
        self.clf = LogisticRegression(C=self.C, max_iter=3000)
        self.clf.fit(self.scaler.transform(Xc), y)
        return self

    def predict(self, X) -> np.ndarray:
        Xc = np.atleast_2d(X)[:, self.cols]
        return self.clf.predict_proba(self.scaler.transform(Xc))[:, 1]

    def logit_state(self, gs: GameState) -> float:
        Xc = np.atleast_2d(features(gs))[:, self.cols]
        return float(self.clf.decision_function(self.scaler.transform(Xc))[0])

    def predict_state(self, gs: GameState) -> float:
        return float(self.predict(features(gs))[0])

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "LiveModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def cross_validate(X, y, groups, ts, Cs=(0.001, 0.003, 0.01, 0.03, 0.1), folds: int = 5,
                   use_core: bool = False) -> dict:
    """Walidacja grupowa (cala gra w jednym foldzie). Zwraca log-loss per C i per przedzial czasu."""
    gkf = GroupKFold(n_splits=folds)
    out = {}
    for C in Cs:
        pred = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            pred[te] = LiveModel(C=C, use_core=use_core).fit(X[tr], y[tr]).predict(X[te])
        out[C] = pred
    return out


def report(pred_by_C: dict, y, ts, prior_logit) -> dict:
    prior_p = 1 / (1 + np.exp(-prior_logit))
    rep = {"prior_only": logloss(prior_p, y), "coinflip": logloss(np.full(len(y), .5), y)}
    for C, p in pred_by_C.items():
        rep[f"C={C}"] = logloss(p, y)
    return rep


def by_time(pred, y, ts, bins=(0, 600, 900, 1200, 1500, 1800, 5000)) -> list[tuple]:
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (ts >= a) & (ts < b)
        if m.sum():
            rows.append((f"{a//60}-{b//60}min", int(m.sum()), logloss(pred[m], y[m]),
                         float(np.mean((pred[m] > .5) == y[m]))))
    return rows


def calibration(pred, y, nbins: int = 8) -> list[tuple]:
    edges = np.linspace(0, 1, nbins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pred >= a) & (pred < b if b < 1 else pred <= b)
        if m.sum():
            rows.append((round(a, 2), round(b, 2), int(m.sum()), float(pred[m].mean()), float(y[m].mean())))
    return rows
