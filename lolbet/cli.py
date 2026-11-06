"""CLI: train | replay | predict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import glob
import pickle

from . import data_oe as O, data_spy as d, db, live as L, market, prior as P
from .gamestate import state_from_dict

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


REPLAY_CUTOFF = "2026-08-20"  # replaye lol_spy sa pozniejsze; core do stackingu uczymy tylko na wczesniejszych


def train_core(files, replay_states_by_game) -> L.LiveModel:
    """Model core na OE: (1) wersja sprzed REPLAY_CUTOFF ustawia core_logit w stanach replayow
    (bez przecieku), (2) wersja koncowa na calym OE zapisuje sie do models/core.pkl."""
    G = O.load_oe(files)
    print(f"OE: {len(G)} gier z {G[0].date[:10]} do {G[-1].date[:10]}")
    X, y, gi, ts = O.checkpoint_dataset(G)
    dates = np.array([G[i].date for i in gi])
    tr = dates < REPLAY_CUTOFF
    if tr.sum() < 500:  # np. same nowe dane: uzyj calosci
        tr = np.ones(len(y), bool)
    te = ~tr
    mk = lambda: L.LiveModel(C=0.1, exclude=L.CORE_EXCLUDE)  # noqa: E731
    pre = mk().fit(X[tr], y[tr])
    if te.sum():
        p = pre.predict(X[te])
        print(f"core OE (poza probka, {te.sum()} wierszy): logloss={P.logloss(p, y[te]):.4f} "
              f"acc={np.mean((p > .5) == y[te]):.3f}")
    for states in replay_states_by_game.values():
        for st in states:
            st.core_logit = pre.logit_state(st)
    final = mk().fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    final.save(MODEL_DIR / "core.pkl")
    return final


def load_models():
    live = L.LiveModel.load(MODEL_DIR / "live.pkl")
    cp = MODEL_DIR / "core.pkl"
    return live, (L.LiveModel.load(cp) if cp.exists() and live.use_core else None)


def predict_p(live, core, gs) -> float:
    if core is not None:
        gs.core_logit = core.logit_state(gs)
    return live.predict_state(gs)


def cmd_train(a):
    conn = db.connect(a.db)
    games = d.load_games(conn)
    print(f"gry z wynikiem i draftem: {len(games)}")
    print("holdout prior (ostatnie 25% gier):", {k: round(v, 4) for k, v in P.holdout_report(games).items()})
    scales = P.tune_scales(games)
    print("skale (prior sd) po tuningu:", {k: round(v, 3) for k, v in scales.items()})
    print("holdout z tuningiem:", {k: round(v, 4) for k, v in P.holdout_report(games, scales=scales).items()})
    priors = P.oof_priors(games, scales)
    ids = d.replay_match_ids(conn)
    S = {i: d.replay_states(conn, i) for i in ids}
    W = {i: d.winner_of(conn, i) for i in ids}
    core = None
    oe_files = a.oe if a.oe is not None else sorted(glob.glob(str(MODEL_DIR.parent / "data" / "oe" / "*.csv")))
    if oe_files:
        core = train_core(oe_files, S)
    X, y, g, ts = L.build_dataset(S, W, priors)
    print(f"model live: {len(ids)} gier, {len(y)} stanow")
    pr = L.cross_validate(X, y, g, ts, use_core=core is not None)
    rep = L.report(pr, y, ts, X[:, 0])
    print("log-loss (walidacja po grach):", {k: round(v, 4) for k, v in rep.items()})
    best = min(pr, key=lambda C: L.logloss(pr[C], y))
    print(f"najlepsze C={best}")
    for r in L.by_time(pr[best], y, ts):
        print("  %-9s n=%-5d logloss=%.3f acc=%.2f" % r)
    print("kalibracja (przedzial, n, srednia pred., czestosc):")
    for r in L.calibration(pr[best], y):
        print("  %.2f-%.2f n=%-5d pred=%.3f real=%.3f" % r)
    MODEL_DIR.mkdir(exist_ok=True)
    L.LiveModel(C=best, use_core=core is not None).fit(X, y).save(MODEL_DIR / "live.pkl")
    with open(MODEL_DIR / "prior.pkl", "wb") as f:
        pickle.dump(P.fit_final(games, scales), f)
    print("zapisano", MODEL_DIR / "live.pkl", "i prior.pkl")


def cmd_replay(a):
    conn = db.connect(a.db)
    model, core = load_models()
    states = d.replay_states(conn, a.match_id, step=a.step)
    win = d.winner_of(conn, a.match_id)
    print(f"mecz {a.match_id}, wygral: {'blue' if win else 'red'} "
          "(UWAGA: gra jest w zbiorze treningowym - wynik zawyzony)")
    for s in states:
        p = predict_p(model, core, s)
        line = f"{int(s.t // 60):2d}:{int(s.t % 60):02d}  P(blue)={p:.3f}  gold_diff={s.gold[0].sum() - s.gold[1].sum():+7.0f}"
        if a.odds_blue and a.odds_red:
            sig = market.evaluate(p, a.odds_blue, a.odds_red)
            if sig:
                line += f"  -> {sig.side} @{sig.odds} edge={sig.edge:+.3f} ev={sig.ev:+.3f} stake={sig.stake:.3f}"
        print(line)


def cmd_predict(a):
    model, core = load_models()
    gs = state_from_dict(json.loads(Path(a.state).read_text(encoding="utf-8")))
    p = predict_p(model, core, gs)
    print(f"P(blue)={p:.3f}  P(red)={1 - p:.3f}")
    if a.odds_blue and a.odds_red:
        sig = market.evaluate(p, a.odds_blue, a.odds_red, trust=a.trust)
        print(sig if sig else "brak value (edge/EV ponizej progu)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lolbet")
    ap.add_argument("--db", help="sciezka do bazy lol_spy")
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--oe", nargs="*", help="pliki CSV Oracle's Elixir (domyslnie data/oe/*.csv)")
    r = sub.add_parser("replay"); r.set_defaults(fn=cmd_replay)
    r.add_argument("match_id", type=int); r.add_argument("--step", type=float, default=120)
    r.add_argument("--odds-blue", type=float); r.add_argument("--odds-red", type=float)
    p = sub.add_parser("predict"); p.set_defaults(fn=cmd_predict)
    p.add_argument("state", help="plik JSON ze stanem gry"); p.add_argument("--trust", type=float, default=0.5)
    p.add_argument("--odds-blue", type=float); p.add_argument("--odds-red", type=float)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
