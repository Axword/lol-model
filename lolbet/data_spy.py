"""Loadery danych z bazy lol_spy (tylko odczyt)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gamestate import ROLES, GameState

_ROLE_IDX = {r: i for i, r in enumerate(ROLES)}


@dataclass
class DraftGame:
    id: int
    date: str
    blue_team: int
    red_team: int
    blue_win: int
    picks: dict  # {(side 0/1, role_idx or None): champion_id}; bez roli: (side, -k)


def load_games(conn: sqlite3.Connection) -> list[DraftGame]:
    """Wszystkie gry z wynikiem, chronologicznie, z pickami wg roli."""
    m = pd.read_sql_query(
        "select id, date, winner, team_blueside_id b, team_redside_id r from matches_match "
        "where winner in ('blue','red') order by date, id", conn)
    p = pd.read_sql_query(
        'select match_id, side, role, "order" ord, champion_id from matches_matchchampionpick', conn)
    by_match = {k: g for k, g in p.groupby("match_id")}
    games = []
    for row in m.itertuples():
        g = by_match.get(row.id)
        if g is None or len(g) != 10:
            continue
        picks = {}
        for pk in g.itertuples():
            side = 0 if pk.side == "blue" else 1
            ridx = _ROLE_IDX.get(pk.role)
            key = (side, ridx) if ridx is not None else (side, -1 - int(pk.ord))
            picks[key] = int(pk.champion_id)
        games.append(DraftGame(row.id, row.date, int(row.b), int(row.r),
                               int(row.winner == "blue"), picks))
    return games


def replay_match_ids(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute(
        "select r.match_id from dynamic_matches_replayimport r "
        "join matches_match m on m.id=r.match_id where m.winner in ('blue','red') order by m.date")]


def replay_states(conn: sqlite3.Connection, match_id: int, step: float = 30.0,
                  start: float = 120.0) -> list[GameState]:
    """Stany gry co `step` sekund z zapisanej telemetrii (bez prioru: prior=0.5)."""
    snap = pd.read_sql_query(
        "select timestamp t, participant_id pid, total_gold g, level l, alive a, kills k "
        "from dynamic_matches_playersnapshot where match_id=? order by timestamp",
        conn, params=(match_id,))
    if snap.empty:
        return []
    obj = pd.read_sql_query(
        "select timestamp t, objective_type o, team from dynamic_matches_objectivekill "
        "where match_id=? order by timestamp", conn, params=(match_id,))
    end = snap.t.max()
    states = []
    # ostatnia obserwacja <= cel: indeks wg znacznika czasu per gracz
    piv = {c: snap.pivot_table(index="t", columns="pid", values=c, aggfunc="last")
           for c in ("g", "l", "a", "k")}
    times = piv["g"].index.to_numpy()
    for tt in np.arange(start, end, step):
        i = np.searchsorted(times, tt, side="right") - 1
        if i < 0:
            continue
        gs = GameState(t=float(tt))
        for c, attr in (("g", "gold"), ("l", "level"), ("a", "alive")):
            row = piv[c].iloc[i]
            if row.isna().any() or len(row) != 10:
                break
            setattr(gs, attr, row.to_numpy(float).reshape(2, 5))
        else:
            kills = piv["k"].iloc[i].to_numpy(float)
            gs.kills = np.array([kills[:5].sum(), kills[5:].sum()])
            _apply_objectives(gs, obj[obj.t <= tt])
            states.append(gs)
    return states


def _apply_objectives(gs: GameState, obj: pd.DataFrame) -> None:
    """`team` w objectivekill = druzyna zabijajaca (blue/red)."""
    for r in obj.itertuples():
        s = 0 if r.team == "blue" else 1
        o = r.o
        if o.startswith("tower"):
            gs.towers[s] += 1
        elif o == "inhibitor":
            gs.inhibitors[s] += 1
        elif o == "dragon_elder":
            gs.dragons[s] += 1
            gs.last_elder_t[s] = r.t
        elif o.startswith("dragon"):
            gs.dragons[s] += 1
        elif o == "baron":
            gs.barons[s] += 1
            gs.last_baron_t[s] = r.t
        elif o in ("herald", "void_grub"):
            gs.grubs_heralds[s] += 1


def winner_of(conn: sqlite3.Connection, match_id: int) -> int:
    w = conn.execute("select winner from matches_match where id=?", (match_id,)).fetchone()[0]
    return int(w == "blue")


def participants(conn: sqlite3.Connection, match_id: int) -> dict:
    raw = conn.execute("select participant_metadata from dynamic_matches_replayimport where match_id=?",
                       (match_id,)).fetchone()
    return json.loads(raw[0]) if raw else {}
