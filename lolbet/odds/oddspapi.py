"""Klient OddsPapi (https://oddspapi.io, v4). Klucz: zmienna ODDSPAPI_KEY lub plik .env.

Darmowy plan ma ~250 zapytan/miesiac, wiec KAZDA odpowiedz jest cache'owana na dysku
(data/odds_cache) i drugi raz to samo zapytanie nie zuzywa limitu.
Format odpowiedzi nie byl jeszcze sprawdzony na zywym kluczu: parser powstaje po `odds-probe`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.oddspapi.io/v4"
LOL_SPORT_ID = 18
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "odds_cache"


def api_key() -> str:
    key = os.environ.get("ODDSPAPI_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ODDSPAPI_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("Brak ODDSPAPI_KEY (zmienna srodowiskowa lub linia w .env)")
    return key


def get(path: str, min_interval: float = 2.0, **params):
    """GET z cache na dysku; klucz nigdy nie trafia do nazwy pliku cache."""
    q = urllib.parse.urlencode(sorted(params.items()))
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(f"{path}?{q}".encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    url = f"{BASE}{path}?{q}&apiKey={urllib.parse.quote(api_key())}"
    time.sleep(min_interval)  # /historical-odds wymaga dluzszej przerwy miedzy wywolaniami
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OddsPapi {path} -> HTTP {e.code}") from None  # bez URL (zawiera klucz)
    f.write_text(json.dumps({"_request": f"{path}?{q}", "data": data}, ensure_ascii=False), encoding="utf-8")
    return {"_request": f"{path}?{q}", "data": data}


def fixtures(date_from: str, date_to: str, sport_id: int = LOL_SPORT_ID):
    return get("/fixtures", sportId=sport_id, **{"from": date_from, "to": date_to})


def odds(fixture_id: str):
    return get("/odds", fixtureId=fixture_id)


def historical(fixture_id: str, bookmakers: str = "pinnacle"):
    return get("/historical-odds", min_interval=6.0, fixtureId=fixture_id, bookmakers=bookmakers)
