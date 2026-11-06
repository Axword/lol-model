"""Dostep tylko-do-odczytu do bazy lol_spy."""
import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(r"C:\programowanie\lol_spy\data\sqlite3-20261101-121705.db")


def connect(path: str | os.PathLike | None = None) -> sqlite3.Connection:
    p = Path(path or os.environ.get("LOLSPY_DB") or DEFAULT_DB)
    if not p.exists():
        raise FileNotFoundError(f"Brak bazy lol_spy: {p} (ustaw LOLSPY_DB)")
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
