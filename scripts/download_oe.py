"""Pobiera pliki Oracle's Elixir z Google Drive; przy limicie Drive ponawia co 2 minuty.

Uruchomienie (niezalezny proces):  python scripts/download_oe.py [2023 2024 ...]
Plik zapisuje sie dopiero, gdy to prawdziwy CSV (naglowek zaczyna sie od "gameid").
Log: data/oe/download.log. Pliki juz pobrane sa pomijane, wiec mozna uruchomic ponownie.
"""
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

IDS = {
    2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "oe"
RETRY_S = 120


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(OUT / "download.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def try_download(year: int) -> bool:
    url = f"https://drive.usercontent.google.com/download?id={IDS[year]}&export=download&confirm=t"
    part = OUT / f"{year}.part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            head = r.read(200)
            if not head.startswith(b"gameid"):
                log(f"{year}: Drive zwraca strone bledu (limit) - ponowie")
                return False
            f.write(head)
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as e:  # siec, timeout itp.
        log(f"{year}: blad {e!r} - ponowie")
        return False
    part.replace(OUT / f"{year}.csv")
    log(f"{year}: OK, {(OUT / f'{year}.csv').stat().st_size} B")
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    years = [int(a) for a in sys.argv[1:]] or sorted(IDS)
    while True:
        todo = [y for y in years if not (OUT / f"{y}.csv").exists()]
        if not todo:
            log("GOTOWE")
            return
        for y in todo:
            try_download(y)
        if any(not (OUT / f"{y}.csv").exists() for y in years):
            time.sleep(RETRY_S)


if __name__ == "__main__":
    main()
