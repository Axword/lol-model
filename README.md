# lolbet

Model P(wygrana) w LoL: draft/Elo (prior) + stan gry live (gold per rola, objectives, czas) + value bety.

```
.venv\Scripts\python -m lolbet.cli train                       # trening na bazie lol_spy (tylko odczyt)
.venv\Scripts\python -m lolbet.cli replay 381 --odds-blue 1.9 --odds-red 1.9
.venv\Scripts\python -m lolbet.cli predict examples_state.json --odds-blue 3.4 --odds-red 1.32
```

Adapter zrodla (OCR / GRID live / Oracle's Elixir) ma zwrocic `GameState` (patrz `state_from_dict`).
Baza: `LOLSPY_DB` lub `--db`. Modele zapisuja sie w `models/`.

Stan: 118 gier z telemetria (Worlds), walidacja po grach: log-loss 0.576 vs 0.693 (moneta).
Prior z draftu jest wylaczony w modelu live (na tym zbiorze pogarszal walidacje).
Brak walidacji EV na prawdziwych kursach - to wymaga zalogowanych kursow live.

## Oracle's Elixir
`python scripts/download_oe.py` (ponawia co 2 min przy limicie Drive) -> `data/oe/<rok>.csv`.
`lolbet train` automatycznie uczy model "core" na OE (gold/level/kills per rola w 10/15/20/25 min)
i podaje jego logit jako cechę modelu live. Stan na 2026-11-06: tylko 2026.csv (9700 gier);
poza probka (po 2026-08-20) core: log-loss 0.468, trafnosc 76%.
Prior z draftu na OE: Elo 0.646, +draft 0.646 (draft nie dodaje nic ponad Elo).
