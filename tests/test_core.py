import numpy as np
import pytest

from lolbet import market
from lolbet.gamestate import FEATURE_NAMES, features, state_from_dict
from lolbet.data_spy import DraftGame
from lolbet.prior import compute_elo, DraftPrior


def test_devig_sums_to_one():
    p = market.devig(1.8, 2.0)
    assert 0.5 < p < 0.6
    assert market.devig(2.0, 2.0) == pytest.approx(0.5)


def test_evaluate_finds_upset_value_and_respects_threshold():
    # rynek ~25% na niebieskich, model 45%: value na niebieskich (upset)
    sig = market.evaluate(0.45, 3.6, 1.30, trust=1.0)
    assert sig and sig.side == "blue" and sig.upset and sig.ev > 0
    assert market.evaluate(0.27, 3.6, 1.30, trust=1.0) is None  # za maly edge
    assert market.evaluate(0.50, 2.0, 2.0) is None  # zgodne z rynkiem


def test_features_antisymmetric_under_side_swap():
    d = {"t": 1200, "blue": {"gold": [8000, 7000, 9000, 10000, 5000], "level": [11, 10, 12, 12, 9]},
         "red": {"gold": [7000, 7000, 8000, 8500, 5200], "level": [10, 10, 11, 11, 9]},
         "kills": [9, 5], "towers": [3, 1], "dragons": [2, 1], "last_baron_t": [1100, -1e9]}
    sw = {"t": 1200, "blue": d["red"], "red": d["blue"], "kills": [5, 9], "towers": [1, 3],
          "dragons": [1, 2], "last_baron_t": [-1e9, 1100]}
    a, b = features(state_from_dict(d)), features(state_from_dict(sw))
    i = FEATURE_NAMES.index("carry_share")
    mask = np.array([n != "prior_logit" for n in FEATURE_NAMES])
    assert np.allclose(a[mask], -b[mask], atol=1e-9), (a[mask] + b[mask])


def test_elo_rewards_winner():
    g = [DraftGame(1, "2026-01-01", 1, 2, 1, {}), DraftGame(2, "2026-01-02", 1, 2, 1, {})]
    pre, final = compute_elo(g)
    assert pre[1] == (1500.0, 1500.0) and pre[2][0] > pre[2][1] and final[1] > final[2]
