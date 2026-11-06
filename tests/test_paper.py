from lolbet.paper import Decision, simulate


def test_no_lookahead_and_bankroll_accounting():
    # rynek 50/50, model mowi 70% na niebieskich: zaklad na blue; przegrana zmniejsza bankroll
    ds = [Decision("g1", 1, 0.70, 2.0, 2.0, blue_win=0, settle_t=10),
          Decision("g2", 2, 0.70, 2.0, 2.0, blue_win=1, settle_t=11)]
    r = simulate(ds, bankroll=100.0)
    assert len(r.bets) == 2
    # g2 zostal postawiony zanim g1 sie rozliczyl: stawka liczona od gotowki pomniejszonej o g1
    assert r.bets[1].stake < r.bets[0].stake * 1.01
    assert r.final == 100 - r.bets[0].stake + r.bets[1].stake


def test_one_bet_per_game_and_skips_without_edge():
    ds = [Decision("g1", 1, 0.5, 2.0, 2.0, 1, 10), Decision("g2", 1, 0.8, 2.0, 2.0, 1, 10),
          Decision("g2", 2, 0.8, 2.0, 2.0, 1, 10)]
    r = simulate(ds)
    assert [b.game_id for b in r.bets] == ["g2"]
