#!/usr/bin/env python3
"""soccer_selftest.py: `python soccer_system.py selftest`. Everything runs offline against soccer_fake.World.

What a PASS means: the mechanics are right (request budget, caching, keys, sanity checks, de-margining, the
one-way cap, every settlement rule on a hand-written match, slips rules, weights that stay at 0 against a book
that is exactly right). What it does NOT mean: that API-Football's live responses or a real vegas.hu board look
like the simulated ones.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="soccer_selftest_"))
    os.environ.update(SOCCER_CACHE=str(tmp / "cache"), SOCCER_OUT=str(tmp / "data"), SOCCER_SLEEP="0", SOCCER_DB=str(tmp / "ledger/s.sqlite"),
                      SOCCER_FD=str(tmp / "fd"))
    os.environ.pop("PROLYT_API_KEY", None)
    import soccer_fake as K
    import soccer_bandit as SB
    import soccer_ingest as I
    import soccer_model as Mo
    import soccer_system as S
    I.CACHE, I.OUT, I.SLEEP, S.OUT, S.DB, SB.FD = tmp / "cache", tmp / "data", 0.0, tmp / "data", str(tmp / "ledger/s.sqlite"), tmp / "fd"
    checks: list = []
    ok = lambda name, cond, info="": checks.append((name, bool(cond), info))
    def run_quiet(fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = fn(*a, **k)
        return res, buf.getvalue()

    # ---------------------------------------------------------------- A. arithmetic
    lf = Mo.line_fracs
    ok("line: 3 goals, over 2.5 won / over 3 push / over 2.75 half won / over 3.25 half lost / under 3.5 won",
       [lf(3, 2.5, "over"), lf(3, 3, "over"), lf(3, 2.75, "over"), lf(3, 3.25, "over"), lf(3, 3.5, "under")] == [(1, 0), (0, 0), (.5, 0), (0, .5), (1, 0)])
    p, over = S.demargin([2.0, 3.5, 3.8])
    ok("de-margin 1X2 sums to 1 and keeps the order", abs(sum(p) - 1) < 1e-9 and p[0] > p[1] > p[2] and over > 1, (p, over))
    true = np.array([0.5, 0.27, 0.23])
    c = 1.25
    q = c * true / (1 - true + c * true)                        # prices made with a known odds ratio
    ok("de-margin recovers the true prices from odds-ratio margined odds", np.allclose(S.demargin(list(1 / q))[0], true, atol=1e-6))
    p, _ = S.demargin([1.30, 1.25, 1.90], 2.0)
    ok("double chance de-margins to a sum of 2", abs(sum(p) - 2) < 1e-9, p)
    ok("one-way price never above 1/odds/1.03, whatever the calibration", all(S.one_way(o, a, b) <= 1 / o / 1.03 + 1e-12 for o in (1.2, 3, 15, 80) for a, b in ((0, 1), (0.5, 1.3), (2, 0.5))))
    gm = Mo.GameModel(1.6, 1.1, -0.05, 0.45)
    ok("model: 1X2 sums to 1; AH -0.5 = home win; DNB = home/(home+away)",
       abs(sum(gm.p("1X2", side=s) for s in "1X2") - 1) < 1e-9 and abs(gm.p("AH", side="1", line=-0.5) - gm.p("1X2", side="1")) < 1e-9
       and abs(gm.p("DNB", side="1") - gm.p("1X2", side="1") / (1 - gm.p("1X2", side="X"))) < 1e-9)
    ok("model: both sides of a quarter line sum to 1 (no push counted twice)", abs(gm.p("TOTAL", side="over", line=2.25) + gm.p("TOTAL", side="under", line=2.25) - 1) < 1e-9
       and abs(gm.p("AH", side="1", line=-1) + gm.p("AH", side="2", line=1) - 1) < 1e-9)

    # ---------------------------------------------------------------- B. settlement on a hand-written match
    # Alpha 2-1 Beta, half-time 1-1. 10' Alpha p11; 45+2' own goal by Alpha p12 (Beta's goal); 70' penalty Alpha p13 (came on 60').
    # Cards: Beta p21 yellow 20', second yellow + red 60'; Alpha p11 yellow 80'; Alpha's coach yellow 85'.
    fx = lambda fid, **k: dict(dict(fixture_id=fid, league_id=40, season=2026, date=pd.Timestamp("2026-11-21 14:00"), status="FT", home_id=1, home="Alpha FC",
                                    away_id=2, away="Beta United", ft_home=2, ft_away=1, ht_home=1, ht_away=1, referee="R", check_goals=True,
                                    check_players=True, check_ht=True, has_events=True), **k)
    E = lambda fid, seq, minute, extra, team, pid, typ, sc=None: dict(fixture_id=fid, seq=seq, minute=minute, extra=extra, team_id=team, player_id=pid, type=typ, scoring_team_id=sc)
    T = dict(
        fixtures=pd.DataFrame([fx(1), fx(2), fx(3, status="PST", ft_home=None, ft_away=None), fx(4, status="ABD")]),
        events=pd.DataFrame([E(1, 0, 10, None, 1, 11, "goal", 1), E(1, 1, 20, None, 2, 21, "yellow"), E(1, 2, 45, 2, 1, 12, "own_goal", 2),
                             E(1, 3, 60, None, 2, 21, "second_yellow"), E(1, 4, 60, None, 2, 21, "red"), E(1, 5, 60, None, 1, 14, "sub"),
                             E(1, 6, 70, None, 1, 13, "penalty", 1), E(1, 7, 80, None, 1, 11, "yellow"), E(1, 8, 85, None, 1, 999, "yellow"),
                             E(2, 0, 5, None, 2, 21, "own_goal", 1), E(2, 1, 30, None, 1, 11, "goal", 1), E(2, 2, 50, None, 2, 22, "goal", 2)]),
        team_match_stats=pd.DataFrame([dict(fixture_id=1, team_id=1, is_home=True, corners=6, shots=12, shots_on_target=5, yellow=1, red=0),
                                       dict(fixture_id=1, team_id=2, is_home=False, corners=3, shots=7, shots_on_target=2, yellow=2, red=1)]),
        player_match_stats=pd.DataFrame([dict(fixture_id=f, team_id=t, player_id=pid, player=f"p{pid}", minutes=m, goals=g, shots=3, shots_on_target=2, started=st)
                                         for f in (1, 2) for t, pid, m, g, st in ((1, 11, 90, 1, True), (1, 12, 90, 0, True), (1, 13, 30, 1, False), (1, 14, 0, 0, False),
                                                                                  (2, 21, 60, 0, True), (2, 22, 90, 0, True))]),
        lineups=pd.DataFrame([dict(fixture_id=f, team_id=t, player_id=pid, starting=st) for f in (1, 2)
                              for t, pid, st in ((1, 11, True), (1, 12, True), (1, 13, False), (1, 14, False), (2, 21, True), (2, 22, True))]),
        coverage=pd.DataFrame([dict(league_id=40, season=2026, events=True, lineups=True, statistics_fixtures=True, statistics_players=True)]),
    )
    facts = S.Facts(T)
    R = lambda typ, side=None, line=None, scope="FT", who=None, pid=None, fid=1: NS(type=typ, side=side, line=line, scope=scope, who=who, player_id=pid,
                                                                               fixture_id=fid, odds=2.0, reason="x")
    want = [
        (R("1X2", "1"), "won", 1), (R("1X2", "X"), "lost", 1), (R("1X2", "X", scope="H1"), "won", 0), (R("DC", "X2"), "lost", 1), (R("DC", "1X"), "won", 1),
        (R("DNB", "1"), "won", 1), (R("DNB", "1", scope="H1"), "push", 0), (R("BTTS", "yes"), "won", 1), (R("BTTS", "yes", scope="H2"), "lost", 0),
        (R("CS", "2:1"), "won", 21), (R("CS", "1:1"), "lost", 21),
        (R("TOTAL", "over", 2.5), "won", 3), (R("TOTAL", "over", 3.0), "push", 3), (R("TOTAL", "over", 2.75), "half_won", 3), (R("TOTAL", "over", 3.25), "half_lost", 3),
        (R("TOTAL", "under", 3.5), "won", 3), (R("TOTAL", "under", 3.25), "half_won", 3),
        (R("TOTAL", "over", 1.5, scope="H1"), "won", 2), (R("TOTAL", "over", 1.5, scope="H2"), "lost", 1), (R("TEAM_TOTAL", "over", 1.5, who="home"), "won", 2),
        (R("TEAM_TOTAL", "over", 1.5, who="away"), "lost", 1),
        (R("AH", "1", -1.0), "push", 1), (R("AH", "1", -0.75), "half_won", 1), (R("AH", "1", -1.25), "half_lost", 1), (R("AH", "1", -0.5), "won", 1),
        (R("AH", "2", 1.5), "won", -1), (R("AH", "2", 0.5), "lost", -1), (R("AH", "2", 1.0), "push", -1), (R("AH", "2", 0.75), "half_lost", -1),
        (R("EH", "X", -1.0), "won", 1), (R("EH", "1", -1.0), "lost", 1),
        (R("CORNERS", "over", 8.5), "won", 9), (R("CORNERS", "over", 6.5, who="home"), "lost", 6), (R("CORNERS", "over", 4.5, scope="H1"), None, None),
        (R("SOT", "under", 7.5), "won", 7), (R("SHOTS", "over", 11.5, who="home"), "won", 12),
        (R("CARDS", "over", 2.5), "won", 3), (R("CARDS", "over", 3.5), None, None), (R("CARDS", "over", 5.5), "lost", 3), (R("CARDS", "over", 0.5, scope="H1"), "won", 1),
        (R("CARDS", "over", 1.5, who="away"), "won", 2), (R("BOOKING_PTS", "over", 40.5), None, None),
        (R("SCORER_ANY", "yes", 0.5, pid=11), "won", 1), (R("SCORER_ANY", "yes", 1.5, pid=11), "lost", 1), (R("SCORER_ANY", "yes", 0.5, pid=12), "lost", 0),
        (R("SCORER_ANY", "yes", 0.5, pid=13), "won", 1), (R("SCORER_ANY", "yes", 0.5, pid=14), "void", 0), (R("SCORER_ANY", "yes", 0.5, pid=None), None, None),
        (R("SCORER_FIRST", "yes", pid=11), "won", 1), (R("SCORER_FIRST", "yes", pid=12), "lost", 0), (R("SCORER_FIRST", "yes", pid=13), None, None),
        (R("SCORER_FIRST", "yes", pid=14), "void", 0), (R("SCORER_FIRST", "yes", pid=11, fid=2), None, None),
        (R("PL_SOT", "over", 1.5, pid=11), "won", 2), (R("PL_SOT", "over", 1.5, pid=14), "void", 0),
        (R("CORNERS_1X2", "1"), "won", 3), (R("CORNERS_1X2", "X"), "lost", 3), (R("SOT_1X2", "2"), "lost", 3), (R("SHOTS_1X2", "1"), "won", 5),
        (R("CORNERS_1X2", "1", scope="H1"), None, None), (R("CARDS_1X2", "2"), None, None),
        (R("PL_CARD", "over", 0.5, pid=11), "won", 1), (R("PL_CARD", "over", 0.5, pid=12), "lost", 0), (R("PL_CARD", "under", 0.5, pid=12), "won", 0),
        (R("PL_CARD", "over", 0.5, pid=14), "void", 0), (R("PL_CARD", "over", 1.5, pid=21), None, None),
        (R("1X2", "1", fid=3), None, None), (R("1X2", "1", fid=4), "void", None), (R("1X2", "1", fid=77), None, None), (R(None), None, None),
    ]
    bad = []
    for r, res, act in want:
        s = S.settle(r, facts)
        if s["result"] != res or (res not in (None, "void") and s["actual"] != act) or (res is None and not s["note"]):
            bad.append((r.type, r.side, r.line, r.scope, r.who, r.player_id, r.fixture_id, "want", res, act, "got", s))
    ok(f"settlement: {len(want)} hand-written cases (quarter lines, pushes, halves, own goals, void, unconfirmed rules)", not bad, bad[:3])
    S.RULES.update(second_yellow_is_two_cards=True, coach_cards_count=False, booking_points=dict(yellow=10, red=25), first_scorer_own_goal="next", first_scorer_late_sub="lost")
    a = [S.settle(R("CARDS", "over", 3.5), facts), S.settle(R("BOOKING_PTS", "over", 40.5), facts), S.settle(R("SCORER_FIRST", "yes", pid=11, fid=2), facts),
         S.settle(R("SCORER_FIRST", "yes", pid=13), facts), S.settle(R("CARDS_1X2", "2"), facts)]
    ok("settlement: once the book rules are entered, the held-back rows grade (cards 3, booking points 45, first scorer after an own goal, cards 1X2 away)",
       [(x["result"], x["actual"]) for x in a] == [("lost", 3.0), ("won", 45.0), ("won", 1.0), ("lost", 0.0), ("won", -1.0)], a)
    S.RULES.update(second_yellow_is_two_cards=None, coach_cards_count=None, booking_points=None, first_scorer_own_goal=None, first_scorer_late_sub=None)

    # ---------------------------------------------------------------- B2. wording read off real vegas.hu boards (EFL, 2026-11-21)
    H, A = dict(id=1, name="Millwall", abbreviation="MIL"), dict(id=2, name="QPR", abbreviation="QPR")
    rr = lambda m, s, sv=None, comp=None, child=None, cc=None: dict(market=m, selection=s, sv=sv, comp=comp, child=child, child_comp=cc)
    real = [
        (rr("1X2", "Millwall", comp=1), ("1X2", "FT", "1", None, None)), (rr("1X2", "Döntetlen"), ("1X2", "FT", "X", None, None)),
        (rr("Hendikep", "Millwall (+0,75)", "+0.75", 1), ("AH", "FT", "1", 0.75, None)), (rr("Hendikep", "QPR (-0,75)", "+0.75", 2), ("AH", "FT", "2", -0.75, None)),
        (rr("Hendikep", "QPR (-0)", "+0", 2), ("AH", "FT", "2", 0.0, None)), (rr("1. félidő - Hendikep", "QPR (+0,5)", "-0.5", 2), ("AH", "H1", "2", 0.5, None)),
        (rr("Hendikep 1X2", "Millwall (0:1)", "0:1", 1), ("EH", "FT", "1", -1.0, None)), (rr("Hendikep 1X2", "Döntetlen (2:0)", "2:0"), ("EH", "FT", "X", 2.0, None)),
        (rr("Szöglet hendikep", "QPR (+0,5)", "-0.5", 2), ("CORNERS_AH", "FT", "2", 0.5, None)),
        (rr("Gólok száma", "2,25 felett", "2.25"), ("TOTAL", "FT", "over", 2.25, None)), (rr("Gólok száma", "2"), (None,) * 5),
        (rr("2. félidő - Gólok száma", "1,5 alatt", "1.5"), ("TOTAL", "H2", "under", 1.5, None)),
        (rr("QPR gólok száma", "1,5 felett", "1.5"), ("TEAM_TOTAL", "FT", "over", 1.5, "away")), (rr("Millwall gólt szerez", "Nem"), ("TEAM_TOTAL", "FT", "under", 0.5, "home")),
        (rr("Mindkét csapat szerez gólt", "Igen"), ("BTTS", "FT", "yes", None, None)), (rr("Mindkét csapat szerez gólt mindkét félidőben", "Igen"), (None,) * 5),
        (rr("Kétesély", "Döntetlen vagy QPR"), ("DC", "FT", "X2", None, None)), (rr("Döntetlennél a tét visszajár", "QPR", comp=2), ("DNB", "FT", "2", None, None)),
        (rr("Pontos végeredmény", "2:1"), ("CS", "FT", "2:1", None, None)), (rr("1. félidő - Pontos végeredmény", "1:0"), ("CS", "H1", "1:0", None, None)),
        (rr("Szögletek száma", "9,5 felett", "9.5"), ("CORNERS", "FT", "over", 9.5, None)), (rr("Szögletek száma", "9-11"), (None,) * 5),
        (rr("Millwall szögletek száma", "5,5 alatt", "5.5"), ("CORNERS", "FT", "under", 5.5, "home")),
        (rr("Büntetőlapok száma", "4,5 felett", "4.5"), ("CARDS", "FT", "over", 4.5, None)), (rr("Büntetőlapok 1X2", "QPR", comp=2), ("CARDS_1X2", "FT", "2", None, None)),
        (rr("Szöglet 1X2", "Döntetlen"), ("CORNERS_1X2", "FT", "X", None, None)), (rr("1X2 kaput eltaláló lövések száma (Rendes játékidő)", "Millwall", comp=1), ("SOT_1X2", "FT", "1", None, None)),
        (rr("1X2 kapura lövések száma (Rendes játékidő)", "QPR", comp=2), ("SHOTS_1X2", "FT", "2", None, None)),
        (rr("Lapok - Csere játékossal együtt", "0,5 felett", "0.5", child="Paul Smyth", cc=2), (None,) * 5), (rr("Ki végez el elsőként 5 szögletet?", "QPR", comp=2), (None,) * 5),
        (rr("Kaput eltaláló lövések száma (Rendes játékidő)", "8,5 felett", "8.5"), ("SOT", "FT", "over", 8.5, None)),
        (rr("QPR összes kaput eltaláló lövések száma (Rendes játékidő)", "3,5 alatt", "3.5|13218"), ("SOT", "FT", "under", 3.5, "away")),
        (rr("Összes Lövés (Rendes játékidő)", "26,5 felett", "26.5"), ("SHOTS", "FT", "over", 26.5, None)),
        (rr("Gólszerző", "Bármikor", "0.5|ws:player:1|1", child="Josh Coburn", cc=1), ("SCORER_ANY", "FT", "yes", 0.5, None)),
        (rr("Gólszerző", "1. gólszerző", child="Josh Coburn", cc=1), ("SCORER_FIRST", "FT", "yes", None, None)),
        (rr("Gólszerző", "Utolsó", child="Josh Coburn", cc=1), (None,) * 5), (rr("Gólszerző (Szupercserével)", "Bármikor", child="Josh Coburn", cc=1), (None,) * 5),
        (rr("Hány gólt szerez?", "2+ gólt szerez", child="Josh Coburn", cc=1), ("SCORER_ANY", "FT", "yes", 1.5, None)),
        (rr("Kaput eltaláló lövések (Rendes játékidő)", "1,5 felett", "1.5", child="Josh Coburn", cc=1), ("PL_SOT", "FT", "over", 1.5, None)),
        (rr("Lövések (Rendes játékidő)", "2,5 felett", "2.5", child="Josh Coburn", cc=1), ("PL_SHOTS", "FT", "over", 2.5, None)),
        (rr("Lapok", "0,5 felett", "0.5", child="Paul Smyth", cc=2), ("PL_CARD", "FT", "over", 0.5, None)),
        (rr("1. Gólszerző + 1X2", "Millwall", child=None), (None,) * 5), (rr("Gólszerző & pontos végeredmény", "1:0"), (None,) * 5),
        (rr("1X2 & gólok száma", "Millwall & 2,5 felett", "2.5"), (None,) * 5), (rr("Első/utolsó gólszerző", "Első vagy utolsó gólszerző", child="X", cc=1), (None,) * 5),
        (rr("Millwall nyeri és nem kap gólt", "Igen"), (None,) * 5), (rr("Félidő/végeredmény", "Millwall/Millwall"), (None,) * 5),
    ]
    badr = []
    for r, want_ in real:
        c = S.classify(r, H, A)
        got = (c["type"], c["scope"], c["side"], c["line"], c["who"]) if c["type"] else (None,) * 5
        if got != want_ or (c["type"] is None and not c["reason"]):
            badr.append((r["market"], r["selection"], "want", want_, "got", got, c["reason"]))
    ok(f"board: {len(real)} market wordings from a real vegas.hu board are read as intended", not badr, badr[:4])

    # ---------------------------------------------------------------- C. ingest against the simulated API (paid plan)
    I.PLAN = "paid"
    clock = [dt.datetime(2026, 11, 20, 9, 0, tzinfo=dt.timezone.utc)]
    I.now_utc = lambda: clock[0]
    I.current_season = lambda d=None: 2026
    mk = lambda **k: K.World(seed=7, n_teams=10, today=clock[0].date(), **k)
    for og in ("scorer", "beneficiary"):                       # whichever team the API lists an own goal under
        W = mk(og_listed_under=og)
        (_, out) = run_quiet(lambda: (I.run(I.Api(key="x", fetch=W.fetch)), I.build()))
        ok(f"ingest: all three sanity checks pass with own goals listed under the {og}", out.count(": OK (150 of 150 games)") == 3, out[-900:])
        if og == "scorer":
            shutil.rmtree(tmp / "cache"), shutil.rmtree(tmp / "data")
    ids_calls = [c for c in W.calls if "ids" in c[1]]
    ok("ingest: 150 games cost 8 batches of 20 ids, and no per-game endpoint is ever called",
       len(ids_calls) == 8 and max(len(c[1]["ids"].split("-")) for c in ids_calls) == 20 and not any("/" in c[0] for c in W.calls), [c[0] for c in W.calls])
    n0 = len(W.calls)
    run_quiet(lambda: I.run(I.Api(key="x", fetch=W.fetch)))
    ok("ingest: a second run fetches no finished game again", not any("ids" in c[1] for c in W.calls[n0:]), W.calls[n0:])
    rel = I.jload(tmp / "data/soccer_relationships.json")
    ok("ingest: every declared key was verified on the data", len(rel["primary_keys"]) == 8 and len(rel["foreign_keys"]) == 10 and not rel["rejected"], rel["rejected"])
    c2, d2 = I.CACHE, I.OUT
    I.CACHE, I.OUT = tmp / "c2", tmp / "d2"
    W2 = mk(block_before=2026)
    W2.left = 9
    (_, out) = run_quiet(lambda: I.run(I.Api(key="x", fetch=W2.fetch)))
    ok("ingest: stops calling when the API says 5 requests are left", W2.left == 5 and "stopped" in out, (W2.left, out[-300:]))
    W2.left = 100
    (_, out) = run_quiet(lambda: I.run(I.Api(key="x", fetch=W2.fetch)))
    ok("ingest: a season the plan refuses is reported and skipped, nothing guessed", out.count("REFUSED BY PLAN") == 2 and "season 2026: " in out, out[-600:])
    W3 = mk(coverage_holes=(79,))
    I.CACHE, I.OUT = tmp / "c3", tmp / "d3"
    (_, out) = run_quiet(lambda: (I.run(I.Api(key="x", fetch=W3.fetch)), I.build()))
    cov = pd.read_csv(tmp / "d3/soccer_2026_coverage.csv").set_index("league_id")
    ok("ingest: the coverage block is recorded per league (no player statistics for league 79)", not cov.statistics_players[79] and cov.statistics_players[40], cov)
    # ---------------------------------------------------------------- C2. the free plan (its rules as seen on 2026-11-20)
    I.PLAN = "free"
    I.CACHE, I.OUT = tmp / "c4", tmp / "d4"
    clock[0] = dt.datetime(2026, 11, 22, 9, 0, tzinfo=dt.timezone.utc)          # Sunday: Saturday's games are D-1
    W4 = mk(free=True)
    I.jsave(I.CACHE / "index.json", {"1": dict(fixture_id=1, league_id=40, season=2024, date="2024-12-01T15:00:00+00:00", ts=1, status="FT",
                                               home_id=1, home="Old", away_id=2, away="Older", ft_home=1, ft_away=0)})   # last season, from a paid-style run
    (_, out) = run_quiet(lambda: (I.run_free(I.Api(key="x", fetch=W4.fetch)), I.build()))
    kinds = sorted({(c[0], tuple(sorted(c[1]))) for c in W4.calls})
    ok("free: only date calls for D-1/D/D+1 and one id call per finished game; never season, ids or a refused call",
       kinds == [("fixtures", ("date",)), ("fixtures", ("id",))] and sum(1 for c in W4.calls if "date" in c[1]) == 3
       and sum(1 for c in W4.calls if "id" in c[1]) == 10 and "REFUSED" not in out, (kinds, out[-500:]))
    ok("free: the three sanity checks pass on the fetched games", out.count(": OK (10 of 10 games)") == 3, out[-700:])
    cov = pd.read_csv(tmp / "d4/soccer_2026_coverage.csv")
    ok("free: coverage inferred from the fetched games and marked as inferred", len(cov) == 2 and cov.source.str.startswith("inferred").all() and cov.events.all(), cov)
    n0 = len(W4.calls)
    run_quiet(lambda: I.run_free(I.Api(key="x", fetch=W4.fetch)))
    ok("free: a second run the same day costs 3 date calls and fetches nothing again", len(W4.calls) - n0 == 3, W4.calls[n0:])
    clock[0] = dt.datetime(2026, 11, 27, 9, 0, tzinfo=dt.timezone.utc)          # Friday: lists Saturday the 28th
    W4 = mk(free=True)
    run_quiet(lambda: I.run_free(I.Api(key="x", fetch=W4.fetch)))
    clock[0] = dt.datetime(2026, 11, 30, 9, 0, tzinfo=dt.timezone.utc)          # Monday: Saturday and Sunday runs were missed
    W4 = mk(free=True)
    (_, out) = run_quiet(lambda: I.run_free(I.Api(key="x", fetch=W4.fetch)))
    ok("free: games whose day left the window unfetched are reported LOST (last season's listed games are not), not requested", "LOST: 10 new, 10 in all" in out and not any("id" in c[1] for c in W4.calls), out[-400:])
    I.CACHE, I.OUT = tmp / "c5", tmp / "d5"
    clock[0] = dt.datetime(2026, 11, 22, 9, 0, tzinfo=dt.timezone.utc)
    W5 = mk(free=True, id_detail=False)
    (_, out) = run_quiet(lambda: (I.run_free(I.Api(key="x", fetch=W5.fetch)), I.build()))
    ok("free: if fixtures?id carries no detail, the four per-game endpoints are used and the checks still pass",
       "switching to 4 calls" in out and out.count(": OK (10 of 10 games)") == 3 and sum(1 for c in W5.calls if c[0] == "fixtures/players") == 10, out[-600:])
    I.CACHE, I.OUT = c2, d2

    # ---------------------------------------------------------------- D. three matchdays end to end, free plan + football-data history
    con = S.connect()
    sat = dt.date(2026, 11, 21)
    truth = []
    for k in range(3):
        day = sat + dt.timedelta(days=7 * k)
        clock[0] = dt.datetime(day.year, day.month, day.day, 9, 0, tzinfo=dt.timezone.utc) - dt.timedelta(days=1)
        W = mk(free=True)
        W.write_fd(tmp / "fd")                                   # football-data: every game played so far
        run_quiet(lambda: (I.run_free(I.Api(key="x", fetch=W.fetch)), I.build()))
        bdir = tmp / "boards" / day.isoformat()
        todays = [f for f, x in W.fx.items() if x["fixture"]["date"][:10] == day.isoformat()]
        for f in todays:
            W.board(f, path=bdir / f"{f}.net.json")
            x = W.fx[f]
            g = Mo.GameModel(*W.lam(x["teams"]["home"]["id"], x["teams"]["away"]["id"]), 0.0, 0.5)
            truth.append((day.isoformat(), f, g))
        if k == 0:
            (_, out_unread) = run_quiet(S.unread, str(bdir))
            first = todays[0]
            xi = [dict(fixture_id=first, team_id=t, player_id=p["id"], player=p["name"], starting=j < 11) for t in (W.fx[first]["teams"]["home"]["id"],)
                  for j, p in enumerate(W.teams[t]["squad"])]
            pd.DataFrame(xi).to_csv(tmp / "data/soccer_prematch_lineups.csv", index=False)
        (D, out_price) = run_quiet(S.price, day.isoformat(), str(bdir), con)
        if k == 0:
            D0, pool0 = D.copy(), S.leg_pool(day.isoformat(), con)
            run_quiet(S.price, day.isoformat(), str(bdir), con)                                       # a second capture of the same board
        (B, out_slips) = run_quiet(S.slips, day.isoformat(), con, (3, 6, 9), None, 1)
        if k == 0:
            B0, legs0 = B.copy(), pd.read_sql("SELECT l.*, p.type, p.subject, p.fixture_id FROM slip_legs l JOIN predictions p USING (matchday, row_key)", con)
            (B7, _) = run_quiet(S.slips, day.isoformat(), con, (3,), None, 1, False, (7,))
            L7 = pd.read_sql("SELECT * FROM slip_legs WHERE matchday = ? AND slip_id LIKE '%7legs'", con, params=[day.isoformat()])
            ok("slips: --legs 7 builds 7-leg slips, one leg per game", len(B7) and (B7.slip_id.str.endswith("7legs")).sum() >= 1
               and len(L7) and all(len(g) == 7 and g.game.nunique() == 7 for _, g in L7.groupby("slip_id")), B7)
            SEp = S.statedge_pool(day.isoformat(), con)
            ok("statedge: statistics legs only, priced 1.5-2.1, model above the book, one per game and quantity",
               len(SEp) > 3 and SEp.type.isin(S.STAT_T).all() and SEp.odds.between(1.5, 2.1).all() and (SEp.p_model > SEp.p_book).all()
               and not SEp.duplicated(["game", "quantity"]).any(), SEp[["game", "type", "odds", "p_book", "p_model"]].head())
            ok("statedge: slips built and their stated chance is the book's, not the model's",
               (B7.strategy == "statedge/legs").any() or (B7.strategy == "statedge").any(), B7)
            run_quiet(S.slips, day.isoformat(), con, (3, 6, 9), None, 1)                    # back to the default set
        clock[0] += dt.timedelta(days=2)
        W = mk(free=True)
        run_quiet(lambda: (I.run_free(I.Api(key="x", fetch=W.fetch)), I.build()))
        if k == 1:                                               # a version bump between pricing and grading
            con.execute("UPDATE predictions SET model_version = 'soccer-0.0' WHERE matchday = ?", [day.isoformat()])
        (_, out_grade) = run_quiet(S.grade, day.isoformat(), con, str(bdir))
        if k == 1:
            ok("grade: rows priced under an older version are re-priced first", "re-priced under" in out_grade
               and con.execute("SELECT count(*) FROM predictions WHERE model_version != ?", [S.MODEL_VERSION]).fetchone()[0] == 0, out_grade[:200])

    T_ = S.load_tables()
    T_["fixtures"] = pd.concat([T_["fixtures"], T_["fixtures"].head(3).assign(fixture_id=[-1, -2, -3], season=2024)])    # last season's games in the tables
    M_ = S.matches_frame(T_)
    ok("model history: older API seasons are left out (football-data has them), this season's API games are in",
       2024 not in set(M_[M_.source == "api"].season) and (M_.source == "api").sum() >= 10, M_.groupby(["source", "season"]).size().to_dict())
    ok("model history: football-data games used, all 20 teams mapped to API ids by name", "teams mapped 20 of 20" in out_price and "NOT MAPPED" not in out_price, out_price[:600])
    T_ = S.load_tables()
    m_, un_ = S.map_fd_teams(T_, SB.load([2026]))
    ok("team map: every football-data name maps to the right API team (checked against the simulation)",
       len(m_) == 20 and all(W.fd_name(v) == k for k, v in m_.items()) and not un_, (len(m_), un_))
    ok("board: every market is typed or listed as unread with its reason", "UNREAD" in out_unread and all(x in out_unread for x in
       ("'Valami egészen új piac'", "[combination market not read] 'Félidő/végeredmény'",
        "[period or time-window market not read] 'Mindkét csapat szerez gólt mindkét félidőben'")), out_unread[-700:])
    nba = tmp / "nba"
    nba.mkdir()
    (nba / "x.net.json").write_text('[{"url": "u", "body": {"sport": {"typeId": 12, "name": "Kosárlabda"}, "markets": [], "odds": [], "competitors": [{}, {}]}}]')
    ok("board: another sport's capture in the folder is skipped", S.read_capture(str(nba / "x.net.json")) == [])
    ok("board: a market listed twice is read once (3 rows of 1X2 per game)", (D0[(D0.type == "1X2") & (D0.scope == "FT")].groupby("game").size() == 3).all())
    typed = D0[D0.type.notna()]
    ok("board: the 14 known market types are all read", set(typed.type) >= {"1X2", "DC", "DNB", "BTTS", "TOTAL", "TEAM_TOTAL", "AH", "CORNERS", "CARDS", "BOOKING_PTS", "CS", "SCORER_ANY"}
       and {"H1"} <= set(typed.scope) and D0.fixture_id.notna().all(), sorted(set(typed.type)))
    tg = {f: g for d, f, g in truth}
    tw = typed[(typed.p_method == "no_vig") & typed.type.isin(["1X2", "DC", "DNB", "BTTS", "TOTAL", "TEAM_TOTAL", "AH"])]
    err = [abs(r.p_book - tg[r.fixture_id].p(r.type, r.scope, r.side, r.line, r.who)) for r in tw.itertuples()]
    ok("price: the de-margined book price is the simulation's true probability (mean error under 1 point, worst under 3.5)",
       len(err) > 250 and np.mean(err) < 0.01 and max(err) < 0.035, (len(err), np.mean(err), max(err)))
    ow = typed[typed.p_method == "one_way"]
    ok("price: one-way rows (correct score, scorers) are capped at 1/odds/1.03 and the model has weight 0", len(ow) > 50 and (ow.p <= 1 / ow.odds / 1.03 + 1e-9).all()
       and (typed.w_used == 0).all() and np.allclose(typed.p, typed.p_book))
    mod = typed[typed.p_model.notna() & (typed.p_method == "no_vig")]
    ok("model: has a number for most two-way rows and sits near the book on average (it was fitted on 150 games)",
       len(mod) > 0.8 * len(typed[typed.p_method == "no_vig"]) - 25 and (mod.p_model - mod.p_book).abs().mean() < 0.09, (len(mod), (mod.p_model - mod.p_book).abs().mean()))
    k2 = con.execute("SELECT count(*), sum(odds_first = odds) FROM predictions WHERE matchday = ?", [sat.isoformat()]).fetchone()
    ok("price: pricing the same board twice does not duplicate rows", k2[0] == len(D0) == k2[1], k2)
    # slips
    ok("slips: every leg is book 60-88%, price >= 1.15, unflagged; no DNB; sides only on a 60% favourite",
       len(pool0) > 10 and pool0.p_book.between(0.60, 0.88).all() and (pool0.odds >= 1.15).all() and pool0.flag.isna().all() and "DNB" not in set(pool0.type)
       and not len(pool0[pool0.type.isin(["1X2", "AH"]) & (pool0.p_book < 0.6)]), len(pool0))
    pl = pool0[pool0.type.isin(S.PLAYER_T)]
    ok("slips: player legs only for players confirmed in a starting XI", set(pl.fixture_id) <= {first}, pl[["game", "subject"]])
    ok("slips: one leg per game in a slip; no game+quantity in two slips of a strategy; none uses a player line",
       len(B0) >= 3 and all(g.game.nunique() == len(g) for _, g in legs0.groupby("slip_id"))
       and all(not legs0[legs0.slip_id.str.startswith(st)].assign(q=lambda x: [S.quantity(t, s) for t, s in zip(x.type, x.subject)]).duplicated(["game", "q"]).any() for st in S.STRATEGIES)
       and not legs0.type.isin(S.PLAYER_T).any(), B0)
    ok("slips: payout within target..target*1.25 and stated chance = product of the legs", all(r.target - 1e-9 <= r.odds <= r.target * 1.25 + 1e-9 for r in B0.itertuples())
       and all(abs(r.p - legs0[legs0.slip_id == r.slip_id].p.prod()) < 1e-9 for r in B0.itertuples()), B0)
    # grading
    G = pd.read_sql("SELECT * FROM predictions", con)
    g = G[G.status == "graded"]
    ok("grade: every graded row records what it was settled on", len(g) > 1500 and g[g.result != "void"].actual.notna().all() and g[g.type.isin(S.PLAYER_T)].player_id.notna().all(), len(g))
    ok("grade: unread rows and booking points are never graded and carry the reason", (G[G.type.isna()].status == "open").all() and G[G.type.isna()].grade_note.notna().all()
       and (G[G.type == "BOOKING_PTS"].status == "open").all())
    TS, EVt = pd.read_csv(tmp / "data/soccer_2026_team_match_stats.csv"), pd.read_csv(tmp / "data/soccer_2026_events.csv")
    corners, ycards = TS.groupby("fixture_id").corners.sum(), EVt[EVt.type == "yellow"].groupby("fixture_id").size()
    gc, gk = g[(g.type == "CORNERS")], g[(g.type == "CARDS")]
    ok("grade: match corners and cards are settled on the match total from the tables, not on one team's",
       len(gc) == 60 and (gc.actual == gc.fixture_id.map(corners)).all() and len(gk) > 40 and (gk.actual >= gk.fixture_id.map(ycards).fillna(0)).all(),
       (len(gc), len(gk)))
    bench = g[(g.type == "SCORER_ANY") & (g.result == "void")]
    ok("grade: the bench player who never came on is void, not lost", len(bench) >= 20 and (bench.actual == 0).all(), len(bench))
    d = S.graded_frame(con)
    twg = d[(d.p_method == "no_vig") & d.p_book.between(0.60, 0.88)]                                  # one side of each pair: the slip band
    z = ((twg.y - twg.p_book) * twg.wt).groupby(twg.cluster).sum()
    zt = z.mean() / (z.std(ddof=1) / np.sqrt(len(z)))
    ok("calibration: in the 60-88% band stated wins and actual wins agree within noise over 30 games (|t| < 3)", len(twg) > 150 and abs(zt) < 3, (len(twg), zt))
    sl = pd.read_sql("SELECT * FROM slips", con)
    ok("slips: every built slip was graded; a lost leg loses the slip", (sl.status == "graded").all() and set(sl.result) <= {"won", "lost", "part", "void"}
       and (sl[sl.result == "lost"].paid == 0).all() and sl.played.sum() >= 3, sl[["slip_id", "result", "paid"]].head())
    TP = S.team_profiles()
    ok("teams: a profile per team (20) with corners, cards, shots, on target and fouls for and against",
       len(TP) == 20 and TP[["corners_for", "corners_against", "cards_for", "sot_for", "fouls_for"]].notna().all().all() and TP.corners_for.between(3, 7).all(),
       TP.head(3).T)
    (Wt, _) = run_quiet(S.fit_weights, con)
    ok("weights: with 30 games no family reaches t > 2 held out, so every weight is 0", len(Wt) > 5 and (Wt.w == 0).all(), Wt[Wt.w > 0])
    (txt, _) = run_quiet(S.report, con)
    ok("report: carries the NOT VERIFIED banner until the hand check is confirmed", "NOT VERIFIED" in txt and "Where the model lost" in txt)
    (_, hc) = run_quiet(S.handcheck, con)
    run_quiet(S.handcheck, con, True)
    (txt, _) = run_quiet(S.report, con)
    ok("report: banner gone after `handcheck --confirm` on five rows of five types", hc.count("settled on actual") == 5 and "NOT VERIFIED" not in txt, hc[:300])

    # ---------------------------------------------------------------- E. API account blocked: football-data.co.uk takes over
    clock[0] = dt.datetime(2026, 12, 11, 9, 0, tzinfo=dt.timezone.utc)
    W = mk(free=True)
    blocked = lambda path, params: ({"errors": {"access": "Your account is suspended, check on https://dashboard.api-football.com."}, "response": []}, {})
    before = pd.read_csv(tmp / "data/soccer_2026_fixtures.csv")
    (_, out_b) = run_quiet(lambda: (I.run_free(I.Api(key="x", fetch=blocked)), I.build()))
    nokey = lambda path, params: ({"errors": {"token": "Error/Missing application key. Go to https://www.api-football.com/documentation-v3"}, "response": []}, {})
    (_, out_k) = run_quiet(lambda: (I.run_free(I.Api(key="x", fetch=nokey)), I.build()))
    ok("rejected key: the ingest stops cleanly with a clear message", "API KEY REJECTED" in out_k, out_k[-300:])
    ok("blocked API: the ingest stops cleanly with a clear message and leaves the tables as they were",
       "API ACCOUNT BLOCKED" in out_b and len(pd.read_csv(tmp / "data/soccer_2026_fixtures.csv")) == len(before), out_b[-400:])
    day4 = dt.date(2026, 12, 5)                                  # as if the API had died before it listed the 5th
    clock[0] = dt.datetime(2026, 12, 4, 9, 0, tzinfo=dt.timezone.utc)
    W = mk(free=True)
    W.write_fd(tmp / "fd")                                       # results up to the 28th
    g4 = [f for f, x in W.fx.items() if x["fixture"]["date"][:10] == day4.isoformat()]
    code = {40: "E1", 79: "D2"}
    pd.DataFrame([dict(Div=code[W.fx[f]["league"]["id"]], Date=day4.strftime("%d/%m/%Y"), Time="15:00",
                       HomeTeam=W.fd_name(W.fx[f]["teams"]["home"]["id"]), AwayTeam=W.fd_name(W.fx[f]["teams"]["away"]["id"]), Referee="")
                  for f in g4]).to_csv(tmp / "fd/fixtures.csv", index=False)
    dE = tmp / "dataE"
    shutil.copytree(tmp / "data", dE)
    for t in dE.glob("soccer_2026_*.csv"):
        x = pd.read_csv(t)
        if "fixture_id" in x and t.name != "soccer_2026_played_fd.csv":
            x[~x.fixture_id.isin(g4)].to_csv(t, index=False)
    (dE / "soccer_2026_played_fd.csv").unlink(missing_ok=True)
    S.OUT, out_old = dE, S.OUT
    con = S.connect(str(tmp / "ledger/e.sqlite"))
    b4 = tmp / "boards" / day4.isoformat()
    (D4, out_p4) = run_quiet(S.price, day4.isoformat(), str(b4), con)
    typed4 = D4[D4.type.notna()]
    ok("blocked API: games the API never listed are taken from football-data's fixtures.csv, matched to their boards and priced",
       len(g4) >= 4 and typed4.fixture_id.notna().all() and (typed4.fixture_id < 0).all() and typed4.fixture_id.nunique() == len(g4)
       and typed4.p_model.notna().sum() > 50, (len(g4), typed4.fixture_id.nunique(), out_p4[:300]))
    clock[0] = dt.datetime(2026, 12, 6, 9, 0, tzinfo=dt.timezone.utc)
    W = mk(free=True)
    W.write_fd(tmp / "fd")                                       # the results file now holds the 5th
    (_, out_g4) = run_quiet(S.grade, day4.isoformat(), con, str(b4))
    G4 = pd.read_sql("SELECT * FROM predictions WHERE matchday = ? AND type IS NOT NULL", con, params=[day4.isoformat()])
    gx = G4[(G4.type == "1X2") & (G4.scope == "FT") & (G4.status == "graded")]
    fx4 = {W.fx[f]["teams"]["home"]["id"]: W.fx[f] for f in g4}
    sc = {r.fixture_id: r for r in S.load_tables()["fixtures"].itertuples()}
    right = all((r.won == 1) == ({"1": sc[r.fixture_id].ft_home > sc[r.fixture_id].ft_away, "X": sc[r.fixture_id].ft_home == sc[r.fixture_id].ft_away,
                                  "2": sc[r.fixture_id].ft_home < sc[r.fixture_id].ft_away}[r.side]) for r in gx.itertuples())
    truth4 = {(W.fx[f]["teams"]["home"]["id"], W.fx[f]["teams"]["away"]["id"]): (W.fx[f]["goals"]["home"], W.fx[f]["goals"]["away"]) for f in g4}
    same = all(truth4.get((sc[i].home_id, sc[i].away_id)) == (sc[i].ft_home, sc[i].ft_away) for i in gx.fixture_id.unique())
    ok("blocked API: results, goal, corner and shots markets graded from football-data; the scores are the simulation's",
       len(gx) == 3 * len(g4) and right and same and (G4[G4.type.isin(["TOTAL", "CORNERS", "SHOTS", "SOT", "AH"]) & (G4.scope == "FT")].status == "graded").all()
       and f"{len(g4)} games graded from football-data" in out_g4, (len(gx), len(g4), right, same, out_g4[:300]))
    pl, h1 = G4[G4.type.isin(list(S.PLAYER_T))], G4[G4.type.isin(["CORNERS", "CARDS"]) & (G4.scope != "FT")]
    ok("blocked API: player markets and statistics by half stay open, with the reason",
       len(pl) > 20 and (pl.status == "open").all() and pl.grade_note.str.contains("football-data has no player data").all()
       and (h1.status == "open").all(), (len(pl), pl.grade_note.value_counts().head(3)))
    ck = G4[(G4.type == "CARDS") & (G4.scope == "FT")]
    ok("blocked API: card lines graded from yellows + reds, a line a red card could flip is held back",
       len(ck) > 10 and ck[ck.status == "graded"].grade_note.str.contains("football-data").all()
       and ck[ck.status == "open"].grade_note.str.contains("red card|not graded from", regex=True).all(), ck.grade_note.value_counts().head())
    S.OUT = out_old

    width = max(len(n) for n, _, _ in checks)
    for n, good, info in checks:
        print(f"  {'ok  ' if good else 'FAIL'} {n:<{width}}" + ("" if good else f"\n       {str(info)[:1500]}"))
    fails = sum(1 for _, good, _ in checks if not good)
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(checks) - fails} of {len(checks)} checks  (simulated data: this proves the mechanics, not the live API or a real board)")
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
