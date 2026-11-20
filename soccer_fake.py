#!/usr/bin/env python3
"""soccer_fake.py: a simulated world for the selftests. NOT data.

It produces (a) responses in the documented shape of API-Football v3 (leagues, fixture lists, fixtures by date,
fixtures by ids with events / lineups / statistics / players) and (b) vegas.hu-style boards in the Altenar
.net.json shape (markets / odds / childMarkets / competitors), priced from the simulation's true probabilities
plus a margin. The Hungarian market names on the fake board are the SPEC'S GUESSES: a passing selftest proves the
mechanics (budget, keys, settlement, de-margining), not that a real board is read correctly.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from soccer_model import GameModel

LEAGUES = {40: ("England", "Championship"), 41: ("England", "League One"), 42: ("England", "League Two"),
           79: ("Germany", "2. Bundesliga"), 136: ("Italy", "Serie B"), 141: ("Spain", "Segunda División")}
STATS = ["Shots on Goal", "Total Shots", "Fouls", "Corner Kicks", "Offsides", "Ball Possession", "Yellow Cards", "Red Cards",
         "Goalkeeper Saves", "expected_goals"]


class World:
    def __init__(self, seed=1, season=2026, n_teams=8, today=None, og_listed_under="scorer", leagues=(40, 79),
                 block_before: int | None = None, coverage_holes=(), free: bool = False, id_detail: bool = True):
        self.seed = seed
        self.season, self.og, self.block_before, self.holes = season, og_listed_under, block_before, set(coverage_holes)
        self.free, self.id_detail = free, id_detail          # free: the rules API-Football's Free plan applies (seen 2026-11-20)
        self.today = today or dt.date(season, 11, 20)
        self.teams, self.fx, self.calls, self.left = {}, {}, [], 100
        tid = 1000 * (season % 100)
        for lid in leagues:
            ids, rng = [], np.random.default_rng([seed, lid])       # results never depend on what else was simulated
            for k in range(n_teams):
                tid += 1
                self.teams[tid] = dict(id=tid, name=f"{LEAGUES[lid][1].split()[-1]} {chr(65 + k)}{chr(97 + k)}town", league=lid,
                                       att=rng.normal(0, 0.25), dfn=rng.normal(0, 0.25),
                                       squad=[dict(id=tid * 100 + j, name=f"P{tid}-{j} Name{j}", w=float(rng.gamma(1.2)) if j else 0.01)
                                              for j in range(18)])
                ids.append(tid)
            self._schedule(lid, ids)

    def lam(self, h, a):
        H, A = self.teams[h], self.teams[a]
        return math.exp(0.15 + 0.25 + H["att"] - A["dfn"]), math.exp(0.15 + A["att"] - H["dfn"])

    def _schedule(self, lid, ids):
        n, fid, d0 = len(ids), lid * 100000 + (self.season % 100) * 1000, dt.date(self.season, 8, 8)
        arr = ids[:]
        rounds = []
        for r in range(n - 1):
            rounds.append([(arr[i], arr[n - 1 - i]) if (r + i) % 2 else (arr[n - 1 - i], arr[i]) for i in range(n // 2)])
            arr = [arr[0]] + [arr[-1]] + arr[1:-1]
        rounds += [[(a, h) for h, a in rd] for rd in rounds]
        for r, rd in enumerate(rounds):
            day = d0 + dt.timedelta(days=7 * r)
            for h, a in rd:
                fid += 1
                self.fx[fid] = self._play(fid, lid, r + 1, day, h, a)

    def _play(self, fid, lid, rnd, day, h, a):
        rng, played = np.random.default_rng([self.seed, fid]), day < self.today
        ts = int(dt.datetime(day.year, day.month, day.day, 14, 0, tzinfo=dt.timezone.utc).timestamp())
        f = dict(fixture=dict(id=fid, referee=f"Ref {int(rng.integers(1, 6))}, {LEAGUES[lid][0]}", timezone="UTC",
                              date=f"{day.isoformat()}T14:00:00+00:00", timestamp=ts, venue=dict(id=h, name=f"Ground {h}", city="X"),
                              status=dict(long="Match Finished" if played else "Not Started", short="FT" if played else "NS",
                                          elapsed=90 if played else None)),
                 league=dict(id=lid, name=LEAGUES[lid][1], country=LEAGUES[lid][0], season=self.season, round=f"Regular Season - {rnd}"),
                 teams=dict(home=dict(id=h, name=self.teams[h]["name"], winner=None), away=dict(id=a, name=self.teams[a]["name"], winner=None)),
                 goals=dict(home=None, away=None),
                 score=dict(halftime=dict(home=None, away=None), fulltime=dict(home=None, away=None), extratime=dict(home=None, away=None),
                            penalty=dict(home=None, away=None)))
        if not played:
            return f
        lh, la = self.lam(h, a)
        ev, score, ht = [], {h: 0, a: 0}, {h: 0, a: 0}
        xi = {t: self.teams[t]["squad"][:11] for t in (h, a)}
        bench = {t: self.teams[t]["squad"][11:] for t in (h, a)}
        mins = {p["id"]: 90 for t in (h, a) for p in xi[t]}
        mins.update({p["id"]: 0 for t in (h, a) for p in bench[t]})
        pgoals, cards = {}, {}
        T = lambda t: dict(id=t, name=self.teams[t]["name"])
        P = lambda p: dict(id=p["id"], name=p["name"])
        for t, o, lam in ((h, a, lh), (a, h, la)):
            for _ in range(rng.poisson(lam)):
                m = int(rng.integers(1, 91))
                extra = int(rng.integers(1, 5)) if m in (45, 90) and rng.random() < 0.5 else None
                u, w = rng.random(), np.array([p["w"] for p in xi[t]])
                if u < 0.06:                                    # own goal by a player of the other side
                    p = xi[o][int(rng.integers(1, 11))]
                    ev.append(dict(time=dict(elapsed=m, extra=extra), team=T(o if self.og == "scorer" else t), player=P(p),
                                   assist=dict(id=None, name=None), type="Goal", detail="Own Goal", comments=None))
                else:
                    p = xi[t][int(rng.choice(11, p=w / w.sum()))]
                    ev.append(dict(time=dict(elapsed=m, extra=extra), team=T(t), player=P(p), assist=dict(id=None, name=None),
                                   type="Goal", detail="Penalty" if u < 0.16 else "Normal Goal", comments=None))
                    pgoals[p["id"]] = pgoals.get(p["id"], 0) + 1
                score[t] += 1
                ht[t] += m <= 45
            if rng.random() < 0.04:
                ev.append(dict(time=dict(elapsed=int(rng.integers(1, 91)), extra=None), team=T(t), player=P(xi[t][9]),
                               assist=dict(id=None, name=None), type="Goal", detail="Missed Penalty", comments=None))
            for _ in range(rng.poisson(1.9)):
                p = xi[t][int(rng.integers(0, 11))]
                c = cards.setdefault(p["id"], dict(y=0, r=0))
                if c["r"]:
                    continue
                m = int(rng.integers(1, 91))
                if c["y"] == 1:                                 # second yellow: the API style of yellow + red is used here
                    c["y"], c["r"] = 2, 1
                    ev.append(dict(time=dict(elapsed=m, extra=None), team=T(t), player=P(p), assist=dict(id=None, name=None),
                                   type="Card", detail="Yellow Card", comments=None))
                    ev.append(dict(time=dict(elapsed=m, extra=None), team=T(t), player=P(p), assist=dict(id=None, name=None),
                                   type="Card", detail="Red Card", comments="Second yellow"))
                else:
                    c["y"] = 1
                    ev.append(dict(time=dict(elapsed=m, extra=None), team=T(t), player=P(p), assist=dict(id=None, name=None),
                                   type="Card", detail="Yellow Card", comments=None))
            if rng.random() < 0.03:
                p = xi[t][int(rng.integers(1, 11))]
                if p["id"] not in cards:
                    cards[p["id"]] = dict(y=0, r=1)
                    ev.append(dict(time=dict(elapsed=int(rng.integers(20, 91)), extra=None), team=T(t), player=P(p),
                                   assist=dict(id=None, name=None), type="Card", detail="Red Card", comments=None))
            for k in range(3):                                  # three substitutions; a scorer may leave after scoring
                off, on, m = xi[t][10 - k], bench[t][k], int(rng.integers(60, 85))
                mins[off["id"]], mins[on["id"]] = m, 90 - m
                ev.append(dict(time=dict(elapsed=m, extra=None), team=T(t), player=P(off), assist=P(on), type="subst",
                               detail=f"Substitution {k + 1}", comments=None))
        ev.sort(key=lambda e: (e["time"]["elapsed"], e["time"]["extra"] or 0))
        f["goals"] = dict(home=score[h], away=score[a])
        f["score"]["fulltime"], f["score"]["halftime"] = dict(home=score[h], away=score[a]), dict(home=ht[h], away=ht[a])
        f["fixture"]["periods"] = dict(first=ts, second=ts + 3600)
        f["events"], f["lineups"], f["statistics"], f["players"] = ev, [], [], []
        for t in (h, a):
            f["lineups"].append(dict(team=T(t), coach=dict(id=t, name=f"Coach {t}"), formation="4-4-2",
                                     startXI=[dict(player=dict(P(p), number=j + 1, pos="G" if j == 0 else "M", grid=None)) for j, p in enumerate(xi[t])],
                                     substitutes=[dict(player=dict(P(p), number=j + 12, pos="M", grid=None)) for j, p in enumerate(bench[t])]))
            sot = score[t] + int(rng.poisson(3))
            shots = sot + int(rng.poisson(7))
            y = sum(min(cards.get(p["id"], {}).get("y", 0), 2) for p in xi[t])
            r = sum(cards.get(p["id"], {}).get("r", 0) for p in xi[t])
            vals = [sot, shots, int(rng.poisson(11)), int(rng.poisson(5.2 if t == h else 4.4)), int(rng.poisson(2)), "50%",
                    y or None, r or None, int(rng.poisson(3)), f"{(lh if t == h else la):.2f}"]
            f["statistics"].append(dict(team=T(t), statistics=[dict(type=k, value=v) for k, v in zip(STATS, vals)]))
            w = np.array([p["w"] for p in xi[t]])
            sh = rng.multinomial(shots, w / w.sum())
            so = rng.multinomial(sot, w / w.sum())
            pl = []
            for j, p in enumerate(xi[t] + bench[t]):
                g = pgoals.get(p["id"], 0)
                pl.append(dict(player=P(p), statistics=[dict(
                    games=dict(minutes=mins[p["id"]] or None, number=j + 1, position="G" if j == 0 else "M", rating=None, captain=False, substitute=j >= 11),
                    shots=dict(total=int(max(sh[j], so[j], g)) if j < 11 else None, on=int(max(so[j], g)) if j < 11 else None),
                    goals=dict(total=g or None, conceded=0, assists=None, saves=None),
                    cards=dict(yellow=min(cards.get(p["id"], {}).get("y", 0), 1), red=cards.get(p["id"], {}).get("r", 0)))]))
            f["players"].append(dict(team=T(t), players=pl))
        if lid in self.holes:                                    # a league with no player statistics
            f["players"] = []
        return f

    # ---------------------------------------------------------------- the API
    def fetch(self, path, params):
        self.calls.append((path, dict(params)))
        self.left -= 1
        head = {"x-ratelimit-requests-remaining": str(self.left), "x-ratelimit-remaining": "9"}
        env = lambda resp, err=None: (dict(get=path, parameters=params, errors=err or [], results=len(resp), response=resp), head)
        season = int(params["season"]) if "season" in params else None
        win = [(self.today + dt.timedelta(days=k)).isoformat() for k in (-1, 0, 1)]
        if self.free:
            if "ids" in params:
                return env([], dict(plan="Free plans do not have access to the Ids parameter."))
            if season is not None and not 2022 <= season <= 2024:
                return env([], dict(plan="Free plans do not have access to this season, try from 2022 to 2024."))
            if "date" in params and params["date"] not in win:
                return env([], dict(plan=f"Free plans do not have access to this date, try from {win[0]} to {win[-1]}."))
            fid = params.get("id") or params.get("fixture")
            if fid is not None and int(fid) in self.fx and self.fx[int(fid)]["fixture"]["date"][:10] not in win:
                return env([], dict(plan=f"Free plans do not have access to this date, try from {win[0]} to {win[-1]}."))
        if season is not None and self.block_before and season < self.block_before:
            return env([], dict(plan=f"Free plans do not have access to this season, try from {self.block_before} to {self.season}."))
        strip = lambda f: {k: v for k, v in f.items() if k not in ("events", "lineups", "statistics", "players")}
        if path == "leagues":
            return env([dict(league=dict(id=lid, name=LEAGUES[lid][1], type="League"), country=dict(name=LEAGUES[lid][0], code="XX"),
                             seasons=[dict(year=season, start=f"{season}-08-01", end=f"{season + 1}-05-30", current=season == self.season,
                                           coverage=dict(fixtures=dict(events=True, lineups=True, statistics_fixtures=True,
                                                                       statistics_players=lid not in self.holes)))])
                        for lid in sorted({t["league"] for t in self.teams.values()})]
                       + [dict(league=dict(id=72, name="Serie B", type="League"), country=dict(name="Brazil", code="BR"), seasons=[])])
        if path == "fixtures" and "id" in params:
            f = self.fx.get(int(params["id"]))
            return env([] if f is None else [f if self.id_detail else strip(f)])
        if path.startswith("fixtures/") and "fixture" in params:
            f = self.fx.get(int(params["fixture"]), {})
            return env(f.get(path.split("/")[1], []))
        if path == "fixtures" and "ids" in params:
            ids = [int(x) for x in str(params["ids"]).split("-")]
            if len(ids) > 20:
                return env([], dict(ids="The Ids field must contain a maximum of 20 fixtures ids."))
            return env([self.fx[i] for i in ids if i in self.fx])
        if path == "fixtures" and "league" in params:
            return env([strip(f) for f in self.fx.values() if f["league"]["id"] == int(params["league"]) and season == self.season])
        if path == "fixtures" and "date" in params:
            return env([strip(f) for f in self.fx.values() if f["fixture"]["date"][:10] == params["date"]])
        return env([], dict(endpoint="This endpoint does not exist."))

    # ---------------------------------------------------------------- football-data.co.uk CSVs
    def fd_name(self, tid: int) -> str:
        """football-data spells clubs its own way and never uses one name twice: 'Championship Aatown' -> 'Aatown',
        'Bundesliga Aatown' -> 'Bund Aatown'."""
        w = self.teams[tid]["name"].split()
        return w[-1] if self.teams[tid]["league"] == 40 else f"{w[0][:4]} {w[-1]}"

    def write_fd(self, root: Path) -> None:
        """This season's played games in football-data's layout, one file per division."""
        import pandas as pd
        code = {40: "E1", 41: "E2", 42: "E3", 79: "D2", 136: "I2", 141: "SP2"}
        rows = {}
        for f in self.fx.values():
            if f["fixture"]["status"]["short"] != "FT":
                continue
            h, a = f["teams"]["home"]["id"], f["teams"]["away"]["id"]
            st = {b["team"]["id"]: {x["type"]: x["value"] for x in b["statistics"]} for b in f["statistics"]}
            g = lambda t, k: int(st[t][k] or 0)
            r = dict(Div=code[f["league"]["id"]], Date=dt.date.fromisoformat(f["fixture"]["date"][:10]).strftime("%d/%m/%Y"),
                     HomeTeam=self.fd_name(h), AwayTeam=self.fd_name(a), FTHG=f["goals"]["home"], FTAG=f["goals"]["away"],
                     HTHG=f["score"]["halftime"]["home"], HTAG=f["score"]["halftime"]["away"], HS=g(h, "Total Shots"), AS=g(a, "Total Shots"),
                     HST=g(h, "Shots on Goal"), AST=g(a, "Shots on Goal"), HC=g(h, "Corner Kicks"), AC=g(a, "Corner Kicks"),
                     HY=g(h, "Yellow Cards"), AY=g(a, "Yellow Cards"), HR=g(h, "Red Cards"), AR=g(a, "Red Cards"))
            if f["league"]["id"] == 40:                          # only the English files carry the referee
                n = f["fixture"]["referee"].split(",")[0].split()
                r["Referee"] = f"{n[0][0]} {n[-1]}"
            rows.setdefault(r["Div"], []).append(r)
        p = root / f"{self.season % 100:02d}{(self.season + 1) % 100:02d}"
        p.mkdir(parents=True, exist_ok=True)
        for c, rr in rows.items():
            pd.DataFrame(rr).to_csv(p / f"{c}.csv", index=False)

    # ---------------------------------------------------------------- a board in the Altenar shape
    def board(self, fid: int, margin=0.06, path: Path | None = None) -> dict:
        f = self.fx[fid]
        h, a = f["teams"]["home"], f["teams"]["away"]
        gm = GameModel(*self.lam(h["id"], a["id"]), 0.0, 0.5)   # the simulation scores uniformly over 90 minutes
        mk, od, kids, n = [], [], [], [fid * 1000]
        H, A = h["name"], a["name"]

        def M(name, sels, child=None):
            """sels: (name, typeId, true p, competitorId, sv). Prices = 1 / (p * (1 + margin)), 2 decimals."""
            ids = []
            for nm, tyid, p, comp, sv in sels:
                n[0] += 1
                price = max(1.01, round(1 / min(p * (1 + margin), 0.99), 2))
                od.append(dict(id=n[0], name=nm, typeId=tyid, price=price, oddStatus=0, competitorId=comp, sv=sv))
                ids.append(n[0])
            n[0] += 1
            if child:
                kids.append(dict(id=n[0], childName=child, competitorId=sels[0][3], desktopOddIds=[ids]))
                return n[0]
            mk.append(dict(id=n[0], name=name, typeId=0, desktopOddIds=[ids], childMarketIds=[]))

        p = gm.p
        M("1X2", [(H, 1, p("1X2", side="1"), h["id"], None), ("Döntetlen", 2, p("1X2", side="X"), None, None), (A, 3, p("1X2", side="2"), a["id"], None)])
        M("Kétesély", [(f"{H} vagy döntetlen", 9, p("DC", side="1X"), None, None), (f"{H} vagy {A}", 10, p("DC", side="12"), None, None),
                       (f"Döntetlen vagy {A}", 11, p("DC", side="X2"), None, None)])
        M("Döntetlennél a tét visszajár", [(H, 1, p("DNB", side="1"), h["id"], None), (A, 3, p("DNB", side="2"), a["id"], None)])
        M("Mindkét csapat szerez gólt", [("Igen", 74, p("BTTS", side="yes"), None, None), ("Nem", 76, p("BTTS", side="no"), None, None)])
        for ln in (1.5, 2.25, 2.5, 3.0, 3.5):
            s = f"{ln:g}".replace(".", ",")
            M("Gólok száma", [(f"{s} felett", 12, p("TOTAL", side="over", line=ln), None, str(ln)), (f"{s} alatt", 13, p("TOTAL", side="under", line=ln), None, str(ln))])
        M("1. félidő - Gólok száma", [("0,5 felett", 12, p("TOTAL", "H1", "over", 0.5), None, "0.5"), ("0,5 alatt", 13, p("TOTAL", "H1", "under", 0.5), None, "0.5")])
        M("1. félidő - 1X2", [(H, 1, p("1X2", "H1", "1"), h["id"], None), ("Döntetlen", 2, p("1X2", "H1", "X"), None, None), (A, 3, p("1X2", "H1", "2"), a["id"], None)])
        M(f"{H} gólok száma", [("1,5 felett", 12, p("TEAM_TOTAL", side="over", line=1.5, who="home"), None, "1.5"),
                               ("1,5 alatt", 13, p("TEAM_TOTAL", side="under", line=1.5, who="home"), None, "1.5")])
        for ln in (-0.5, -0.75, -1.0, 0.25):
            M("Hendikep", [(f"{H} ({ln:+g})", 1714, p("AH", side="1", line=ln), h["id"], str(ln)),
                                  (f"{A} ({-ln:+g})", 1715, p("AH", side="2", line=-ln), a["id"], str(ln))])
        cm = lambda s, k, ln: float(poisson.sf(ln, k) if s == "over" else poisson.cdf(ln, k))
        M("Szögletek száma", [("9,5 felett", 12, cm("over", 9.6, 9.5), None, "9.5"), ("9,5 alatt", 13, cm("under", 9.6, 9.5), None, "9.5")])
        M("Büntetőlapok száma", [("3,5 felett", 12, cm("over", 3.9, 3.5), None, "3.5"), ("3,5 alatt", 13, cm("under", 3.9, 3.5), None, "3.5")])
        M("Büntetőpontok", [("40,5 felett", 12, 0.5, None, "40.5"), ("40,5 alatt", 13, 0.5, None, "40.5")])
        M("Pontos végeredmény", [(f"{i}:{j}", 500, p("CS", score=(i, j)), None, None) for i in range(3) for j in range(3)])
        M("Félidő/végeredmény", [(f"{H}/{H}", 600, 0.3, None, None), ("Döntetlen/Döntetlen", 601, 0.15, None, None)])
        M("Mindkét csapat szerez gólt mindkét félidőben", [("Igen", 704, 0.2, None, None), ("Nem", 705, 0.8, None, None)])
        M("Valami egészen új piac", [("Igen", 700, 0.5, None, None), ("Nem", 701, 0.5, None, None)])
        M("Mindkét csapat szerez gólt és 2,5 felett", [("Igen", 702, 0.4, None, None), ("Nem", 703, 0.6, None, None)])
        ids = []
        for t, lam_t in ((h, gm.lh), (a, gm.la)):
            sq = self.teams[t["id"]]["squad"]
            w = np.array([q["w"] for q in sq[:11]])
            for q, wq in list(zip(sq[:11], w / w.sum()))[7:11] + [(sq[15], 0.02)]:   # four starters and a bench player who never comes on
                ids.append(M(None, [("Bármikor", 800, 1 - math.exp(-lam_t * 0.84 * wq) if wq > 0.02 else 0.05, t["id"], None)], child=q["name"]))
        mk.append(dict(id=n[0] + 1, name="Gólszerző", typeId=0, desktopOddIds=[], childMarketIds=ids))
        mk.append(dict(mk[0]))                                   # the real page lists some markets twice under one id
        ev = dict(id=fid, name=f"{H} vs. {A}", startDate=f["fixture"]["date"].replace("+00:00", "Z"), sport="Labdarúgás",
                  champ=f["league"]["name"], category=f["league"]["country"],
                  competitors=[dict(id=h["id"], name=H, abbreviation=H[:3].upper()), dict(id=a["id"], name=A, abbreviation=A[:3].upper())],
                  markets=mk, odds=od, childMarkets=kids)
        cap = [dict(url="https://sb2frontend-altenar2.biahosted.com/api/widget/GetEventDetails", body=ev),
               dict(url="https://vegas.hu/other", body="not json")]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cap, ensure_ascii=False), encoding="utf-8")
        return ev
