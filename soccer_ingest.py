#!/usr/bin/env python3
"""soccer_ingest.py: API-Football v3 -> CSV tables for six second-tier leagues (spec section 2).

    python soccer_ingest.py run                  # daily: yesterday/today/tomorrow, detail of finished games, build, check, publish
    python soccer_ingest.py build                # rebuild tables + checks from the cache, no API calls
    python soccer_ingest.py lineups --date D     # pre-match starting XIs of that day's games (slip rule 6), ~2-4 calls
    python soccer_ingest.py status               # what is cached, what is queued, calls used today

Env: APIFOOTBALL_KEY (required for run/lineups), SOCCER_PLAN (free | paid, default free), SOCCER_CACHE (cache),
     SOCCER_OUT (data), SOCCER_BUDGET (95 calls/day), PROLYT_API_KEY, PROLYT_URL (https://analytiq.dcentai.tech).

FREE PLAN (tested 2026-11-20 with the user's key; confirmed by API-Football's helper): no current season by season
parameter, no `ids`, `date` only D-1 / D / D+1. So, per run:
    fixtures?date=D-1, D, D+1        3 calls: every game in the window (filtered to our six leagues)
    fixtures?id=X                    1 call per finished game: events, lineups, team and player statistics
A busy Saturday is about 50 of our games: ~55 calls. A game whose day has left the window can never be fetched
again on this plan, so the run must happen every day (the workflow runs twice a day). History before the first
run comes from football-data.co.uk (soccer_bandit.py download), not from this API.

PAID PLAN (SOCCER_PLAN=paid): the batched design below.
    leagues?season=S                 1 call: ids, names and the coverage block of every league that season
    fixtures?league=L&season=S       1 call per league season: the whole fixture list with FT and HT scores
    fixtures?date=D                  1 call per day: every game that day
    fixtures?ids=a-b-c (max 20)      1 call per 20 games: events, lineups, team statistics, player statistics
Never used: fixtures/statistics, fixtures/events, fixtures/players (one call per game), predictions, odds.
A finished game is fetched once and kept on disk for ever.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import pandas as pd

BASE = os.environ.get("APIFOOTBALL_BASE", "https://v3.football.api-sports.io")
CACHE = Path(os.environ.get("SOCCER_CACHE", "cache"))
OUT = Path(os.environ.get("SOCCER_OUT", "data"))
BUDGET = int(os.environ.get("SOCCER_BUDGET", "95"))
SLEEP = float(os.environ.get("SOCCER_SLEEP", "7"))          # free plan: 10 requests a minute
STOP_AT = 5                                                 # stop when the API says this many requests are left
PLAN = os.environ.get("SOCCER_PLAN", "free")

# id (from memory, VERIFIED on every run against the live /leagues response), country, name as the API spells it,
# football-data.co.uk division code
LEAGUES = {
    40: ("England", "Championship", "E1"),
    41: ("England", "League One", "E2"),
    42: ("England", "League Two", "E3"),
    79: ("Germany", "2. Bundesliga", "D2"),
    136: ("Italy", "Serie B", "I2"),
    141: ("Spain", "Segunda División", "SP2"),
}
FINISHED = {"FT", "AET", "PEN"}
DEAD = {"PST", "CANC", "ABD", "AWD", "WO"}
STAT = {"Total Shots": "shots", "Shots on Goal": "shots_on_target", "Corner Kicks": "corners", "Fouls": "fouls",
        "Offsides": "offsides", "Yellow Cards": "yellow", "Red Cards": "red", "Ball Possession": "possession",
        "Goalkeeper Saves": "saves", "expected_goals": "xg"}
TABLES = ("fixtures", "team_match_stats", "events", "player_match_stats", "lineups", "teams", "players", "coverage")
PK = {"fixtures": ["fixture_id"], "team_match_stats": ["fixture_id", "team_id"], "events": ["fixture_id", "seq"],
      "player_match_stats": ["fixture_id", "player_id"], "lineups": ["fixture_id", "player_id"],
      "teams": ["team_id", "season"], "players": ["player_id"], "coverage": ["league_id", "season"]}
FK = [("team_match_stats", "fixture_id", "fixtures", "fixture_id"), ("events", "fixture_id", "fixtures", "fixture_id"),
      ("player_match_stats", "fixture_id", "fixtures", "fixture_id"), ("lineups", "fixture_id", "fixtures", "fixture_id"),
      ("team_match_stats", "team_id", "teams", "team_id"), ("player_match_stats", "player_id", "players", "player_id"),
      ("lineups", "player_id", "players", "player_id"), ("fixtures", "league_id", "coverage", "league_id"),
      ("fixtures", "home_id", "teams", "team_id"), ("fixtures", "away_id", "teams", "team_id")]


class Stop(Exception):
    """Out of budget for today: stop calling, build from what is on disk."""


class PlanError(Exception):
    """The API refused this request for plan reasons (e.g. a season the free plan does not include)."""


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def current_season(d: dt.date | None = None) -> int:
    d = d or now_utc().date()
    return d.year if d.month >= 7 else d.year - 1


def jload(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def jsave(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# ------------------------------------------------------------------ API
class Api:
    def __init__(self, key: str | None = None, fetch=None):
        self.key, self.fetch = key or os.environ.get("APIFOOTBALL_KEY", ""), fetch or self._http
        self.state = jload(CACHE / "state.json", {}) or {}
        self.today = now_utc().date().isoformat()
        self.used = int(self.state.get("calls", {}).get(self.today, 0))
        self.left = None                                   # from the response header, once seen
        self.last = 0.0

    def _http(self, path: str, params: dict):
        req = urllib.request.Request(f"{BASE}/{path}?{urllib.parse.urlencode(params)}", headers={"x-apisports-key": self.key})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()), {k.lower(): v for k, v in r.headers.items()}

    def get(self, path: str, **params) -> list:
        if self.used >= BUDGET or (self.left is not None and self.left <= STOP_AT):
            raise Stop(f"budget: {self.used} calls used today, API says {self.left} left")
        wait = SLEEP - (time.time() - self.last)
        if self.last and wait > 0:
            time.sleep(wait)
        try:
            body, head = self.fetch(path, params)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise Stop("HTTP 429: rate limit") from e
            raise
        finally:
            self.last = time.time()
            self.used += 1
            self.state.setdefault("calls", {})[self.today] = self.used
        rem = head.get("x-ratelimit-requests-remaining")
        if rem is not None and str(rem).lstrip("-").isdigit():
            self.left = int(rem)
        with (CACHE / "calls.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{now_utc().isoformat(timespec='seconds')} {path} {params} results={body.get('results')} left={self.left}\n")
        jsave(CACHE / "state.json", self.state)
        err = body.get("errors")
        if err:                                             # the API answers 200 with an errors object
            msg = json.dumps(err, ensure_ascii=False)
            low = msg.lower()
            if "plan" in low or "subscription" in low:
                raise PlanError(msg)
            if "limit" in low or "requests" in low:
                raise Stop(f"API: {msg}")
            raise RuntimeError(f"API error on {path} {params}: {msg}")
        return body.get("response", [])


# ------------------------------------------------------------------ fetch steps
def slim(f: dict) -> dict:
    """The part of a fixture kept in the index (list and date calls)."""
    fx, lg, tm, sc = f["fixture"], f["league"], f["teams"], f.get("score", {})
    ht, ft = sc.get("halftime") or {}, sc.get("fulltime") or {}
    return dict(fixture_id=fx["id"], league_id=lg["id"], season=lg["season"], round=lg.get("round"), date=fx["date"],
                ts=fx.get("timestamp"), status=(fx.get("status") or {}).get("short"), referee=fx.get("referee"),
                venue=(fx.get("venue") or {}).get("name"), home_id=tm["home"]["id"], home=tm["home"]["name"],
                away_id=tm["away"]["id"], away=tm["away"]["name"],
                ft_home=ft.get("home") if ft.get("home") is not None else (f.get("goals") or {}).get("home"),
                ft_away=ft.get("away") if ft.get("away") is not None else (f.get("goals") or {}).get("away"),
                ht_home=ht.get("home"), ht_away=ht.get("away"))


def verify_leagues(api: Api | None, season: int, quiet: bool = False) -> dict:
    """One call. Returns {league_id: coverage row}. Ids are matched by country + name, never trusted from memory:
    a wrong id in LEAGUES is corrected here and reported."""
    p = CACHE / f"leagues_{season}.json"
    resp = jload(p)
    if resp is None:
        resp = [r for r in api.get("leagues", season=season) if r.get("country", {}).get("name") in {c for c, _, _ in LEAGUES.values()}]
        jsave(p, resp)
    found, say = {}, (lambda *a: None) if quiet else print
    for lid, (country, name, code) in LEAGUES.items():
        hit = [r for r in resp if r["country"]["name"] == country and r["league"]["name"].lower() == name.lower()]
        if len(hit) != 1:
            hit = [r for r in resp if r["league"]["id"] == lid and r["country"]["name"] == country]
            if len(hit) == 1:
                say(f"  NOTE {season}: id {lid} is called '{hit[0]['league']['name']}' by the API (expected '{name}')")
        if len(hit) != 1:
            say(f"  ERROR {season}: {country} / {name} not found in /leagues: league skipped. Names on offer: "
                  f"{sorted(r['league']['name'] for r in resp if r['country']['name'] == country)[:12]}")
            continue
        r = hit[0]
        if r["league"]["id"] != lid:
            say(f"  WARNING {season}: {name} has id {r['league']['id']}, not {lid}: using the API's id")
        s = next((x for x in r.get("seasons", []) if x.get("year") == season), None)
        if s is None:
            say(f"  ERROR {season}: {name} has no season {season}")
            continue
        cov = (s.get("coverage") or {}).get("fixtures") or {}
        found[r["league"]["id"]] = dict(league_id=r["league"]["id"], season=season, league=name, country=country, fd_code=code,
                                        start=s.get("start"), end=s.get("end"), current=s.get("current"),
                                        events=bool(cov.get("events")), lineups=bool(cov.get("lineups")),
                                        statistics_fixtures=bool(cov.get("statistics_fixtures")),
                                        statistics_players=bool(cov.get("statistics_players")))
    return found


def fetch_list(api: Api, index: dict, lid: int, season: int, max_age_days: float | None) -> None:
    """The whole fixture list of a league season. The current season's list is fetched again every `max_age_days`
    (postponed and re-arranged games); a list read back from disk never overwrites newer entries of the index."""
    p, st = CACHE / f"fixtures_{lid}_{season}.json", api.state.setdefault("lists", {})
    age = (now_utc() - dt.datetime.fromisoformat(st[p.name])).total_seconds() / 86400 if p.name in st else None
    fresh = not p.exists() or age is None or (max_age_days is not None and age >= max_age_days)
    if fresh:
        resp = api.get("fixtures", league=lid, season=season)
        jsave(p, resp)
        st[p.name] = now_utc().isoformat()
    else:
        resp = jload(p)
    for f in resp:
        if fresh or str(f["fixture"]["id"]) not in index:
            index[str(f["fixture"]["id"])] = slim(f)


def fetch_dates(api: Api, index: dict, ids: set, upto: dt.date) -> None:
    """One call per missing day, at most 5 days back (a longer gap is covered by the weekly list refresh)."""
    last = api.state.get("last_date")
    d = max(dt.date.fromisoformat(last) + dt.timedelta(days=1), upto - dt.timedelta(days=4)) if last else upto
    while d <= upto:
        for f in api.get("fixtures", date=d.isoformat()):
            if f["league"]["id"] in ids:
                index[str(f["fixture"]["id"])] = slim(f)
        api.state["last_date"] = d.isoformat()
        d += dt.timedelta(days=1)


def detail_path(fid) -> Path:
    return CACHE / "match" / f"{fid}.json"


def queue(index: dict, seasons: list[int]) -> list[int]:
    """Finished games without detail: current season first (newest first), then the older seasons."""
    cut = now_utc().timestamp() - 6 * 3600                  # statistics settle a few hours after the whistle
    todo = [f for f in index.values() if f["status"] in FINISHED and (f.get("ts") or 0) < cut
            and f["season"] in seasons and not detail_path(f["fixture_id"]).exists()]
    todo.sort(key=lambda f: (seasons.index(f["season"]), -(f.get("ts") or 0)))
    return [f["fixture_id"] for f in todo]


def work_queue(api: Api, ids: list[int]) -> int:
    miss, done = api.state.setdefault("missing", {}), 0
    ids = [i for i in ids if miss.get(str(i), 0) < 3]
    for k in range(0, len(ids), 20):
        chunk = ids[k:k + 20]
        resp = api.get("fixtures", ids="-".join(map(str, chunk)))
        got = set()
        for f in resp:
            if (f["fixture"].get("status") or {}).get("short") in FINISHED:
                jsave(detail_path(f["fixture"]["id"]), f)
                got.add(f["fixture"]["id"])
        for i in set(chunk) - got:
            miss[str(i)] = miss.get(str(i), 0) + 1
        done += len(got)
    return done


def run(api: Api) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cur = current_season()
    seasons = [cur, cur - 1, cur - 2]
    index = jload(CACHE / "index.json", {}) or {}
    blocked = api.state.setdefault("plan_blocked", {})
    try:
        for s in seasons:
            if str(s) in blocked and blocked[str(s)].get("day") == api.today:
                continue
            try:
                cov = verify_leagues(api, s)
                for lid in cov:
                    fetch_list(api, index, lid, s, 7 if s == cur else None)
                if s == cur:
                    fetch_dates(api, index, set(cov), now_utc().date() - dt.timedelta(days=1))
                jsave(CACHE / "index.json", index)
                n = work_queue(api, queue(index, [s]))
                print(f"season {s}: {n} games fetched this run")
            except PlanError as e:
                blocked[str(s)] = dict(day=api.today, msg=str(e))
                print(f"season {s}: REFUSED BY PLAN: {e}\n  -> this key cannot read season {s}. Nothing was guessed; "
                      f"the season is skipped today.")
    except Stop as e:
        print(f"stopped: {e}")
    finally:
        jsave(CACHE / "index.json", index)
        jsave(CACHE / "queue.json", queue(index, seasons))
        jsave(CACHE / "state.json", api.state)
    print(f"calls used today: {api.used} (API says {api.left} left); queue: {len(queue(index, seasons))} games still to fetch")


def detail_ok(f: dict) -> bool:
    """A fixtures?id response carries the match detail (it did in the documentation; not yet seen live on Free)."""
    return any(k in f for k in ("events", "statistics", "players", "lineups"))


def fetch_detail(api: Api, fid: int) -> dict | None:
    """One finished game. fixtures?id first; if that answer has no detail blocks, the four per-game endpoints."""
    mode = api.state.get("detail_mode", "id")
    if mode == "id":
        resp = api.get("fixtures", id=fid)
        if not resp:
            return None
        f = resp[0]
        if detail_ok(f):
            return f
        api.state["detail_mode"] = "split"
        print("  NOTE: fixtures?id returned no events/statistics: switching to 4 calls per game (events, lineups, statistics, players)")
    else:
        resp = api.get("fixtures", id=fid)
        if not resp:
            return None
        f = resp[0]
    for key, path in (("events", "fixtures/events"), ("lineups", "fixtures/lineups"), ("statistics", "fixtures/statistics"), ("players", "fixtures/players")):
        f[key] = api.get(path, fixture=fid)
    return f


def run_free(api: Api) -> None:
    """Free plan: the D-1 / D / D+1 window, then one detail call per finished game of our leagues."""
    CACHE.mkdir(parents=True, exist_ok=True)
    index = jload(CACHE / "index.json", {}) or {}
    today = now_utc().date()
    window = [today - dt.timedelta(days=1), today, today + dt.timedelta(days=1)]
    ours, seen = set(LEAGUES), api.state.setdefault("league_check", {})
    fetched = lost_now = 0
    try:
        for d in window:
            day = api.get("fixtures", date=d.isoformat())
            for f in day:
                lid, cty, nm = f["league"]["id"], f["league"].get("country"), f["league"].get("name")
                for want, (c, n, _) in LEAGUES.items():                    # the ids are checked on every response
                    if c == cty and n.lower() == str(nm).lower() and lid != want:
                        print(f"  WARNING: {c} {n} appears with id {lid}, not {want}: fix LEAGUES")
                if lid in ours:
                    seen[str(lid)] = nm
                    index[str(f["fixture"]["id"])] = slim(f)
        jsave(CACHE / "index.json", index)
        first = window[0].isoformat()
        cut = now_utc().timestamp() - 3 * 3600                             # statistics settle soon after the whistle
        todo = sorted((f for f in index.values() if f["status"] in FINISHED and f["season"] >= current_season() and not detail_path(f["fixture_id"]).exists()
                       and (f.get("ts") or 0) < cut), key=lambda f: f.get("ts") or 0)
        lost = api.state.setdefault("lost", {})
        for f in index.values():                                           # listed once, now out of the window for good
            if (f["season"] >= current_season() and f["date"][:10] < first and f["status"] not in DEAD
                    and not detail_path(f["fixture_id"]).exists() and str(f["fixture_id"]) not in lost):
                lost[str(f["fixture_id"])] = f["date"][:10]
                lost_now += 1
        for f in todo:
            if f["date"][:10] < first:
                continue
            try:
                g = fetch_detail(api, f["fixture_id"])
            except PlanError as e:
                print(f"  REFUSED BY PLAN for fixture {f['fixture_id']} ({f['date'][:10]}): {e}")
                continue
            if g and (g["fixture"].get("status") or {}).get("short") in FINISHED:
                jsave(detail_path(f["fixture_id"]), g)
                fetched += 1
    except Stop as e:
        print(f"stopped: {e}")
    except PlanError as e:
        print(f"REFUSED BY PLAN: {e}")
    finally:
        jsave(CACHE / "index.json", index)
        jsave(CACHE / "state.json", api.state)
    missing = [l for l in LEAGUES if str(l) not in seen]
    print(f"window {window[0]}..{window[-1]}: {sum(1 for f in index.values() if f['date'][:10] >= window[0].isoformat())} of our games listed, "
          f"{fetched} finished games fetched; calls used today: {api.used} (API says {api.left} left)")
    if missing:
        print(f"  leagues not seen in any window yet (no games, or a wrong id): {[LEAGUES[l][1] for l in missing]}")
    if lost_now or api.state.get("lost"):
        print(f"  LOST: {lost_now} new, {len(api.state['lost'])} in all: finished games whose date left the window before they were "
              f"fetched (a missed run). They are graded from nothing and kept out of the player model.")


def lineups_cmd(api: Api, date: str, limit: int = 20) -> None:
    """Pre-match lineups: written to soccer_prematch_lineups.csv (never into the finished-game cache)."""
    index = jload(CACHE / "index.json", {}) or {}
    ours = set(LEAGUES) if PLAN == "free" else set(verify_leagues(api, current_season(dt.date.fromisoformat(date))))
    day = [f for f in api.get("fixtures", date=date) if f["league"]["id"] in ours]
    for f in day:
        index[str(f["fixture"]["id"])] = slim(f)
    jsave(CACHE / "index.json", index)
    now = now_utc().timestamp()
    rows, ids = [], [f["fixture"]["id"] for f in day if (f["fixture"].get("status") or {}).get("short") == "NS"
                     and 0 < (f["fixture"].get("timestamp") or 0) - now < 2 * 3600]           # kick-off within 2 hours
    try:
        if PLAN == "free":
            for fid in ids[:limit]:                                         # 1 call per game on Free: capped
                rows += lineup_rows(dict(fixture=dict(id=fid), lineups=api.get("fixtures/lineups", fixture=fid)))
        else:
            for k in range(0, len(ids), 20):
                for f in api.get("fixtures", ids="-".join(map(str, ids[k:k + 20]))):
                    rows += lineup_rows(f)
    except Stop as e:
        print(f"stopped: {e}")
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OUT / "soccer_prematch_lineups.csv") if (OUT / "soccer_prematch_lineups.csv").exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(new):
        new["captured"] = now_utc().isoformat(timespec="seconds")
        old = old[~old.fixture_id.isin(new.fixture_id)] if len(old) else old
    pd.concat([old, new]).to_csv(OUT / "soccer_prematch_lineups.csv", index=False)
    print(f"{date}: {len(ids)} games, lineups out for {new.fixture_id.nunique() if len(new) else 0} "
          f"(they appear about an hour before kickoff); calls used today: {api.used}")


# ------------------------------------------------------------------ tables
def lineup_rows(f: dict) -> list[dict]:
    out = []
    for lu in f.get("lineups") or []:
        for key, start in (("startXI", True), ("substitutes", False)):
            for x in lu.get(key) or []:
                p = x.get("player") or {}
                if p.get("id") is None:
                    continue
                out.append(dict(fixture_id=f["fixture"]["id"], team_id=lu["team"]["id"], player_id=p["id"], player=p.get("name"),
                                number=p.get("number"), position=p.get("pos"), starting=start, formation=lu.get("formation")))
    return out


def _num(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().rstrip("%")
        try:
            return float(v)
        except ValueError:
            return None
    return float(v)


def event_type(e: dict) -> str:
    t, d = (e.get("type") or "").lower(), (e.get("detail") or "").lower()
    if t == "goal":
        return "missed_penalty" if "missed" in d else "own_goal" if "own" in d else "penalty" if "penalty" in d else "goal"
    if t == "card":
        return "second_yellow" if "second" in d else "red" if "red" in d else "yellow"
    return "sub" if t == "subst" else "var" if t == "var" else "other"


def parse_match(f: dict) -> dict:
    """One cached fixture -> rows for every table, plus the three sanity checks for that game."""
    s = slim(f)
    fid, hid, aid = s["fixture_id"], s["home_id"], s["away_id"]
    other = {hid: aid, aid: hid}
    lus = lineup_rows(f)
    team_of = {r["player_id"]: r["team_id"] for r in lus}
    prow = []
    for blk in f.get("players") or []:
        tid = blk["team"]["id"]
        for x in blk.get("players") or []:
            st, p = (x.get("statistics") or [{}])[0], x.get("player") or {}
            if p.get("id") is None:
                continue
            g, sh, gl, cd = st.get("games") or {}, st.get("shots") or {}, st.get("goals") or {}, st.get("cards") or {}
            team_of.setdefault(p["id"], tid)
            prow.append(dict(fixture_id=fid, team_id=tid, player_id=p["id"], player=p.get("name"), minutes=int(g.get("minutes") or 0),
                             goals=int(gl.get("total") or 0), assists=int(gl.get("assists") or 0), shots=int(sh.get("total") or 0),
                             shots_on_target=int(sh.get("on") or 0), yellow=int(cd.get("yellow") or 0), red=int(cd.get("red") or 0),
                             started=not bool(g.get("substitute")), position=g.get("position")))
    ev, og_rule = [], {}
    for k, e in enumerate(f.get("events") or []):
        typ, tid, pid = event_type(e), (e.get("team") or {}).get("id"), (e.get("player") or {}).get("id")
        row = dict(fixture_id=fid, seq=k, minute=(e.get("time") or {}).get("elapsed"), extra=(e.get("time") or {}).get("extra"),
                   team_id=tid, player_id=pid, player=(e.get("player") or {}).get("name"), assist_id=(e.get("assist") or {}).get("id"),
                   type=typ, detail=e.get("detail"), comments=e.get("comments"), scoring_team_id=None)
        if typ in ("goal", "penalty"):
            row["scoring_team_id"] = tid
        elif typ == "own_goal":
            own = team_of.get(pid)                             # the scorer's own team, read from the lineups
            row["scoring_team_id"] = other.get(own) if own in other else None
            og_rule[k] = "lineup" if own in other else "unknown"
        ev.append(row)
    goals = lambda t: sum(1 for r in ev if r["scoring_team_id"] == t)
    unk = [r for r in ev if r["type"] == "own_goal" and r["scoring_team_id"] is None]
    if unk and s["ft_home"] is not None:                       # no lineup to read the scorer's team from: the only
        for guess in ("event team scored", "event team conceded"):   # assignment that reproduces the score, or none
            for r in unk:
                r["scoring_team_id"] = r["team_id"] if guess == "event team scored" else other.get(r["team_id"])
            if goals(hid) == s["ft_home"] and goals(aid) == s["ft_away"]:
                break
        else:
            for r in unk:
                r["scoring_team_id"] = None
    seen_y = set()
    for r in ev:                                               # a second yellow listed as a plain yellow
        if r["type"] == "yellow":
            if (r["team_id"], r["player_id"]) in seen_y:
                r["type"] = "second_yellow"
            seen_y.add((r["team_id"], r["player_id"]))
    tstat = []
    for blk in f.get("statistics") or []:
        row = dict(fixture_id=fid, team_id=blk["team"]["id"], is_home=blk["team"]["id"] == hid,
                   goals=s["ft_home"] if blk["team"]["id"] == hid else s["ft_away"])
        row.update({c: None for c in STAT.values()})
        for x in blk.get("statistics") or []:
            if x.get("type") in STAT:
                row[STAT[x["type"]]] = _num(x.get("value"))
        for c in ("yellow", "red", "corners", "offsides", "saves", "fouls"):   # the API sends null for zero
            if row[c] is None and row["shots"] is not None:
                row[c] = 0.0
        tstat.append(row)
    chk = dict(check_goals=None, check_players=None, check_ht=None)
    if ev or (s["ft_home"] == 0 and s["ft_away"] == 0 and f.get("events") is not None):
        chk["check_goals"] = goals(hid) == s["ft_home"] and goals(aid) == s["ft_away"]
        if s["ht_home"] is not None:
            h1 = lambda t: sum(1 for r in ev if r["scoring_team_id"] == t and (r["minute"] or 0) <= 45)
            chk["check_ht"] = h1(hid) == s["ht_home"] and h1(aid) == s["ht_away"]
    if prow:
        pg = lambda t: sum(r["goals"] for r in prow if r["team_id"] == t)
        og = lambda t: sum(1 for r in ev if r["type"] == "own_goal" and r["scoring_team_id"] == t)
        chk["check_players"] = pg(hid) + og(hid) == s["ft_home"] and pg(aid) + og(aid) == s["ft_away"]
    fx = dict(s, has_events=bool(ev), has_stats=bool(tstat), has_players=bool(prow), has_lineups=bool(lus), **chk)
    return dict(fixtures=fx, team_match_stats=tstat, events=ev, player_match_stats=prow, lineups=lus)


def build() -> dict:
    """Cache -> one CSV per table and season, DuckDB, relationships JSON. Prints the sanity checks."""
    OUT.mkdir(parents=True, exist_ok=True)
    index = jload(CACHE / "index.json", {}) or {}
    cov = {}
    for p in sorted(CACHE.glob("leagues_*.json")):
        season = int(p.stem.split("_")[1])
        cov.update({(lid, season): row for lid, row in verify_leagues(None, season, quiet=True).items()})
    acc = {t: [] for t in TABLES}
    for f in index.values():
        p = detail_path(f["fixture_id"])
        if p.exists():
            m = parse_match(jload(p))
            acc["fixtures"].append(m["fixtures"])
            for t in ("team_match_stats", "events", "player_match_stats", "lineups"):
                acc[t] += [dict(r, season=f["season"]) for r in m[t]]
        else:
            acc["fixtures"].append(dict(f, has_events=False, has_stats=False, has_players=False, has_lineups=False,
                                        check_goals=None, check_players=None, check_ht=None))
        for side in ("home", "away"):
            acc["teams"].append(dict(team_id=f[f"{side}_id"], team=f[side], league_id=f["league_id"], season=f["season"]))
    # Free plan: no coverage block for the current season. Inferred from what the detail calls returned (a league
    # season counts as covered for a block when at least 80% of its fetched games have it). Recorded as such.
    F0 = pd.DataFrame(acc["fixtures"])
    if len(F0):
        got = F0[F0[["has_events", "has_stats", "has_players", "has_lineups"]].any(axis=1)]
        for (lid, season), g in got.groupby(["league_id", "season"]):
            if (lid, season) not in cov:
                c, n, code = LEAGUES.get(lid, ("?", "?", None))
                cov[(lid, season)] = dict(league_id=lid, season=season, league=n, country=c, fd_code=code, start=None, end=None, current=None,
                                          events=bool(g.has_events.mean() >= 0.8), lineups=bool(g.has_lineups.mean() >= 0.8),
                                          statistics_fixtures=bool(g.has_stats.mean() >= 0.8), statistics_players=bool(g.has_players.mean() >= 0.8),
                                          source=f"inferred from {len(g)} games")
    for k in cov:
        cov[k].setdefault("source", "api coverage block")
    acc["coverage"] = list(cov.values())
    D = {t: pd.DataFrame(acc[t]) for t in TABLES}
    if len(D["teams"]):
        D["teams"] = D["teams"].drop_duplicates(["team_id", "season"])
    if len(D["fixtures"]):
        D["fixtures"]["date"] = pd.to_datetime(D["fixtures"]["date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        D["fixtures"] = D["fixtures"].sort_values(["date", "fixture_id"])
    pl = pd.concat([D[t][["player_id", "player", "team_id", "fixture_id"]] for t in ("lineups", "player_match_stats") if len(D[t])]) \
        if len(D["lineups"]) or len(D["player_match_stats"]) else pd.DataFrame(columns=["player_id", "player", "team_id", "fixture_id"])
    D["players"] = pl.sort_values("fixture_id").drop_duplicates("player_id", keep="last").drop(columns="fixture_id")
    # A lost cache must never shrink the published tables: a season that had more finished games with detail on disk
    # than it has now is not overwritten.
    for p in sorted(OUT.glob("soccer_*_fixtures.csv")):
        old = pd.read_csv(p, usecols=["season", "has_events"])
        new = D["fixtures"][D["fixtures"].season == old.season.iloc[0]] if len(D["fixtures"]) and len(old) else pd.DataFrame()
        if len(old) and int(old.has_events.sum()) > (int(new.has_events.sum()) if len(new) else 0) and not os.environ.get("SOCCER_ALLOW_SHRINK"):
            sys.exit(f"ERROR: {p.name} holds {int(old.has_events.sum())} games with detail, the cache now has {int(new.has_events.sum()) if len(new) else 0}. "
                     f"The cache was lost or cut. Tables left untouched. Restore the cache, or set SOCCER_ALLOW_SHRINK=1 to rebuild from what is there.")
    files = []
    for t in TABLES:
        df = D[t]
        if df.empty:
            continue
        if "season" in df and t not in ("players",):
            for s, g in df.groupby("season"):
                files.append(OUT / f"soccer_{s}_{t}.csv")
                g.to_csv(files[-1], index=False)
        else:
            files.append(OUT / f"soccer_all_{t}.csv")
            df.to_csv(files[-1], index=False)
    rel = verify_keys(D)
    jsave(OUT / "soccer_relationships.json", rel)
    sanity(D)
    try:
        import duckdb
        con = duckdb.connect(str(OUT / "soccer.duckdb"))
        for t in TABLES:
            if not D[t].empty:
                df = D[t]                                    # noqa: F841 (read by duckdb)
                con.execute(f"CREATE OR REPLACE TABLE soccer_{t} AS SELECT * FROM df")
        con.close()
    except ImportError:
        print("duckdb not installed: CSVs written, no .duckdb file")
    return dict(files=files, tables=D)


def verify_keys(D: dict) -> dict:
    """A key is declared only after it has been checked on the data."""
    rel = dict(primary_keys={}, foreign_keys=[], rejected=[])
    for t, cols in PK.items():
        df = D[t]
        if df.empty:
            continue
        bad = int(df[cols].isna().any(axis=1).sum() + df.duplicated(cols).sum())
        if bad == 0:
            rel["primary_keys"][f"soccer_{t}"] = cols
        else:
            rel["rejected"].append(f"PK soccer_{t}{cols}: {bad} null or duplicate rows")
    for t, c, pt, pc in FK:
        if D[t].empty or D[pt].empty:
            continue
        orphans = int((~D[t][c].dropna().isin(D[pt][pc])).sum())
        if orphans == 0:
            rel["foreign_keys"].append(dict(table=f"soccer_{t}", column=c, references=f"soccer_{pt}", ref_column=pc))
        else:
            rel["rejected"].append(f"FK soccer_{t}.{c} -> soccer_{pt}.{pc}: {orphans} orphans")
    print(f"keys: {len(rel['primary_keys'])} primary, {len(rel['foreign_keys'])} foreign declared; rejected: {rel['rejected'] or 'none'}")
    return rel


def sanity(D: dict) -> bool:
    F = D["fixtures"]
    if F.empty:
        print("Sanity: no fixtures yet")
        return True
    ok = True
    names = dict(check_goals="1. goals in events (incl. own goals) = full-time score",
                 check_players="2. player goals + opponent's own goals = team goals",
                 check_ht="3. half-time score = goals up to minute 45(+)")
    for c, label in names.items():
        x = F[F[c].notna()]
        bad = x[x[c] == False]                               # noqa: E712
        print(f"Sanity {label}: {'OK' if bad.empty else 'FAIL'} ({len(x) - len(bad)} of {len(x)} games)"
              + ("" if bad.empty else f"  first failures: {bad.fixture_id.head(8).tolist()}"))
        ok &= bad.empty
    E = D["events"]
    if len(E):
        og = E[E.type == "own_goal"]
        same = int((og.scoring_team_id == og.team_id).sum())
        print(f"Own goals: {len(og)}; the API lists {same} under the team that got the goal, {len(og) - same - int(og.scoring_team_id.isna().sum())} "
              f"under the scorer's own team, {int(og.scoring_team_id.isna().sum())} unreadable")
    done = F[F.status.isin(FINISHED)]
    print(f"Fixtures: {len(F)} listed, {len(done)} finished, {int(done.has_events.sum())} with detail; "
          f"with stats {int(done.has_stats.sum())}, players {int(done.has_players.sum())}, lineups {int(done.has_lineups.sum())}")
    if not ok:
        print("A failed check means a field is misread: fix it before anything is graded (the grader skips failed games).")
    return ok


def publish(files: list[Path]) -> bool:
    key, url = os.environ.get("PROLYT_API_KEY"), os.environ.get("PROLYT_URL", "https://analytiq.dcentai.tech").rstrip("/")
    if not key:
        print("PROLYT_API_KEY not set: upload skipped")
        return True
    ok = True
    for f in files:
        b = uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n1\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{f.name}\"\r\nContent-Type: text/csv\r\n\r\n").encode() \
            + f.read_bytes() + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(f"{url}/upload?overwrite=1", data=body, method="POST",
                                     headers={"X-API-Key": key, "Content-Type": f"multipart/form-data; boundary={b}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                print(f"  uploaded {f.name}: HTTP {r.status}")
        except Exception as e:                                 # noqa: BLE001
            ok = False
            print(f"  UPLOAD FAILED {f.name}: {e}")
    return ok


def status() -> None:
    index, st = jload(CACHE / "index.json", {}) or {}, jload(CACHE / "state.json", {}) or {}
    F = pd.DataFrame(index.values())
    if F.empty:
        print("nothing cached yet")
        return
    F["detail"] = [detail_path(i).exists() for i in F.fixture_id]
    F["finished"] = F.status.isin(FINISHED)
    print(F.groupby(["season", "league_id"]).agg(listed=("fixture_id", "size"), finished=("finished", "sum"), detail=("detail", "sum")).to_string())
    print(f"calls today: {st.get('calls', {}).get(now_utc().date().isoformat(), 0)}; last date call: {st.get('last_date')}; "
          f"plan refusals: {st.get('plan_blocked') or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run")
    sub.add_parser("build")
    sub.add_parser("status")
    s = sub.add_parser("lineups")
    s.add_argument("--date", required=True)
    s.add_argument("--max", type=int, default=20, help="free plan: at most this many games (1 request each)")
    a = ap.parse_args()
    cmd = a.cmd or "run"
    if cmd in ("run", "lineups") and not os.environ.get("APIFOOTBALL_KEY"):
        sys.exit("APIFOOTBALL_KEY is not set (keep the key in an env var or a GitHub secret, never in the repo)")
    if cmd == "status":
        return status()
    if cmd == "lineups":
        return lineups_cmd(Api(), a.date, a.max)
    if cmd == "run":
        print(f"plan: {PLAN}")
        (run_free if PLAN == "free" else run)(Api())
    res = build()
    if cmd == "run" and not publish(res["files"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
