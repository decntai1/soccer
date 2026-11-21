#!/usr/bin/env python3
"""College basketball (NCAAB) grading model + bandit for vegas.hu boards (Altenar .net.json captures made by
capture_board.py). One file: stdlib + numpy + pandas + tabulate. Spec: NCAAB_MODEL.md.

    python ncaab_grade.py selftest                                   # must print PASS first
    python ncaab_grade.py markets BOARDS_DIR                         # every market name and how it is typed
    python ncaab_grade.py price   --date 2026-11-18 --boards DIR [--capture 1730] [--teams teams.csv]
    python ncaab_grade.py results --date 2026-11-18 --out results/2026-11-18.txt     # ESPN (spec 5), UNTESTED
    python ncaab_grade.py grade   --date 2026-11-18 --results results/2026-11-18.txt
    python ncaab_grade.py report                                     # ledger report, open-vs-close, bandit
    python ncaab_grade.py slate   --date 2026-11-18 --out slate/2026-11-18.txt       # ESPN: who plays, neutral courts
    python ncaab_grade.py results --date 2025-11-03 --to 2026-04-07 --out results    # backfill a range (one file a day)
    python ncaab_grade.py fit     --results results --out ratings    # own ratings from results -> ratings/teams.csv
    python ncaab_grade.py picks   --date 2026-11-18                  # predicted winners with odds and expected return
    python ncaab_grade.py pack    --boards DIR --out boards/2026-11-18/1730.rows.csv.gz   # small file for git
    python ncaab_grade.py lines   --from 2025-11-01 --out lines     # CollegeBasketballData.com lines, needs CBBD_API_KEY
    python ncaab_grade.py slips   --date 2026-11-18                  # ten paper multis at about 3x / 7x / 10x
    python ncaab_grade.py sync    [--backfill-from 2025-11-03]       # everything, in order: price, slips, grade, report

Stored in ncaab.sqlite next to this file (override: NCAAB_DB=path). Several captures of one night are kept apart
(--capture, default = the boards' file time HHMM). For every GAME the last capture taken before its tip is the one
that is pooled; earlier ones are only compared with it. Captures are ordered by the time they were taken
(captured_utc: the boards' file time, or --captured-at), not by their label, so a 00:30 capture follows a 23:00 one. Principles: the book is the anchor (p = p_book + w * lean), games are the unit of
evidence, every number carries its n, nothing is a bet until the gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB = os.environ.get("NCAAB_DB", str(Path(__file__).with_name("ncaab.sqlite")))
MODEL_VERSION = "ncaab-0.10"
HALF = re.compile(r"(\d)\.\s*félidő", re.I)                    # '1. félidő' = first half
START_W = {"main": 0.05, "half": 0.0, "other": 0.0}
LEAN_CAP, MIN_GAMES, PRIOR_SD = 0.05, 30, 0.04
HCA, SD_MARGIN, SD_TOTAL = 3.2, 11.0, 17.0                     # unfitted starting values (spec 6)
MIN_TEAM_GAMES, RIDGE, HALF_LIFE = 4, 0.15, 150.0             # fit: fewest games to rate a team, shrinkage, days
#   ridge / half-life chosen 2026-11-18 on 7,205 real games (walk-forward, Jan-Feb and Nov): see CHANGELOG
HOME, DRAW, AWAY, OVER, UNDER, HCP_H, HCP_A, ODD, EVEN = 1, 2, 3, 12, 13, 1714, 1715, 70, 72   # site's own codes
NEITHER = 7                                                    # "Egyik sem" in a race market
RACE = re.compile(r"verseny a\(z\) \d+ pontért|ki ér el elsőként \d+ pontot")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (night TEXT, capture TEXT, row_key TEXT, game TEXT, home TEXT, away TEXT,
  market TEXT, type TEXT, scope TEXT, ot INTEGER, subject TEXT, selection TEXT, side TEXT, line REAL, odds REAL,
  p_book REAL, p_model REAL, p REAL, w_used REAL, p_method TEXT, overround REAL, model_version TEXT,
  status TEXT DEFAULT 'open', result TEXT, actual REAL, start TEXT, captured_utc TEXT,
  PRIMARY KEY (night, capture, row_key));
CREATE TABLE IF NOT EXISTS slips (night TEXT, capture TEXT, slip TEXT, target REAL, plan TEXT, legs TEXT, odds REAL,
  p REAL, p_model REAL, exp_ret REAL, PRIMARY KEY (night, slip));
CREATE TABLE IF NOT EXISTS weights (as_of TEXT, family TEXT, games INTEGER, w_fit REAL, w_se REAL, w_used REAL);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
    for col in ("start", "captured_utc"):                      # ledgers made before ncaab-0.7
        if col not in have:
            con.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT")
    return con


def _utc(v) -> str | None:
    """Any timestamp -> 'YYYY-MM-DDTHH:MM' in UTC (sorts as text), or None."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return None
    t = pd.Timestamp(v)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return f"{t:%Y-%m-%dT%H:%M}"


def mark_captures(d: pd.DataFrame) -> pd.DataFrame:
    """Adds cap_key (what captures of a night are ordered by) and is_first / is_last PER GAME. A night whose captures
    all carry captured_utc is ordered by it; otherwise by the capture label (old ledgers)."""
    d = d.copy()
    has_ts = d.groupby("night").captured_utc.transform(lambda x: x.notna().all())
    d["cap_key"] = np.where(has_ts, d.captured_utc.fillna("") + "|", "") + d.capture.astype(str)   # label breaks ties
    g = d.groupby(["night", "game"]).cap_key
    d["is_last"], d["is_first"] = d.cap_key == g.transform("max"), d.cap_key == g.transform("min")
    return d


def clock_warning(files: list) -> str | None:
    """Capture files are named ..._HHMM.net.json (local time) and carry the browser's timezoneOffset in their first
    request. If the files' own time does not match that, they were copied and their time is not the capture time."""
    off_n = n = 0
    for f in files:
        m = re.search(r"_(\d{2})(\d{2})\.(?:net\.)?json$", Path(f).name)
        try:
            with open(f, encoding="utf-8") as fh:
                z = re.search(r"timezoneOffset=(-?\d+)", fh.read(6000))
        except OSError:
            z = None
        if not m or not z:
            continue
        local = (int(os.path.getmtime(f)) // 60 - int(z.group(1))) % 1440
        gap = abs(local - (int(m.group(1)) * 60 + int(m.group(2))))
        n, off_n = n + 1, off_n + (min(gap, 1440 - gap) > 10)
    if n and off_n > n / 2:
        return (f"   WARNING: {off_n} of {n} files have a file time that does not match the capture time in their name: "
                f"they look copied. The capture time may be wrong - pass --captured-at.")
    return None


def norm_team(s: str) -> str:
    """One spelling per team: the board's full name, trimmed ('Baylor Bears'). Results must use the same."""
    return re.sub(r"\s+", " ", str(s)).strip()


def _aliases() -> dict:
    """results_aliases.csv next to this file (espn,board): ESPN's spelling -> the board's. Applied when a results or
    slate file is READ, so files already scraped are fixed too."""
    f = Path(__file__).with_name("results_aliases.csv")
    return {norm_team(a): norm_team(b) for a, b in pd.read_csv(f).values} if f.exists() else {}


ALIAS = _aliases()


def ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ------------------------------------------------------------------ boards
def load_event(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and "markets" in d:
        return d
    hits = [g["body"] for g in d if isinstance(g.get("body"), dict) and "markets" in g["body"] and "odds" in g["body"]]
    if not hits:
        raise ValueError(f"{path}: no GetEventDetails response in it")
    return max(hits, key=lambda b: len(b["odds"]))


def shown(price: float) -> float:
    return math.floor(price * 100 + 1e-6) / 100             # the site shows and pays prices floored to 2 decimals


def _line(o: dict):
    sv = str(o.get("sv") or "").split("|")[0].replace(",", ".")
    try:
        return float(sv)
    except ValueError:
        m = re.search(r"([+\-]?\d+(?:[.,]\d+)?)", o.get("name", ""))
        return float(m.group(1).replace(",", ".")) if m else None


def classify(name: str, tids: set, teams: tuple[str, str]) -> tuple[str, str, int, str | None]:
    """-> (type, scope, ot, team). scope G / H1 / H2. Decided by WHICH selection codes the market has; the name
    gives only scope, stat and team. Anything not recognised is UNPARSED: stored, never graded."""
    low = name.lower()
    bare = low
    for t in teams:                                            # 'William & Mary Tribe' must not look like a combo
        bare = bare.replace(t.lower(), " ")
    bare = re.sub(r"\(.*?\)", "", bare)
    m = HALF.search(low)
    scope = f"H{m.group(1)}" if m else "G"
    team = next((t for t in teams if t.lower() in low), None)
    ot = int(scope == "G" and "nélkül" not in low)             # full-game basketball markets include overtime
    if (scope == "G" and re.match(r"\s*\d\s*\.", low)) or scope not in ("G", "H1", "H2"):
        return "UNPARSED", scope, ot, team                     # quarters, numbered parts that are not halves
    if RACE.search(low):                                       # who gets to N points first (half: + 'neither')
        return ("RACE" if scope in ("G", "H1") and tids and tids <= {HOME, AWAY, NEITHER} else "UNPARSED"), scope, ot, None
    if re.search(r" és | & |pontos|multi|tartomány|különbség|melyik|hányadik|első|utolsó|ki ér el|verseny|félidő/", bare):
        return "UNPARSED", scope, ot, team
    if tids == {HOME, AWAY}:
        return ("DNB" if scope != "G" or "döntetlennél" in low else "ML"), scope, ot, None
    if tids == {HOME, DRAW, AWAY}:
        return "1X2", scope, 0, None                           # 3-way: regulation only
    if tids and tids <= {HCP_H, HCP_A}:
        return "HANDICAP", scope, ot, None
    if tids and tids <= {OVER, UNDER}:
        if "pont" not in bare or re.search(r"tripla|hárompont|büntet|lepattan|gólpassz", bare):
            return "UNPARSED", scope, ot, team
        return ("TEAM_TOTAL" if team else "TOTAL"), scope, ot, team
    if tids == {ODD, EVEN}:
        return "ODD_EVEN", scope, ot, team
    return "UNPARSED", scope, ot, team


def board_rows(path) -> list[dict]:
    ev = load_event(path)
    comp = ev["competitors"][:2]
    names = (norm_team(comp[0]["name"]), norm_team(comp[1]["name"]))
    home, away = names
    cid = {comp[0]["id"]: home, comp[1]["id"]: away}
    odds = {o["id"]: o for o in ev["odds"]}
    start = _utc(ev.get("startDate"))
    rows, seen = [], set()
    for m in ev["markets"]:
        if m.get("typeId") == 31046 or m.get("childMarketIds"):   # boosted container; player tables (none in NCAAB)
            continue
        if m["id"] in seen:                                    # real boards list most markets twice (48 of 59 files)
            continue
        seen.add(m["id"])
        live = [odds[i] for col in m.get("desktopOddIds", []) for i in col
                if i in odds and odds[i].get("oddStatus", 0) == 0 and shown(odds[i].get("price", 0)) > 1]
        if not live:
            continue
        name = m["name"].strip()
        typ, scope, ot, team = classify(name, {o.get("typeId") for o in live}, names)
        for o in live:
            t, nm, line, side, subject = o.get("typeId"), o.get("name", ""), _line(o), None, None
            if typ in ("ML", "DNB", "1X2"):
                side, subject = {HOME: "home", DRAW: "draw", AWAY: "away"}.get(t), cid.get(o.get("competitorId"))
                line = None                                    # no line here ('Philadelphia 76ers' is not a 76)
            elif typ == "HANDICAP":
                side = "home" if t == HCP_H else "away"
                subject = home if side == "home" else away
                own = re.search(r"\(?([+\-]\d+(?:[.,]\d+)?)\)?\s*$", nm)    # the selection's OWN line is in its name;
                if own:                                                     # the sv field is the home line for both
                    line = float(own.group(1).replace(",", "."))
                elif side == "away" and line is not None:
                    line = -line
            elif typ in ("TOTAL", "TEAM_TOTAL"):
                side, subject = ("over" if t == OVER else "under"), team
            elif typ == "ODD_EVEN":
                side, subject, line = ("odd" if t == ODD else "even"), team, None
            elif typ == "RACE":
                side, subject = {HOME: "home", AWAY: "away", NEITHER: "neither"}.get(t), cid.get(o.get("competitorId"))
                tgt = re.search(r"(\d+) pont", name)
                line = float(tgt.group(1)) if tgt else None
                side = side if line is not None else None
            ty = typ if (side is not None or typ == "UNPARSED") else "UNPARSED"
            key = hashlib.md5(f"{away}@{home}|{ty}|{scope}|{subject}|{side}|{line}|{name}|{nm if ty == 'UNPARSED' else ''}".encode()).hexdigest()[:16]
            rows.append(dict(row_key=key, game=f"{away}@{home}", home=home, away=away, market=name, type=ty, scope=scope,
                             ot=ot, subject=subject, selection=nm, side=side, line=line, odds=shown(o["price"]), mid=m["id"],
                             start=start))
    return rows


def devig(B: pd.DataFrame) -> pd.DataFrame:
    """Margin removed. Groups that cover every outcome (2-way by market; over/under by line; handicap home L with
    away -L; 3-way) and whose implied chances add to 1.00-1.25 are normalised ('no_vig'). Everything else is divided
    by the board's median overround ('assumed_margin'): graded, reported, never pooled into calibration or arms."""
    B = B.copy()
    B["imp"] = 1 / B.odds
    ln = np.where((B.type == "HANDICAP") & (B.side == "away"), -B.line.fillna(0), B.line.fillna(0))
    lined = B.type.isin(["TOTAL", "TEAM_TOTAL", "HANDICAP", "RACE"])
    B["grp"] = B.mid.astype(str) + np.where(lined, "|" + pd.Series(ln, index=B.index).round(2).astype(str), "")
    g = B.groupby("grp")
    B["overround"], n, sides = g.imp.transform("sum"), g.imp.transform("size"), g.side.transform("nunique")
    need = np.where((B.type == "1X2") | ((B.type == "RACE") & (B.scope != "G")), 3, 2)   # a half race needs 'neither'

    ok = (n == need) & (sides == need) & B.overround.between(1.0, 1.25) & (B.type != "UNPARSED")
    med = B.loc[ok, "overround"].median() if ok.any() else 1.07
    B["p_book"] = np.where(ok, B.imp / B.overround, np.minimum(B.imp / med, 0.999))
    B["p_method"] = np.where(ok, "no_vig", f"assumed_margin_{med:.3f}")
    B.loc[~ok, "overround"] = med
    return B


# ------------------------------------------------------------------ model: tempo-free ratings -> normal margin / total
def expected_points(teams_csv: str):
    """teams.csv: team,adj_o,adj_d,tempo (points per 100 possessions for / against, possessions per game - the
    layout of Bart Torvik's and KenPom's ratings). `team` must be spelled as on the board; aliases.csv (board,ratings)
    maps the rest. A model.json next to it (written by `fit`) replaces HCA / SD_MARGIN / SD_TOTAL with fitted
    values. Returns f(home, away, neutral) -> (home points, away points), or None when a team is missing or has
    fewer than MIN_TEAM_GAMES games behind its rating (unknown is never zero)."""
    global HCA, SD_MARGIN, SD_TOTAL
    t = pd.read_csv(teams_csv)
    mj = Path(teams_csv).with_name("model.json")
    if mj.exists():
        mp = json.loads(mj.read_text())
        HCA = mp["hca"] if mp.get("hca") is not None else HCA
        SD_MARGIN, SD_TOTAL = mp.get("sd_margin") or SD_MARGIN, mp.get("sd_total") or SD_TOTAL
    ali = Path(teams_csv).with_name("aliases.csv")
    alias = dict(pd.read_csv(ali).values) if ali.exists() else {}
    R = {norm_team(r.team): r for r in t.itertuples()}
    eff, tempo = t.adj_o.mean(), t.tempo.mean()

    def f(home, away, neutral=False):
        h, a = R.get(alias.get(home, home)), R.get(alias.get(away, away))
        if h is None or a is None or min(getattr(h, "games", 99), getattr(a, "games", 99)) < MIN_TEAM_GAMES:
            return None
        poss, hc = h.tempo * a.tempo / tempo, 0.0 if neutral else HCA / 2
        return h.adj_o * a.adj_d / eff * poss / 100 + hc, a.adj_o * h.adj_d / eff * poss / 100 - hc
    return f


def model_p(r, ph: float, pa: float) -> float | None:
    k = 1.0 if r.scope == "G" else 0.5
    m, tot, sdm, sdt = (ph - pa) * k, (ph + pa) * k, SD_MARGIN * math.sqrt(k), SD_TOTAL * math.sqrt(k)
    sd_team = math.sqrt(sdm ** 2 + sdt ** 2) / 2
    t, s, L = r.type, r.side, r.line
    if t == "RACE":                                            # v0, unfitted: the lead after the share of the game
        if L is None or L != L:                                # it takes the leader to reach L points
            return None
        frac = min(L / ((ph + pa) / 2), 1.0 if r.scope == "G" else 0.5)
        first = ncdf((ph - pa) * math.sqrt(frac) / SD_MARGIN)
        if r.scope == "G":
            return first if s == "home" else 1 - first if s == "away" else None
        sh = math.sqrt(SD_MARGIN ** 2 + SD_TOTAL ** 2) / 2 / math.sqrt(2)
        none = ncdf((L - 0.5 - ph / 2) / sh) * ncdf((L - 0.5 - pa / 2) / sh)
        return none if s == "neither" else (1 - none) * (first if s == "home" else 1 - first)
    if t in ("ML", "DNB"):
        p = ncdf(m / sdm)
        return p if s == "home" else 1 - p
    if L is None or L % 1 != 0.5:
        return None                                            # whole-number lines can push: left to the book
    if t == "HANDICAP":
        return ncdf(((m if s == "home" else -m) + L) / sdm)
    if t == "TOTAL":
        p = 1 - ncdf((L - tot) / sdt)
        return p if s == "over" else 1 - p
    if t == "TEAM_TOTAL":
        p = 1 - ncdf((L - (ph if r.subject == r.home else pa) * k) / sd_team)
        return p if s == "over" else 1 - p
    return None


def family(t: str, scope: str) -> str:
    return "half" if scope != "G" else "main" if t in ("ML", "HANDICAP", "TOTAL", "TEAM_TOTAL") else "other"


def price(night: str, boards: str, capture: str | None, teams_csv: str | None, con, slate: str | None = None,
          captured_at: str | None = None) -> pd.DataFrame:
    """`boards` = a folder of .json captures, or one file made by `pack`. `slate` = a file made by `slate`: games
    it marks NEUTRAL get no home-court advantage from the model. `captured_at` = when the boards were pulled (any
    timestamp; default: the pack's own captured_utc column, or the .json files' time). Games that had already
    tipped at that moment are not stored: their prices know the score."""
    frames, skipped, cap_utc = [], [], _utc(captured_at)
    if Path(boards).is_file():
        capture = capture or Path(boards).name.split(".")[0]
        raw = pd.read_csv(boards)
        if cap_utc is None and "captured_utc" in raw and raw.captured_utc.notna().any():
            cap_utc = _utc(raw.captured_utc.dropna().iloc[0])
        frames = [devig(x) for _, x in raw.drop(columns=["captured_utc"], errors="ignore").groupby("game")]
    else:
        files = sorted(Path(boards).glob("*.json"))
        if not files:
            sys.exit(f"no .json boards in {boards}")
        mt = pd.Timestamp(max(os.path.getmtime(f) for f in files), unit="s")
        if cap_utc is None and (wn := clock_warning(files)):
            print(wn)
        capture, cap_utc = capture or mt.strftime("%H%M"), cap_utc or _utc(mt)
        for f in files:
            try:
                rows = board_rows(f)
                if rows:
                    frames.append(devig(pd.DataFrame(rows)))
            except Exception as e:
                skipped.append(f"{f.name}: {e}")
    card, neutral = read_slate(slate) if slate and Path(slate).exists() else (set(), set())
    venue = slate_home(slate) if slate and Path(slate).exists() else {}
    B = pd.concat(frames, ignore_index=True).drop_duplicates("row_key", keep="last")   # the key holds the game
    if "start" not in B:
        B["start"] = None                                      # packs made before ncaab-0.7
    B["captured_utc"] = cap_utc
    live = B.start.notna() & (B.start.astype(str) <= (cap_utc or "")) if cap_utc else pd.Series(False, index=B.index)
    if live.any():
        print(f"   {B[live].game.nunique()} games had already tipped at {cap_utc}Z: not stored ({sorted(B[live].game.unique())[:5]})")
        B = B[~live].reset_index(drop=True)
    if B.empty:
        sys.exit(f"{night} capture {capture}: every game had already tipped at {cap_utc}Z, nothing stored. If that is "
                 f"not when the boards were pulled (copied files lose their time), pass --captured-at.")
    B["p_model"] = np.nan
    if teams_csv:
        f, miss, flipped = expected_points(teams_csv), set(), set()
        for g, x in B.groupby("game"):
            bh, ba = x.home.iloc[0], x.away.iloc[0]
            if venue.get(frozenset((bh, ba))) == ba:           # the board lists home/away the other way round from
                e = f(ba, bh, False)                           # ESPN: home court goes to ESPN's home team
                e = None if e is None else (e[1], e[0])
                flipped.add(g)
            else:
                e = f(bh, ba, frozenset((bh, ba)) in neutral)
            if e is None:
                miss.add(g)
                continue
            B.loc[x.index, "p_model"] = [np.nan if (v := model_p(r, *e)) is None else v for r in x.itertuples()]
        pairs = {frozenset((h, a)): g for g, h, a in B.drop_duplicates("game")[["game", "home", "away"]].values}
        print(f"model priced {B.game.nunique() - len(miss)} of {B.game.nunique()} games ({len(neutral & set(pairs))} on a "
              f"neutral court, {len(flipped)} home/away flipped vs ESPN); no rating for: {sorted(miss)[:8]}")
        if card:
            print(f"   not on ESPN's card under these names (add to results_aliases.csv): {sorted(g for k, g in pairs.items() if k not in card)}")
    w = dict(START_W)
    for fam, wu in con.execute("SELECT family, w_used FROM weights w WHERE as_of = (SELECT max(as_of) FROM weights WHERE family = w.family)"):
        w[fam] = wu
    B["w_used"] = [w[family(t, s)] for t, s in zip(B.type, B.scope)]
    lean = (B.p_model - B.p_book).clip(-LEAN_CAP, LEAN_CAP)
    B["p"] = np.where(B.p_model.notna(), B.p_book + B.w_used * lean, B.p_book)
    bad = B[(B.p <= 0) | (B.p >= 1) | (B.odds <= 1) | ((B.p_method == "no_vig") & ~B.overround.between(1.0, 1.25))]
    cols = ["row_key", "game", "home", "away", "market", "type", "scope", "ot", "subject", "selection", "side", "line",
            "odds", "p_book", "p_model", "p", "w_used", "p_method", "overround", "start", "captured_utc"]
    con.execute("DELETE FROM predictions WHERE night = ? AND capture = ?", [night, capture])
    con.executemany(f"INSERT INTO predictions (night, capture, model_version, {','.join(cols)}) VALUES "
                    f"(?, ?, ?, {','.join('?' * len(cols))})",
                    [(night, capture, MODEL_VERSION, *[None if isinstance(v, float) and math.isnan(v) else v for v in r])
                     for r in B[cols].astype(object).values.tolist()])
    con.commit()
    nv = B[B.p_method == "no_vig"]
    print(f"{night} capture {capture} (taken {cap_utc or 'time unknown'}Z): {B.game.nunique()} games, {len(B):,} rows, "
          f"price audit {len(bad)} broken; skipped files {len(skipped)}")
    print(nv.groupby(["type", "scope"]).agg(rows=("odds", "size"), games=("game", "nunique"),
                                            cut=("overround", lambda s: s.median() - 1)).round(3).to_string())
    print(f"UNPARSED rows: {(B.type == 'UNPARSED').sum()} in {B[B.type == 'UNPARSED'].market.nunique()} market names")
    return B


# ------------------------------------------------------------------ results
def read_results(path: str) -> tuple[dict, list[str]]:
    """GAME <away>@<home>   (team names exactly as on the board)
       H <team>: h1 h2 [ot1 ot2 ...]      one line per team
       NEUTRAL                            optional: neutral court (used by `fit` and by the model, not by grading)
       RACE 10:H1 15:A1 20:A1 ...         optional, from play-by-play: target points : H/A = the home/away team of
                                          the GAME line got there first, then the period (1, 2, 3+ = overtime)
    Rejected: a tie without overtime, overtime without a regulation tie, unequal period counts, unknown team."""
    return parse_results(Path(path).read_text(encoding="utf-8"))


def read_slate(path: str) -> set:
    """-> (every game, neutral-court games) of a slate or results file, each as a frozenset of its two teams: on
    neutral courts the board and ESPN often disagree about who is 'home'."""
    G = parse_results(Path(path).read_text(encoding="utf-8"))[0].values()
    return {frozenset((d["home"], d["away"])) for d in G}, {frozenset((d["home"], d["away"])) for d in G if d["neutral"]}


def slate_home(path: str) -> dict:
    """{frozenset of the two teams: ESPN's home team} for games NOT on a neutral court."""
    G = parse_results(Path(path).read_text(encoding="utf-8"))[0].values()
    return {frozenset((d["home"], d["away"])): d["home"] for d in G if not d["neutral"]}


def parse_results(text: str) -> tuple[dict, list[str]]:
    games, errs, g = {}, [], None
    for n, ln in enumerate(text.splitlines(), 1):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        try:
            if ln.startswith("GAME "):
                g = ln[5:].strip()
                a, h = (ALIAS.get(norm_team(x), norm_team(x)) for x in g.split("@"))
                g = f"{a}@{h}"
                games[g] = dict(home=h, away=a, per={}, neutral=False, ok=True, race={})
            elif ln == "NEUTRAL":
                games[g]["neutral"] = True
            elif ln.startswith("RACE "):                       # RACE 10:H1 15:A1 ...  = target:who got there first + period
                for tok in ln[5:].split():
                    n_, w_ = tok.split(":")
                    games[g]["race"][int(n_)] = (games[g]["home" if w_[0] == "H" else "away"], int(w_[1:]))
            elif ln.startswith("H "):
                t, v = ln[2:].rsplit(":", 1)
                t = ALIAS.get(norm_team(t), norm_team(t))
                assert t in (games[g]["home"], games[g]["away"]), "team is not in this game"
                games[g]["per"][t] = [int(x) for x in v.split()]
            else:
                raise ValueError("unknown line")
        except Exception as e:
            errs.append(f"line {n}: {ln!r} ({e})")
    for g, d in games.items():
        n0, d["ok"] = len(errs), False
        if set(d["per"]) != {d["home"], d["away"]}:
            errs.append(f"{g}: needs one H line per team"); continue
        h, a = d["per"][d["home"]], d["per"][d["away"]]
        if len(h) != len(a) or len(h) < 2:
            errs.append(f"{g}: both teams need the same number of periods (2 halves + overtimes)"); continue
        if len(h) == 2 and sum(h) == sum(a):
            errs.append(f"{g}: tied with no overtime")
        if len(h) > 2 and sum(h[:2]) != sum(a[:2]):
            errs.append(f"{g}: overtime listed but regulation was not tied")
        if len(h) > 2 and sum(h) == sum(a):
            errs.append(f"{g}: still tied after the listed overtimes")
        if not all(5 <= x <= 95 for x in h[:2] + a[:2]):
            errs.append(f"{g}: a half outside 5-95 points - check the numbers")
        d["ok"] = len(errs) == n0
    return games, errs


def settle(r, d: dict) -> tuple[str | None, float | None]:
    h, a = d["per"][d["home"]], d["per"][d["away"]]
    if r.scope == "G":
        hg, ag = (sum(h), sum(a)) if r.ot else (sum(h[:2]), sum(a[:2]))
    else:
        i = int(r.scope[1]) - 1
        hg, ag = h[i], a[i]
    t, s, L = r.type, r.side, r.line
    yn = lambda c: "won" if c else "lost"
    if t in ("ML", "1X2"):
        return yn((hg > ag) if s == "home" else (hg < ag) if s == "away" else (hg == ag)), hg - ag
    if t == "DNB":
        return ("push" if hg == ag else yn((hg > ag) == (s == "home"))), hg - ag
    if t in ("TOTAL", "TEAM_TOTAL"):
        x = hg + ag if t == "TOTAL" else (hg if r.subject == d["home"] else ag)
        return ("push" if x == L else yn((x > L) == (s == "over"))), x
    if t == "HANDICAP":
        m = (hg - ag) if s == "home" else (ag - hg)
        return ("push" if m + L == 0 else yn(m + L > 0)), m
    if t == "ODD_EVEN":
        team = r.subject if isinstance(r.subject, str) and r.subject else None   # pandas 3 reads NULL as NaN (truthy)
        x = hg + ag if team is None else (hg if team == d["home"] else ag)
        return yn((x % 2 == 1) == (s == "odd")), x
    if t == "RACE":
        if not d.get("race") or L is None or L != L:                     # no play-by-play for this game: unknown, not graded
            return None, None
        who, per = d["race"].get(int(L), (None, 0))
        if r.scope == "H1" and per != 1:
            who = None                                         # nobody got there before half-time
        if who is None and r.scope == "G":
            return None, None                                  # nobody reached it in the whole game: left ungraded
        return yn(s == ("neither" if who is None else "home" if who == d["home"] else "away")), float(per)
    return None, None


def grade(night: str, results: str, con) -> None:
    games, errs = read_results(results)
    for e in errs:
        print("ERROR", e)
    if errs:
        sys.exit("results file rejected: fix it, nothing was graded")
    P = pd.read_sql("SELECT * FROM predictions WHERE night = ?", con, params=[night])
    if P.empty:
        sys.exit(f"no predictions stored for {night}")
    out, by_pair = [], {frozenset((d["home"], d["away"])): d for d in games.values()}
    for r in P.itertuples():
        d = by_pair.get(frozenset((r.home, r.away)))
        if d is not None:
            d = dict(d, home=r.home, away=r.away)              # the board's home/away, whatever ESPN calls them
        res, act = (None, None) if (d is None or r.type == "UNPARSED") else settle(r, d)
        st = "unparsed" if r.type == "UNPARSED" else "graded" if res else "no_result"
        out.append((st, res, act, night, r.capture, r.row_key))
    con.executemany("UPDATE predictions SET status=?, result=?, actual=? WHERE night=? AND capture=? AND row_key=?", out)
    con.commit()
    seen = {frozenset((h, a)): g for g, h, a in P.drop_duplicates("game")[["game", "home", "away"]].values}
    miss = sorted(g for k, g in seen.items() if k not in by_pair)
    print(f"{night}: {len(games)} games in results, {P.game.nunique()} on the boards, {len(miss)} boards without a result")
    for m in miss[:15]:
        print("   no result:", m)
    extra = sorted(g for g, d in games.items() if frozenset((d["home"], d["away"])) not in seen)
    if extra:
        print(f"   {len(extra)} results match no board (name spelling?): {extra[:8]}")
    print(pd.Series([o[0] + (":" + o[1] if o[1] else "") for o in out]).value_counts().to_string())


# ------------------------------------------------------------------ report + bandit
def decided(con, last_only: bool = True) -> pd.DataFrame:
    d = pd.read_sql("SELECT * FROM predictions WHERE status = 'graded' AND result IN ('won', 'lost')", con)
    if d.empty:
        return d
    d = mark_captures(d)
    op = d[d.is_first & ~d.is_last & (d.p_method == "no_vig")][["night", "game", "row_key", "p_book", "odds"]]
    d = d.merge(op.rename(columns=dict(p_book="p_open", odds="odds_open")), on=["night", "game", "row_key"], how="left")
    if last_only:
        d = d[d.is_last]                                       # per GAME: a noon game is not in the evening capture
    d = d.copy()
    d["y"] = (d.result == "won").astype(float)
    d["ret"] = d.y * d.odds - 1
    d["book_ret"] = d.p_book * d.odds - 1
    d["cluster"] = d.night + "|" + d.game
    d["family"] = [family(t, s) for t, s in zip(d.type, d.scope)]
    d["dist"] = (d.p_book - 0.5).abs()
    d["mid_line"] = d.dist == d.groupby(["cluster", "type", "scope"]).dist.transform("min")     # the line nearest to even
    two = d.p_book.where(d.side.isin(["home", "away"]))
    d["race_fav"] = (d.type == "RACE") & (two == two.groupby([d.cluster, d.market]).transform("max"))
    d["edge"] = d.p_model - d.p_book
    return d


def cstat(x: pd.DataFrame) -> dict:
    g = x.groupby("cluster").ret.mean()
    n = len(g)
    return dict(bets=len(x), games=n, hit=x.y.mean(), need=(1 / x.odds).mean(), ret=g.mean(),
                se=g.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan"))


def cbrier(p, y, cl) -> float:
    return float(((p - y) ** 2).groupby(cl).mean().mean()) if len(y) else float("nan")


def bandit(d: pd.DataFrame) -> pd.DataFrame:
    """P(arm's true return per bet > 0), starting from 'the book's price is right' (prior mean = the return the
    book's own no-vig price implies, sd PRIOR_SD), updated with the per-game mean return. Tiers need 100 games."""
    x = d[(d.p_method == "no_vig") & d.odds.between(1.15, 6.0)].copy()
    fav = x.p_book >= 0.5
    arms = {f"{t} {sc} {s}": g for (t, sc, s), g in x.groupby(["type", "scope", "side"])}
    ml = x[x.type == "ML"]
    arms.update({"* Favourite to win": ml[ml.p_book >= 0.5], "* Underdog to win": ml[ml.p_book < 0.5],
                 "* Home side, handicap": x[(x.type == "HANDICAP") & (x.side == "home") & (x.scope == "G")],
                 "* Away side, handicap": x[(x.type == "HANDICAP") & (x.side == "away") & (x.scope == "G")],
                 "* Big favourite -15 or more, handicap": x[(x.type == "HANDICAP") & (x.scope == "G") & (x.line <= -15)],
                 "* Big underdog +15 or more, handicap": x[(x.type == "HANDICAP") & (x.scope == "G") & (x.line >= 15)],
                 "* Short legs (book 70%+)": x[x.p_book >= 0.7], "* Longshot legs (book under 30%)": x[x.p_book < 0.3],
                 "* Favourite side, any market": x[fav & x.type.isin(["ML", "HANDICAP"])]})
    hc, rc = x[x.type == "HANDICAP"], x[(x.type == "RACE") & x.side.isin(["home", "away"])]
    arms.update({                                              # set 2026-11-18, before any NCAAB game was graded
        "* Handicap, line nearest to even: favourite": hc[hc.mid_line & (hc.scope == "G") & (hc.line < 0)],
        "* Handicap, line nearest to even: underdog": hc[hc.mid_line & (hc.scope == "G") & (hc.line > 0)],
        "* Handicap, 1st half: favourite": hc[(hc.scope == "H1") & (hc.line < 0)],
        "* Handicap, 1st half: underdog": hc[(hc.scope == "H1") & (hc.line > 0)],
        "* Handicap, model likes it by 5+ pts": hc[hc.edge >= 0.05],
        "* Handicap, model against it by 5+ pts": hc[hc.edge <= -0.05],
        "* Race: favourite": rc[rc.race_fav], "* Race: underdog": rc[~rc.race_fav],
        "* Race to 25 or less: favourite": rc[rc.race_fav & (rc.line <= 25)],
        "* Race to 25 or less: underdog": rc[~rc.race_fav & (rc.line <= 25)],
        "* Race to 30 or more: favourite": rc[rc.race_fav & (rc.line >= 30)],
        "* Race to 30 or more: underdog": rc[~rc.race_fav & (rc.line >= 30)],
        "* Race, 1st half: neither gets there": x[(x.type == "RACE") & (x.side == "neither")],
        "* Race, model likes it by 5+ pts": rc[rc.edge >= 0.05]})
    rows = []
    for name, a in arms.items():
        g = a.groupby("cluster").ret.mean()
        n = len(g)
        if n < 12 or len(a) < 30:
            continue
        se = max(g.std(ddof=1) / math.sqrt(n) * (1 + 2 / n), 0.005)      # widened for few games
        mu0 = a.groupby("cluster").book_ret.mean().mean()
        w = (1 / se ** 2) / (1 / se ** 2 + 1 / PRIOR_SD ** 2)
        pm, ps = w * g.mean() + (1 - w) * mu0, math.sqrt(1 / (1 / se ** 2 + 1 / PRIOR_SD ** 2))
        pb = ncdf(pm / ps)
        c = a[a.p_open.notna()]                                # bets also priced at the game's first capture
        clv = (c.p_book - c.p_open).groupby(c.cluster).mean()
        rows.append(dict(arm=name, bets=len(a), games=n, hit=a.y.mean(), need=(1 / a.odds).mean(), seen=g.mean(),
                         seen_se=se, book_says=mu0, belief=pm, p_beats=pb,
                         clv=clv.mean() if len(clv) else float("nan"), clv_games=len(clv),
                         status="CANDIDATE" if pb >= 0.95 and n >= 100 else "SMALL STAKE" if pb >= 0.80 and n >= 100
                         else "watch" if pb >= 0.65 else "-"))
    return pd.DataFrame(rows).sort_values("p_beats", ascending=False) if rows else pd.DataFrame()


def refit(d: pd.DataFrame, con) -> None:
    rows, now = [], pd.Timestamp.now("UTC").isoformat()
    for fam, x in d[d.p_method == "no_vig"].dropna(subset=["p_model"]).groupby("family"):
        nc = x.cluster.nunique()
        if nc < MIN_GAMES:
            print(f"  {fam}: {nc} games with a model price (< {MIN_GAMES}): weight stays {START_W[fam]}")
            continue
        k = 1 / x.groupby("cluster").cluster.transform("size")
        dx, e0 = x.p_model - x.p_book, x.y - x.p_book
        sxx = float((k * dx * dx).sum())
        w = float((k * dx * e0).sum() / sxx)
        gs = (k * dx * (e0 - w * dx)).groupby(x.cluster).sum()
        se = max(math.sqrt(float((gs ** 2).sum()) * nc / (nc - 1)) / sxx, 0.03)
        wu = float(np.clip((w / se ** 2 + START_W[fam] / 0.15 ** 2) / (1 / se ** 2 + 1 / 0.15 ** 2), 0, 1.2))
        rows.append((now, fam, nc, w, se, round(wu, 3)))
        print(f"  {fam}: {nc} games, fitted weight {w:+.2f} +- {se:.2f} -> used {wu:.3f}")
    if rows:
        con.executemany("INSERT INTO weights VALUES (?,?,?,?,?,?)", rows)
        con.commit()


def report(con) -> None:
    d = decided(con)
    if d.empty:
        sys.exit("nothing graded yet")
    pct = lambda c: lambda v: f"{v:+.1%}" if c else f"{v:.1%}"
    L = [f"# NCAAB ledger report\n\n{len(d):,} decided bets, **{d.cluster.nunique()} games**, nights {d.night.min()} .. "
         f"{d.night.max()} (for each game: the last capture before its tip). Games are the unit of evidence.\n"]
    B = bandit(d)
    best = f"{B.p_beats.max():.0%}" if len(B) else "n/a"
    L += ["## Verdict\n", f"- Real-money verdict: **{'NO BET' if not len(B) or not B.status.isin(['CANDIDATE', 'SMALL STAKE']).any() else 'see tiers below'}**"
          f" (best arm at {best}; SMALL STAKE needs 80%, CANDIDATE 95%, both 100 games).\n"]
    m = d[d.p_method == "no_vig"].dropna(subset=["p_model"])
    L.append("## Negatives first: book vs model vs blend (Brier, lower is better)\n")
    if len(m):
        T = pd.DataFrame([dict(type=t, scope=s, n=len(x), games=x.cluster.nunique(), book=cbrier(x.p_book, x.y, x.cluster),
                               model=cbrier(x.p_model, x.y, x.cluster), blend=cbrier(x.p, x.y, x.cluster))
                          for (t, s), x in m.groupby(["type", "scope"])])
        L += [T.assign(model_minus_book=T.model - T.book).sort_values("model_minus_book", ascending=False).round(4).to_markdown(index=False), ""]
    else:
        L.append("No model prices graded (price with --teams).\n")
    nv = d[d.p_method == "no_vig"]
    C = nv.groupby(pd.cut(nv.p_book, np.arange(0, 1.01, 0.1), include_lowest=True), observed=True).agg(
        n=("y", "size"), games=("cluster", "nunique"), p_book=("p_book", "mean"), hit=("y", "mean"))
    L += ["## Is the book's price right? (two-way priced rows)\n", C.assign(gap=C.hit - C.p_book).round(3).to_markdown(), ""]
    R = pd.DataFrame([dict(type=t, scope=sc, side=s, **cstat(x)) for (t, sc, s), x in nv.groupby(["type", "scope", "side"])])
    L += ["## Return by bet type and side\n", R.sort_values("ret").round(3).to_markdown(index=False), ""]

    mm = nv[nv.p_open.notna()]                                 # games priced at two or more captures before tip
    L.append("## Open vs close (games with two or more captures before tip)\n")
    if len(mm):
        mv = mm.p_book - mm.p_open
        row = lambda nm, x: dict(group=nm, bets=len(x), games=x.cluster.nunique(), brier_open=cbrier(x.p_open, x.y, x.cluster),
                                 brier_close=cbrier(x.p_book, x.y, x.cluster), ret_at_open=(x.y * x.odds_open - 1).mean())
        rows = [row("all", mm)] + [row(nm, x) for nm, x in (("price shortened 3+ pts", mm[mv >= 0.03]),
                                                           ("price drifted 3+ pts", mm[mv <= -0.03])) if len(x)]
        L += [pd.DataFrame(rows).round(4).to_markdown(index=False),
              "\nCLV of an arm = how far its bets shortened between a game's first and last capture (the `clv` column "
              "below). Positive CLV is the fastest honest signal.\n"]
    else:
        L.append("None yet: price the same night twice (morning, then about an hour before the first tip).\n")
    L.append("## Bandit (arms marked * are hypotheses set before any data)\n")
    if len(B):
        T = B.copy()
        for c in ("hit", "need", "p_beats"):
            T[c] = T[c].map(pct(False))
        for c in ("seen", "seen_se", "book_says", "belief"):
            T[c] = T[c].map(pct(True))
        T["clv"] = T.clv.map(lambda v: "" if pd.isna(v) else f"{100 * v:+.1f} pts")
        L += [T.to_markdown(index=False), "",
              "`seen` = raw return so far; `belief` = what is left after starting from 'the book is right'. With dozens of "
              "arms some will look good by chance: only p_beats with 100+ games counts. `clv` = how far the book moved these "
              "bets between a game's first and last capture (+ = they shortened), over `clv_games` games; a real edge shows "
              "here first. Before real stakes an arm needs clv > 0 as well as its tier."]
    else:
        L.append("No arm has 12 games yet.")
    SR = slip_results(con)
    if len(SR):
        done = SR[SR.result != "open"]
        L += ["\n## Paper slips\n", SR.round(3).to_markdown(index=False)]
        if len(done):
            L.append(f"\n{len(done)} settled: {int((done.result == 'won').sum())} won, {done.win_chance.sum():.1f} expected; "
                     f"{len(done)} unit stakes returned {done.paid.sum():.2f} ({(done.odds * done.win_chance).sum():.2f} expected). "
                     f"A slip result says little: the evidence is in the legs, in the bandit table above.")
    out = Path(DB).with_name("ncaab_report.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"{d.cluster.nunique()} games. Best arm P(beats) {best}. Weight refit:")
    refit(d, con)
    print(out)


# ------------------------------------------------------------------ ESPN public scoreboard: results and slate
def race_line(event_id: str, hp: int, ap: int) -> str | None:
    """ESPN play-by-play -> 'RACE 5:H1 10:A1 ...' (who reached each multiple of 5 first, and in which period). None
    when the play-by-play is missing, runs backwards, or does not end on the final score: unknown, never guessed."""
    import urllib.request
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={event_id}"
    js = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
    return race_from_plays(js.get("plays") or [], hp, ap)


def race_from_plays(plays: list, hp: int, ap: int) -> str | None:
    got, h0, a0 = {}, 0, 0
    for pl in plays:
        try:
            h, a, per = int(pl["homeScore"]), int(pl["awayScore"]), int(pl["period"]["number"])
        except (KeyError, TypeError, ValueError):
            continue
        if h < h0 or a < a0:
            return None
        for n in range(5, 201, 5):
            if n not in got and (h >= n or a >= n) and not (h0 >= n or a0 >= n):
                if h >= n and a >= n:
                    return None                                # both crossed on one line of the feed: order unknown
                got[n] = f"{n}:{'H' if h >= n else 'A'}{per}"
        h0, a0 = h, a
    if (h0, a0) != (hp, ap) or not got:
        return None
    return "RACE " + " ".join(got[n] for n in sorted(got))


def fetch_results(date: str, out: str, slate: bool = False, pbp: bool = False) -> int:
    """Finished games of a US Eastern date, score by half (slate=True: every game on the card, no scores, to learn
    which are on a neutral court before tip). Team names are ESPN's 'displayName' ('Baylor Bears'); where the board
    spells a team differently, add a line to results_aliases.csv (espn,board) next to this file. A game whose
    numbers do not add up is written commented out (# SKIPPED), never guessed. Field layout checked against the
    live feed 2026-11-18; the fetch itself must run somewhere that can reach ESPN."""
    import urllib.request
    url = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
           f"?dates={date.replace('-', '')}&groups=50&limit=500")
    js = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
    ali = Path(__file__).with_name("results_aliases.csv")
    alias = dict(pd.read_csv(ali).values) if ali.exists() else {}
    nm = lambda x: alias.get(x["team"]["displayName"], x["team"]["displayName"])
    per = lambda x: " ".join(str(int(float(p.get("value", p.get("displayValue", 0))))) for p in x.get("linescores", []))
    L = [f"# NCAAB {'slate' if slate else 'results'} {date} (US Eastern date) from ESPN scoreboard, fetched {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M}Z"]
    n = skip = races = 0
    for ev in js.get("events", []):
        c = ev["competitions"][0]
        t = {x.get("homeAway"): x for x in c.get("competitors", [])}
        if set(t) != {"home", "away"}:
            continue
        blk = [f"GAME {nm(t['away'])}@{nm(t['home'])}"] + (["NEUTRAL"] if c.get("neutralSite") else [])
        if not slate:
            if not c.get("status", {}).get("type", {}).get("completed"):
                continue
            blk += [f"H {nm(t['home'])}: {per(t['home'])}", f"H {nm(t['away'])}: {per(t['away'])}"]
            errs = parse_results("\n".join(blk))[1]
            if errs:
                L += ["", f"# SKIPPED: {errs[0]}"] + ["# " + b for b in blk]
                skip += 1
                continue
            if pbp:
                try:
                    import time
                    time.sleep(0.15)
                    rl = race_line(ev["id"], *(sum(int(x) for x in per(t[k]).split()) for k in ("home", "away")))
                except Exception:
                    rl = None
                blk += [rl] if rl else []
                races += bool(rl)
        L += [""] + blk
        n += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{out}: {n} games written, {skip} skipped as inconsistent" + (f", {races} with play-by-play races" if pbp else ""))
    return n


# ------------------------------------------------------------------ own ratings, fitted from the results files
def load_games(results_dir: str) -> pd.DataFrame:
    """Every consistent game in results/<YYYY-MM-DD>.txt. Points are scaled to 40 minutes for overtime games."""
    rows = []
    for f in sorted(Path(results_dir).glob("*.txt")):
        try:
            day = pd.Timestamp(f.stem)
        except ValueError:
            continue
        for d in read_results(str(f))[0].values():
            if d["ok"]:
                h, a = d["per"][d["home"]], d["per"][d["away"]]
                k = 40 / (40 + 5 * (len(h) - 2))
                rows.append(dict(date=day, home=d["home"], away=d["away"], hp=sum(h), ap=sum(a), k=k, neutral=d["neutral"]))
    return pd.DataFrame(rows)


def fit_ratings(G: pd.DataFrame, ref: pd.Timestamp, lam: float, half_life: float) -> dict:
    """log(points) = league + offence[team] + defence[opponent] +- home/2. Ridge-shrunk to the league average (lam is
    roughly 'games of average play added to every team'), recent games weighted more (weight halves every half_life
    days, so last season acts as a weak prior)."""
    teams = sorted(set(G.home) | set(G.away))
    ix, n = {t: i for i, t in enumerate(teams)}, len(teams)
    h, a = G.home.map(ix).to_numpy(), G.away.map(ix).to_numpy()
    hc = np.where(G.neutral.to_numpy(bool), 0.0, 0.5)
    z = np.zeros(len(G), int)
    I = np.concatenate([np.c_[z, 1 + h, 1 + n + a, z + 1 + 2 * n], np.c_[z, 1 + a, 1 + n + h, z + 1 + 2 * n]])
    V = np.concatenate([np.c_[z + 1.0, z + 1.0, z + 1.0, hc], np.c_[z + 1.0, z + 1.0, z + 1.0, -hc]])
    y = np.log(np.clip(np.concatenate([G.hp * G.k, G.ap * G.k]).astype(float), 20, None))
    w = np.tile(0.5 ** ((ref - G.date).dt.days.to_numpy() / half_life), 2)
    A, b = np.zeros((2 * n + 2, 2 * n + 2)), np.zeros(2 * n + 2)
    np.add.at(A, (I[:, :, None], I[:, None, :]), w[:, None, None] * V[:, :, None] * V[:, None, :])
    np.add.at(b, I, (w * y)[:, None] * V)
    A[np.arange(1, 2 * n + 1), np.arange(1, 2 * n + 1)] += lam
    A[-1, -1] += 1e-9
    beta = np.linalg.solve(A, b)
    cnt = pd.concat([G.home, G.away]).value_counts()
    return dict(mu=float(np.exp(beta[0])), hca=float(beta[-1]), off=dict(zip(teams, beta[1:n + 1])),
                de=dict(zip(teams, beta[n + 1:2 * n + 1])), games=cnt.to_dict())


def predict(F: dict, home: str, away: str, neutral: bool):
    if min(F["games"].get(home, 0), F["games"].get(away, 0)) < MIN_TEAM_GAMES:
        return None
    hc = 0.0 if neutral else F["hca"] / 2
    return (F["mu"] * math.exp(F["off"][home] + F["de"][away] + hc), F["mu"] * math.exp(F["off"][away] + F["de"][home] - hc))


def fit(results_dir: str, out_dir: str, as_of: str | None = None, lam: float = RIDGE, half_life: float = HALF_LIFE,
        days: int = 21) -> bool:
    """Writes <out>/teams.csv (the layout expected_points reads) and <out>/model.json (home court, SDs, and an honest
    walk-forward test: each of the last `days` game days predicted from games before it only)."""
    G = load_games(results_dir)
    if as_of and len(G):
        G = G[G.date < pd.Timestamp(as_of)]
    if len(G) < 150:
        print(f"fit: {len(G)} usable games in {results_dir} (< 150): no ratings written")
        return False
    ref = pd.Timestamp(as_of) if as_of else G.date.max() + pd.Timedelta(days=1)
    F = fit_ratings(G, ref, lam, half_life)
    mp = dict(as_of=f"{ref:%Y-%m-%d}", games=len(G), teams=len(F["off"]), first=f"{G.date.min():%Y-%m-%d}",
              hca=round(F["mu"] * 2 * math.sinh(F["hca"] / 2), 2), sd_margin=None, sd_total=None, ridge=lam,
              half_life=half_life, walk_forward=None)
    W = []
    for day in (sorted(G.date.unique())[-days:] if days > 0 else []):
        tr = G[G.date < day]
        if len(tr) < 150:
            continue
        Fd = fit_ratings(tr, pd.Timestamp(day), lam, half_life)
        for r in G[G.date == day].itertuples():
            e = predict(Fd, r.home, r.away, r.neutral)
            if e:
                W.append((e[0] - e[1], e[0] + e[1], r.hp - r.ap, r.hp + r.ap))
    W = pd.DataFrame(W, columns=["pm", "pt", "m", "t"])
    if len(W) >= 200:
        sdm, sdt = float((W.m - W.pm).std()), float((W.t - W.pt).std())
        won = (W.m > 0).astype(float)
        mp.update(sd_margin=round(sdm, 2), sd_total=round(sdt, 2), walk_forward=dict(
            games=len(W), winner_hit=round(float(((W.pm > 0) == (W.m > 0)).mean()), 4), home_wins=round(float(won.mean()), 4),
            margin_mae=round(float((W.m - W.pm).abs().mean()), 2), total_mae=round(float((W.t - W.pt).abs().mean()), 2),
            brier=round(float(((W.pm / sdm).map(ncdf) - won).pow(2).mean()), 4)))
    T, c = 68.0, float(np.mean(np.exp(list(F["off"].values()))))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if mp["sd_margin"] is None and (out / "model.json").exists():          # no test this run: keep the last fitted SDs
        old = json.loads((out / "model.json").read_text())
        mp.update(sd_margin=old.get("sd_margin"), sd_total=old.get("sd_total"))
    pd.DataFrame([dict(team=t, adj_o=round(100 * F["mu"] * math.exp(F["off"][t]) / T, 3),
                       adj_d=round(100 * F["mu"] * c * math.exp(F["de"][t]) / T, 3), tempo=T, games=F["games"].get(t, 0))
                  for t in F["off"]]).to_csv(out / "teams.csv", index=False)
    (out / "model.json").write_text(json.dumps(mp, indent=1))
    wf = mp["walk_forward"]
    print(f"fit as of {mp['as_of']}: {len(G)} games, {mp['teams']} teams, home court {mp['hca']:+.1f} pts; " +
          (f"walk-forward on {wf['games']} games: winner right {wf['winner_hit']:.1%} (home side wins {wf['home_wins']:.1%}), "
           f"margin off by {wf['margin_mae']} on average, SD margin {mp['sd_margin']} / total {mp['sd_total']}, Brier {wf['brier']}"
           if wf else f"no walk-forward test this run ({len(W)} predicted games, needs 200): SD margin / total = "
                      f"{mp['sd_margin'] or SD_MARGIN} / {mp['sd_total'] or SD_TOTAL}"))
    return True


# ------------------------------------------------------------------ picks: predicted winners, with what they cost
def picks(night: str, capture: str | None, con) -> None:
    P = pd.read_sql("SELECT * FROM predictions WHERE night = ?", con, params=[night])
    if P.empty:
        sys.exit(f"no predictions stored for {night}")
    P = mark_captures(P)
    cap = capture or P.sort_values("cap_key").capture.iloc[-1]
    P = P[(P.capture == cap) & (P.p_method == "no_vig") & (P.scope == "G")].copy()
    P["exp_ret"], P["gap"] = P.p * P.odds - 1, P.p_model - P.p_book
    ml = P[P.type == "ML"]
    W = ml.loc[ml.groupby("game").p.idxmax()].sort_values("p", ascending=False) if len(ml) else ml
    f = lambda v, sign=False: "" if pd.isna(v) else f"{v:+.1%}" if sign else f"{v:.1%}"
    L = [f"# NCAAB picks {night} (capture {cap})\n",
         ("- Verdict: **NO BET**. Every price at odds 1.15-6.0 has a negative expected return"
          if not (P[P.odds.between(1.15, 6.0)].exp_ret > 0).any() else
          f"- Verdict: **NO BET yet**. {int((P[P.odds.between(1.15, 6.0)].exp_ret > 0).sum())} prices show a positive expected return on paper") +
         "; nothing is a bet until an arm in ncaab_report.md reaches SMALL STAKE (80%, 100 games).",
         f"- {P.game.nunique()} games on the boards, {W.game.nunique()} with a to-win market, "
         f"{P[P.p_model.notna()].game.nunique()} priced by the model.\n", "## Predicted winners (highest win chance first)\n"]
    if len(W):
        L += [pd.DataFrame(dict(game=W.game, winner=W.subject.fillna(W.selection), odds=W.odds, book=W.p_book.map(f),
                                model=W.p_model.map(f), blend=W.p.map(f), break_even=(1 / W.odds).map(f),
                                exp_return=W.exp_ret.map(lambda v: f(v, True)))).to_markdown(index=False), ""]
    else:
        L.append("No to-win markets on these boards.\n")
    m = P[P.p_model.notna() & P.type.isin(["ML", "HANDICAP", "TOTAL"])].copy()
    L.append("## Where the model disagrees with the book most (watch list, not bets)\n")
    if len(m):
        m["d"] = (m.p_book - 0.5).abs()
        m = m[(m.d == m.groupby(["game", "type"]).d.transform("min")) & (m.gap > 0)].sort_values("gap", ascending=False).head(15)
        L += [pd.DataFrame(dict(game=m.game, bet=m.type, selection=m.selection, odds=m.odds, book=m.p_book.map(f),
                                model=m.p_model.map(f), gap=m.gap.map(lambda v: f(v, True)),
                                exp_return=m.exp_ret.map(lambda v: f(v, True)))).to_markdown(index=False),
              "\nThe middle line of each game. A gap is more often the model missing news than the book being wrong."]
    else:
        L.append("No model prices (no ratings yet, or the teams are not in them).")
    rc = P[(P.type == "RACE") & P.side.isin(["home", "away"])]
    if len(rc):
        rc = rc.loc[rc.groupby(["game", "line"]).p_book.idxmax()].sort_values("p_book", ascending=False)
        L += ["\n## First to 20 points, full game (book's favourite in each race)\n",
              pd.DataFrame(dict(game=rc.game, first_to=rc.line.astype(int), team=rc.subject, odds=rc.odds, book=rc.p_book.map(f),
                                model=rc.p_model.map(f), exp_return=rc.exp_ret.map(lambda v: f(v, True)))).to_markdown(index=False),
              "\nFirst-half races (to 20-60, with 'neither') are in the ledger; their cut is about 10%, the highest on the board."]
    out = Path(DB).with_name("ncaab_picks.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6 if len(W) else 5])[:6000])
    print(out)


SLIP_PLAN = [  # (target odds, legs, which legs, how they are ranked) - fixed 2026-11-18 before any slip was graded
    (3, 2, "any", "cut"), (3, 3, "handicap", "cut"), (3, 4, "race", "p"), (3, 3, "any", "model"),
    (7, 3, "handicap", "cut"), (7, 4, "mix", "cut"), (7, 5, "any", "p"),
    (10, 4, "any", "cut"), (10, 5, "mix", "model"), (10, 6, "any", "p")]
SLIP_COLS = ["night", "capture", "slip", "target", "plan", "legs", "odds", "p", "p_model", "exp_ret"]


def slips(night: str, capture: str | None, con) -> pd.DataFrame:
    """Ten paper multis at about 3x / 7x / 10x from one capture (default: the latest). One leg per game and every
    game in one slip only, so the slips are independent. Legs: two-way-priced (no_vig) to-win, handicap (.5 lines),
    first-half draw-no-bet and races, odds 1.15-2.60. 'cut' = cheapest legs first (least margin), 'p' = likeliest
    legs, 'model' = largest model-over-book gap. Win chance and expected return use the blended price p; every slip
    loses on average. Stored in the ledger AND in slips/<night>.csv next to it (row keys + odds), so they survive a
    rebuilt ledger."""
    P = pd.read_sql("SELECT * FROM predictions WHERE night = ?", con, params=[night])
    if P.empty:
        sys.exit(f"no predictions stored for {night}")
    cap = capture or mark_captures(P).sort_values("cap_key").capture.iloc[-1]
    C = P[(P.capture == cap) & (P.p_method == "no_vig") & P.odds.between(1.15, 2.60) & P.type.isin(["ML", "HANDICAP", "DNB", "RACE"])
          & P.side.isin(["home", "away"]) & ((P.type != "HANDICAP") | ((P.line % 1) == 0.5))].copy()
    C["val"], C["gap"] = C.p * C.odds, (C.p_model - C.p_book).fillna(-1)
    used, out = set(), []
    for i, (T, n, kind, rank) in enumerate(SLIP_PLAN, 1):
        pool = C[~C.game.isin(used)]
        pool = pool[pool.type.isin({"handicap": ["HANDICAP"], "race": ["RACE"], "mix": ["HANDICAP", "RACE"],
                                    "any": ["ML", "HANDICAP", "DNB", "RACE"]}[kind])]
        legs, left = [], float(T)
        for k in range(n, 0, -1):
            t = left ** (1 / k)
            q = pool[~pool.game.isin([l.game for l in legs]) & ((pool.odds / t - 1).abs() <= (0.12 if k > 1 else 0.10))]
            if q.empty:
                q = pool[~pool.game.isin([l.game for l in legs])].assign(d=lambda x: (x.odds / t - 1).abs()).nsmallest(5, "d")
            if q.empty:
                break
            l = q.sort_values({"cut": "val", "p": "p", "model": "gap"}[rank], ascending=False).iloc[0]
            if kind == "mix" and legs and all(x.type == l.type for x in legs) and k == 1:
                alt = q[q.type != l.type]
                l = alt.sort_values("val", ascending=False).iloc[0] if len(alt) else l
            legs.append(l)
            left /= l.odds
        if len(legs) < n:
            continue
        used |= {l.game for l in legs}
        o, pw = float(np.prod([l.odds for l in legs])), float(np.prod([l.p for l in legs]))
        pm = float(np.prod([l.p_model for l in legs])) if all(pd.notna(l.p_model) for l in legs) else float("nan")
        out.append(dict(night=night, capture=cap, slip=f"S{i:02d}", target=T, plan=f"{n} legs, {kind}, by {rank}",
                        legs=json.dumps([l.row_key for l in legs]), odds=round(o, 3), p=pw, p_model=pm, exp_ret=o * pw - 1,
                        _legs=legs))
    S = pd.DataFrame(out)
    con.execute("DELETE FROM slips WHERE night = ?", [night])
    if len(S):
        con.executemany(f"INSERT INTO slips VALUES ({','.join('?' * len(SLIP_COLS))})",
                        [[None if isinstance(v, float) and math.isnan(v) else v for v in r] for r in S[SLIP_COLS].astype(object).values.tolist()])
        sd = Path(DB).with_name("slips")
        sd.mkdir(exist_ok=True)
        S[SLIP_COLS].to_csv(sd / f"{night}.csv", index=False)
    con.commit()
    f = lambda v, sg=False: "n/a" if v is None or v != v else f"{v:+.1%}" if sg else f"{v:.1%}"
    L = [f"# NCAAB paper slips {night} (capture {cap})\n",
         "**Paper only: NO BET.** Every leg already pays the book's cut, a multi pays it once per leg. No game is in two "
         "slips, so each wins or loses on its own. Expected return is what a slip makes on average.\n"]
    if not len(S):
        L.append("No slip could be built from this capture (too few two-way-priced legs).")
    else:
        L += [pd.DataFrame([dict(slip=r["slip"], target=f"{r['target']:g}x", plan=r["plan"], odds=f"{r['odds']:.2f}",
                                 win_chance=f(r["p"]), model_says=f(r["p_model"]), break_even=f(1 / r["odds"]),
                                 exp_return=f(r["exp_ret"], True)) for r in out]).to_markdown(index=False), ""]
        for r in out:
            L.append(f"\n### {r['slip']}  {r['odds']:.2f}x  win chance {f(r['p'])}  ({r['plan']})\n")
            L.append(pd.DataFrame([dict(game=l.game, market=l.market, pick=l.selection, odds=l.odds, book=f(l.p_book),
                                        model=f(l.p_model)) for l in r["_legs"]]).to_markdown(index=False))
        L.append(f"\nOn average {sum(r['p'] for r in out):.1f} of these {len(out)} slips win. {len(out)} unit stakes return "
                 f"about {sum(r['odds'] * r['p'] for r in out):.2f} units on average.")
    (Path(DB).with_name("ncaab_slips.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:3]))
    return S


def load_slips(con) -> None:
    """slips/<night>.csv next to the ledger -> the slips table, for nights the table does not hold (rebuilt ledger)."""
    have = {n for (n,) in con.execute("SELECT DISTINCT night FROM slips")}
    for f in sorted(Path(DB).with_name("slips").glob("*.csv")):
        if f.stem not in have:
            S = pd.read_csv(f, dtype={"night": str, "capture": str})
            con.executemany(f"INSERT OR IGNORE INTO slips VALUES ({','.join('?' * len(SLIP_COLS))})",
                            [[None if isinstance(v, float) and math.isnan(v) else v for v in r] for r in S[SLIP_COLS].astype(object).values.tolist()])
    con.commit()


def slip_results(con) -> pd.DataFrame:
    """Graded slips: a pushed leg counts as odds 1.00 (dropped), any lost leg loses the slip, any ungraded leg = open."""
    load_slips(con)
    S = pd.read_sql("SELECT * FROM slips ORDER BY night, slip", con)
    rows = []
    for r in S.itertuples():
        keys = json.loads(r.legs)
        G = pd.read_sql(f"SELECT row_key, odds, status, result FROM predictions WHERE night = ? AND capture = ? AND row_key IN "
                        f"({','.join('?' * len(keys))})", con, params=[r.night, r.capture, *keys])
        res = ("lost" if (G.result == "lost").any() else "open" if len(G) < len(keys) or (G.status != "graded").any()
               else "won")
        paid = float(np.prod(G[G.result == "won"].odds)) if res == "won" else 0.0
        rows.append(dict(night=r.night, slip=r.slip, target=r.target, plan=r.plan, odds=r.odds, win_chance=r.p, result=res,
                         legs_won=int((G.result == "won").sum()), legs=len(keys), paid=paid))
    return pd.DataFrame(rows)


LINE_COLS = ["gameId", "season", "seasonType", "startDate", "homeTeam", "awayTeam", "homeConference", "awayConference",
             "homeScore", "awayScore", "provider", "spread", "overUnder", "homeMoneyline", "awayMoneyline", "spreadOpen",
             "overUnderOpen"]


def fetch_lines(first: str, out_dir: str, refetch_days: int = 40) -> None:
    """Betting lines from CollegeBasketballData.com into <out>/<YYYY-MM>.csv, one call per month (the API returns at
    most 3,000 games a call; a busy month has about 1,600). A month is fetched once; the current and the previous
    month are fetched again on every run (late lines and scores). The key is read from CBBD_API_KEY and never printed.
    Field layout from the API reference (2026-11-20); the spread's sign is checked against the scores in `report`."""
    import urllib.error
    import urllib.request
    key = os.environ.get("CBBD_API_KEY", "").strip()
    if not key:
        print("lines: CBBD_API_KEY is not set - nothing fetched")
        return
    base = os.environ.get("CBBD_API", "https://api.collegebasketballdata.com")
    out, now = Path(out_dir), pd.Timestamp.now("UTC").tz_localize(None)
    out.mkdir(parents=True, exist_ok=True)
    calls = 0
    for m in pd.period_range(pd.Timestamp(first), now, freq="M"):
        f = out / f"{m}.csv"
        if 5 <= m.month <= 10 and m != pd.Period(now, "M"):     # off-season: no games, no call
            continue
        if f.exists() and m.end_time < now - pd.Timedelta(days=refetch_days):
            continue
        q = f"startDateRange={m.start_time:%Y-%m-%dT00:00:00Z}&endDateRange={m.end_time:%Y-%m-%dT23:59:59Z}"
        req = urllib.request.Request(f"{base}/lines?{q}", headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                                                                   "User-Agent": "ncaab_grade"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                js = json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"lines {m}: HTTP {e.code} " + {401: "(key wrong or not active yet)", 429: "(monthly call limit reached)"}.get(e.code, ""))
        calls += 1
        rows = [dict({k: g.get(k) for k in LINE_COLS[:10]}, **{k: (ln or {}).get(k) for k in LINE_COLS[10:]})
                for g in js for ln in (g.get("lines") or [None])]
        df = pd.DataFrame(rows, columns=LINE_COLS)
        df.to_csv(f, index=False)
        n_games = df.gameId.nunique()
        print(f"{f}: {n_games} games, {df.spread.notna().sum()} lines with a spread, providers {sorted(df.provider.dropna().unique())[:6]}"
              + ("  WARNING: 3,000 games = the API's cap, this month may be cut short" if n_games >= 3000 else ""))
    print(f"lines: {calls} API calls")


def pack(boards: str, out: str, captured_at: str | None = None) -> None:
    """A folder of .json captures -> one small .rows.csv.gz with every priced selection (what `price` needs), each
    game's tip time and when the boards were pulled (captured_utc: --captured-at, else the files' own time - so run
    `pack` on the machine that made the capture, before the files are copied anywhere)."""
    frames, skipped = [], []
    files = sorted(Path(boards).glob("*.json"))
    if not captured_at and (wn := clock_warning(files)):
        print(wn)
    cap_utc = _utc(captured_at) or (_utc(pd.Timestamp(max(os.path.getmtime(f) for f in files), unit="s")) if files else None)
    for f in files:
        try:
            frames.append(pd.DataFrame(board_rows(f)))
        except Exception as e:
            skipped.append(f"{f.name}: {e}")
    if not frames:
        sys.exit(f"no readable .json boards in {boards}")
    B = pd.concat(frames, ignore_index=True)
    B["captured_utc"] = cap_utc
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    B.to_csv(out, index=False, compression="gzip")
    print(f"{out}: {B.game.nunique()} games, {len(B):,} rows, {Path(out).stat().st_size / 1024:.0f} KB, taken {cap_utc}Z "
          f"(wrong? pass --captured-at); skipped {skipped}")


def sync(root: str, backfill_from: str | None = None) -> None:
    """Everything in order, safe to rerun: fetch MISSING results (existing files are never overwritten) -> price
    boards/<date>/<label>.rows.csv.gz files not priced yet (ratings fitted only on games BEFORE that date) -> grade
    every finished night that has a results file -> refit -> report. One run rebuilds an empty ledger completely."""
    import time
    root, con = Path(root), connect()
    today = pd.Timestamp.now("America/New_York").tz_localize(None).normalize()

    def get(date: str, slate: bool = False, pbp: bool = False) -> Path:
        out = root / ("slate" if slate else "results") / f"{date}.txt"
        if out.exists() and out.stat().st_size:                # the scrape job owns these files: a refetch from here
            return out                                         # would drop the play-by-play (RACE) lines
        try:
            fetch_results(date, str(out), slate, pbp)
        except Exception as e:
            print(f"   could not fetch {'slate' if slate else 'results'} {date}: {e}")
        return out

    if backfill_from:
        for d in pd.date_range(backfill_from, today - pd.Timedelta(days=1)):
            if not (root / "results" / f"{d:%Y-%m-%d}.txt").exists():
                get(f"{d:%Y-%m-%d}")
                time.sleep(0.4)
    if (root / "results").exists():                             # first, so the fitted SDs exist before any pricing
        fit(str(root / "results"), str(root / "ratings"))
    for f in sorted((root / "boards").glob("*/*.rows.csv.gz")):
        night, cap = f.parent.name, f.name.split(".")[0]
        if con.execute("SELECT 1 FROM predictions WHERE night = ? AND capture = ? LIMIT 1", [night, cap]).fetchone():
            continue
        sl = get(night, slate=True)
        rated = (root / "results").exists() and fit(str(root / "results"), str(root / "ratings"), as_of=night, days=0)
        try:
            price(night, str(f), cap, str(root / "ratings" / "teams.csv") if rated else None, con, str(sl))
            picks(night, None, con)
            load_slips(con)
            if not con.execute("SELECT 1 FROM slips WHERE night = ? LIMIT 1", [night]).fetchone():
                slips(night, cap, con)                         # once per night, from its first priced capture
        except SystemExit as e:
            print("  ", e)
    # grade AFTER pricing, so a ledger rebuilt from nothing is graded in the same run. Every night with a results
    # file is graded once ('open' rows), whatever its age; a night with gaps (no_result) is retried for a week.
    todo = con.execute("""SELECT night, sum(status = 'open') FROM predictions WHERE status IN ('open', 'no_result')
                          GROUP BY night ORDER BY night""").fetchall()
    for n, n_open in todo:
        if not pd.Timestamp(n) < today or not (n_open or pd.Timestamp(n) >= today - pd.Timedelta(days=7)):
            continue
        f = get(n, pbp=True) if pd.Timestamp(n) >= today - pd.Timedelta(days=4) else root / "results" / f"{n}.txt"
        if f.exists():
            try:
                grade(n, str(f), con)
            except SystemExit as e:
                print("  ", e)
    if (root / "results").exists():
        fit(str(root / "results"), str(root / "ratings"))
    try:
        report(con)
    except SystemExit as e:
        print(e)


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    import tempfile
    od, mk, n = [], [], [0]

    def M(name, sels):
        ids = []
        for nm, tid, pr, cid, sv in sels:
            n[0] += 1
            od.append(dict(id=n[0], name=nm, typeId=tid, price=pr, oddStatus=0, competitorId=cid, sv=sv))
            ids.append(n[0])
        mk.append(dict(id=1000 + len(mk), name=name, typeId=len(mk), desktopOddIds=[ids]))

    H, A = "Baylor Bears", "Jackson State Tigers"
    M("Győztes (hosszabbítást beleértve)", [(H, 1, 1.20, 1, None), (A, 3, 4.60, 2, None)])
    M("Hendikep (hosszabbítást beleértve)", [(f"{H} -9,5", 1714, 1.90, 1, "-9.5"), (f"{A} +9,5", 1715, 1.90, 2, "-9.5"),
                                            (f"{H} -12,5", 1714, 2.30, 1, "-12.5"), (f"{A} +12,5", 1715, 1.60, 2, "-12.5")])
    M("Pontok száma (hosszabbítást beleértve)", [("150,5 felett", 12, 1.86, None, "150.5"), ("150,5 alatt", 13, 1.95, None, "150.5")])
    M(f"{H} pontok száma (hosszabbítást beleértve)", [("80,5 felett", 12, 1.83, None, "80.5"), ("80,5 alatt", 13, 1.83, None, "80.5")])
    M("1. félidő - Hendikep", [(f"{H} -5,5", 1714, 1.90, 1, "-5.5"), (f"{A} +5,5", 1715, 1.90, 2, "-5.5")])
    M("1. félidő - pontok száma", [("70,5 felett", 12, 1.90, None, "70.5"), ("70,5 alatt", 13, 1.90, None, "70.5")])
    M("1. félidő - 1X2", [(H, 1, 1.50, 1, None), ("Döntetlen", 2, 12.0, None, None), (A, 3, 3.20, 2, None)])
    M("Félidő/végeredmény", [("x/y", 900, 5.0, None, None)])
    M("Pontok száma páratlan/páros (hosszabbítást beleértve)", [("Páratlan", 70, 1.90, None, None), ("Páros", 72, 1.90, None, None)])
    M("1. félidő - pontok száma páratlan/páros", [("Páratlan", 70, 1.90, None, None), ("Páros", 72, 1.90, None, None)])
    M("1.félidő - verseny a(z) 20 pontért", [(H, 1, 1.65, 1, "20"), ("Egyik sem", 7, 81.0, None, "20"), (A, 3, 2.10, 2, "20")])
    M("1.félidő - verseny a(z) 45 pontért", [(H, 1, 8.0, 1, "45"), ("Egyik sem", 7, 1.11, None, "45"), (A, 3, 12.0, 2, "45")])
    M("1.félidő - verseny a(z) 30 pontért", [(H, 1, 1.60, 1, "30"), (A, 3, 2.20, 2, "30")])          # 'neither' missing
    M("Ki ér el elsőként 20 pontot (hosszabbítást beleértve)?", [(H, 1, 1.62, 1, "20"), (A, 3, 2.15, 2, "20")])
    M("1. negyed - pontok száma", [("30,5 felett", 12, 1.9, None, "30.5"), ("30,5 alatt", 13, 1.9, None, "30.5")])
    ev = dict(markets=mk, odds=od, competitors=[dict(id=1, name=H), dict(id=2, name=A)], startDate="2026-01-02T00:00:00Z")
    TAKEN = {"0900": "2026-01-01T14:00Z", "1730": "2026-01-01T22:30Z"}
    tmp = Path(tempfile.mkdtemp())
    for cap, bump in (("0900", 0.0), ("1730", 0.10)):                       # second capture: home favourite shortens
        e2 = json.loads(json.dumps(ev))
        e2["odds"][0]["price"], e2["odds"][1]["price"] = 1.20 - bump / 2, 4.60 + bump * 8
        (tmp / cap).mkdir()
        (tmp / cap / "BAY-JKST.net.json").write_text(json.dumps([dict(url="GetEventDetails", body=e2)]))
    (tmp / "teams.csv").write_text(f"team,adj_o,adj_d,tempo\n{H},118,95,68\n{A},100,108,70\nOther,108,104,68\n")
    # Baylor 40+38 = 78, Jackson State 36+42 = 78 -> overtime 11:6 -> final 89:84
    (tmp / "res.txt").write_text(f"GAME {A}@{H}\nH {H}: 40 38 11\nH {A}: 36 42 6\nRACE 10:H1 20:A1 30:H1 40:H1 45:A2 50:A2\n")
    global DB
    DB = str(tmp / "t.sqlite")
    con = connect()
    for cap in ("0900", "1730"):
        price("2026-01-01", str(tmp / cap), cap, str(tmp / "teams.csv"), con, captured_at=TAKEN[cap])
    grade("2026-01-01", str(tmp / "res.txt"), con)
    g = pd.read_sql("SELECT * FROM predictions WHERE capture = '1730'", con)
    R = {(r.type, r.scope, r.side, None if pd.isna(r.line) else r.line): r.result for r in g.itertuples()}
    want = {("ML", "G", "home", None): "won", ("ML", "G", "away", None): "lost",
            ("HANDICAP", "G", "home", -9.5): "lost", ("HANDICAP", "G", "away", 9.5): "won",       # won by 5
            ("HANDICAP", "G", "away", 12.5): "won",
            ("TOTAL", "G", "over", 150.5): "won",                                                 # 173 with overtime
            ("TEAM_TOTAL", "G", "over", 80.5): "won",                                             # Baylor 89
            ("HANDICAP", "H1", "home", -5.5): "lost", ("HANDICAP", "H1", "away", 5.5): "won",     # 40:36
            ("TOTAL", "H1", "over", 70.5): "won", ("1X2", "H1", "home", None): "won", ("1X2", "H1", "draw", None): "lost",
            ("RACE", "H1", "away", 20.0): "won", ("RACE", "H1", "home", 20.0): "lost", ("RACE", "H1", "neither", 20.0): "lost",
            ("RACE", "H1", "neither", 45.0): "won", ("RACE", "H1", "away", 45.0): "lost",        # 45 reached only after half-time
            ("RACE", "G", "away", 20.0): "won", ("RACE", "G", "home", 20.0): "lost", ("RACE", "H1", "home", 30.0): "won",
            ("ODD_EVEN", "G", "odd", None): "won", ("ODD_EVEN", "G", "even", None): "lost",       # 173: odd
            ("ODD_EVEN", "H1", "even", None): "won", ("ODD_EVEN", "H1", "odd", None): "lost"}    # 40 + 36 = 76: even
    bad = [(k, v, R.get(k)) for k, v in want.items() if R.get(k) != v]
    d = decided(con)
    (tmp / "bad.txt").write_text(f"GAME {A}@{H}\nH {H}: 40 38\nH {A}: 36 42\n")
    checks = [("settlement", not bad, bad),
              ("quarters and combos unparsed", (g.type == "UNPARSED").sum() == 3 and set(g[g.type == "UNPARSED"].status) == {"unparsed"}, None),
              ("no-vig pair sums to 1", abs(g[(g.type == "ML")].p_book.sum() - 1) < 1e-9, None),
              ("handicap lines pair up", (g[g.type == "HANDICAP"].p_method == "no_vig").all(), None),
              ("model favours Baylor", float(g[(g.type == "ML") & (g.side == "home")].p_model.iloc[0]) > 0.8, None),
              ("only the last capture is pooled", set(d.capture) == {"1730"} and d.cluster.nunique() == 1, None),
              ("tie without overtime rejected", bool(read_results(str(tmp / "bad.txt"))[1]), None),
              ("race: 3-way and 2-way priced, 2 of 3 not pooled", (g[(g.type == "RACE") & (g.line != 30)].p_method == "no_vig").all()
               and (g[(g.type == "RACE") & (g.line == 30)].p_method != "no_vig").all()
               and abs(g[(g.type == "RACE") & (g.line == 45)].p_book.sum() - 1) < 1e-9, None),
              ("race model: sums to 1, favours Baylor", abs(g[(g.type == "RACE") & (g.line == 20) & (g.scope == "H1")].p_model.sum() - 1) < 1e-9
               and float(g[(g.type == "RACE") & (g.scope == "G") & (g.side == "home")].p_model.iloc[0]) > 0.6, None),
              ("play-by-play -> race line", race_from_plays(
                  [dict(homeScore=h, awayScore=a, period=dict(number=q)) for h, a, q in
                   ((0, 0, 1), (3, 0, 1), (3, 5, 1), (9, 5, 1), (9, 11, 1), (12, 11, 2))], 12, 11) == "RACE 5:A1 10:A1"
               and race_from_plays([dict(homeScore=3, awayScore=0, period=dict(number=1))], 70, 60) is None, None)]
    report(con)
    txt = Path(DB).with_name("ncaab_report.md").read_text()
    checks.append(("open-vs-close section filled", "price shortened" in txt, None))
    # own ratings: 40 teams of known strength, 30 days x 20 games, every 5th game on a neutral court
    rng, T = np.random.default_rng(7), [f"Team {i:02d}" for i in range(40)]
    st = rng.normal(0, 8, 40)
    (tmp / "results").mkdir()
    for day in range(30):
        o, L = rng.permutation(40), []
        for j in range(20):
            h, a, neu = o[2 * j], o[2 * j + 1], j % 5 == 0
            hp = int(round(72 + (st[h] - st[a]) / 2 + (0 if neu else 1.6) + rng.normal(0, 8)))
            ap = int(round(72 - (st[h] - st[a]) / 2 - (0 if neu else 1.6) + rng.normal(0, 8)))
            hp += hp == ap
            L += [f"GAME {T[a]}@{T[h]}"] + ["NEUTRAL"] * neu + [f"H {T[h]}: {hp // 2} {hp - hp // 2}", f"H {T[a]}: {ap // 2} {ap - ap // 2}"]
        (tmp / "results" / f"{pd.Timestamp('2025-12-01') + pd.Timedelta(days=day):%Y-%m-%d}.txt").write_text("\n".join(L) + "\n")
    fitted = fit(str(tmp / "results"), str(tmp / "ratings"))
    tc, mp = pd.read_csv(tmp / "ratings" / "teams.csv").set_index("team"), json.loads((tmp / "ratings" / "model.json").read_text())
    f = expected_points(str(tmp / "ratings" / "teams.csv"))
    e_home, e_neu = f(T[0], T[1]), f(T[0], T[1], True)
    checks += [("fit recovers team strength", fitted and np.corrcoef((tc.adj_o - tc.adj_d).reindex(T), st)[0, 1] > 0.9, None),
               ("fit: walk-forward beats a coin", (mp.get("walk_forward") or {}).get("winner_hit", 0) > 0.6, mp.get("walk_forward")),
               ("fit: home court about 3.2", 1.5 < mp["hca"] < 5.0, mp["hca"]),
               ("neutral court removes home edge", abs((e_home[0] - e_home[1]) - (e_neu[0] - e_neu[1]) - mp["hca"]) < 1e-6, None),
               ("ratings predict the true margin", abs((e_neu[0] - e_neu[1]) - (st[0] - st[1])) < 6, (e_neu, st[0] - st[1]))]
    pack(str(tmp / "1730"), str(tmp / "boards" / "2026-01-02" / "1730.rows.csv.gz"), TAKEN["1730"])
    price("2026-01-02", str(tmp / "boards" / "2026-01-02" / "1730.rows.csv.gz"), None, str(tmp / "teams.csv"), con)
    g2 = pd.read_sql("SELECT * FROM predictions WHERE night = '2026-01-02'", con)
    same = g2.merge(g, on="row_key", suffixes=("", "_j"))
    checks.append(("packed boards price the same", len(g2) == len(g) == len(same) and set(g2.capture) == {"1730"}
                   and np.allclose(same.p_book, same.p_book_j) and np.allclose(same.p.astype(float), same.p_j.astype(float)), None))
    picks("2026-01-01", None, con)
    ptxt = Path(DB).with_name("ncaab_picks.md").read_text()
    checks.append(("picks names the winner and says NO BET", "Baylor Bears" in ptxt and "NO BET" in ptxt, None))
    import shutil
    stamp = lambda fl, when: os.utime(fl, (pd.Timestamp(when).timestamp(),) * 2)
    cw = tmp / "cw"
    cw.mkdir()
    (cw / "2026-01-01_A-B_1131.net.json").write_text(json.dumps([dict(url="https://x/api?timezoneOffset=300&y=1", status=200, body=ev)]))
    stamp(cw / "2026-01-01_A-B_1131.net.json", "2026-01-01T16:31:40Z")
    cw_ok = clock_warning(list(cw.glob("*.json")))
    stamp(cw / "2026-01-01_A-B_1131.net.json", "2026-01-03T09:02:00Z")
    checks.append(("copied capture files are noticed", cw_ok is None and bool(clock_warning(list(cw.glob("*.json")))), cw_ok))
    # ncaab-0.7: captures ordered by time and per game; started games dropped; one sync run grades an empty ledger
    def one(Hn, An, tip):
        o = [dict(id=1, name=Hn, typeId=1, price=1.5, oddStatus=0, competitorId=1, sv=None),
             dict(id=2, name=An, typeId=3, price=2.5, oddStatus=0, competitorId=2, sv=None)]
        return [dict(url="GetEventDetails", body=dict(markets=[dict(id=10, name="Győztes (hosszabbítást beleértve)", typeId=1,
                desktopOddIds=[[1, 2]])], odds=o, competitors=[dict(id=1, name=Hn), dict(id=2, name=An)], startDate=tip))]
    noon, late = ("Noon Home", "Noon Away", "2026-02-01T17:00:00Z"), ("Late Home", "Late Away", "2026-02-02T02:00:00Z")
    for lab, games in (("1130", (noon, late)), ("2300", (noon, late)), ("0030", (late,))):
        (tmp / "t" / lab).mkdir(parents=True)
        for i, gm in enumerate(games):
            (tmp / "t" / lab / f"{i}.net.json").write_text(json.dumps(one(*gm)))
    out3 = {}
    for lab, at in (("1130", "2026-02-01T10:30Z"), ("2300", "2026-02-01T22:00Z"), ("0030", "2026-02-01T23:30Z")):
        out3[lab] = price("2026-02-01", str(tmp / "t" / lab), lab, None, con, captured_at=at)
    (tmp / "r3.txt").write_text("GAME Noon Away@Noon Home\nH Noon Home: 40 40\nH Noon Away: 30 30\n\n"
                                "GAME Late Away@Late Home\nH Late Home: 40 40\nH Late Away: 30 30\n")
    grade("2026-02-01", str(tmp / "r3.txt"), con)
    d3 = decided(con)
    d3 = d3[d3.night == "2026-02-01"]
    used = dict(d3.drop_duplicates("game")[["game", "capture"]].values)
    wm = classify("William & Mary Tribe pontok száma (hosszabbítást beleértve)", {OVER, UNDER}, ("William & Mary Tribe", "Duke Blue Devils"))
    db_keep, DB = DB, str(tmp / "fresh.sqlite")                 # an empty ledger, a night 30 days old, one sync run
    old = f"{pd.Timestamp.now('America/New_York').tz_localize(None).normalize() - pd.Timedelta(days=30):%Y-%m-%d}"
    root = tmp / "root"
    (root / "results").mkdir(parents=True)
    pack(str(tmp / "t" / "0030"), str(root / "boards" / old / "1730.rows.csv.gz"), "2020-01-01T00:00Z")
    (root / "results" / f"{old}.txt").write_text("GAME Late Away@Late Home\nH Late Home: 40 40\nH Late Away: 30 30\nRACE 20:H1\n")
    (root / "slate" / f"{old}.txt").parent.mkdir()
    (root / "slate" / f"{old}.txt").write_text("GAME Late Away@Late Home\n")
    sync(str(root))
    fresh = pd.read_sql("SELECT status FROM predictions", connect()).status.value_counts().to_dict()
    DB = db_keep
    Bn = bandit(pd.concat([d.assign(cluster=d.cluster + str(i)) for i in range(15)]))     # enough 'games' to list arms
    sl = slips("2026-02-01", "1130", con)                      # two games, two to-win markets -> the 2-leg slip only
    sr = slip_results(con)
    con.execute("DELETE FROM slips")
    con.commit()
    sr2 = slip_results(con)                                    # reloaded from slips/2026-02-01.csv
    checks += [("slips: one leg per game, odds = product of legs", len(sl) >= 1 and all(
                    len({l.game for l in r}) == len(r) and abs(np.prod([l.odds for l in r]) - o) < 1e-3 for r, o in zip(sl._legs, sl.odds)), None),
               ("slips: settled from their legs (both home favourites won)", len(sr) >= 1 and sr.result.iloc[0] == "won"
                and abs(sr.paid.iloc[0] - 2.25) < 1e-6, sr.to_dict("records")),
               ("slips survive a wiped table (csv next to the ledger)", len(sr2) == len(sr) and (sr2.result == sr.result).all(), None)]
    checks += [("a game that tipped before the late capture is still pooled", used.get("Noon Away@Noon Home") == "1130", used),
               ("a capture after midnight counts as the last one", used.get("Late Away@Late Home") == "0030", used),
               ("games already tipped at capture time are not stored", set(out3["2300"].game) == {"Late Away@Late Home"}, None),
               ("team name with ' & ' is typed", wm[0] == "TEAM_TOTAL", wm),
               ("one sync run grades an empty ledger, old nights too", fresh == {"graded": 2}, fresh),
               ("sync leaves existing results files alone", "RACE 20:H1" in (root / "results" / f"{old}.txt").read_text(), None),
               ("opening price attached; the favourite that shortened has positive clv",
                bool(d.p_open.notna().any()) and len(Bn) and bool((Bn.clv_games > 0).any())
                and float((d[(d.type == "ML") & (d.side == "home")].p_book - d[(d.type == "ML") & (d.side == "home")].p_open).iloc[0]) > 0,
                Bn[["arm", "games", "clv", "clv_games"]].to_dict("records") if len(Bn) else "no arms")]
    for nm, ok, info in checks:
        print(("ok   " if ok else "FAIL ") + nm + (f"  {info}" if info and not ok else ""))
    print("PASS" if all(c[1] for c in checks) else "FAIL")
    sys.exit(0 if all(c[1] for c in checks) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("report")
    s = sub.add_parser("markets"); s.add_argument("boards")
    s = sub.add_parser("price"); s.add_argument("--date", required=True); s.add_argument("--boards", required=True)
    s.add_argument("--capture"); s.add_argument("--teams"); s.add_argument("--slate")
    s.add_argument("--captured-at", help="when the boards were pulled, e.g. 2026-11-19T00:30+01:00 (default: file time)")
    s = sub.add_parser("results"); s.add_argument("--date", required=True); s.add_argument("--out", required=True)
    s.add_argument("--to", help="last date of a range; --out is then a folder, one file per date")
    s.add_argument("--pbp", action="store_true", help="also read each game's play-by-play (needed to grade race markets)")
    s = sub.add_parser("slate"); s.add_argument("--date", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("fit"); s.add_argument("--results", default="results"); s.add_argument("--out", default="ratings")
    s.add_argument("--as-of", help="use only games before this date"); s.add_argument("--ridge", type=float, default=RIDGE)
    s.add_argument("--half-life", type=float, default=HALF_LIFE); s.add_argument("--days", type=int, default=21)
    s = sub.add_parser("picks"); s.add_argument("--date", required=True); s.add_argument("--capture")
    s = sub.add_parser("pack"); s.add_argument("--boards", required=True); s.add_argument("--out", required=True)
    s.add_argument("--captured-at", help="when the boards were pulled (default: the .json files' own time)")
    s = sub.add_parser("slips"); s.add_argument("--date", required=True); s.add_argument("--capture")
    s = sub.add_parser("lines"); s.add_argument("--from", dest="first", default="2025-11-01"); s.add_argument("--out", default="lines")
    s = sub.add_parser("sync"); s.add_argument("--root", default=str(Path(__file__).parent)); s.add_argument("--backfill-from")
    s = sub.add_parser("grade"); s.add_argument("--date", required=True); s.add_argument("--results", required=True)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "markets":
        B = pd.concat([pd.DataFrame(board_rows(f)) for f in sorted(Path(a.boards).glob("*.json"))])
        pd.set_option("display.width", 220, "display.max_colwidth", 90, "display.max_rows", 2000)
        print(B.groupby(["type", "scope", "market"]).agg(rows=("odds", "size"), games=("game", "nunique")).reset_index()
              .sort_values(["type", "rows"], ascending=[True, False]).to_string(index=False))
        print(f"\n{B.game.nunique()} games; {(B.type != 'UNPARSED').mean():.0%} of {len(B):,} rows typed")
    elif a.cmd == "price":
        price(a.date, a.boards, a.capture, a.teams, connect(), a.slate, a.captured_at)
    elif a.cmd == "results":
        if a.to:
            import time
            for d in pd.date_range(a.date, a.to):
                fetch_results(f"{d:%Y-%m-%d}", str(Path(a.out) / f"{d:%Y-%m-%d}.txt"), pbp=a.pbp)
                time.sleep(0.4)
        else:
            fetch_results(a.date, a.out, pbp=a.pbp)
    elif a.cmd == "slate":
        fetch_results(a.date, a.out, slate=True)
    elif a.cmd == "fit":
        fit(a.results, a.out, a.as_of, a.ridge, a.half_life, a.days)
    elif a.cmd == "picks":
        picks(a.date, a.capture, connect())
    elif a.cmd == "pack":
        pack(a.boards, a.out, a.captured_at)
    elif a.cmd == "slips":
        slips(a.date, a.capture, connect())
    elif a.cmd == "lines":
        fetch_lines(a.first, a.out)
    elif a.cmd == "sync":
        sync(a.root, a.backfill_from)
    elif a.cmd == "grade":
        grade(a.date, a.results, connect())
    elif a.cmd == "report":
        report(connect())


if __name__ == "__main__":
    main()
