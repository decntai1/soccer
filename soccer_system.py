#!/usr/bin/env python3
"""soccer_system.py: price, grade and build paper slips for vegas.hu soccer boards (spec sections 3-6).

    python soccer_system.py selftest                              # must end with PASS before anything is trusted
    python soccer_system.py unread  --boards DIR                  # FIRST thing on a real board: what could not be read
    python soccer_system.py price   --date D --boards DIR         # every row: book price without margin, model, blend
    python soccer_system.py slips   --date D [--targets 3 6 9]    # paper slips by rule, per strategy; bandit picks one
    python soccer_system.py grade   --date D [--boards DIR]       # settle rows and slips from the ingest tables
    python soccer_system.py handcheck [--confirm]                 # five graded rows of different types to check by hand
    python soccer_system.py report                                # soccer_report.md: hits vs stated, where the model lost

Env: SOCCER_OUT (ingest tables, default data), SOCCER_DB (ledger, default ledger/soccer.sqlite).
Nothing here places a bet. Every slip is paper.

Rules (spec section 3): 1 the stated probability is the book's price with the margin removed; 2 the model has
weight 0 until held-out games give it one; 3 what cannot be read is kept, typed None, never priced or graded;
4 rows are graded under the current MODEL_VERSION; 5 every graded row records `actual` and `player_id`;
6 a player with 0 minutes is void; 7 half lines are their own markets; 8 the report says where the model lost.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_model import LeagueModel, PlayerModel, line_fracs, newcomer_starts

MODEL_VERSION = "soccer-0.2"
OUT = Path(os.environ.get("SOCCER_OUT", "data"))
DB = os.environ.get("SOCCER_DB", "ledger/soccer.sqlite")
LEG_P, LEG_MIN_ODDS, SIDE_MIN = (0.60, 0.88), 1.15, 0.60
STAT_VOTE, ONEWAY_CAP = 0.65, 1.03
BOOK_MARGIN = -0.08                                         # bandit 2 prior: every strategy loses the margin

# vegas.hu rules. None = NOT CONFIRMED from the book's rules page: a row whose result depends on it is not graded.
RULES = dict(
    second_yellow_is_two_cards=None,     # yellow + second yellow + red: 2 cards (usual) or 3?
    coach_cards_count=None,              # cards shown to the bench
    booking_points=None,                 # e.g. dict(yellow=10, red=25)
    first_scorer_own_goal=None,          # "next" = an own goal is skipped
    first_scorer_late_sub=None,          # player comes on after the first goal: "void" or "lost"
    quarter_split=True,                  # ASSUMED (standard Asian rule): 2.25 = half on 2 and half on 2.5
)

SIDE_T, GOAL_T = {"1X2", "DNB", "AH", "EH", "DC"}, {"TOTAL", "TEAM_TOTAL", "BTTS", "CS"}
OU_T = {"TOTAL", "TEAM_TOTAL", "CORNERS", "CARDS", "SHOTS", "SOT", "BOOKING_PTS", "PL_SOT", "PL_SHOTS", "PL_CARD"}
PLAYER_T = {"SCORER_ANY", "SCORER_FIRST", "PL_SOT", "PL_SHOTS", "PL_CARD"}
STAT3_T = {"CORNERS_1X2", "CARDS_1X2", "SHOTS_1X2", "SOT_1X2"}                 # which team has more
STAT_T = {"CORNERS", "CORNERS_AH", "CARDS", "SHOTS", "SOT"} | STAT3_T          # the statistics markets (statedge pool)
STATEDGE_ODDS = (1.5, 2.1)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  matchday TEXT, row_key TEXT, fixture_id INTEGER, league INTEGER, game TEXT, kickoff TEXT, market TEXT, selection TEXT,
  type TEXT, scope TEXT, side TEXT, line REAL, who TEXT, subject TEXT, player_id INTEGER, odds REAL, odds_first REAL,
  p_book REAL, p_method TEXT, overround REAL, p_model REAL, w_used REAL, p REAL, flag TEXT, reason TEXT, captured TEXT,
  model_version TEXT, status TEXT DEFAULT 'open', result TEXT, won REAL, lost REAL, ret REAL, actual REAL, grade_note TEXT,
  checked INTEGER DEFAULT 0, PRIMARY KEY (matchday, row_key));
CREATE TABLE IF NOT EXISTS weights (as_of TEXT, family TEXT, games INTEGER, w_fit REAL, t REAL, w REAL, model_version TEXT);
CREATE TABLE IF NOT EXISTS calib (as_of TEXT, games INTEGER, a REAL, b REAL);
CREATE TABLE IF NOT EXISTS slips (matchday TEXT, slip_id TEXT, strategy TEXT, target REAL, odds REAL, p REAL, legs INTEGER,
  played INTEGER DEFAULT 0, status TEXT DEFAULT 'open', result TEXT, paid REAL, PRIMARY KEY (matchday, slip_id));
CREATE TABLE IF NOT EXISTS slip_legs (matchday TEXT, slip_id TEXT, row_key TEXT, game TEXT, market TEXT, selection TEXT,
  odds REAL, p REAL, votes TEXT, PRIMARY KEY (matchday, slip_id, row_key));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def connect(db: str | None = None) -> sqlite3.Connection:
    db = db or DB
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    return con


# ------------------------------------------------------------------ small helpers
def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


CLUB_NOISE = {"fc", "cf", "sc", "sv", "ssc", "ss", "ac", "as", "us", "ud", "cd", "sd", "rc", "afc", "bsc", "tsv", "vfl", "vfb",
              "fsv", "spvgg", "1", "club", "calcio", "de", "the", "town", "city", "united", "real", "sporting", "athletic"}


def team_key(s) -> str:
    toks = [t for t in norm(s).split() if t not in CLUB_NOISE]
    return " ".join(toks) or norm(s)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def demargin(odds: list[float], target: float = 1.0) -> tuple[list[float], float]:
    """Odds-ratio method: every price in a complete group is shortened by the same odds ratio c, i.e.
    p_i = q_i / (c + q_i - c q_i) with q_i = 1/odds_i, and c chosen so the p_i sum to `target` (1; 2 for double
    chance, where exactly two of the three outcomes win). Returns (p, overround = sum q / target)."""
    q = np.array([1 / o for o in odds])
    f = lambda c: (q / (c + q - c * q)).sum() - target
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    c = math.sqrt(lo * hi)
    return list(q / (c + q - c * q)), float(q.sum() / target)


def one_way(odds: float, a: float = 0.0, b: float = 1.0) -> float:
    """Calibrated one-way price, never above 1/odds/1.03: a calibration may remove margin, never create an edge."""
    return float(min(sigmoid(a + b * logit(1 / odds)), 1 / odds / ONEWAY_CAP))


def family(typ: str, scope: str | None = None) -> str:
    """What the weights and the record are kept by."""
    return f"{typ}/{scope or 'FT'}"


def quantity(typ: str, subject=None) -> str:
    """Legs with the same game + quantity fail together and count as one quantity in slips."""
    if typ in PLAYER_T:
        return f"player:{subject}"
    if typ in STAT_T or typ == "BOOKING_PTS":
        return {"BOOKING_PTS": "cards"}.get(typ, typ.split("_")[0].lower())     # corners, corner handicap, corner 1X2: one quantity
    return "side" if typ in SIDE_T else "goals" if typ in GOAL_T else typ.lower()


# ------------------------------------------------------------------ ingest tables
def load_tables(out: Path | None = None) -> dict:
    out = Path(out or OUT)

    def rd(name):
        fs = sorted(glob.glob(str(out / f"soccer_*_{name}.csv")))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

    T = {n: rd(n) for n in ("fixtures", "team_match_stats", "events", "player_match_stats", "lineups", "teams", "players", "coverage")}
    pre = out / "soccer_prematch_lineups.csv"
    T["prematch"] = pd.read_csv(pre) if pre.exists() else pd.DataFrame()
    if len(T["fixtures"]):
        T["fixtures"]["date"] = pd.to_datetime(T["fixtures"]["date"], utc=True).dt.tz_localize(None)
    return T


def ref_key(r) -> str | None:
    """'Andy Davies, England' (API) and 'A Davies' (football-data) -> 'a davies'."""
    if not isinstance(r, str) or not r.strip() or r.strip().lower() in ("nan", "none"):
        return None
    t = norm(r.split(",")[0]).split()
    return f"{t[0][0]} {t[-1]}" if len(t) >= 2 else (t[0] if t else None)


WEAK_NOISE = {"fc", "cf", "sc", "afc", "ssc", "ss", "ac", "as", "us", "cd", "sd", "ud", "rc", "bsc", "tsv", "vfl", "vfb", "fsv", "sv",
              "spvgg", "1", "club", "calcio", "de", "the", "cp", "sad"}
SYN = {"ii": "b", "utd": "united", "munchen": "munich", "1860": "1860"}


def _toks(s: str) -> list[str]:
    return [SYN.get(t, t) for t in norm(s).split() if t not in WEAK_NOISE] or norm(s).split()


def _tok_match(x: str, y: str) -> float:
    if x == y:
        return 1.0
    a, b = (x, y) if len(x) <= len(y) else (y, x)
    if len(a) >= 2 and b.startswith(a):
        return 1.0
    it = iter(b)
    return 0.8 if len(a) >= 2 and a[0] == b[0] and all(c in it for c in a) else 0.0        # 'rvs' in 'rovers'


def _sim(a: str, b: str) -> float:
    """Name similarity for one club written two ways ('Sheffield Wednesday' / 'Sheffield Weds', 'QPR', 'Sp Gijon').
    Every word of the shorter name must match a word of the longer one: 'Sheffield Weds' scores 1.0 against
    'Sheffield Wednesday' and 0.5 against 'Sheffield United'."""
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    for x in short:                                                   # 'MK Dons': 'mk' = initials of 'milton keynes'
        for k in range(len(long_)):
            for n in (2, 3):
                run = long_[k:k + n]
                if len(run) == n and len(x) == n and x == "".join(w[0] for w in run):
                    long_ = long_[:k] + [x] + long_[k + n:]
                    break
    tok = sum(max(_tok_match(x, y) for y in long_) for x in short) / len(short)
    if len(short) == 1 and len(long_) > 2:
        tok *= 0.9                                                   # one word against a long name: a little less sure
    ini = lambda t: "".join(w[0] for w in t)
    abbr = 0.95 if (len(short) == 1 and len(short[0]) >= 2 and short[0] in (ini(long_), ini(long_)[:3])) else 0.0
    seq = difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(tok, abbr, seq)


def map_fd_teams(T: dict, FD: pd.DataFrame, out: Path | None = None, quiet: bool = True) -> tuple[dict, list]:
    """football-data team name -> API team id, per league of the current season. Evidence: games both sources hold
    (same league, date +-1, same score) count 1 each; name similarity breaks ties; one name per id (assignment).
    Overrides in data/soccer_team_map.csv (fd_name,team_id). Returns (map, unmapped names)."""
    from scipy.optimize import linear_sum_assignment
    import soccer_ingest as SI
    code_of = {lid: v[2] for lid, v in SI.LEAGUES.items()}
    F = T["fixtures"]
    m, unmapped = {}, []
    over = Path(out or OUT) / "soccer_team_map.csv"
    if over.exists():
        m.update({str(k): int(v) for k, v in pd.read_csv(over)[["fd_name", "team_id"]].values})
    if F.empty or FD.empty:
        return m, unmapped
    for lid, G in F.groupby("league_id"):
        code = code_of.get(lid)
        season = G.season.max()
        X = FD[(FD.league == code) & (FD.season == season)]
        if X.empty:
            continue
        api = pd.concat([G[["home_id", "home"]].set_axis(["id", "name"], axis=1), G[["away_id", "away"]].set_axis(["id", "name"], axis=1)]).drop_duplicates("id")
        names = sorted(set(X.home) | set(X.away))
        ai = {t: i for i, t in enumerate(api.id)}
        ni = {n: j for j, n in enumerate(names)}
        S = np.array([[_sim(a, n) for n in names] for a in api.name])
        votes = np.zeros_like(S)
        done = G[G.status.isin(["FT", "AET", "PEN"]) & G.ft_home.notna()]
        for r in done.itertuples():
            c = X[(X.hg == r.ft_home) & (X.ag == r.ft_away) & ((X.date - r.date.normalize()).abs() <= pd.Timedelta(days=1))]
            for x in c.itertuples():
                votes[ai[r.home_id], ni[x.home]] += 1 / len(c)
                votes[ai[r.away_id], ni[x.away]] += 1 / len(c)
        score = S + votes
        rows, cols = linear_sum_assignment(-score)
        taken = {v for v in m.values()}
        for i, j in zip(rows, cols):
            n, tid = names[j], int(api.id.iloc[i])
            if n in m or tid in taken:
                continue
            if votes[i, j] >= 1.5 or (votes[i, j] >= 0.5 and S[i, j] >= 0.3) or S[i, j] >= 0.6:
                m[n] = tid
            else:
                unmapped.append(f"{n} (best guess {api.name.iloc[i]}, similarity {S[i, j]:.2f}, results {votes[i, j]:.0f})")
        unmapped += [n for n in names if n not in m and not any(u.startswith(n + " (") for u in unmapped)]
    return m, unmapped


def fd_matches(T: dict, out: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """football-data.co.uk games (current and the two seasons before), teams translated to API ids. Teams that are
    not in our current API leagues keep a stable negative id: they are opponents in the history only."""
    try:
        import soccer_bandit as SB
        import soccer_ingest as SI
    except ImportError:
        return pd.DataFrame(), {}
    F = T["fixtures"]
    cur = int(F.season.max()) if len(F) else None
    if cur is None:
        return pd.DataFrame(), dict(note="no fixtures yet")
    FD = SB.load([cur - 2, cur - 1, cur]) if Path(SB.FD).exists() else pd.DataFrame()
    played = Path(out or OUT) / f"soccer_{cur}_played_fd.csv"
    if FD.empty and played.exists():                             # the copy the ingest commits to data/ (this season only)
        FD = pd.read_csv(played, parse_dates=["date"])
    if FD.empty:
        return FD, dict(note="no football-data files (run soccer_bandit.py download)")
    if FD.empty:
        return FD, dict(note="football-data files empty")
    m, unmapped = map_fd_teams(T, FD, out)
    lid_of = {v[2]: lid for lid, v in SI.LEAGUES.items()}
    tid = lambda n: m.get(n, -(int(hashlib.md5(n.encode()).hexdigest()[:7], 16)))
    X = pd.DataFrame(dict(fixture_id=None, date=FD.date, league=FD.league.map(lid_of), season=FD.season, home=FD.home.map(tid), away=FD.away.map(tid),
                          hg=FD.hg, ag=FD.ag, hthg=FD.hthg, htag=FD.htag, referee=FD.referee, h_corners=FD.hc, a_corners=FD.ac,
                          h_cards=FD.hy + FD.hr, a_cards=FD.ay + FD.ar, h_shots=FD.hsh, a_shots=FD.ash, h_sot=FD.hst, a_sot=FD.ast,
                          h_fouls=FD.hf if "hf" in FD else np.nan, a_fouls=FD.af if "af" in FD else np.nan, source="football-data"))
    cur_names = set(FD[FD.season == cur].home) | set(FD[FD.season == cur].away)
    info = dict(games=len(X), current=int((X.season == cur).sum()), mapped=len([n for n in cur_names if n in m]), teams=len(cur_names), unmapped=unmapped)
    return X, info


def matches_frame(T: dict, with_fd: bool = True) -> pd.DataFrame:
    """One row per finished game whose goal check did not fail: what the model is fitted on. API games first; on
    the free plan the API only reaches back to yesterday, so football-data.co.uk fills the rest of the history (a
    game both sources hold is kept once, from the API)."""
    F = T["fixtures"]
    if F.empty:
        return pd.DataFrame()
    cur = F.season.max()                                         # the current season: listed games, played or not
    F = F[F.status.isin(["FT", "AET", "PEN"]) & F.ft_home.notna() & (F.check_goals.astype(str) != "False")]
    if with_fd:
        F = F[F.season == cur]                                   # older seasons come from football-data, once
    M = pd.DataFrame(dict(fixture_id=F.fixture_id, date=F.date, league=F.league_id, season=F.season, home=F.home_id, away=F.away_id,
                          hg=F.ft_home, ag=F.ft_away, hthg=F.ht_home, htag=F.ht_away, referee=F.referee))
    S = T["team_match_stats"]
    if len(S):
        S = S.assign(cards=S.yellow.fillna(0) + S.red.fillna(0)).rename(columns={"shots_on_target": "sot"})
        for flag, pre in ((True, "h_"), (False, "a_")):
            x = S[S.is_home == flag][["fixture_id", "corners", "cards", "shots", "sot", "fouls"]].add_prefix(pre).rename(columns={pre + "fixture_id": "fixture_id"})
            M = M.merge(x, on="fixture_id", how="left")
    M["source"] = "api"
    if with_fd:
        X, info = T["fd"] if "fd" in T else fd_matches(T)
        T["fd"], T["fd_info"] = (X, info), info
        if len(X):
            key = lambda d: set(zip(d.league, d.home, d.away, d.date.dt.normalize()))
            have = key(M) | key(M.assign(date=M.date - pd.Timedelta(days=1))) | key(M.assign(date=M.date + pd.Timedelta(days=1)))
            keep = [k not in have for k in zip(X.league, X.home, X.away, X.date.dt.normalize())]
            M = pd.concat([M, X[keep]], ignore_index=True)
    M["referee"] = M.referee.map(ref_key)
    return M.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ board reading (Altenar .net.json captures)
def read_capture(path: str) -> list[dict]:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    out = []
    for g in d if isinstance(d, list) else [d]:
        b = g.get("body") if isinstance(g, dict) else None
        if isinstance(b, dict) and isinstance(b.get("markets"), list) and isinstance(b.get("odds"), list) and len(b.get("competitors") or []) == 2:
            sp = b.get("sport") or {}
            sp = sp if isinstance(sp, dict) else dict(name=sp)
            if sp and sp.get("typeId") != 1 and sp.get("name") != "Labdarúgás":    # another sport's board in the folder
                continue
            out.append(b)
    return out


def raw_rows(ev: dict) -> list[dict]:
    odds = {o["id"]: o for o in ev["odds"]}
    kids = {k["id"]: k for k in ev.get("childMarkets") or []}
    rows, seen = [], set()                     # a market can be listed twice (two sections of the page): each price once

    def add(m, holder, child, child_comp=None):
        for grp in holder.get("desktopOddIds") or []:
            for oid in grp if isinstance(grp, list) else [grp]:
                o = odds.get(oid)
                if o is None or not isinstance(o.get("price"), (int, float)) or oid in seen:
                    continue
                seen.add(oid)
                rows.append(dict(market_id=holder["id"], market=str(m.get("name") or "").strip(), child=child, selection=str(o.get("name") or "").strip(),
                                 sv=o.get("sv"), comp=o.get("competitorId"), child_comp=child_comp, odds=float(o["price"]),
                                 active=o.get("oddStatus", 0) == 0))

    for m in ev["markets"]:
        add(m, m, None)
        for cid in m.get("childMarketIds") or []:
            if cid in kids:
                add(m, kids[cid], str(kids[cid].get("childName") or "").strip() or None, kids[cid].get("competitorId"))
    return rows


_NUM = r"[-+]?\d+(?:[.,]\d+)?"


def _f(s):
    try:
        return float(str(s).replace(",", "."))
    except (TypeError, ValueError):
        return None


def classify(r: dict, home: dict, away: dict) -> dict:
    """market + selection wording -> type, scope, side, line, who, subject. Patterns read off real vegas.hu boards
    (EFL Championship, 2026-11-21 captures). Anything not understood: type None and the reason; never forced."""
    out = dict(type=None, scope="FT", side=None, line=None, who=None, subject=None, reason=None)
    bad = lambda why: dict(out, type=None, reason=why)
    raw, sel = r["market"], r["selection"]
    for t, tag in ((home, " hteam "), (away, " ateam ")):              # team names in the market name -> tokens
        if t.get("name") and t["name"] in raw:
            raw = raw.replace(t["name"], tag)
    m = norm(raw)
    g = re.match(r"(1|2) felido (.*)", m)
    if g:
        out["scope"], m = f"H{g.group(1)}", g.group(2)
    combo = r"&|(?<![\d(])\+(?!\d)"                                     # 'A + B', 'A & B'; not '(+0,75)' or '2+'
    if re.search(combo + "|/", raw) or re.search(combo, sel) or re.search(r"\b(es|vagy)\b", m):
        return bad("combination market not read")
    if re.search(r"\b(perc|percben|percig|negyed|felido|felidoben|felidot)\b", m):
        return bad("period or time-window market not read")
    if "hteam" in m.split() and "ateam" in m.split():
        return bad("both teams in the market name: not read")
    for tok, w in (("hteam", "home"), ("ateam", "away")):
        if tok in m.split():
            out["who"], m = w, re.sub(r"\s+", " ", m.replace(tok, " ")).strip()
    s = norm(sel)
    ou = "over" if re.search(r"\bfelett\b", s) else "under" if re.search(r"\balatt\b", s) else None
    num = re.search(_NUM, sel.replace(home["name"], "").replace(away["name"], ""))
    sv = _f(str(r["sv"]).split("|")[0]) if r["sv"] is not None else None
    base = norm(re.sub(r"\(.*?\)", "", sel))
    side = ("X" if re.fullmatch(r"x|dontetlen", base) else "1" if (r["comp"] == home["id"] or base == norm(home["name"])) else
            "2" if (r["comp"] == away["id"] or base == norm(away["name"])) else None)

    def over_under(typ):
        line = sv if sv is not None else _f(num.group()) if num else None
        if ou is None or line is None:
            return bad("not an over/under selection (exact count, range or odd/even)")
        return dict(out, type=typ, side=ou, line=line)

    # ---- player markets (one child market per player)
    if r["child"]:
        out["subject"] = r["child"]
        if "szupercser" in m:
            return bad("super-sub variant: book rules not read")
        if m == "golszerzo":
            kind = {"barmikor": "any", "1 golszerzo": "first", "utolso": "last"}.get(s)
            if kind == "any":
                return dict(out, type="SCORER_ANY", side="yes", line=0.5, who=None)
            if kind == "first":
                return dict(out, type="SCORER_FIRST", side="yes", who=None)
            return bad("last scorer not read" if kind == "last" else "scorer selection not read")
        if m == "hany golt szerez":
            lad = re.match(r"(\d+)\+", sel.strip())
            return dict(out, type="SCORER_ANY", side="yes", line=int(lad.group(1)) - 0.5, who=None) if lad else bad("scorer ladder not read")
        if m == "golt szerez" and out["scope"] == "FT":                   # older board wording
            return dict(out, type="SCORER_ANY", side="yes", line=0.5, who=None)
        if m == "kaput eltalalo lovesek rendes jatekido":
            return dict(over_under("PL_SOT"), subject=r["child"], who=None)
        if m == "lovesek rendes jatekido":
            return dict(over_under("PL_SHOTS"), subject=r["child"], who=None)
        if m == "lapok":                                                     # player booked ('csere jatekossal' variant is not read)
            x = over_under("PL_CARD")
            return dict(x, subject=r["child"], who=None) if x["type"] and abs(x["line"] - 0.5) < 1e-9 else bad("player cards line other than 0.5")
        return bad("player market not read")
    # ---- match and team markets
    if m in ("1x2", "vegeredmeny") and out["who"] is None:
        return dict(out, type="1X2", side=side) if side else bad("1X2 selection not read")
    if m == "ketesely":
        dc = s.upper().replace(" ", "") if s in ("1x", "12", "x2") else None
        if dc is None:
            has = (norm(home["name"]) in s, "dontetlen" in s, norm(away["name"]) in s)
            dc = {(True, True, False): "1X", (True, False, True): "12", (False, True, True): "X2"}.get(has)
        return dict(out, type="DC", side=dc) if dc else bad("double-chance selection not read")
    if m == "dontetlennel a tet visszajar":
        return dict(out, type="DNB", side=side) if side in ("1", "2") else bad("team not read")
    if m == "mindket csapat szerez golt":
        yn = {"igen": "yes", "nem": "no"}.get(s)
        return dict(out, type="BTTS", side=yn) if yn else bad("yes/no not read")
    if m in ("pontos vegeredmeny", "pontos eredmeny", "vegeredmeny xl"):
        g = re.fullmatch(r"\s*(\d+)\s*[:\-]\s*(\d+)\s*", sel)
        return dict(out, type="CS", side=f"{g.group(1)}:{g.group(2)}") if g else bad("correct-score selection not read")
    if m == "golok szama":
        return over_under("TEAM_TOTAL" if out["who"] else "TOTAL")
    if m == "golt szerez" and out["who"]:
        yn = {"igen": "over", "nem": "under"}.get(s)
        return dict(out, type="TEAM_TOTAL", side=yn, line=0.5) if yn else bad("yes/no not read")
    if m in ("hendikep", "azsiai hendikep", "szoglet hendikep"):
        if side not in ("1", "2") or sv is None:
            return bad("handicap side or line not read")
        own = _f(num.group()) if num else None                              # the number shown next to the team
        if own is not None and abs(abs(own) - abs(sv)) > 1e-9:
            return bad("handicap line in the name and in the data disagree")
        typ = "CORNERS_AH" if m.startswith("szoglet") else "AH"
        return dict(out, type=typ, side=side, line=sv if side == "1" else -sv)    # sv is the home line on both sides
    if m == "hendikep 1x2":
        g = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", str(r["sv"] or ""))
        if not g or side is None:
            return bad("3-way handicap not read")
        return dict(out, type="EH", side=side, line=float(int(g.group(1)) - int(g.group(2))))   # home starts a:b
    if m == "szogletek szama":
        return over_under("CORNERS")
    if m in ("buntetolapok szama", "lapok szama"):
        return over_under("CARDS")
    if m.startswith("buntetopont"):
        return over_under("BOOKING_PTS")
    if m in ("kaput eltalalo lovesek szama rendes jatekido", "osszes kaput eltalalo lovesek szama rendes jatekido", "kapura lovesek"):
        return over_under("SOT")
    if m == "osszes loves rendes jatekido":
        return over_under("SHOTS")
    stat3 = {"szoglet 1x2": "CORNERS_1X2", "buntetolapok 1x2": "CARDS_1X2", "1x2 kapura lovesek szama rendes jatekido": "SHOTS_1X2",
             "1x2 kaput eltalalo lovesek szama rendes jatekido": "SOT_1X2"}.get(m)
    if stat3:
        return dict(out, type=stat3, side=side) if side else bad("1X2 selection not read")
    return bad("unknown market")


class Matcher:
    """Board event -> fixture row, by kickoff day (+-1) and both team names. No unique match = no match."""

    def __init__(self, F: pd.DataFrame, aliases: dict | None = None):
        self.F, self.alias = F, {norm(k): v for k, v in (aliases or {}).items()}

    def _score(self, board: str, api: str, tid) -> float:
        if self.alias.get(norm(board)) == tid:
            return 1.0
        return _sim(board, api)

    def find(self, ev: dict, day: str):
        if self.F.empty:
            return None, "no fixtures table"
        d = pd.Timestamp(str(ev.get("startDate") or day)[:10])
        C = self.F[(self.F.date >= d - pd.Timedelta(days=1)) & (self.F.date < d + pd.Timedelta(days=2))]
        h, a = ev["competitors"][0]["name"], ev["competitors"][1]["name"]
        pairs = [(self._score(h, r.home, r.home_id), self._score(a, r.away, r.away_id), r.fixture_id) for r in C.itertuples()]
        sc = sorted(((min(x, y), f) for x, y, f in pairs), reverse=True)
        if not sc or sc[0][0] < 0.72:
            # a team plays once a day: one side certain, the other plausible, and no other candidate
            sure = [(max(x, y), f) for x, y, f in pairs if max(x, y) >= 0.95 and min(x, y) >= 0.4]
            if len(sure) == 1:
                return C[C.fixture_id == sure[0][1]].iloc[0], None
            return None, f"no fixture found for '{h}' v '{a}' around {d.date()}"
        if len(sc) > 1 and sc[1][0] > sc[0][0] - 0.08:
            return None, f"two fixtures fit '{h}' v '{a}'"
        return C[C.fixture_id == sc[0][1]].iloc[0], None


def find_player(name: str, roster: pd.DataFrame):
    """Exact name, else surname + first initial, inside the two squads. Two candidates = no match, never a guess."""
    if roster.empty:
        return None
    n = norm(re.sub(r"\(.*?\)", "", name))
    if "," in name:
        n = norm(" ".join(reversed([x.strip() for x in name.split(",", 1)])))
    keys = roster.player.map(norm)
    hit = roster[keys == n]
    if len(hit.player_id.unique()) != 1 and n:
        hit = roster[[k.split()[-1:] == n.split()[-1:] and k[:1] == n[:1] for k in keys]]
    return hit.iloc[0] if len(hit.player_id.unique()) == 1 else None


# ------------------------------------------------------------------ price
def current(con, table: str, default):
    r = con.execute(f"SELECT * FROM {table} ORDER BY as_of DESC LIMIT 1").fetchone()
    return r if r else default


def weights_now(con) -> dict:
    q = "SELECT family, w FROM weights WHERE as_of = (SELECT max(as_of) FROM weights) AND model_version = ?"
    return dict(con.execute(q, [MODEL_VERSION]).fetchall())


def price_event(ev: dict, day: str, T: dict, M: pd.DataFrame, match: Matcher, models: dict, w: dict, cal: tuple, captured: str) -> list[dict]:
    home, away = ev["competitors"][0], ev["competitors"][1]
    fx, why = match.find(ev, day)
    game = f"{home['name']} v {away['name']}"
    roster = pd.DataFrame()
    gm = pm = None
    if fx is not None:
        P = T["players"]
        roster = P[P.team_id.isin([fx.home_id, fx.away_id])] if len(P) else P
        key = (fx.league_id, day)
        if key not in models:
            starts = newcomer_starts(M) if len(M) else {}
            lm = LeagueModel(M, fx.league_id, day, starts, fx.season) if len(M) else None
            PS = T["player_match_stats"]
            if len(PS):
                PS = PS.merge(T["fixtures"][["fixture_id", "date", "league_id"]], on="fixture_id")
                PS = PS[PS.league_id == fx.league_id]
            models[key] = (lm, PlayerModel(PS, T["events"], day) if len(PS) else None)
        lm, pm = models[key]
        gm = lm.game(fx.home_id, fx.away_id, ref_key(fx.referee)) if lm is not None and lm.dc is not None else None
    rows = []
    for r in raw_rows(ev):
        c = classify(r, home, away)
        row = dict(r, **c, game=game, fixture_id=None if fx is None else int(fx.fixture_id), league=None if fx is None else int(fx.league_id),
                   kickoff=ev.get("startDate"), player_id=None, flag=None, p_book=None, p_method=None, overround=None, p_model=None)
        if not r["active"] and row["type"]:
            row.update(type=None, reason="price suspended on the board")
        if row["type"] and r["odds"] <= 1.0:
            row.update(type=None, reason="price not above 1.00")
        if row["type"] in PLAYER_T:
            team = None if fx is None else {home["id"]: fx.home_id, away["id"]: fx.away_id}.get(r.get("child_comp"))
            hit = find_player(row["subject"], roster[roster.team_id == team] if team is not None and len(roster) else roster)
            tag = re.search(r"\(([A-Za-z]{2,4})\)", r["selection"])
            if hit is None:
                row.update(type=None, reason="player not matched to one player of the two squads (no team tag: not priced)")
            else:
                side_team = "home" if hit.team_id == fx.home_id else "away"
                abbr = (home if side_team == "home" else away).get("abbreviation") or ""
                if tag and abbr and norm(tag.group(1)) != norm(abbr):
                    row.update(type=None, reason="team tag on the board and the squad disagree")
                else:
                    row.update(player_id=int(hit.player_id), who=side_team)
        if fx is None and row["type"]:
            row["flag"] = why                                    # readable, but cannot be modelled or graded
        rows.append(row)
    # ---- complete groups are de-margined together; everything else is one-way
    G: dict = {}
    for i, row in enumerate(rows):
        if row["type"] is None:
            continue
        t = row["type"]
        lk = None if row["line"] is None else round(row["line"] if t not in ("AH", "CORNERS_AH") or row["side"] == "1" else -row["line"], 2)
        G.setdefault((row["market_id"], t, row["scope"], row["who"] if t not in PLAYER_T else None, row["subject"], lk), []).append(i)
    for (mid, t, *_), idx in G.items():
        sides = [rows[i]["side"] for i in idx]
        want = ({"1", "X", "2"} if t in ("1X2", "EH") or t in STAT3_T else {"1X", "12", "X2"} if t == "DC" else {"1", "2"} if t in ("DNB", "AH", "CORNERS_AH") else
                {"yes", "no"} if t == "BTTS" else {"over", "under"} if t in OU_T else None)
        if t == "EH" and set(sides) == {"1", "2"} and len(idx) == 2:               # a plain "Hendikep" with two sides is Asian
            for i in idx:
                rows[i].update(type="AH", line=rows[i]["line"] if rows[i]["side"] == "1" else -rows[i]["line"])
            t, want = "AH", {"1", "2"}
        if want and set(sides) == want and len(sides) == len(want):
            ps, over = demargin([rows[i]["odds"] for i in idx], 2.0 if t == "DC" else 1.0)
            for i, p in zip(idx, ps):
                rows[i].update(p_book=float(p), p_method="no_vig", overround=over)
                if not 1.0 <= over <= 1.25:
                    rows[i]["flag"] = rows[i]["flag"] or f"overround {over:.3f} outside 1.00-1.25: check the reading"
        else:
            for i in idx:
                if want and t not in ("SCORER_ANY",):
                    rows[i]["flag"] = rows[i]["flag"] or "incomplete group: other side missing"
                rows[i].update(p_book=one_way(rows[i]["odds"], *cal), p_method="one_way", overround=None)
    # ---- the model's number, and the blend (weight 0 until the weights table says otherwise)
    for row in rows:
        t = row["type"]
        if t is None:
            row.update(p=None, w_used=None)
            continue
        pmod = None
        if gm is not None:
            if t in PLAYER_T and pm is not None and row["player_id"] is not None:
                lam = gm.lh if row["who"] == "home" else gm.la
                if t == "SCORER_ANY":
                    pmod = pm.p_anytime(row["player_id"], lam, int(row["line"] + 0.5))
                elif t == "SCORER_FIRST":
                    pmod = pm.p_first(row["player_id"], lam, gm.lh + gm.la)
                elif t == "PL_CARD":
                    pmod = pm.p_card(row["player_id"], row["line"], row["side"]) if row["scope"] == "FT" else None
                elif row["scope"] == "FT":
                    pmod = pm.p_count(row["player_id"], "shots_on_target" if t == "PL_SOT" else "shots", row["line"], row["side"])
            elif t == "CS":
                pmod = gm.p("CS", row["scope"], score=tuple(int(x) for x in row["side"].split(":")))
            elif t not in PLAYER_T and t != "BOOKING_PTS":
                pmod = gm.p(t, row["scope"], row["side"], row["line"], row["who"])
        if pmod is not None and not (isinstance(pmod, float) and math.isnan(pmod)):
            row["p_model"] = float(min(max(pmod, 0.0), 1.0))
        wt = float(w.get(family(t, row["scope"]), 0.0)) if row["p_model"] is not None else 0.0
        p = row["p_book"] + wt * ((row["p_model"] or 0) - row["p_book"]) if wt else row["p_book"]
        if row["p_method"] == "one_way":
            p = min(p, 1 / row["odds"] / ONEWAY_CAP)
        if row["p_model"] is not None and abs(row["p_model"] - row["p_book"]) > 0.25 and row["p_method"] == "no_vig":
            row["flag"] = row["flag"] or "model more than 25 points from the book: check the reading"
        row.update(p=float(min(max(p, 0.001), 0.999)), w_used=wt)
    for row in rows:
        row["row_key"] = hashlib.md5("|".join([norm(game), row["market"], row["child"] or "", row["selection"], str(row["sv"])]).encode()).hexdigest()[:16]
        row["captured"] = captured
    return rows


COLS = ["matchday", "row_key", "fixture_id", "league", "game", "kickoff", "market", "selection", "type", "scope", "side", "line", "who",
        "subject", "player_id", "odds", "odds_first", "p_book", "p_method", "overround", "p_model", "w_used", "p", "flag", "reason",
        "captured", "model_version"]


def price(day: str, boards: str, con, out: Path | None = None, quiet: bool = False) -> pd.DataFrame:
    T = load_tables(out)
    M = matches_frame(T)
    aliases = {}
    ap = Path(out or OUT) / "soccer_aliases.csv"
    if ap.exists():
        aliases = dict(pd.read_csv(ap)[["board_name", "team_id"]].values)
    match, models, w = Matcher(T["fixtures"], aliases), {}, weights_now(con)
    cal = current(con, "calib", (None, 0, 0.0, 1.0))[2:4]
    latest = {}
    for f in sorted(glob.glob(os.path.join(boards, "*.net.json"))):          # the newest capture of a game is its closing price
        for ev in read_capture(f):
            latest[norm(ev.get("name") or f)] = (f, ev)
    allrows = []
    for f, ev in latest.values():
        cap = pd.Timestamp(os.path.getmtime(f), unit="s").isoformat(timespec="seconds")
        allrows += price_event(ev, day, T, M, match, models, w, cal, cap)
    if not allrows:
        sys.exit(f"no readable captures in {boards}")
    D = pd.DataFrame(allrows).drop_duplicates("row_key")
    D["matchday"], D["model_version"] = day, MODEL_VERSION
    D["subject"] = D.subject.where(D.subject.notna(), D.child)
    old = dict(con.execute("SELECT row_key, odds_first FROM predictions WHERE matchday = ?", [day]).fetchall())
    D["odds_first"] = [old.get(k) or o for k, o in zip(D.row_key, D.odds)]
    con.executemany("DELETE FROM predictions WHERE matchday = ? AND status = 'open' AND row_key = ?", [(day, k) for k in D.row_key])
    graded = {r[0] for r in con.execute("SELECT row_key FROM predictions WHERE matchday = ? AND status != 'open'", [day])}
    N = D[~D.row_key.isin(graded)]
    con.executemany(f"INSERT INTO predictions ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                    N[COLS].astype(object).where(N[COLS].notna(), None).values.tolist())
    con.commit()
    if not quiet:
        typed = D.type.notna()
        print(f"{day}: {len(latest)} games, {len(D)} rows; typed {typed.mean():.0%} ({int(typed.sum())}), two-way de-margined {int((D.p_method == 'no_vig').sum())}, "
              f"one-way {int((D.p_method == 'one_way').sum())}, with a model number {int(D.p_model.notna().sum())}, flagged {int(D.flag.notna().sum())}")
        fi = T.get("fd_info") or {}
        if fi.get("note"):
            print(f"  model history: API games only ({fi['note']})")
        elif fi:
            print(f"  model history: {fi['games']} football-data games ({fi['current']} this season); teams mapped {fi['mapped']} of {fi['teams']}"
                  + (f"; not mapped yet (a team maps once the API window has listed it; a wrong pair goes in data/soccer_team_map.csv): "
                     f"{[u.split(' (')[0] for u in fi['unmapped']][:12]}" if fi["unmapped"] else ""))
        nf = D[D.fixture_id.isna()].game.unique()
        if len(nf):
            print(f"  NOT MATCHED to a fixture (kept, not graded): {list(nf)}")
        print_unread(D)
    return D


def print_unread(D: pd.DataFrame) -> None:
    U = D[D.type.isna()]
    if U.empty:
        print("  unread: none")
        return
    g = U.groupby(["reason", "market"]).agg(rows=("selection", "size"), example=("selection", "first")).reset_index().sort_values(["reason", "rows"], ascending=[True, False])
    print(f"  UNREAD: {len(U)} rows in {U.market.nunique()} markets (kept with the reason; never priced, never graded):")
    for r in g.itertuples():
        print(f"    [{r.reason}] {r.market!r} x{r.rows}  e.g. {r.example!r}")


# ------------------------------------------------------------------ grade
class Facts:
    def __init__(self, T: dict):
        self.T = T
        self.F = T["fixtures"].set_index("fixture_id") if len(T["fixtures"]) else pd.DataFrame()
        self.cov = {(r.league_id, r.season): r for r in T["coverage"].itertuples()} if len(T["coverage"]) else {}
        self._c = {}

    def game(self, fid):
        if fid not in self._c:
            sub = lambda n: self.T[n][self.T[n].fixture_id == fid] if len(self.T[n]) else pd.DataFrame()
            self._c[fid] = dict(fx=self.F.loc[fid], ts=sub("team_match_stats"), ev=sub("events"), ps=sub("player_match_stats"), lu=sub("lineups"))
        return self._c[fid]


def settle(r, facts: Facts) -> dict:
    """One row -> result. Returns dict(result, won, lost, actual, note). result None = not graded, with the reason in
    note. won/lost are fractions of the stake (quarter lines: 0.5/0; push: 0/0)."""
    no = lambda why: dict(result=None, won=None, lost=None, actual=None, note=why)
    if not isinstance(r.type, str):
        return no(f"not read: {r.reason}")
    for k in ("who", "side", "line", "player_id", "fixture_id"):
        if isinstance(getattr(r, k, None), float) and math.isnan(getattr(r, k)):
            return no(f"row field {k} is NaN: pass None for 'not set'")
    if r.fixture_id is None or int(r.fixture_id) not in facts.F.index:
        return no("no fixture matched")
    g = facts.game(int(r.fixture_id))
    fx = g["fx"]
    if fx.status in ("ABD", "CANC", "AWD", "WO"):
        return dict(result="void", won=0.0, lost=0.0, actual=None, note=f"match status {fx.status}")
    if fx.status not in ("FT", "AET", "PEN"):
        return no(f"match status {fx.status}: no result yet")
    if str(fx.check_goals) == "False":
        return no("ingest check failed for this game (events do not add up to the score)")
    cov = facts.cov.get((fx.league_id, fx.season))
    fh, fa = int(fx.ft_home), int(fx.ft_away)
    if r.scope in ("H1", "H2"):
        if pd.isna(fx.ht_home) or str(fx.check_ht) == "False":
            return no("half-time score missing or failed its check")
        h, a = (int(fx.ht_home), int(fx.ht_away)) if r.scope == "H1" else (fh - int(fx.ht_home), fa - int(fx.ht_away))
    else:
        h, a = fh, fa

    def done(won, lost, actual, note=None):
        res = "won" if won == 1 else "lost" if lost == 1 else "push" if won == lost == 0 else "half_won" if won else "half_lost"
        return dict(result=res, won=float(won), lost=float(lost), actual=None if actual is None else float(actual), note=note)

    def ou(x, note=None):
        q = round(r.line * 4) % 2 == 1
        return done(*line_fracs(x, r.line, r.side), x, "quarter line: split-stake rule assumed" if q and note is None else note)

    t = r.type
    if t == "1X2":
        return done(*(1, 0) if {"1": h > a, "X": h == a, "2": h < a}[r.side] else (0, 1), h - a)
    if t == "DC":
        return done(*(1, 0) if {"1X": h >= a, "12": h != a, "X2": h <= a}[r.side] else (0, 1), h - a)
    if t == "DNB":
        return done(0, 0, 0) if h == a else done(*(1, 0) if (h > a) == (r.side == "1") else (0, 1), h - a)
    if t == "BTTS":
        return done(*(1, 0) if (h > 0 and a > 0) == (r.side == "yes") else (0, 1), min(h, a))
    if t == "CS":
        return done(*(1, 0) if r.side == f"{h}:{a}" else (0, 1), h * 10 + a, "actual = home*10 + away")
    if t == "TOTAL":
        return ou(h + a)
    if t == "TEAM_TOTAL":
        return ou(h if r.who == "home" else a)
    if t == "AH":
        mg = (h - a) if r.side == "1" else (a - h)
        q = round(r.line * 4) % 2 == 1
        return done(*line_fracs(mg, -r.line, "over"), mg, "quarter line: split-stake rule assumed" if q else None)
    if t == "EH":
        adj = h - a + r.line
        return done(*(1, 0) if {"1": adj > 0, "X": adj == 0, "2": adj < 0}[r.side] else (0, 1), h - a)
    if t in STAT3_T:
        col = {"CORNERS_1X2": "corners", "SHOTS_1X2": "shots", "SOT_1X2": "shots_on_target", "CARDS_1X2": None}[t]
        if r.scope != "FT":
            return no("no statistics by half in the data")
        if col:
            if cov is None or not cov.statistics_fixtures or len(g["ts"]) != 2 or g["ts"][col].isna().any():
                return no(f"{col} not covered for this league/game")
            v = g["ts"].set_index("is_home")[col]
            dh = [float(v[True] - v[False])] * 2
        else:
            rh, ra = _card_range(g, fx.home_id), _card_range(g, fx.away_id)
            if rh is None or ra is None:
                return no("card events not covered for this league/game")
            dh = [rh[0] - ra[1], rh[1] - ra[0]]                              # lowest and highest possible difference
        res = {"1" if d > 0 else "X" if d == 0 else "2" for d in dh}
        if len(res) > 1:
            return no(f"result depends on an unconfirmed card rule (difference {dh[0]:g} to {dh[1]:g})")
        return done(*(1, 0) if res == {r.side} else (0, 1), dh[0])
    if t == "CORNERS_AH":
        if r.scope != "FT":
            return no("no corners by half in the data")
        if cov is None or not cov.statistics_fixtures or len(g["ts"]) != 2 or g["ts"]["corners"].isna().any():
            return no("corners not covered for this league/game")
        v = g["ts"].set_index("is_home")["corners"]
        mg = (v[True] - v[False]) if r.side == "1" else (v[False] - v[True])
        return done(*line_fracs(mg, -r.line, "over"), mg, "quarter line: split-stake rule assumed" if round(r.line * 4) % 2 == 1 else None)
    if t in ("CORNERS", "SHOTS", "SOT"):
        col = {"CORNERS": "corners", "SHOTS": "shots", "SOT": "shots_on_target"}[t]
        if r.scope != "FT":
            return no(f"no {col} by half in the data")
        if cov is None or not cov.statistics_fixtures or len(g["ts"]) != 2 or g["ts"][col].isna().any():
            return no(f"{col} not covered for this league/game")
        v = g["ts"].set_index("is_home")[col]
        return ou(v.sum() if r.who is None else v[r.who == "home"])
    if t in ("CARDS", "BOOKING_PTS"):
        if cov is None or not cov.events or g["ev"].empty and str(fx.has_events) != "True":
            return no("card events not covered for this league/game")
        E = g["ev"][g["ev"].type.isin(["yellow", "second_yellow", "red"])]
        if r.scope == "H1":
            E = E[E.minute <= 45]
        elif r.scope == "H2":
            E = E[E.minute > 45]
        if r.who:
            E = E[E.team_id == (fx.home_id if r.who == "home" else fx.away_id)]
        squad = set(g["lu"].player_id) if len(g["lu"]) else None
        lo = hi = 0.0
        pts = RULES["booking_points"]
        if t == "BOOKING_PTS" and pts is None:
            return no("booking-points values not confirmed from the book's rules")
        for pid, ce in E.groupby("player_id", dropna=False):
            y, off = int(ce.type.isin(["yellow", "second_yellow"]).sum()), bool(ce.type.isin(["red", "second_yellow"]).any())
            if t == "CARDS":
                # yellow then sent off = 2 cards for certain. Was the red a second yellow, and does the book then count
                # 3? Seen as two yellow events: known second yellow. One yellow + red: could be either.
                base, extra = min(y, 1) + off, 1.0 if (off and y >= 1) else 0.0
                if RULES["second_yellow_is_two_cards"] is True:
                    extra = 0.0
                elif RULES["second_yellow_is_two_cards"] is False and y >= 2:
                    base, extra = base + 1, 0.0
            else:
                base, extra = min(y, 1) * pts["yellow"] + off * pts["red"], 0.0
            staff = squad is not None and (pd.isna(pid) or pid not in squad)
            if staff and RULES["coach_cards_count"] is not True:
                if RULES["coach_cards_count"] is None:
                    hi += base + extra                           # counted or not: both possibilities kept
                continue
            lo, hi = lo + base, hi + base + extra
        a_, b_ = line_fracs(lo, r.line, r.side), line_fracs(hi, r.line, r.side)
        if a_ != b_:
            return no(f"result depends on an unconfirmed card rule (count is {lo:g} or {hi:g})")
        return ou(lo, None if lo == hi else f"card count {lo:g}-{hi:g} under unconfirmed rules: same result either way")
    if t in PLAYER_T:
        if r.player_id is None or pd.isna(r.player_id):
            return no("player not matched")
        pid = int(r.player_id)
        ps = g["ps"][g["ps"].player_id == pid] if len(g["ps"]) else g["ps"]
        covered = cov is not None and cov.statistics_players and len(g["ps"]) > 0
        goals_ev = g["ev"][g["ev"].type.isin(["goal", "penalty"])] if len(g["ev"]) else g["ev"]
        if covered:
            minutes = int(ps.minutes.iloc[0]) if len(ps) else 0
        else:
            lu = g["lu"][g["lu"].player_id == pid] if len(g["lu"]) else g["lu"]
            scored = len(goals_ev) and (goals_ev.player_id == pid).any()
            if len(lu) and bool(lu.starting.iloc[0]) or scored:
                minutes = 1
            elif len(g["lu"]) and lu.empty:
                minutes = 0                                      # not in the matchday squad
            else:
                return no("no player statistics for this league: cannot tell whether he played")
        if minutes == 0:
            return dict(result="void", won=0.0, lost=0.0, actual=0.0, note="did not play (0 minutes)")
        if t == "PL_CARD":
            if r.scope != "FT" or abs(r.line - 0.5) > 1e-9:
                return no("player card line other than 0.5 not read")
            if cov is None or not cov.events or str(fx.has_events) != "True":
                return no("card events not covered for this league/game")
            booked = len(g["ev"]) > 0 and bool(((g["ev"].player_id == pid) & g["ev"].type.isin(["yellow", "second_yellow", "red"])).any())
            return done(*(1, 0) if booked == (r.side == "over") else (0, 1), float(booked))
        if t in ("PL_SOT", "PL_SHOTS"):
            if not covered or r.scope != "FT":
                return no("player shots not covered")
            return ou(float(ps["shots_on_target" if t == "PL_SOT" else "shots"].iloc[0]))
        if cov is None or not cov.events or str(fx.check_goals) != "True":
            return no("goal events not covered or not checked")
        if r.scope != "FT":
            return no("scorer by half not read")
        n = int((goals_ev.player_id == pid).sum())               # own goals never count for a player
        if t == "SCORER_ANY":
            return done(*(1, 0) if n > r.line else (0, 1), n)
        allg = g["ev"][g["ev"].type.isin(["goal", "penalty", "own_goal"])].sort_values(["minute", "extra", "seq"], na_position="first")
        if allg.empty:
            return done(0, 1, 0, "no goal in the match")
        first = allg.iloc[0]
        if first.type == "own_goal":
            if RULES["first_scorer_own_goal"] != "next":
                return no("first goal was an own goal: book rule not confirmed")
            rest = allg[allg.type != "own_goal"]
            if rest.empty:
                return done(0, 1, 0, "only own goals")
            first = rest.iloc[0]
        if first.player_id == pid:
            return done(1, 0, 1)
        started = bool(ps.started.iloc[0]) if covered and len(ps) else None
        if started is not True and RULES["first_scorer_late_sub"] is None:
            return no("player did not start and did not score first: book rule on late substitutes not confirmed")
        if started is not True and RULES["first_scorer_late_sub"] == "void":
            return dict(result="void", won=0.0, lost=0.0, actual=0.0, note="came on after kick-off: void by rule")
        return done(0, 1, 0)
    return no(f"no settlement for type {t}")


def _card_range(g: dict, team_id) -> tuple | None:
    """A team's card count in one game: (lowest, highest) under the rules still unconfirmed (second yellow counted as
    one card or two; cards to the bench). None when the game has no card events."""
    if str(g["fx"].has_events) != "True":
        return None
    E = g["ev"][g["ev"].type.isin(["yellow", "second_yellow", "red"]) & (g["ev"].team_id == team_id)] if len(g["ev"]) else g["ev"]
    squad = set(g["lu"].player_id) if len(g["lu"]) else None
    lo = hi = 0.0
    for pid, ce in E.groupby("player_id", dropna=False):
        y, off = int(ce.type.isin(["yellow", "second_yellow"]).sum()), bool(ce.type.isin(["red", "second_yellow"]).any())
        base, extra = min(y, 1) + off, 1.0 if (off and y >= 1) else 0.0
        if RULES["second_yellow_is_two_cards"] is True:
            extra = 0.0
        elif RULES["second_yellow_is_two_cards"] is False and y >= 2:
            base, extra = base + 1, 0.0
        staff = squad is not None and (pd.isna(pid) or pid not in squad)
        if staff and RULES["coach_cards_count"] is not True:
            if RULES["coach_cards_count"] is None:
                hi += base + extra
            continue
        lo, hi = lo + base, hi + base + extra
    return lo, hi


def grade(day: str, con, boards: str | None = None, out: Path | None = None, quiet: bool = False) -> None:
    stale = con.execute("SELECT count(*) FROM predictions WHERE matchday = ? AND status = 'open' AND model_version != ?", [day, MODEL_VERSION]).fetchone()[0]
    if stale:
        if boards and glob.glob(os.path.join(boards, "*.net.json")):
            print(f"{stale} open rows were priced under an older version: re-priced under {MODEL_VERSION} before grading")
            price(day, boards, con, out, quiet=True)
        else:
            print(f"WARNING: {stale} open rows are from an older model version and no boards were given: graded, but kept out of the weights")
    facts = Facts(load_tables(out))
    P = pd.read_sql("SELECT * FROM predictions WHERE matchday = ? AND status = 'open'", con, params=[day])
    if P.empty:
        print(f"{day}: nothing open to grade")
        return
    up = []
    P = P.astype(object).where(P.notna(), None)              # SQLite NULLs come back as NaN: a NaN `who` is not "the match"
    for r in P.itertuples():
        s = settle(r, facts)
        if s["result"] is None:
            up.append((None, None, None, None, None, s["note"], "open", day, r.row_key))
        else:
            ret = s["won"] * (r.odds - 1) - s["lost"]
            up.append((s["result"], s["won"], s["lost"], ret, s["actual"], s["note"], "graded", day, r.row_key))
    con.executemany("UPDATE predictions SET result=?, won=?, lost=?, ret=?, actual=?, grade_note=?, status=? WHERE matchday=? AND row_key=?", up)
    con.commit()
    grade_slips(day, con, quiet)
    if not quiet:
        G = pd.read_sql("SELECT * FROM predictions WHERE matchday = ?", con, params=[day])
        print(f"{day}: graded {int((G.status == 'graded').sum())} of {len(G)} rows in {G[G.status == 'graded'].game.nunique()} games")
        ng = G[G.status == "open"].groupby("grade_note").size().sort_values(ascending=False)
        for note, n in ng.items():
            print(f"    not graded x{n}: {note}")
        if not handchecked(con):
            print("  NOT VERIFIED: no number from this ledger is to be reported until `handcheck` has been done on five results.")


# ------------------------------------------------------------------ what the record says (rule 2)
def graded_frame(con, current_only: bool = True) -> pd.DataFrame:
    q = "SELECT * FROM predictions WHERE status = 'graded' AND result != 'void' AND won + lost > 0"
    d = pd.read_sql(q + (" AND model_version = ?" if current_only else ""), con, params=[MODEL_VERSION] if current_only else [])
    if len(d):
        d["y"], d["wt"] = d.won / (d.won + d.lost), d.won + d.lost
        d["cluster"] = d.matchday + "|" + d.game
        d["family"] = [family(t, s) for t, s in zip(d.type, d.scope)]
    return d


def fit_weights(con, k: int = 5, quiet: bool = False) -> pd.DataFrame:
    """Per family: w fitted on other matchdays, scored on held-out matchdays; t is across GAMES (a goals ladder is
    fifteen rows about one number). w = w_fit * games/(games+40) * max(0, 1 - 4/t^2): nothing below t = 2."""
    d = graded_frame(con)
    rows = []
    if len(d):
        d = d[d.p_model.notna() & (d.p_method == "no_vig")]
    for fam, x in (d.groupby("family") if len(d) else []):
        days = sorted(x.matchday.unique())
        games = x.cluster.nunique()
        diff, err, wt = (x.p_model - x.p_book).values, (x.y - x.p_book).values, x.wt.values
        wfit = lambda m: float(np.clip((wt[m] * diff[m] * err[m]).sum() / max((wt[m] * diff[m] ** 2).sum(), 1e-12), 0, 1))
        w_all, t = wfit(np.ones(len(x), bool)), 0.0
        if len(days) >= k and games >= 10:
            fold = x.matchday.map({dy: i % k for i, dy in enumerate(days)}).values
            gain = np.zeros(len(x))
            for f in range(k):
                te = fold == f
                wk = wfit(~te)
                gain[te] = wt[te] * (err[te] ** 2 - (err[te] - wk * diff[te]) ** 2)      # Brier gain of the blend, held out
            gg = pd.Series(gain).groupby(x.cluster.values).sum()
            t = float(gg.mean() / (gg.std(ddof=1) / math.sqrt(len(gg)))) if len(gg) > 1 and gg.std(ddof=1) > 0 else 0.0
        w = w_all * games / (games + 40) * max(0.0, 1 - 4 / t ** 2) if t > 2 else 0.0
        rows.append(dict(family=fam, games=games, w_fit=w_all, t=t, w=w))
    W = pd.DataFrame(rows, columns=["family", "games", "w_fit", "t", "w"])
    now = pd.Timestamp.now("UTC").isoformat()
    con.executemany("INSERT INTO weights VALUES (?,?,?,?,?,?,?)", [(now, r.family, int(r.games), r.w_fit, r.t, r.w, MODEL_VERSION) for r in W.itertuples()])
    con.commit()
    if not quiet and len(W):
        print(W.round(3).to_string(index=False))
    return W


def fit_calib(con, lam: float = 40.0, min_games: int = 150):
    """One-way prices: p = sigmoid(a + b * logit(1/odds)), each game weighing 1, ridge toward (0, 1) worth ~40 games.
    Stored only from `min_games` graded games on; the 1/odds/1.03 cap applies whatever is fitted."""
    d = graded_frame(con)
    d = d[d.p_method == "one_way"] if len(d) else d
    if len(d) == 0 or d.cluster.nunique() < min_games:
        return None
    kw = (d.wt / d.groupby("cluster").cluster.transform("size")).values
    z, y, a, b = logit(1 / d.odds.values), d.y.values, 0.0, 1.0
    for _ in range(60):
        p = sigmoid(a + b * z)
        v = kw * p * (1 - p) + 1e-12
        g0, g1 = (kw * (y - p)).sum() - lam * a, (kw * (y - p) * z).sum() - lam * (b - 1)
        h00, h01, h11 = v.sum() + lam, (v * z).sum(), (v * z * z).sum() + lam
        det = h00 * h11 - h01 ** 2
        a += float(np.clip((h11 * g0 - h01 * g1) / det, -0.25, 0.25))
        b += float(np.clip((-h01 * g0 + h00 * g1) / det, -0.25, 0.25))
    con.execute("INSERT INTO calib VALUES (?,?,?,?)", [pd.Timestamp.now("UTC").isoformat(), int(d.cluster.nunique()), a, b])
    con.commit()
    return a, b


def record(con) -> dict:
    """family -> (stated wins, actual wins, games): vote 3 of the filter."""
    d = graded_frame(con, current_only=False)
    if d.empty:
        return {}
    d = d[d.p_book.between(*LEG_P)]
    return {f: (float((x.p_book * x.wt).sum()), float((x.y * x.wt).sum()), x.cluster.nunique()) for f, x in d.groupby("family")}


# ------------------------------------------------------------------ slips (section 5) and bandit 2 (section 6)
def leg_pool(day: str, con, out: Path | None = None) -> pd.DataFrame:
    P = pd.read_sql("SELECT * FROM predictions WHERE matchday = ? AND status = 'open' AND type IS NOT NULL", con, params=[day])
    if P.empty:
        return P
    ok = (P.p_method == "no_vig") & P.p_book.between(*LEG_P) & (P.odds >= LEG_MIN_ODDS) & P.flag.isna() & P.fixture_id.notna()   # rules 1, 2
    ok &= ~P.type.isin(["DNB", "BOOKING_PTS"])                   # DNB's price is "given no draw"; booking points cannot be graded yet
    side1 = P[(P.type == "1X2") & (P.scope == "FT")].sort_values("p_book").groupby("game").tail(1).set_index("game")
    fav = {g: (r.side if r.p_book >= SIDE_MIN else None) for g, r in side1.iterrows()}
    is_side = P.type.isin(["1X2", "AH", "EH"])
    ok &= ~is_side | np.array([fav.get(g) is not None and s == fav.get(g) for g, s in zip(P.game, P.side)], dtype=bool)                               # rule 3
    pre = load_tables(out)["prematch"]
    xi = set(zip(pre[pre.starting == True].fixture_id, pre[pre.starting == True].player_id)) if len(pre) else set()            # noqa: E712
    is_pl = P.type.isin(PLAYER_T)
    ok &= ~is_pl | np.array([(f, p) in xi for f, p in zip(P.fixture_id, P.player_id)], dtype=bool)                                                    # rule 6
    P = P[ok].copy()
    rec = record(con)
    P["family"] = [family(t, s) for t, s in zip(P.type, P.scope)]
    P["quantity"] = [quantity(t, s) for t, s in zip(P.type, P.subject)]
    P["v_stat"] = P.p_model >= STAT_VOTE
    P["v_rec"] = [f in rec and rec[f][2] >= 30 and rec[f][1] >= rec[f][0] for f in P.family]
    op = Path(out or OUT) / "soccer_outside.csv"                 # optional: fixture_id,type,side from a public model
    agree = set(map(tuple, pd.read_csv(op)[["fixture_id", "type", "side"]].astype(str).values)) if op.exists() else None
    P["v_out"] = [None if agree is None else (str(int(f)), t, str(s)) in agree for f, t, s in zip(P.fixture_id, P.type, P.side)]
    P["v_stat"] = P.v_stat.fillna(False).astype(bool)
    P["votes"] = 1 + P.v_stat.astype(int) + P.v_rec.astype(int) + np.array([bool(v) for v in P.v_out], dtype=int)
    P["value"] = P.p * P.odds - 1
    # one leg per quantity and game: the line that costs the least margin
    return P.sort_values("value", ascending=False).groupby(["game", "quantity"]).head(1).reset_index(drop=True)


def build_slip(d: pd.DataFrame, target: float, score: str, tol: float = 0.25, width: int = 300, max_legs: int = 14):
    """One leg per game. Among slips paying target .. target*(1+tol), the highest sum of `score` per leg (log p +
    log odds = least margin paid, for the book strategy)."""
    if d.empty:
        return None
    d = d.reset_index(drop=True)
    lo, hi = math.log(target) - 1e-9, math.log(target * (1 + tol))
    lo_o, sc, gm = np.log(d.odds.values), d[score].values, d.game.values
    beam, best = [((), 0.0, 0.0)], None
    for _ in range(max_legs):
        nxt = []
        for legs, o, s in beam:
            gs = {gm[i] for i in legs}
            for j in range(legs[-1] + 1 if legs else 0, len(d)):
                if gm[j] in gs or o + lo_o[j] > hi:
                    continue
                c = (legs + (j,), o + lo_o[j], s + sc[j])
                if c[1] >= lo:
                    best = c if best is None or c[2] / len(c[0]) > best[2] / len(best[0]) else best
                else:
                    nxt.append(c)
        if not nxt:
            break
        beam = sorted(nxt, key=lambda c: c[2] / len(c[0]), reverse=True)[:width]
    return None if best is None else d.loc[list(best[0])]


STRATEGIES = {
    "book": lambda P: P[~P.type.isin(PLAYER_T)].assign(score=lambda x: np.log(x.p_book * x.odds)),
    "stats": lambda P: P[~P.type.isin(PLAYER_T) & P.v_stat].assign(score=lambda x: np.log(x.p_model.clip(1e-6) * x.odds)),
    "votes": lambda P: P[~P.type.isin(PLAYER_T) & (P.votes >= 3)].assign(score=lambda x: x.votes + np.log(x.p_book * x.odds)),
}


def statedge_pool(day: str, con, band=STATEDGE_ODDS) -> pd.DataFrame:
    """Statistics legs only (corners, corner handicap and 1X2, cards and cards 1X2, shots, shots on target), priced
    inside `band`, where the team-statistics model is above the book's price without margin. Ranked by the model's
    view of the leg's value, model x odds - 1. The stated chance stays the BOOK's (rule 1): the model's weight is 0
    until the record gives it one, so these are paper legs that test the model, not legs the system believes in."""
    P = pd.read_sql("SELECT * FROM predictions WHERE matchday = ? AND status = 'open' AND type IS NOT NULL", con, params=[day])
    if P.empty:
        return P
    ok = (P.p_method == "no_vig") & P.type.isin(STAT_T) & P.odds.between(*band) & P.flag.isna() & P.fixture_id.notna() & P.p_model.notna()
    P = P[ok & (P.p_model > P.p_book)].copy()
    P["family"] = [family(t, s) for t, s in zip(P.type, P.scope)]
    P["quantity"] = [quantity(t, s) for t, s in zip(P.type, P.subject)]
    P["edge"] = P.p_model * P.odds - 1
    P["votes"] = 1 + (P.p_model >= STAT_VOTE).astype(int)
    P["score"] = np.log1p(P.edge.clip(lower=-0.99))
    return P.sort_values("edge", ascending=False).groupby(["game", "quantity"]).head(1).reset_index(drop=True)


def arm_stats(con) -> pd.DataFrame:
    S = pd.read_sql("SELECT strategy, target, paid FROM slips WHERE status = 'graded' AND result != 'void'", con)
    rows = []
    for st in list(STRATEGIES) + ["statedge"]:
        for tg in sorted(set(S.target) | {3.0, 6.0, 9.0}):
            x = S[(S.strategy == st) & (S.target == tg)].paid - 1
            n, sd0 = len(x), math.sqrt(max(tg - 1, 0.5))
            sd = float(x.std(ddof=1)) if n >= 10 and x.std(ddof=1) > 0 else sd0
            mean = (2 * BOOK_MARGIN + x.sum()) / (2 + n)          # prior: the book's margin, worth two matchdays
            se = sd / math.sqrt(2 + n)
            rows.append(dict(strategy=st, target=tg, n=n, ret=float(x.mean()) if n else float("nan"), post_mean=mean, post_sd=se,
                             stopped=bool(mean + 1.645 * se < BOOK_MARGIN)))
    return pd.DataFrame(rows)


def fixed_legs(d: pd.DataFrame, n: int) -> pd.DataFrame | None:
    """Exactly n legs, one per game: each game's best leg by score, then the n best games."""
    if d.empty or d.game.nunique() < n:
        return None
    best = d.sort_values("score", ascending=False).groupby("game").head(1)
    return best.head(n)


def slips(day: str, con, targets=(3, 6, 9), out: Path | None = None, seed: int | None = None, quiet: bool = False,
          legs: tuple = ()) -> pd.DataFrame:
    pool = leg_pool(day, con, out)
    con.execute("DELETE FROM slip_legs WHERE matchday = ? AND slip_id IN (SELECT slip_id FROM slips WHERE matchday = ? AND status = 'open')", [day, day])
    con.execute("DELETE FROM slips WHERE matchday = ? AND status = 'open'", [day])
    A = arm_stats(con)
    rng = np.random.default_rng(seed)
    A["draw"] = rng.normal(A.post_mean, A.post_sd)
    A.loc[A.stopped, "draw"] = -np.inf
    built = []
    if not quiet:
        n_games = pool.game.nunique() if len(pool) else 0
        print(f"{day}: {len(pool)} candidate legs in {n_games} games (book 60-88%, price >= 1.15, no flags, one per quantity). ALL SLIPS ARE PAPER.")
        if len(pool):
            show = pool.sort_values(["votes", "value"], ascending=False).head(12)
            print("  legs by votes (book / statistics / record / outside):")
            for r in show.itertuples():
                v = f"B{'S' if r.v_stat else '-'}{'R' if r.v_rec else '-'}{'O' if r.v_out else '-' if r.v_out is not None else '?'}"
                print(f"    {r.votes}/4 {v}  {r.game[:34]:<34} {r.market[:28]:<28} {r.selection[:22]:<22} @{r.odds:.2f}  book {r.p_book:.0%}  model {'' if pd.isna(r.p_model) else format(r.p_model, '.0%')}")
    SE = statedge_pool(day, con)
    if not quiet:
        print(f"  statedge pool: {len(SE)} statistics legs at {STATEDGE_ODDS[0]}-{STATEDGE_ODDS[1]} where the team numbers rate them above the book"
              + (f" (best: {SE.game.iloc[0][:30]} {SE.market.iloc[0][:30]} {SE.selection.iloc[0][:18]} @{SE.odds.iloc[0]:.2f}, "
                 f"book {SE.p_book.iloc[0]:.0%}, model {SE.p_model.iloc[0]:.0%})" if len(SE) else ""))
    for st, pick in list(STRATEGIES.items()) + [("statedge", lambda P: SE)]:
        used = set()                                             # no quantity in two slips of one strategy
        cand = pick(pool) if (len(pool) or st == "statedge") else pool
        for kind, tg in [("x", t) for t in targets] + [("legs", n) for n in legs]:
            c = cand[[(g, q) not in used for g, q in zip(cand.game, cand.quantity)]] if len(cand) else cand
            L = (build_slip(c, tg, "score") if kind == "x" else fixed_legs(c, int(tg))) if len(c) else None
            if L is None:
                if not quiet and kind == "legs":
                    print(f"\n  {st}-{tg:g}legs: not enough games with a candidate leg ({c.game.nunique() if len(c) else 0})")
                continue
            used |= set(zip(L.game, L.quantity))
            sid, strat = (f"{st}-{tg:g}x", st) if kind == "x" else (f"{st}-{tg:g}legs", f"{st}/legs")
            o, p = float(np.prod(L.odds)), float(np.prod(L.p))
            con.execute("INSERT INTO slips (matchday, slip_id, strategy, target, odds, p, legs) VALUES (?,?,?,?,?,?,?)", [day, sid, strat, tg, o, p, len(L)])
            con.executemany("INSERT INTO slip_legs VALUES (?,?,?,?,?,?,?,?,?)",
                            [(day, sid, r.row_key, r.game, r.market, r.selection, r.odds, r.p, f"{r.votes}/4") for r in L.itertuples()])
            built.append(dict(slip_id=sid, strategy=strat, target=tg, odds=o, p=p, legs=len(L)))
            if not quiet:
                print(f"\n  {sid}: pays {o:.2f}, {len(L)} legs, stated chance {p:.1%} (1 in {1 / p:.1f}), expected return {p * o - 1:+.1%}")
                for r in L.sort_values("kickoff").itertuples():
                    extra = f"  model {r.p_model:.0%}" if st == "statedge" else f"  votes {r.votes}/4"
                    print(f"     {str(r.kickoff)[5:16]:<11} {r.game[:32]:<32} {r.market[:28]:<28} {r.selection[:22]:<22} @{r.odds:.2f}  {r.p:.0%}{extra}")
    B = pd.DataFrame(built)
    for tg in targets:                                           # bandit 2: which strategy would be played at each target
        a = A[(A.target == tg) & A.strategy.isin(B[B.target == tg].strategy if len(B) else [])]
        if len(a) and np.isfinite(a.draw.max()):
            st = a.sort_values("draw").iloc[-1].strategy
            con.execute("UPDATE slips SET played = 1 WHERE matchday = ? AND slip_id = ?", [day, f"{st}-{tg:g}x"])
            if not quiet:
                print(f"\n  bandit pick at {tg:g}x: {st} (posterior {a[a.strategy == st].post_mean.iloc[0]:+.1%} +- {a[a.strategy == st].post_sd.iloc[0]:.1%} after {int(a[a.strategy == st].n.iloc[0])} matchdays)")
    con.commit()
    if not quiet and B.empty:
        print("  no slip reachable from this pool")
    return B


def grade_slips(day: str, con, quiet: bool = False) -> None:
    """Leg multiplier = won * odds + refunded part; one fully lost leg loses the slip. A void / pushed leg counts 1.00."""
    S = pd.read_sql("SELECT * FROM slips WHERE matchday = ? AND status = 'open'", con, params=[day])
    P = pd.read_sql("SELECT row_key, status, result, won, lost, odds FROM predictions WHERE matchday = ?", con, params=[day]).set_index("row_key")
    for s in S.itertuples():
        L = pd.read_sql("SELECT row_key FROM slip_legs WHERE matchday = ? AND slip_id = ?", con, params=[day, s.slip_id])
        legs = P.reindex(L.row_key)
        if (legs.status != "graded").any():
            continue
        mult = legs.won * legs.odds + (1 - legs.won - legs.lost)
        paid = float(mult.prod())
        res = "void" if (legs.won + legs.lost == 0).all() else "lost" if paid == 0 else "won" if paid >= 1 else "part"
        con.execute("UPDATE slips SET status = 'graded', result = ?, paid = ? WHERE matchday = ? AND slip_id = ?", [res, paid, day, s.slip_id])
        if not quiet:
            print(f"  slip {s.slip_id}{' (bandit pick)' if s.played else ''}: {res}, stake 1 -> {paid:.2f} (stated {s.p:.1%})")
    con.commit()


# ------------------------------------------------------------------ hand check and report
def handchecked(con) -> bool:
    return con.execute("SELECT v FROM meta WHERE k = 'handcheck'").fetchone() is not None


def handcheck(con, confirm: bool = False) -> None:
    """Spec section 1: five results across different market types, checked by hand against the match report."""
    if confirm:
        n = con.execute("SELECT count(*), count(DISTINCT type) FROM predictions WHERE checked = 1").fetchone()
        if n[0] < 5 or n[1] < 5:
            sys.exit("run `handcheck` first: it marks the five rows to check")
        con.execute("INSERT OR REPLACE INTO meta VALUES ('handcheck', ?)", [pd.Timestamp.now("UTC").isoformat()])
        con.commit()
        print("recorded: five results checked by hand. Reports may now carry numbers.")
        return
    G = pd.read_sql("SELECT * FROM predictions WHERE status = 'graded' AND result != 'void'", con)
    if G.type.nunique() < 5:
        sys.exit(f"only {G.type.nunique()} market types graded so far: five different ones are needed")
    order = ["AH", "TOTAL", "CORNERS", "CARDS", "SCORER_ANY", "1X2", "BTTS", "TEAM_TOTAL", "SOT", "PL_SOT", "DC", "CS", "EH", "DNB", "SCORER_FIRST"]
    types = sorted(G.type.unique(), key=lambda t: order.index(t) if t in order else 99)[:5]
    pick = pd.concat([G[(G.type == t)].sort_values("result").groupby("scope").head(1).head(1) for t in types])
    con.execute("UPDATE predictions SET checked = 0")
    con.executemany("UPDATE predictions SET checked = 1 WHERE matchday = ? AND row_key = ?", pick[["matchday", "row_key"]].values.tolist())
    con.commit()
    print("Check each line against the match report, then run `handcheck --confirm`:\n")
    for r in pick.itertuples():
        print(f"  {r.matchday}  {r.game}  (fixture {int(r.fixture_id)})\n     {r.market} | {r.selection} @ {r.odds:.2f}\n     read as {r.type}/{r.scope} side={r.side} line={r.line} who={r.who}"
              f"{'' if pd.isna(r.player_id) else f' player_id={int(r.player_id)}'}\n     settled on actual = {r.actual:g} -> {r.result}{'  [' + r.grade_note + ']' if isinstance(r.grade_note, str) and r.grade_note else ''}\n")


def _cstat(x: pd.DataFrame) -> dict:
    """Stated vs happened and flat return, standard error across games."""
    g = x.groupby("cluster").agg(ret=("ret", "sum"), n=("ret", "size"))
    n = len(x)
    se = float(g.ret.std(ddof=1) * math.sqrt(len(g)) / n) if len(g) > 1 else float("nan")
    return dict(games=len(g), rows=n, stated=float((x.p_book * x.wt).sum()), won=float((x.y * x.wt).sum()), ret=float(x.ret.sum() / n), se=se)


def report(con, path: str | None = None) -> str:
    L = [f"# Soccer ledger report ({MODEL_VERSION}, {pd.Timestamp.now('UTC'):%Y-%m-%d})", ""]
    ok = handchecked(con)
    if not ok:
        L += ["**NOT VERIFIED. Five results across market types have not been hand-checked (`handcheck`). "
              "Do not quote any number below.**", ""]
    d = graded_frame(con, current_only=False)
    A = pd.read_sql("SELECT status, type, grade_note, result FROM predictions", con)
    L += [f"Rows: {len(A)} stored, {int((A.status == 'graded').sum())} graded ({int((A.result == 'void').sum())} void), "
          f"{int(A.type.isna().sum())} unread, {int(((A.status == 'open') & A.type.notna()).sum())} read but not graded.", ""]
    if d.empty:
        L.append("Nothing graded yet.")
    else:
        games = d.cluster.nunique()
        L += [f"Graded games: {games}. Games, not rows, are the unit: with fewer than ~300 nothing here is evidence.", "",
              "## By market type: stated (book, margin removed) vs happened", "",
              "| family | games | rows | stated wins | actual wins | return per row | +-se |", "|---|---|---|---|---|---|---|"]
        tw = d[d.p_method == "no_vig"]
        L.insert(-2, "Only the side the book rates 50% or more: both sides of a pair together always add up, which says nothing.")
        L.insert(-2, "")
        for fam, x in sorted(tw[tw.p_book >= 0.5].groupby("family"), key=lambda kv: -kv[1].cluster.nunique()):
            s = _cstat(x)
            L.append(f"| {fam} | {s['games']} | {s['rows']} | {s['stated']:.1f} | {s['won']:.1f} | {s['ret']:+.1%} | {s['se']:.1%} |")
        L += ["", "## Probability bands (two-way rows)", "", "| book says | games | rows | stated | happened | return | +-se |", "|---|---|---|---|---|---|---|"]
        for lo, hi in ((0, .4), (.4, .6), (.6, .7), (.7, .8), (.8, .88), (.88, 1.01)):
            x = tw[(tw.p_book >= lo) & (tw.p_book < hi)]
            if len(x):
                s = _cstat(x)
                L.append(f"| {lo:.0%}-{min(hi, 1):.0%} | {s['games']} | {s['rows']} | {s['stated'] / x.wt.sum():.1%} | {s['won'] / x.wt.sum():.1%} | {s['ret']:+.1%} | {s['se']:.1%} |")
        L += ["", "## Where the model lost (honest negatives)", ""]
        mm = tw[tw.p_model.notna()]
        neg = []
        for fam, x in mm.groupby("family"):
            g = (x.wt * ((x.y - x.p_book) ** 2 - (x.y - x.p_model) ** 2)).groupby(x.cluster).sum()
            t = float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))) if len(g) > 2 and g.std(ddof=1) > 0 else float("nan")
            neg.append((t, fam, len(g)))
        if neg:
            L += ["| family | games | model vs book, t (negative = the model was worse) |", "|---|---|---|"]
            L += [f"| {f} | {n} | {t:+.2f} |" for t, f, n in sorted(neg, key=lambda z: (math.isnan(z[0]), z[0]))]
            worse = sum(1 for t, _, _ in neg if t < 0)
            L += ["", f"The model was worse than the book in {worse} of {len(neg)} families. Its weight stays 0 where t <= 2."]
        else:
            L.append("No graded row has a model number yet.")
        ow = d[d.p_method == "one_way"]
        if len(ow):
            s = _cstat(ow)
            L += ["", f"One-way rows (scorers, correct score, ladders): {s['rows']} rows in {s['games']} games, stated {s['stated']:.1f} wins, "
                      f"happened {s['won']:.1f}, return {s['ret']:+.1%} +- {s['se']:.1%}."]
        pl = d[d.type.isin(PLAYER_T)]
        if len(pl):
            s = _cstat(pl)
            L += ["", f"Player lines (kept in a separate list, never in slips until this turns): {s['rows']} rows, return {s['ret']:+.1%} +- {s['se']:.1%}."]
    if len(d):
        st = d[d.type.isin(STAT_T) & (d.p_method == "no_vig")].copy()
        L += ["", "## Statistics markets: book, model and what happened", ""]
        if st.empty:
            L.append("None graded yet.")
        else:
            L += ["| market | games | rows | book said | model said | happened | model minus book, t |", "|---|---|---|---|---|---|---|"]
            for fam, x in sorted(st.groupby("family"), key=lambda kv: -kv[1].cluster.nunique()):
                m = x[x.p_model.notna()]
                g = (m.wt * ((m.y - m.p_book) ** 2 - (m.y - m.p_model) ** 2)).groupby(m.cluster).sum() if len(m) else pd.Series(dtype=float)
                t = float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))) if len(g) > 2 and g.std(ddof=1) > 0 else float("nan")
                L.append(f"| {fam} | {x.cluster.nunique()} | {len(x)} | {(x.p_book * x.wt).sum() / x.wt.sum():.1%} | "
                         f"{(m.p_model * m.wt).sum() / m.wt.sum() if len(m) else float('nan'):.1%} | {(x.y * x.wt).sum() / x.wt.sum():.1%} | {t:+.2f} |")
            tm = st[st.who.notna()].copy()                          # team lines: corners / cards / shots of one team
            if len(tm):
                tm["team"] = [g_.split(" v ")[0 if w == "home" else 1] if " v " in g_ else g_ for g_, w in zip(tm.game, tm.who)]
                tm["over"] = tm.side == "over"
                agg = tm[tm.over].groupby(["team", "type"]).agg(games=("cluster", "nunique"), stated=("p_book", "mean"), happened=("y", "mean"))
                agg = agg[agg.games >= 3].sort_values("games", ascending=False)
                L += ["", "Team lines, the OVER side (a team whose overs keep winning beyond what the book stated is the kind of thing this "
                          "is looking for; under ~15 games per team it is noise):", ""]
                L += [agg.head(25).round(3).to_markdown()] if len(agg) else ["No team has 3 graded games yet."]
        SEL = pd.read_sql("SELECT l.matchday, l.row_key FROM slip_legs l JOIN slips s USING (matchday, slip_id) WHERE s.strategy LIKE 'statedge%'", con)
        if len(SEL):
            x = d.merge(SEL.drop_duplicates(), on=["matchday", "row_key"])
            if len(x):
                s_ = _cstat(x)
                L += ["", f"statedge legs graded: {s_['rows']} in {s_['games']} games. Book said {(x.p_book * x.wt).sum() / x.wt.sum():.1%}, "
                          f"model said {(x.p_model * x.wt).sum() / x.wt.sum():.1%}, happened {(x.y * x.wt).sum() / x.wt.sum():.1%}; "
                          f"return per leg {s_['ret']:+.1%} +- {s_['se']:.1%}. The model earns weight only when this beats the book on held-out weeks."]
    S = pd.read_sql("SELECT * FROM slips WHERE status = 'graded' AND result != 'void'", con)
    L += ["", "## Slips: hits against expected hits", ""]
    if S.empty:
        L.append("No graded slips yet.")
    else:
        L += ["| strategy | target | slips | expected wins | wins | return per slip |", "|---|---|---|---|---|---|"]
        for (st, tg), x in S.groupby(["strategy", "target"]):
            L.append(f"| {st} | {tg:g}{' legs' if st.endswith('/legs') else 'x'} | {len(x)} | {x.p.sum():.1f} | {int((x.result == 'won').sum())} | {x.paid.mean() - 1:+.1%} |")
        Aa = arm_stats(con)
        L += ["", "Bandit 2 (prior: every arm loses the book's margin, -8%, worth two matchdays):", "",
              "| arm | matchdays | posterior mean | +-sd | status |", "|---|---|---|---|---|"]
        L += [f"| {r.strategy}-{r.target:g}x | {r.n} | {r.post_mean:+.1%} | {r.post_sd:.1%} | {'STOPPED' if r.stopped else 'running'} |" for r in Aa.itertuples()]
        L += ["", "A 6x slip wins about 1 time in 7: these returns say nothing for months. The table exists to stop a losing arm."]
    W = pd.read_sql("SELECT family, games, w_fit, t, w FROM weights WHERE as_of = (SELECT max(as_of) FROM weights)", con)
    L += ["", "## Model weights in use", ""]
    L += [W.round(3).to_markdown(index=False) if len(W) and W.w.gt(0).any() else "All 0: the stated probability is the book's price with the margin removed."]
    ng = A[(A.status == "open") & A.grade_note.notna()].groupby("grade_note").size().sort_values(ascending=False).head(12)
    if len(ng):
        L += ["", "## Not graded, and why", ""] + [f"- {n} rows: {k}" for k, n in ng.items()]
    unc = [k for k, v in RULES.items() if v is None]
    L += ["", "## Book rules still unconfirmed", "", ", ".join(unc) + ". `quarter_split` is assumed (standard Asian rule)." if unc else "none"]
    text = "\n".join(L) + "\n"
    out = Path(path or Path(DB).with_name("soccer_report.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def team_profiles(out: Path | None = None, league=None, last: int = 6) -> pd.DataFrame:
    """Per team, this season: games and per-game corners, cards, shots, shots on target and fouls, for and against,
    home and away, and the last `last` games. From football-data plus the API's own games (no API requests)."""
    T = load_tables(out)
    M = matches_frame(T)
    if M.empty:
        sys.exit("no games in the tables yet")
    cur = M.season.max()
    M = M[M.season == cur]
    names = {}
    if len(T["fixtures"]):
        F = T["fixtures"]
        names = dict(zip(F.home_id, F.home)) | dict(zip(F.away_id, F.away))
    fd = T.get("fd")
    if fd is not None and len(fd[0]):
        _, info = fd
    rows = []
    stats = ("corners", "cards", "shots", "sot", "fouls")
    for (lg, team), _ in pd.concat([M[["league", "home"]].set_axis(["lg", "t"], axis=1), M[["league", "away"]].set_axis(["lg", "t"], axis=1)]).groupby(["lg", "t"]):
        if league is not None and lg != league:
            continue
        H, A = M[(M.league == lg) & (M.home == team)], M[(M.league == lg) & (M.away == team)]
        r = dict(league=int(lg), team_id=int(team), team=names.get(team, str(team)), games=len(H) + len(A), home=len(H), away=len(A))
        both = pd.concat([H.assign(side="h"), A.assign(side="a")]).sort_values("date")
        for s_ in stats:
            fo = np.where(both.side == "h", both.get(f"h_{s_}"), both.get(f"a_{s_}")).astype(float) if f"h_{s_}" in both else np.full(len(both), np.nan)
            ag = np.where(both.side == "h", both.get(f"a_{s_}"), both.get(f"h_{s_}")).astype(float) if f"h_{s_}" in both else np.full(len(both), np.nan)
            r[f"{s_}_for"], r[f"{s_}_against"] = np.nanmean(fo) if np.isfinite(fo).any() else np.nan, np.nanmean(ag) if np.isfinite(ag).any() else np.nan
            r[f"{s_}_for_home"] = np.nanmean(fo[both.side.values == "h"]) if (both.side == "h").any() and np.isfinite(fo[both.side.values == "h"]).any() else np.nan
            r[f"{s_}_for_away"] = np.nanmean(fo[both.side.values == "a"]) if (both.side == "a").any() and np.isfinite(fo[both.side.values == "a"]).any() else np.nan
            r[f"{s_}_for_last{last}"] = np.nanmean(fo[-last:]) if np.isfinite(fo[-last:]).any() else np.nan
        rows.append(r)
    P = pd.DataFrame(rows)
    p = Path(out or OUT) / f"soccer_{cur}_team_profiles.csv"
    P.round(2).to_csv(p, index=False)
    return P


def print_profiles(P: pd.DataFrame) -> None:
    import soccer_ingest as SI
    for lg, x in P.groupby("league"):
        print(f"\n{SI.LEAGUES.get(lg, ('', str(lg)))[1]} ({len(x)} teams), per game this season: for / against (last 6 for)")
        print(f"  {'team':<24}{'gp':>3}  {'corners':>15}  {'cards':>13}  {'shots':>15}  {'on target':>13}  {'fouls':>13}")
        for r in x.sort_values("corners_for", ascending=False).itertuples():
            f = lambda s_: f"{getattr(r, s_ + '_for'):4.1f}/{getattr(r, s_ + '_against'):4.1f} ({getattr(r, s_ + '_for_last6'):4.1f})"
            print(f"  {str(r.team)[:23]:<24}{r.games:>3}  {f('corners'):>15}  {f('cards'):>13}  {f('shots'):>15}  {f('sot'):>13}  {f('fouls'):>13}")


def unread(boards: str) -> None:
    """No ledger, no ingest: how every market on the boards is read. Run this on the first real board."""
    rows = []
    for f in sorted(glob.glob(os.path.join(boards, "*.net.json"))):
        evs = read_capture(f)
        if not evs:
            print(f"  {os.path.basename(f)}: no event with markets/odds/competitors found (keys differ from the NFL captures?)")
        for ev in evs:
            h, a = ev["competitors"]
            print(f"  {os.path.basename(f)}: {h.get('name')} v {a.get('name')}  start {ev.get('startDate')}  {len(ev['markets'])} markets, {len(ev['odds'])} prices")
            rows += [dict(r, **classify(r, h, a)) for r in raw_rows(ev)]
    if not rows:
        sys.exit(f"nothing read from {boards}")
    D = pd.DataFrame(rows)
    print(f"\n{len(D)} rows, typed {D.type.notna().mean():.0%}\n\nHow each market name is read:")
    for (mk, t, sc), x in sorted(D.groupby(["market", D.type.fillna("-"), "scope"]), key=lambda kv: (kv[0][1] == "-", kv[0][0])):
        print(f"  {t:<13}{sc:<4}{mk!r} x{len(x)}  e.g. {x.selection.iloc[0]!r} sv={x.sv.iloc[0]!r}")
    print()
    print_unread(D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("unread"); s.add_argument("--boards", required=True)
    s = sub.add_parser("price"); s.add_argument("--date", required=True); s.add_argument("--boards", required=True)
    s = sub.add_parser("slips"); s.add_argument("--date", required=True); s.add_argument("--targets", nargs="+", type=float, default=[3, 6, 9])
    s.add_argument("--legs", nargs="*", type=int, default=[], help="also build slips of exactly this many legs, e.g. --legs 7")
    s = sub.add_parser("grade"); s.add_argument("--date", required=True); s.add_argument("--boards")
    s = sub.add_parser("handcheck"); s.add_argument("--confirm", action="store_true")
    s = sub.add_parser("teams"); s.add_argument("--league", type=int)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "selftest":
        import soccer_selftest
        sys.exit(soccer_selftest.main())
    if a.cmd == "unread":
        return unread(a.boards)
    if a.cmd == "teams":
        P = team_profiles(league=a.league)
        print_profiles(P)
        return print(f"\nwritten: {Path(OUT) / ('soccer_' + str(int(load_tables()['fixtures'].season.max())) + '_team_profiles.csv')}")
    con = connect()
    if a.cmd == "price":
        price(a.date, a.boards, con)
    elif a.cmd == "slips":
        slips(a.date, con, a.targets, legs=tuple(a.legs))
    elif a.cmd == "grade":
        grade(a.date, con, a.boards)
        fit_weights(con, quiet=True)
        fit_calib(con)
    elif a.cmd == "handcheck":
        handcheck(con, a.confirm)
    elif a.cmd == "report":
        fit_weights(con, quiet=True)
        text = report(con)
        print(text if handchecked(con) else "NOT VERIFIED: soccer_report.md was written, but no number is shown until `handcheck --confirm` "
              "records that five results across market types were checked by hand.")


if __name__ == "__main__":
    main()
