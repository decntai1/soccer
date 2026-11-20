#!/usr/bin/env python3
"""soccer_bandit.py: bandit 1 of the spec (section 6): a Thompson-sampling backtest on closing lines.

    python soccer_bandit.py download --seasons 2015-2025       # football-data.co.uk CSVs -> fd/<season>/<code>.csv (free, no API calls)
    python soccer_bandit.py backtest [--seasons 2016-2025] [--book avg|b365|pinnacle] [--runs 300] [--show 2025]
    python soccer_bandit.py simulate                           # self-check on made-up seasons with a known answer

Each game, three decisions: 1X2 (home / draw / away / no bet), over or under 2.5, Asian handicap side.
Context comes from EARLIER matchdays only. Learner: P(outcome) = p_book + x.theta, prior theta ~ N(0, I/25): the
book is right until shown otherwise. One theta draw per matchday; the posterior is updated after the matchday from
EVERY outcome (the result of every arm is visible, so nothing is thrown away). Reward: units won at the offered odds.
Every season is run separately from a fresh prior. Believe a result only if it is in profit in MOST seasons.

A matchday here is a calendar block: Friday-Monday or Tuesday-Thursday (football-data has no round numbers).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_model import GameModel, LeagueModel, line_fracs, newcomer_starts

CODES = {"E1": "Championship", "E2": "League One", "E3": "League Two", "D2": "2. Bundesliga", "I2": "Serie B", "SP2": "La Liga 2"}
FD = Path(os.environ.get("SOCCER_FD", "fd"))
PRIOR_PREC = 25.0
# column triples in order of preference per book family; the first complete one is used and its name recorded
ODDS = {
    "avg": dict(x12=[("AvgCH", "AvgCD", "AvgCA"), ("BbAvH", "BbAvD", "BbAvA"), ("AvgH", "AvgD", "AvgA"), ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")],
                ou=[("AvgC>2.5", "AvgC<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("Avg>2.5", "Avg<2.5"), ("B365C>2.5", "B365C<2.5"), ("B365>2.5", "B365<2.5")],
                ah=[("AHCh", "AvgCAHH", "AvgCAHA"), ("BbAHh", "BbAvAHH", "BbAvAHA"), ("AHh", "AvgAHH", "AvgAHA"), ("AHCh", "B365CAHH", "B365CAHA"), ("AHh", "B365AHH", "B365AHA")]),
    "b365": dict(x12=[("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")], ou=[("B365C>2.5", "B365C<2.5"), ("B365>2.5", "B365<2.5")],
                 ah=[("AHCh", "B365CAHH", "B365CAHA"), ("AHh", "B365AHH", "B365AHA"), ("B365AH", "B365AHH", "B365AHA")]),
    "pinnacle": dict(x12=[("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA")], ou=[("PC>2.5", "PC<2.5"), ("P>2.5", "P<2.5")],
                     ah=[("AHCh", "PCAHH", "PCAHA"), ("AHh", "PAHH", "PAHA")]),
}
CLOSING = {"AvgCH", "B365CH", "PSCH", "AvgC>2.5", "B365C>2.5", "PC>2.5", "AHCh"}


def season_code(y: int) -> str:
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def parse_seasons(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def download(seasons: list[int]) -> None:
    for y in seasons:
        for code in CODES:
            p = FD / season_code(y) / f"{code}.csv"
            if p.exists() and y < seasons[-1]:
                continue
            url = f"https://www.football-data.co.uk/mmz4281/{season_code(y)}/{code}.csv"
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                print(f"  {p}")
            except Exception as e:                              # noqa: BLE001
                print(f"  FAILED {url}: {e}")


def fd_date(col: pd.Series) -> pd.Series:
    """football-data dates are dd/mm/yy in old files and dd/mm/yyyy in new ones."""
    s = col.astype(str).str.strip()
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce").fillna(pd.to_datetime(s, format="%d/%m/%y", errors="coerce"))


def demargin(q: np.ndarray) -> np.ndarray:
    """Odds-ratio method on rows of implied probabilities (same rule as the live system)."""
    lo, hi = np.full(len(q), 1e-3), np.full(len(q), 1e3)
    for _ in range(60):
        c = np.sqrt(lo * hi)[:, None]
        over = (q / (c + q - c * q)).sum(1) > 1
        lo, hi = np.where(over, c[:, 0], lo), np.where(over, hi, c[:, 0])
    c = np.sqrt(lo * hi)[:, None]
    return q / (c + q - c * q)


def load(seasons: list[int], book: str = "avg") -> pd.DataFrame:
    out = []
    for y in seasons:
        for code in CODES:
            p = FD / season_code(y) / f"{code}.csv"
            if not p.exists():
                continue
            try:
                d = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip")
            except UnicodeDecodeError:
                d = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip")
            d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            if d.empty:
                continue
            g = pd.DataFrame(dict(league=code, season=y, date=fd_date(d["Date"]), home=d.HomeTeam.str.strip(),
                                  away=d.AwayTeam.str.strip(), hg=d.FTHG.astype(int), ag=d.FTAG.astype(int)))
            for new, old in (("hthg", "HTHG"), ("htag", "HTAG"), ("hst", "HST"), ("ast", "AST"), ("hc", "HC"), ("ac", "AC")):
                g[new] = pd.to_numeric(d[old], errors="coerce") if old in d else np.nan
            for key, names in (("x12", ("o_h", "o_d", "o_a")), ("ou", ("o_over", "o_under")), ("ah", ("ah_line", "o_ah_h", "o_ah_a"))):
                for n in names:
                    g[n] = np.nan
                g[f"src_{key}"] = None
                for cols in ODDS[book][key]:
                    if not all(c in d for c in cols):
                        continue
                    v = d[list(cols)].apply(pd.to_numeric, errors="coerce")
                    price_cols = cols[1:] if key == "ah" else cols
                    okrow = v.notna().all(axis=1) & (v[list(price_cols)] > 1.0).all(axis=1) & g[names[-1]].isna()
                    for n, c in zip(names, cols):
                        g.loc[okrow, n] = v.loc[okrow, c]
                    g.loc[okrow, f"src_{key}"] = cols[0]
            out.append(g.dropna(subset=["date"]))
    if not out:
        return pd.DataFrame()
    M = pd.concat(out, ignore_index=True).sort_values(["date", "league", "home"]).reset_index(drop=True)
    wd = M.date.dt.weekday                                     # Fri-Mon = one block, Tue-Thu = the other
    M["block"] = M.date - pd.to_timedelta(np.where(wd.isin([4, 5, 6, 0]), (wd - 4) % 7, wd - 1), unit="D")
    return M


# ------------------------------------------------------------------ context (earlier matchdays only)
FEATS = ["dc_diff", "ppg", "gd", "sot", "newcomer", "rest", "matchday"]


def season_frame(M: pd.DataFrame, season: int, verbose: bool = False) -> pd.DataFrame:
    """Rows of `season` with book prices without margin, model prices and form, each built only from games before
    the row's matchday block."""
    S = M[M.season == season].copy()
    hist = M[(M.season >= season - 3) & (M.season <= season)]
    starts = newcomer_starts(M[M.season < season]) if (M.season < season).any() else {}
    prev = M[M.season == season - 1]
    prev_teams = {lg: set(g.home) | set(g.away) for lg, g in prev.groupby("league")}
    q = 1 / S[["o_h", "o_d", "o_a"]].values
    S[["pb_h", "pb_d", "pb_a"]] = demargin(np.nan_to_num(q, nan=1 / 3))
    qo = 1 / S[["o_over", "o_under"]].values
    S["pb_o"] = demargin(np.nan_to_num(qo, nan=0.5))[:, 0]
    qa = 1 / S[["o_ah_h", "o_ah_a"]].values
    S["pb_ah"] = demargin(np.nan_to_num(qa, nan=0.5))[:, 0]
    for c in ("pm_h", "pm_d", "pm_a", "pm_o", "pm_ah", "f_ppg", "f_gd", "f_sot", "f_rest", "f_md"):
        S[c] = np.nan
    S["f_new"] = 0.0
    log: dict = {}                                              # team -> list of (date, points, goal diff, sot diff)
    for r in M[(M.season == season - 1)].itertuples():          # last season's games seed the form
        for t, gf, ga, sf, sa in ((r.home, r.hg, r.ag, r.hst, r.ast), (r.away, r.ag, r.hg, r.ast, r.hst)):
            log.setdefault(t, []).append((r.date, 3 * (gf > ga) + (gf == ga), gf - ga, sf - sa))
    played: dict = {}
    for b, B in S.groupby("block", sort=True):
        for lg, G in B.groupby("league"):
            lm = LeagueModel(hist, lg, b, starts, season)
            for r in G.itertuples():
                lam = lm.lambdas(r.home, r.away) if lm.dc is not None else None
                if lam is not None:
                    gm = GameModel(lam[0], lam[1], lm.dc["rho"], lm.h1)
                    S.loc[r.Index, ["pm_h", "pm_d", "pm_a", "pm_o"]] = [gm.p("1X2", side="1"), gm.p("1X2", side="X"), gm.p("1X2", side="2"),
                                                                       gm.p("TOTAL", side="over", line=2.5)]
                    if not math.isnan(r.ah_line):
                        S.loc[r.Index, "pm_ah"] = gm.p("AH", side="1", line=r.ah_line)
                f = {}
                for side, t in (("h", r.home), ("a", r.away)):
                    last = log.get(t, [])[-6:]
                    f[side] = (np.mean([x[1] for x in last]) if last else 1.35, np.mean([x[2] for x in last]) if last else 0.0,
                               np.nanmean([x[3] for x in last]) if last and not all(np.isnan(x[3]) for x in last) else 0.0,
                               min(max((r.date - log[t][-1][0]).days, 2), 14) if log.get(t) else 7, played.get(t, 0))
                new = lambda t: float(lg in prev_teams and t not in prev_teams[lg])
                S.loc[r.Index, ["f_ppg", "f_gd", "f_sot", "f_rest", "f_md", "f_new"]] = [
                    f["h"][0] - f["a"][0], (f["h"][1] - f["a"][1]) / 2, (f["h"][2] - f["a"][2]) / 4, (f["h"][3] - f["a"][3]) / 7,
                    (f["h"][4] + f["a"][4]) / 92 - 0.5, new(r.home) - new(r.away)]
        for r in B.itertuples():                                 # only now do this block's results exist
            for t, gf, ga, sf, sa in ((r.home, r.hg, r.ag, r.hst, r.ast), (r.away, r.ag, r.hg, r.ast, r.hst)):
                log.setdefault(t, []).append((r.date, 3 * (gf > ga) + (gf == ga), gf - ga, sf - sa))
                played[t] = played.get(t, 0) + 1
        if verbose:
            print(f"    {season} block {b.date()} done", end="\r")
    return S


def design(S: pd.DataFrame, pm: str, pb: str) -> np.ndarray:
    """x for one outcome: league dummies (the intercepts) + model-minus-book for that outcome + form."""
    lg = np.stack([(S.league == c).values.astype(float) for c in CODES], 1)
    dc = (S[pm] - S[pb]).fillna(0).clip(-0.3, 0.3).values[:, None]
    rest = S[["f_ppg", "f_gd", "f_sot", "f_new", "f_rest", "f_md"]].fillna(0).values
    return np.hstack([lg, dc, rest])


# ------------------------------------------------------------------ the bandit
def run_season(S: pd.DataFrame, runs: int = 300, seed: int = 0, state: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    out_k = [("h", "pm_h", "pb_h"), ("d", "pm_d", "pb_d"), ("a", "pm_a", "pb_a"), ("o", "pm_o", "pb_o"), ("ah", "pm_ah", "pb_ah")]
    X = {k: design(S, pm, pb) for k, pm, pb in out_k}
    d = X["h"].shape[1]
    hg, ag = S.hg.values, S.ag.values
    res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    ahw = np.array([line_fracs(h - a, -ln, "over") if not math.isnan(ln) else (np.nan, np.nan) for h, a, ln in zip(hg, ag, S.ah_line.values)])
    # y and weight per outcome model
    Y = dict(h=(res == 0) * 1.0, d=(res == 1) * 1.0, a=(res == 2) * 1.0, o=(hg + ag > 2.5) * 1.0,
             ah=np.where(ahw.sum(1) > 0, ahw[:, 0] / np.where(ahw.sum(1) > 0, ahw.sum(1), 1), 0.0))
    Wt = dict(h=S.o_h.notna().values * 1.0, d=S.o_h.notna().values * 1.0, a=S.o_h.notna().values * 1.0, o=S.o_over.notna().values * 1.0,
              ah=np.nan_to_num(ahw.sum(1)) * S.o_ah_h.notna().values)
    PB = {k: S[pb].values for k, _, pb in out_k}
    A = {k: np.eye(d) * PRIOR_PREC for k in X} if state is None else state["A"]      # state: what earlier seasons taught (--carry)
    bvec = {k: np.zeros(d) for k in X} if state is None else state["b"]
    O = S[["o_h", "o_d", "o_a"]].values
    rows = []
    for b, idx in S.groupby("block", sort=True).indices.items():
        th = {}
        for k in X:                                              # one draw per matchday and run
            cov = np.linalg.inv(A[k])
            th[k] = rng.multivariate_normal(cov @ bvec[k], cov, size=runs)
            th[k][0] = cov @ bvec[k]                             # run 0 is "greedy": the posterior mean, no exploration
        sh = {k: X[k][idx] @ th[k].T for k in X}                 # games x runs: shift from the book's price
        rec = dict(block=b, games=len(idx))
        # ---- 1X2: best expected value among home / draw / away, or no bet
        ok = ~np.isnan(O[idx]).any(1)
        P = np.stack([np.clip(PB[k][idx][:, None] + sh[k], 0.01, 0.99) for k in ("h", "d", "a")], 2)      # games x runs x 3
        ev = P * O[idx][:, None, :] - 1
        pick = ev.argmax(2)
        bet = (np.take_along_axis(ev, pick[..., None], 2)[..., 0] > 0) & ok[:, None]
        win = pick == res[idx][:, None]
        o_p = np.take_along_axis(np.broadcast_to(O[idx][:, None, :], ev.shape), pick[..., None], 2)[..., 0]
        rec.update(x12_n=bet.sum(0), x12_right=(bet & win).sum(0), x12_units=np.where(bet, np.where(win, o_p - 1, -1.0), 0.0).sum(0),
                   x12_short=(bet & (pick == np.nanargmin(O[idx], 1)[:, None] if ok.any() else False)).sum(0),
                   x12_shift=np.where(bet, np.abs(np.take_along_axis(np.stack([sh[k] for k in ("h", "d", "a")], 2), pick[..., None], 2)[..., 0]), 0).sum(0))
        # ---- over / under 2.5: always one side
        oo, ou = S.o_over.values[idx], S.o_under.values[idx]
        ok = ~np.isnan(oo)
        po = np.clip(PB["o"][idx][:, None] + sh["o"], 0.01, 0.99)
        over = po * oo[:, None] - 1 >= (1 - po) * ou[:, None] - 1
        y = (Y["o"][idx] == 1)[:, None]
        win = over == y
        o_p = np.where(over, oo[:, None], ou[:, None])
        rec.update(ou_n=np.full(runs, ok.sum()), ou_right=(win & ok[:, None]).sum(0), ou_units=np.where(ok[:, None], np.where(win, o_p - 1, -1.0), 0.0).sum(0),
                   ou_short=((o_p <= np.minimum(oo, ou)[:, None]) & ok[:, None]).sum(0), ou_shift=np.where(ok[:, None], np.abs(sh["o"]), 0).sum(0))
        # ---- Asian handicap: always one side, settled with pushes and half results
        oh, oa = S.o_ah_h.values[idx], S.o_ah_a.values[idx]
        ok = ~np.isnan(oh) & ~np.isnan(ahw[idx, 0])
        pa = np.clip(PB["ah"][idx][:, None] + sh["ah"], 0.01, 0.99)
        home = pa * oh[:, None] - 1 >= (1 - pa) * oa[:, None] - 1
        w_h, l_h = np.nan_to_num(ahw[idx, 0])[:, None], np.nan_to_num(ahw[idx, 1])[:, None]
        won, lost = np.where(home, w_h, l_h), np.where(home, l_h, w_h)      # the away side's fractions are the mirror
        o_p = np.where(home, oh[:, None], oa[:, None])
        decided = (won != lost) & ok[:, None]
        rec.update(ah_n=np.full(runs, ok.sum()), ah_decided=decided.sum(0), ah_right=((won > lost) & ok[:, None]).sum(0),
                   ah_units=np.where(ok[:, None], won * (o_p - 1) - lost, 0.0).sum(0),
                   ah_short=((o_p <= np.minimum(oh, oa)[:, None]) & ok[:, None]).sum(0), ah_shift=np.where(ok[:, None], np.abs(sh["ah"]), 0).sum(0))
        rows.append(rec)
        for k in X:                                              # learn from every outcome of the matchday
            x, w = X[k][idx], Wt[k][idx] / np.clip(PB[k][idx] * (1 - PB[k][idx]), 0.1, None)
            good = w > 0
            A[k] += (x[good] * w[good, None]).T @ x[good]
            bvec[k] += (x[good] * w[good, None]).T @ (Y[k][idx][good] - PB[k][idx][good])
    # ---- baselines on the same games and prices
    ok = ~np.isnan(O).any(1)
    r3 = np.where(np.arange(3)[None, :] == res[:, None], O - 1, -1.0)[ok]
    fav = np.nanargmin(np.where(np.isnan(O), np.inf, O), 1)[ok]
    oko = S.o_over.notna().values
    base = dict(home=r3[:, 0].mean(), draw=r3[:, 1].mean(), away=r3[:, 2].mean(), favourite=r3[np.arange(len(r3)), fav].mean(), random=r3.mean(),
                over=np.where(Y["o"][oko] == 1, S.o_over.values[oko] - 1, -1.0).mean() if oko.any() else np.nan,
                margin_1x2=float((1 / O[ok]).sum(1).mean() - 1) if ok.any() else np.nan)
    theta = {k: np.linalg.solve(A[k], bvec[k]) for k in X}
    return dict(blocks=rows, base=base, theta=theta, games=len(S), state=dict(A=A, b=bvec))


def summarise(season: int, R: dict) -> dict:
    out = dict(season=season, games=R["games"])
    tot_u = tot_n = 0
    for k in ("x12", "ou", "ah"):
        n = np.sum([b[f"{k}_n"] for b in R["blocks"]], 0).astype(float)
        u = np.sum([b[f"{k}_units"] for b in R["blocks"]], 0)
        right = np.sum([b[f"{k}_right"] for b in R["blocks"]], 0)
        den = np.sum([b["ah_decided"] for b in R["blocks"]], 0).astype(float) if k == "ah" else n
        ret = u / np.maximum(n, 1)
        out[f"{k}_greedy"] = ret[0]
        n, u, right, den, ret = n[1:], u[1:], right[1:], den[1:], ret[1:]
        out.update({f"{k}_picks": n.mean(), f"{k}_hit": (right / np.maximum(den, 1)).mean(), f"{k}_ret": ret.mean(), f"{k}_lo": np.quantile(ret, 0.05),
                    f"{k}_hi": np.quantile(ret, 0.95), f"{k}_short": (np.sum([b[f"{k}_short"] for b in R["blocks"]], 0)[1:] / np.maximum(n, 1)).mean(),
                    f"{k}_shift": (np.sum([b[f"{k}_shift"] for b in R["blocks"]], 0)[1:] / np.maximum(n, 1)).mean()})
        tot_u, tot_n = tot_u + np.sum([b[f"{k}_units"] for b in R["blocks"]], 0), tot_n + np.sum([b[f"{k}_n"] for b in R["blocks"]], 0).astype(float)
    ret = tot_u / np.maximum(tot_n, 1)
    out.update(all_ret=ret[1:].mean(), all_lo=np.quantile(ret[1:], 0.05), all_hi=np.quantile(ret[1:], 0.95), all_units=tot_u[1:].mean(),
               share_runs_in_profit=(tot_u[1:] > 0).mean(), greedy_ret=ret[0])
    out.update({f"base_{k}": v for k, v in R["base"].items()})
    return out


def matchday_table(R: dict) -> pd.DataFrame:
    rows, cum = [], dict(n=0.0, right=0.0, units=0.0)
    cum_u = 0
    for b in R["blocks"]:
        n = b["x12_n"] + b["ou_n"] + b["ah_n"]
        right, units = b["x12_right"] + b["ou_right"] + b["ah_right"], b["x12_units"] + b["ou_units"] + b["ah_units"]
        den = b["x12_n"] + b["ou_n"] + b["ah_decided"]
        cum_u = cum_u + units
        cum["n"], cum["right"], cum["units"] = cum["n"] + den.mean(), cum["right"] + right.mean(), cum["units"] + units.mean()
        rows.append(dict(matchday=b["block"].date(), games=b["games"], picks=n.mean(), right=right.mean(), hit=right.mean() / max(den.mean(), 1),
                         units=units.mean(), cum_hit=cum["right"] / max(cum["n"], 1), cum_units=cum["units"], cum_p5=np.quantile(cum_u, 0.05),
                         cum_p95=np.quantile(cum_u, 0.95), on_shorter_price=(b["x12_short"] + b["ou_short"] + b["ah_short"]).mean() / max(n.mean(), 1),
                         avg_shift=(b["x12_shift"] + b["ou_shift"] + b["ah_shift"]).mean() / max(n.mean(), 1)))
    return pd.DataFrame(rows)


def backtest(seasons: list[int], book: str = "avg", runs: int = 300, show: int | None = None, outdir: str = ".", quiet: bool = False, carry: bool = False):
    M = load(list(range(seasons[0] - 3, seasons[-1] + 1)), book)
    if M.empty:
        sys.exit(f"no CSVs under {FD}/ : run `download` first (needs open internet: your machine or GitHub Actions)")
    rows, last, state = [], None, None
    for y in seasons:
        if not (M.season == y).any():
            print(f"  season {y}: no data")
            continue
        S = season_frame(M, y)
        cov = dict(x12=S.o_h.notna().mean(), ou=S.o_over.notna().mean(), ah=S.o_ah_h.notna().mean(), model=S.pm_h.notna().mean(),
                   closing=S.src_x12.isin(CLOSING).mean())
        R = run_season(S, runs, seed=y, state=state if carry else None)
        state = R["state"]
        r = summarise(y, R)
        r.update({f"cov_{k}": v for k, v in cov.items()}, src=",".join(sorted({str(x) for x in S.src_x12.dropna().unique()})))
        rows.append(r)
        if show == y or (show is None and y == seasons[-1]):
            last = (y, matchday_table(R), R["theta"])
        if not quiet:
            print(f"  {y}/{(y + 1) % 100:02d}: {r['games']} games  1X2 {r['x12_ret']:+.1%} ({r['x12_picks']:.0f} picks)  O/U {r['ou_ret']:+.1%}  AH {r['ah_ret']:+.1%}  "
                  f"all {r['all_ret']:+.1%} [{r['all_lo']:+.1%} .. {r['all_hi']:+.1%}]  closing prices {cov['closing']:.0%}  model on {cov['model']:.0%}")
    T = pd.DataFrame(rows)
    if T.empty:
        sys.exit("no season could be run")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    T.to_csv(Path(outdir) / "bandit1_seasons.csv", index=False)
    L = [f"# Bandit 1 backtest ({book} prices, {runs} runs per season, "
         + ("what it learned is CARRIED from season to season)" if carry else "fresh prior every season)"), "",
         "| season | games | 1X2 picks | 1X2 hit | 1X2 return | O/U hit | O/U return | AH hit | AH return | all bets | 5-95% of runs | runs in profit |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        L.append(f"| {r.season}/{(r.season + 1) % 100:02d} | {r.games} | {r.x12_picks:.0f} | {r.x12_hit:.1%} | {r.x12_ret:+.1%} | {r.ou_hit:.1%} | {r.ou_ret:+.1%} | {r.ah_hit:.1%} | "
                 f"{r.ah_ret:+.1%} | {r.all_ret:+.1%} | {r.all_lo:+.1%} .. {r.all_hi:+.1%} | {r.share_runs_in_profit:.0%} |")
    prof = int((T.all_ret > 0).sum())
    L += ["", f"Without exploration (bets on the posterior mean): {T.greedy_ret.mean():+.1%} per bet, in profit in {int((T.greedy_ret > 0).sum())} of {len(T)} seasons.",
          "", f"**In profit in {prof} of {len(T)} seasons** (mean return per bet {T.all_ret.mean():+.1%}). "
              + ("That is most seasons: worth a closer look, starting with whether the prices were really available." if prof > len(T) / 2 and len(T) >= 5
                 else "That is not most seasons: no edge shown. The average does not count."),
          "", "Baselines on the same games and prices (return per bet):", "",
          "| season | always home | always draw | always away | always favourite | random 1X2 | always over | 1X2 margin |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.season}/{(r.season + 1) % 100:02d} | {r.base_home:+.1%} | {r.base_draw:+.1%} | {r.base_away:+.1%} | {r.base_favourite:+.1%} | {r.base_random:+.1%} | "
          f"{r.base_over:+.1%} | {r.base_margin_1x2:.1%} |" for r in T.itertuples()]
    L += ["", "| season | picks on the shorter price (1X2 / O/U / AH) | average shift from the book's price (1X2 / O/U / AH) | 1X2 price source |", "|---|---|---|---|"]
    L += [f"| {r.season}/{(r.season + 1) % 100:02d} | {r.x12_short:.0%} / {r.ou_short:.0%} / {r.ah_short:.0%} | {r.x12_shift:.3f} / {r.ou_shift:.3f} / {r.ah_shift:.3f} | {r.src} |"
          for r in T.itertuples()]
    if last:
        y, D, theta = last
        D.to_csv(Path(outdir) / f"bandit1_matchdays_{y}.csv", index=False)
        L += ["", f"## Season {y}/{(y + 1) % 100:02d} by matchday (means over the runs)", "", D.round(3).to_markdown(index=False), "",
              "What it had learned by the end of that season (posterior mean; league columns are the intercepts):", "",
              pd.DataFrame(theta, index=list(CODES) + FEATS).round(3).to_markdown()]
    text = "\n".join(L) + "\n"
    (Path(outdir) / "bandit1_report.md").write_text(text, encoding="utf-8")
    if not quiet:
        print(f"\nIn profit in {prof} of {len(T)} seasons. Report: {Path(outdir) / 'bandit1_report.md'}")
    return T


# ------------------------------------------------------------------ self-check on made-up seasons
def simulate() -> int:
    """Made-up seasons in football-data's own CSV layout (old and new column names). (a) the book is exactly right:
    the bandit must lose about the margin and must not be 'in profit in most seasons'. (b) the book prices every
    home win 10 points too low: the bandit must find it. (c) the same with 4 points: reported as a measure of what
    the design can detect, not as a pass/fail. (d) a quarter-line check on the handicap settlement."""
    global FD
    keep, tmp = FD, Path(tempfile.mkdtemp(prefix="soccer_bandit_sim_"))
    checks, power = [], {}
    for name, bias, carry in (("fair", 0.0, False), ("bias10", 0.10, False), ("bias4", 0.04, False), ("bias4c", 0.04, True)):
        FD = tmp / name
        rng = np.random.default_rng(11)
        strength = {}
        for y in range(2017, 2024):
            old = y < 2019                                       # the pre-2019 files have no closing columns
            for code in ("E1", "D2"):
                teams = [f"{code} Team {i}" for i in range(16)]
                if y > 2017:
                    teams = teams[:14] + [f"{code} New {y}-{i}" for i in range(2)]
                for t in teams:
                    strength.setdefault(t, (rng.normal(0, 0.22), rng.normal(0, 0.22)))
                rows, day = [], pd.Timestamp(y, 8, 7)
                order = list(teams)
                for rnd in range(30):
                    if rnd % 15 == 0:
                        rng.shuffle(order)
                    rot = [order[0]] + order[1 + rnd % 15:] + order[1:1 + rnd % 15]
                    for i in range(8):
                        h, a = (rot[i], rot[15 - i]) if rnd < 15 else (rot[15 - i], rot[i])
                        lh, la = math.exp(0.1 + 0.25 + strength[h][0] - strength[a][1]), math.exp(0.1 + strength[a][0] - strength[h][1])
                        g = GameModel(lh, la, 0.0, 0.45)
                        hg, ag = rng.poisson(lh), rng.poisson(la)
                        p3 = np.array([g.p("1X2", side=s) for s in "1X2"])
                        pb = p3 + np.array([-bias, bias / 2, bias / 2])                       # what the book believes
                        o3 = 1 / (demarg_inv(pb, 1.06))
                        po = g.p("TOTAL", side="over", line=2.5)
                        line = float(np.clip(-round((lh - la) * 4) / 4, -1.5, 1.5))
                        pah = g.p("AH", side="1", line=line)
                        oo, oah = 1 / demarg_inv(np.array([po, 1 - po]), 1.05), 1 / demarg_inv(np.array([pah, 1 - pah]), 1.04)
                        r = dict(Div=code, Date=(day + pd.Timedelta(days=int(rng.integers(0, 3)))).strftime("%d/%m/%Y" if not old else "%d/%m/%y"), HomeTeam=h,
                                 AwayTeam=a, FTHG=hg, FTAG=ag, FTR="H" if hg > ag else "D" if hg == ag else "A", HTHG=min(hg, rng.binomial(hg, 0.45)),
                                 HTAG=rng.binomial(ag, 0.45), HST=hg + rng.poisson(3), AST=ag + rng.poisson(3), HC=rng.poisson(5), AC=rng.poisson(4))
                        names = (("BbAvH", "BbAvD", "BbAvA"), ("BbAv>2.5", "BbAv<2.5"), ("BbAHh", "BbAvAHH", "BbAvAHA")) if old else \
                            (("AvgCH", "AvgCD", "AvgCA"), ("AvgC>2.5", "AvgC<2.5"), ("AHCh", "AvgCAHH", "AvgCAHA"))
                        r.update(dict(zip(names[0], np.round(o3, 2))), **dict(zip(names[1], np.round(oo, 2))), **dict(zip(names[2], [line] + list(np.round(oah, 2)))))
                        rows.append(r)
                    day += pd.Timedelta(days=7)
                p = FD / season_code(y) / f"{code}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig" if not old else "latin-1")
        T = backtest(list(range(2018, 2024)), runs=120, outdir=str(tmp / name), quiet=True, carry=carry)
        prof = int((T.all_ret > 0).sum())
        print(f"  {name:<7} book: 1X2 {T.x12_ret.mean():+.1%} (no exploration: {T.x12_greedy.mean():+.1%}, in profit {(T.x12_greedy > 0).sum()} of {len(T)})  O/U {T.ou_ret.mean():+.1%}  AH {T.ah_ret.mean():+.1%}  all {T.all_ret.mean():+.1%}; "
              f"in profit {prof} of {len(T)} seasons; always-home baseline {T.base_home.mean():+.1%}; margin {T.base_margin_1x2.mean():.1%}")
        if name == "fair":
            checks += [("fair book: every decision loses (between -12% and -1% per bet)", all(-0.12 < T[c].mean() < -0.01 for c in ("x12_ret", "ou_ret", "ah_ret"))),
                       ("fair book: not in profit in most seasons", prof <= len(T) / 2),
                       ("both CSV layouts are read (old Bb* columns and new closing columns), prices on every game", (T.cov_x12 == 1).all() and (T.cov_ah == 1).all()
                        and set(",".join(T.src).split(",")) == {"AvgCH", "BbAvH"}),
                       ("the model has a number for nearly every game, built from earlier matchdays", (T.cov_model > 0.9).all())]
        elif name == "bias10":
            checks += [("book prices every home win 10 points too low: the bandit's 1X2 bets are in profit in most seasons", (T.x12_ret > 0).sum() > len(T) / 2 and T.x12_ret.mean() > 0.02),
                       ("same book: the unbiased markets (O/U, AH) still lose", T.ou_ret.mean() < 0 and T.ah_ret.mean() < 0)]
        else:
            power[name] = (T.x12_ret.mean(), int((T.x12_ret > 0).sum()), len(T), T.base_home.mean())
    FD = keep
    a = [line_fracs(1, 1.0, "over"), line_fracs(1, 0.75, "over"), line_fracs(1, 1.25, "over")]
    checks.append(("handicap -1 / -0.75 / -1.25 with a one-goal win: push / half won / half lost", a == [(0, 0), (0.5, 0), (0, 0.5)]))
    for n, good in checks:
        print(f"  {'ok  ' if good else 'FAIL'} {n}")
    for nm, lab in (("bias4", "fresh prior each season (the spec)"), ("bias4c", "carried across seasons (--carry)")):
        r, k, n, home = power[nm]
        print(f"  POWER, not a pass/fail: home wins priced 4 points too low (always-home earns {home:+.1%}), 480 games a season, {lab}: "
              f"bandit 1X2 {r:+.1%}, in profit {k} of {n} seasons")
    bad = sum(1 for _, g in checks if not g)
    print(f"{'PASS' if not bad else 'FAIL'}: {len(checks) - bad} of {len(checks)} checks (made-up seasons: this proves the mechanics, not a result)")
    return 1 if bad else 0


def demarg_inv(p: np.ndarray, over: float) -> np.ndarray:
    """True probabilities -> implied probabilities with overround `over`, by a common odds ratio (the reverse of demargin)."""
    lo, hi = 1.0, 50.0
    for _ in range(60):
        c = (lo + hi) / 2
        q = c * p / (1 - p + c * p)
        lo, hi = (c, hi) if q.sum() < over else (lo, c)
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download"); s.add_argument("--seasons", default="2012-2026")
    s = sub.add_parser("backtest"); s.add_argument("--seasons", default="2015-2025"); s.add_argument("--book", default="avg", choices=list(ODDS))
    s.add_argument("--runs", type=int, default=300); s.add_argument("--show", type=int); s.add_argument("--out", default="results")
    s.add_argument("--carry", action="store_true", help="keep what was learned from season to season (the spec's default is a fresh prior)")
    sub.add_parser("simulate")
    a = ap.parse_args()
    if a.cmd == "download":
        download(parse_seasons(a.seasons))
    elif a.cmd == "backtest":
        backtest(parse_seasons(a.seasons), a.book, a.runs, a.show, a.out, carry=a.carry)
    else:
        sys.exit(simulate())


if __name__ == "__main__":
    main()
