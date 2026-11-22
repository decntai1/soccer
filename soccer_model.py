#!/usr/bin/env python3
"""soccer_model.py: the "statistics" vote of the soccer system (spec section 4, "Model").

Deliberately simple, and expected NOT to beat the book:
  goals    Dixon-Coles (Poisson per team, attack/defence per team, home advantage per league, low-score
           correction), time-decay weights, newcomers start at the average of past newcomers to that league
  halves   first-half share of goals fitted per league
  counts   corners / cards / shots / shots on target: negative binomial, rate from the team's own rate and the
           opponent's conceded rate; cards also carry a heavily shrunk referee factor
  players  per-90 rates x expected minutes; penalties handled as a separate share

Input is one `matches` frame (one row per finished game) built by soccer_system.load_tables() from the ingest
CSVs. Nothing here reads the board or the ledger.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import nbinom, poisson

MAXG = 10                    # goals per team carried in the score matrix
HALF_LIFE_DAYS = 365         # time decay: a game one season old weighs half
PRIOR_SD = 0.25              # prior sd of a team's log attack / defence strength around its start value
COUNT_STATS = ("corners", "cards", "shots", "sot", "fouls")
STAT_3WAY = ("CORNERS_1X2", "CARDS_1X2", "SHOTS_1X2", "SOT_1X2")


# ------------------------------------------------------------------ line settlement (shared with the grader)
def line_fracs(x: float, line: float, side: str) -> tuple[float, float]:
    """(won fraction, lost fraction) of the stake for `side` ('over'/'under') of `line` when the quantity is x.
    Whole line: x == line is a push (0, 0). Quarter line (2.25, 2.75): the stake is split over the two neighbouring
    half/whole lines, so 0.5 won + 0.5 pushed = (0.5, 0). Never both non-zero."""
    q = round(line * 4)
    parts = [line] if q % 2 == 0 else [line - 0.25, line + 0.25]
    w = l = 0.0
    for p in parts:
        d = (x - p) if side == "over" else (p - x)
        if d > 1e-9:
            w += 1 / len(parts)
        elif d < -1e-9:
            l += 1 / len(parts)
    return w, l


def p_eff(values, probs, line: float, side: str) -> float:
    """Chance the bet wins GIVEN it is not refunded: W / (W + L). This is the number a de-margined two-way price
    states, so model and book are compared like with like (no push counted twice)."""
    W = L = 0.0
    for x, p in zip(values, probs):
        if p <= 0:
            continue
        w, l = line_fracs(float(x), line, side)
        W += p * w
        L += p * l
    return W / (W + L) if W + L > 0 else float("nan")


# ------------------------------------------------------------------ Dixon-Coles
def score_matrix(lh: float, la: float, rho: float = 0.0, maxg: int = MAXG) -> np.ndarray:
    """P[i, j] = P(home scores i, away scores j)."""
    g = np.arange(maxg + 1)
    P = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    if rho:
        P[0, 0] *= max(1 - lh * la * rho, 1e-9)
        P[0, 1] *= max(1 + lh * rho, 1e-9)
        P[1, 0] *= max(1 + la * rho, 1e-9)
        P[1, 1] *= max(1 - rho, 1e-9)
    return P / P.sum()


def fit_dc(M: pd.DataFrame, as_of=None, start: dict | None = None, half_life: float = HALF_LIFE_DAYS) -> dict | None:
    """One league. M: date, home, away, hg, ag (finished games only, all strictly before as_of).
    log(home goals) = c + h + att[home] - dfn[away];  log(away goals) = c + att[away] - dfn[home].
    `start`: team -> (att0, dfn0), the prior centre (newcomers); everyone else 0 = league average."""
    M = M.dropna(subset=["hg", "ag"])
    if as_of is not None:
        M = M[M.date < pd.Timestamp(as_of)]
    if len(M) < 30:
        return None
    as_of = pd.Timestamp(as_of) if as_of is not None else M.date.max() + pd.Timedelta(days=1)
    teams = sorted(set(M.home) | set(M.away))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = M.home.map(ix).values, M.away.map(ix).values
    hg, ag = M.hg.values.astype(float), M.ag.values.astype(float)
    w = 0.5 ** ((as_of - M.date).dt.days.values / half_life)
    a0, d0 = np.zeros(n), np.zeros(n)
    for t, (a, d) in (start or {}).items():
        if t in ix:
            a0[ix[t]], d0[ix[t]] = a, d
    lam = 1 / PRIOR_SD ** 2

    def nll(x):
        c, h, att, dfn = x[0], x[1], x[2:2 + n], x[2 + n:]
        lh, la = np.exp(c + h + att[hi] - dfn[ai]), np.exp(c + att[ai] - dfn[hi])
        f = (w * (lh - hg * np.log(lh) + la - ag * np.log(la))).sum() + 0.5 * lam * (((att - a0) ** 2).sum() + ((dfn - d0) ** 2).sum())
        rh, ra = w * (lh - hg), w * (la - ag)
        g = np.empty_like(x)
        g[0], g[1] = (rh + ra).sum(), rh.sum()
        g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + lam * (att - a0)
        g[2 + n:] = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + lam * (dfn - d0)
        return f, g

    x0 = np.concatenate([[math.log(max(M.ag.mean(), 0.3)), 0.2], a0, d0])
    r = minimize(nll, x0, jac=True, method="L-BFGS-B")
    c, h, att, dfn = r.x[0], r.x[1], r.x[2:2 + n], r.x[2 + n:]
    lh, la = np.exp(c + h + att[hi] - dfn[ai]), np.exp(c + att[ai] - dfn[hi])

    def nll_rho(rho):                                          # only the four low scores depend on rho
        t = np.ones(len(M))
        t = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, t)
        t = np.where((hg == 0) & (ag == 1), 1 + lh * rho, t)
        t = np.where((hg == 1) & (ag == 0), 1 + la * rho, t)
        t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
        return -(w * np.log(np.clip(t, 1e-9, None))).sum()

    rho = float(minimize_scalar(nll_rho, bounds=(-0.2, 0.2), method="bounded").x)
    return dict(c=float(c), h=float(h), rho=rho, att=dict(zip(teams, att)), dfn=dict(zip(teams, dfn)),
                n_games=len(M), start=dict(start or {}))


def newcomer_starts(M: pd.DataFrame) -> dict:
    """(league, origin) -> (att0, dfn0): how teams new to a league did in their first season there, in log goals
    relative to the league. origin = the league (of the ones we hold) the team played in the season before, or
    'new' when it came from outside our data (so English promotions and relegations are told apart; elsewhere all
    newcomers share one start)."""
    out, acc = {}, {}
    seas = sorted(M.season.unique())
    where = {(s, t): lg for (s, lg), g in M.groupby(["season", "league"]) for t in set(g.home) | set(g.away)}
    for lg, G in M.groupby("league"):
        for s in seas[1:]:
            g, prev = G[G.season == s], G[G.season == seas[seas.index(s) - 1]]
            if g.empty or prev.empty:
                continue
            old = set(prev.home) | set(prev.away)
            per_team = (g.hg.sum() + g.ag.sum()) / (2 * len(g))
            for t in (set(g.home) | set(g.away)) - old:
                gf = g.hg[g.home == t].sum() + g.ag[g.away == t].sum()
                ga = g.ag[g.home == t].sum() + g.hg[g.away == t].sum()
                k = (g.home == t).sum() + (g.away == t).sum()
                if k < 10:
                    continue
                origin = where.get((seas[seas.index(s) - 1], t), "new")
                v = (math.log(max(gf / k, 0.2) / per_team), -math.log(max(ga / k, 0.2) / per_team))
                acc.setdefault((lg, origin), []).append(v)
                acc.setdefault((lg, "any"), []).append(v)
    for k, v in acc.items():
        if len(v) >= 5 or k[1] == "any":
            out[k] = (float(np.mean([a for a, _ in v])), float(np.mean([d for _, d in v])))
    return out


# ------------------------------------------------------------------ counts
def nb_pmf(mean: float, phi: float, kmax: int = 60) -> np.ndarray:
    """Negative binomial with variance = phi * mean (Poisson when phi <= 1.02)."""
    k = np.arange(kmax + 1)
    mean = max(mean, 1e-6)
    if phi <= 1.02:
        p = poisson.pmf(k, mean)
    else:
        p = nbinom.pmf(k, mean / (phi - 1), 1 / phi)
    return p / p.sum()


def _shrunk_ratio(total, n, avg, k=6.0):
    return (total + k * avg) / ((n + k) * avg) if avg > 0 else 1.0


class GameModel:
    """Everything the model says about one game. p() returns None for anything it has no number for."""

    def __init__(self, lh, la, rho, h1_share, counts=None):
        self.lh, self.la, self.rho, self.h1 = lh, la, rho, h1_share
        self.counts = counts or {}                             # stat -> dict(home=mean, away=mean, phi_team, phi_total)

    def matrix(self, scope="FT") -> np.ndarray:
        if scope == "FT":
            return score_matrix(self.lh, self.la, self.rho)
        s = self.h1 if scope == "H1" else 1 - self.h1
        return score_matrix(self.lh * s, self.la * s, 0.0)

    def _goal_dist(self, scope, who):
        P = self.matrix(scope)
        g = np.arange(P.shape[0])
        if who == "home":
            return g, P.sum(1)
        if who == "away":
            return g, P.sum(0)
        i, j = np.meshgrid(g, g, indexing="ij")
        v = (i + j) if who == "total" else (i - j)
        vals = np.unique(v)
        return vals, np.array([P[v == x].sum() for x in vals])

    def p(self, typ, scope="FT", side=None, line=None, who=None, score=None):
        """typ: 1X2 (side 1/X/2), DC (1X/12/X2), DNB (1/2), BTTS (yes/no), TOTAL / TEAM_TOTAL (over/under, who=home/away),
        AH (side 1/2, line = that side's handicap), EH (side 1/X/2, line = home handicap, 3-way), CS (score=(h, a)),
        CORNERS / CARDS / SHOTS / SOT (over/under; who = None for the match, home/away for a team),
        CORNERS_1X2 / CARDS_1X2 / SHOTS_1X2 / SOT_1X2 (which team has more: 1/X/2), CORNERS_AH (side 1/2, line = that side's)."""
        if typ in STAT_3WAY or typ == "CORNERS_AH":
            c = self.counts.get(typ.split("_")[0].lower())
            if c is None or scope != "FT":
                return None
            ph, pa = nb_pmf(c["home"], c["phi_team"]), nb_pmf(c["away"], c["phi_team"])
            J = np.outer(ph, pa)                                   # the two teams' counts taken as independent
            k = np.arange(len(ph))
            d = k[:, None] - k[None, :]
            if typ == "CORNERS_AH":
                if line is None or side not in ("1", "2"):
                    return None
                vals = np.arange(-len(ph) + 1, len(ph))
                pr = np.array([J[d == v].sum() for v in vals])
                return p_eff(vals if side == "1" else -vals, pr, -line, "over")
            return float({"1": J[d > 0].sum(), "X": J[d == 0].sum(), "2": J[d < 0].sum()}.get(side, float("nan")))
        if typ in ("CORNERS", "CARDS", "SHOTS", "SOT"):
            c = self.counts.get(typ.lower())
            if c is None or scope != "FT" or line is None:
                return None
            mean = c["home"] + c["away"] if who is None else c[who]
            pm = nb_pmf(mean, c["phi_total"] if who is None else c["phi_team"])
            return p_eff(np.arange(len(pm)), pm, line, side)
        P = self.matrix(scope)
        hw, dr, aw = np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()
        if typ == "1X2":
            return {"1": hw, "X": dr, "2": aw}.get(side)
        if typ == "DC":
            return {"1X": hw + dr, "12": hw + aw, "X2": dr + aw}.get(side)
        if typ == "DNB":
            return {"1": hw, "2": aw}.get(side, float("nan")) / (hw + aw) if side in ("1", "2") else None
        if typ == "BTTS":
            yes = P[1:, 1:].sum()
            return {"yes": yes, "no": 1 - yes}.get(side)
        if typ == "CS":
            return float(P[score[0], score[1]]) if score and max(score) <= MAXG else None
        if line is None:
            return None
        if typ == "TOTAL":
            return p_eff(*self._goal_dist(scope, "total"), line, side)
        if typ == "TEAM_TOTAL":
            return p_eff(*self._goal_dist(scope, who), line, side) if who in ("home", "away") else None
        vals, pr = self._goal_dist(scope, "diff")                # home minus away
        if typ == "AH":                                         # side's margin + handicap > 0
            return p_eff(vals if side == "1" else -vals, pr, -line, "over") if side in ("1", "2") else None
        if typ == "EH":                                         # 3-way: home margin + line vs 0
            adj = vals + line
            return float({"1": pr[adj > 0].sum(), "X": pr[adj == 0].sum(), "2": pr[adj < 0].sum()}[side]) if side in ("1", "X", "2") else None
        return None


class LeagueModel:
    """Fitted from games strictly before `as_of`. game(home, away, referee) -> GameModel or None (unknown team)."""

    def __init__(self, M: pd.DataFrame, league, as_of, starts: dict | None = None, season=None):
        self.league, self.as_of = league, pd.Timestamp(as_of)
        A = M[(M.league == league) & (M.date < self.as_of)].dropna(subset=["hg", "ag"])
        self.n = len(A)
        start = {}
        if starts and season is not None and len(A):
            seen = set(A[A.season < season].home) | set(A[A.season < season].away)
            cur = M[(M.league == league) & (M.season == season)]
            prev_where = {t: lg for lg, g in M[M.season == season - 1].groupby("league") for t in set(g.home) | set(g.away)}
            for t in (set(cur.home) | set(cur.away)) - seen:
                s = starts.get((league, prev_where.get(t, "new"))) or starts.get((league, "any"))
                if s:
                    start[t] = s
        self.newcomers = set(start)
        self.dc = fit_dc(A, self.as_of, start)
        ht = A.dropna(subset=["hthg", "htag"]) if "hthg" in A else A.iloc[:0]
        tot = (ht.hg + ht.ag).sum() if len(ht) else 0
        self.h1 = float(((ht.hthg + ht.htag).sum() + 0.445 * 200) / (tot + 200)) if tot else 0.445
        self.A, self.cnt = A, {}
        recent = A[A.date >= self.as_of - pd.Timedelta(days=550)]
        for s in COUNT_STATS:
            hc, ac = f"h_{s}", f"a_{s}"
            if hc not in recent or recent[hc].notna().sum() < 40:
                continue
            R = recent.dropna(subset=[hc, ac])
            mh, ma = R[hc].mean(), R[ac].mean()
            tot_ = R[hc] + R[ac]
            self.cnt[s] = dict(R=R, mh=mh, ma=ma, phi_total=max(1.0, tot_.var() / tot_.mean()),
                               phi_team=max(1.0, (R[hc].var() / mh + R[ac].var() / ma) / 2))
        self.ref = {}
        if "cards" in self.cnt and "referee" in A:
            R = self.cnt["cards"]["R"]
            avg = (R.h_cards + R.a_cards).mean()
            for ref, g in R.groupby("referee"):
                if ref and isinstance(ref, str):
                    self.ref[ref] = float(((g.h_cards + g.a_cards).sum() + 15 * avg) / ((len(g) + 15) * avg))   # 15 pseudo-games

    def lambdas(self, home, away):
        d = self.dc
        if d is None or home not in d["att"] and home not in d["start"] or away not in d["att"] and away not in d["start"]:
            return None
        a = lambda t: d["att"].get(t, d["start"].get(t, (0, 0))[0])
        f = lambda t: d["dfn"].get(t, d["start"].get(t, (0, 0))[1])
        return math.exp(d["c"] + d["h"] + a(home) - f(away)), math.exp(d["c"] + a(away) - f(home))

    def game(self, home, away, referee=None) -> GameModel | None:
        lam = self.lambdas(home, away)
        if lam is None:
            return None
        counts = {}
        for s, c in self.cnt.items():
            R, hc, ac = c["R"], f"h_{s}", f"a_{s}"
            avg = (c["mh"] + c["ma"]) / 2

            def ratio(team, own: bool):
                hh, aa = R[R.home == team], R[R.away == team]
                tot = (hh[hc].sum() + aa[ac].sum()) if own else (hh[ac].sum() + aa[hc].sum())
                return _shrunk_ratio(tot, len(hh) + len(aa), avg)

            mh = c["mh"] * ratio(home, True) * ratio(away, False)
            ma = c["ma"] * ratio(away, True) * ratio(home, False)
            if s == "cards":
                k = self.ref.get(referee, 1.0)
                mh, ma = mh * k, ma * k
            counts[s] = dict(home=mh, away=ma, phi_team=c["phi_team"], phi_total=c["phi_total"])
        return GameModel(lam[0], lam[1], self.dc["rho"], self.h1, counts)


# ------------------------------------------------------------------ players
class PlayerModel:
    """Per-90 rates x expected minutes. PS: player_match_stats joined with the game date and the team's goals;
    EV: events (for penalties). Rates are shrunk toward the league rate with 900 minutes of prior."""

    def __init__(self, PS: pd.DataFrame, EV: pd.DataFrame | None, as_of):
        P = PS[PS.date < pd.Timestamp(as_of)].copy()
        self.ok = len(P) > 0
        if not self.ok:
            return
        for c in ("minutes", "goals", "shots", "shots_on_target", "yellow", "red"):
            P[c] = pd.to_numeric(P[c], errors="coerce").fillna(0) if c in P else 0.0
        P["cards"] = ((P.yellow > 0) | (P.red > 0)).astype(float)            # booked at least once
        pens = {}
        self.team_pen_rate, self.pen_share = {}, {}
        if EV is not None and len(EV):
            E = EV[EV.fixture_id.isin(P.fixture_id)]
            pk = E[E.type.isin(["penalty", "missed_penalty"])]
            sc = E[E.type == "penalty"]
            pens = sc.groupby(["fixture_id", "player_id"]).size().to_dict()
            games = P.groupby("team_id").fixture_id.nunique()
            for t, g in pk.groupby("team_id"):
                self.team_pen_rate[t] = len(sc[sc.team_id == t]) / max(games.get(t, 1), 1)
                last = g.tail(8)
                for pid, n in last.groupby("player_id").size().items():
                    self.pen_share[(t, pid)] = n / len(last)
        P["pen_goals"] = [pens.get((f, p), 0) for f, p in zip(P.fixture_id, P.player_id)]
        P["np_goals"] = (P.goals - P.pen_goals).clip(lower=0)
        tot = P[["minutes", "np_goals", "shots", "shots_on_target", "cards"]].sum()
        self.lg = {k: tot[k] / max(tot.minutes, 1) * 90 for k in ("np_goals", "shots", "shots_on_target", "cards")}
        g = P.groupby("player_id")[["minutes", "np_goals", "shots", "shots_on_target", "cards"]].sum()
        self.rate = {k: ((g[k] + self.lg[k] * 10) / (g.minutes + 900) * 90).to_dict() for k in self.lg}
        self.exp_min, self.team_of = {}, {}
        for t, T in P.groupby("team_id"):
            last = sorted(T.fixture_id.unique(), key=lambda f: T.date[T.fixture_id == f].iloc[0])[-6:]
            L = T[T.fixture_id.isin(last)]
            for pid, m in L.groupby("player_id").minutes.sum().items():
                self.exp_min[pid] = m / len(last)
                self.team_of[pid] = t
        self.team_gpg = (P.groupby(["team_id", "fixture_id"]).goals.sum().groupby("team_id").mean()).to_dict()

    def exp_goals(self, pid, team_lambda: float | None):
        """Expected goals of the player in this game, or None when the model has never seen him."""
        if not self.ok or pid not in self.exp_min:
            return None
        t = self.team_of[pid]
        scale = (team_lambda / self.team_gpg[t]) if team_lambda and self.team_gpg.get(t, 0) > 0.2 else 1.0
        m = self.exp_min[pid] / 90
        return m * (self.rate["np_goals"].get(pid, 0) * scale + self.team_pen_rate.get(t, 0) * self.pen_share.get((t, pid), 0))

    def p_anytime(self, pid, team_lambda=None, n: int = 1):
        x = self.exp_goals(pid, team_lambda)
        return None if x is None else float(poisson.sf(n - 1, x))

    def p_first(self, pid, team_lambda, match_lambda):
        """Share of the game's goals x chance there is a goal at all."""
        x = self.exp_goals(pid, team_lambda)
        if x is None or not match_lambda:
            return None
        return float(min(x / match_lambda, 1) * (1 - math.exp(-match_lambda)))

    def p_card(self, pid, line: float, side: str):
        """Booked at least once (line 0.5 only): 1 - exp(-rate per 90 x expected minutes / 90)."""
        if not self.ok or pid not in self.exp_min or abs(line - 0.5) > 1e-9:
            return None
        mu = self.rate["cards"].get(pid, 0) * self.exp_min[pid] / 90
        p = 1 - math.exp(-mu)
        return p if side == "over" else 1 - p

    def p_count(self, pid, stat: str, line: float, side: str):
        """stat: shots / shots_on_target. Poisson on rate x expected minutes."""
        if not self.ok or pid not in self.exp_min or stat not in self.rate:
            return None
        mu = self.rate[stat].get(pid, 0) * self.exp_min[pid] / 90
        k = np.arange(0, 30)
        return p_eff(k, poisson.pmf(k, mu), line, side)
