#!/usr/bin/env python3
"""ledger_csv.py: move the ledger between chats as plain CSV tables (small, readable, any upload accepts them).

    python ledger_csv.py export soccer.sqlite ledger_csv/     # every table to CSV; unread rows become a count per reason
    python ledger_csv.py import ledger_csv/ soccer.sqlite     # rebuild the SQLite ledger from those CSVs

Unread rows (combination markets and the rest, about 90% of a board) carry no price and no grade; only their count
per day and reason travels (table unread_summary). Everything priced, graded or used in a slip travels in full.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

TABLES = ("predictions", "slips", "slip_legs", "weights", "calib", "meta", "unread_summary")


def export(db: str, out: str) -> None:
    con, o = sqlite3.connect(db), Path(out)
    o.mkdir(parents=True, exist_ok=True)
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for t in TABLES:
        if t == "unread_summary":
            df = pd.read_sql("SELECT matchday, reason, count(*) AS n FROM predictions WHERE type IS NULL GROUP BY matchday, reason", con)
            if "unread_summary" in have:                      # a ledger imported earlier: keep the counts it brought
                old = pd.read_sql("SELECT * FROM unread_summary", con)
                df = pd.concat([old, df]).groupby(["matchday", "reason"], as_index=False).n.sum()
        elif t in have:
            q = "SELECT * FROM predictions WHERE type IS NOT NULL" if t == "predictions" else f"SELECT * FROM {t}"
            df = pd.read_sql(q, con)
        else:
            continue
        df.to_csv(o / f"{t}.csv", index=False)
        print(f"  {t}: {len(df)} rows")


def load(src: str, db: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import soccer_system as S                                # creates the tables with the right schema
    con = S.connect(db)
    for t in TABLES:
        p = Path(src) / f"{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if t == "unread_summary":
            df.to_sql(t, con, if_exists="replace", index=False)
        else:
            con.execute(f"DELETE FROM {t}")
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            df = df[[c for c in cols if c in df.columns]]
            df.astype(object).where(df.notna(), None).to_sql(t, con, if_exists="append", index=False)
        print(f"  {t}: {len(df)} rows")
    con.commit()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("export", "import"):
        sys.exit(__doc__)
    (export if sys.argv[1] == "export" else load)(sys.argv[2], sys.argv[3])
