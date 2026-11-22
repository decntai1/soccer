# Soccer system: handoff to the next agent

You are taking over a running paper-betting ledger for six second divisions (Championship, League One, League Two,
2. Bundesliga, Serie B, Segunda), priced against vegas.hu boards. Read `SOCCER_MODEL_SPEC.md` rules (in
`SOCCER_SYSTEM.md`) before changing anything. Nothing here places bets. Every slip is paper.

## How the work is split

- **GitHub `decntai1/soccer` (public) only fetches data.** Its workflow runs at 05:23 and 23:41 UTC: football-data
  download, API-Football ingest (free plan), tables committed to `data/`. Its price/grade steps find no boards and
  skip. Do not ask the user to upload boards to GitHub: they decided boards come to the chat.
- **The chat does pricing, slips and grading.** The user sends one zip of vegas.hu captures per day
  (`<league>_<date>.zip`, folders `<YYYY-MM-DD>/*.net.json`; other sports may be mixed in and are skipped).
- **The ledger travels as `ledger_csv/`** (this pack; rebuild with `ledger_csv.py import`). It is the only copy of every
  priced row, slip and grade. Unread rows travel as a count per day and reason (`unread_summary`). To hand it on:
  `python3 ledger_csv.py export soccer.sqlite ledger_csv/` and zip it with the scripts (keep the pack well under 30 MB).

## Resume in 4 steps

```bash
pip install --break-system-packages pandas numpy scipy duckdb tabulate
unzip soccer_handoff.zip -d ~/soccer && cd ~/soccer
python3 ledger_csv.py import ledger_csv/ soccer.sqlite          # rebuild the ledger from the CSV tables
python3 soccer_system.py selftest | tail -1                     # must say PASS
git clone --depth 1 https://github.com/decntai1/soccer /tmp/repo   # fresh data/ from GitHub (works from the workspace)
export SOCCER_OUT=/tmp/repo/data SOCCER_DB=~/soccer/soccer.sqlite SOCCER_FD=/nonexistent
```

`SOCCER_FD` pointing nowhere makes the model use `data/soccer_<season>_played_fd.csv` (football-data history the
ingest commits). Then, per day:

```bash
python3 soccer_system.py price  --date D --boards DIR           # latest capture per game counts; re-run on recaptures
python3 soccer_system.py slips  --date D --targets 3 6 9 --legs 7
python3 soccer_system.py grade  --date D --boards DIR           # after GitHub's run has fetched the day's games
python3 soccer_system.py report                                  # numbers stay hidden until handcheck --confirm
```

Pull the repo again before every grade: results arrive with the 23:41 and 05:23 UTC runs.

## State at handoff

Written 2026-11-22 08:13 UTC.

| matchday | rows | read | graded |
|---|---|---|---|
| 2026-11-20 | 16806 | 1647 | 1597 |
| 2026-11-21 | 120433 | 13568 | 13224 |
| 2026-11-22 | 44916 | 3683 | 0 |

Slips: 26 built, 14 graded, 2 won.
Hand check (five results by hand): NOT DONE: report numbers stay hidden.

Open questions and next steps: see the last chat message of the previous agent, if the user passes it on.
