# Changelog: soccer system (decntai1/soccer)

## 2026-11-24 slips: no legs football-data cannot grade
- First live boards with the API gone (28 games, 24-25 Nov): all matched from fd_fixtures.csv (negative ids), all 132
  teams mapped, priced as before.
- Two slips held a first-half corners leg (book-9x 24 Nov, book-7legs 25 Nov): football-data has no statistics by half,
  so those slips could never have been graded. New gradable(): on a game known only from football-data, leg_pool and
  statedge_pool drop player markets, statistics by half and booking points. Removed 7 such legs on 24 Nov, 2 on 25 Nov.
- selftest: 61 checks (new check fails with the filter off).

## 2026-11-23 fix: football-data fixtures file read as an API table
- data/soccer_fd_fixtures.csv matched the pattern soccer_*_fixtures.csv: the ingest's shrink guard crashed on it
  (ValueError: Usecols ... has_events, season) and load_tables would have mixed it into the fixtures.
- Renamed to data/fd_fixtures.csv; both patterns now only match a four-digit season (soccer_2026_fixtures.csv) or
  soccer_all_*. Delete data/soccer_fd_fixtures.csv from the repo (harmless if left: no longer matched).
- selftest: 60 checks (new: a stale fd fixtures file in data/ during a blocked ingest; fails on the old code).

## 2026-11-23 ingest: rejected key is a clean stop
- Since 2026-11-23 the API answers {"token": "Error/Missing application key"} (key no longer valid after the suspension). The ingest now stops cleanly on it (API KEY REJECTED) like on the suspension, so runs stay green and grading uses football-data.co.uk.
- selftest: 59 checks.

## 2026-11-22 workflow: commits signed as github-actions[bot]
- The commit step used soccer-bot@users.noreply.github.com, which GitHub credits to whoever owns the account "soccer-bot". Now GitHub's own Actions bot identity.

## 2026-11-22 football-data.co.uk fallback (API-Football account suspended)
- The API answered every call with "Your account is suspended". soccer_ingest.py now treats that as a clean stop
  (prints API ACCOUNT BLOCKED, builds the tables from what it has, exits 0) instead of a traceback.
- soccer_system.load_tables fills from football-data.co.uk every game the API has no result for, from the first day
  the API holds onwards: score, half-time score and full-time team statistics (shots, on target, corners, fouls,
  yellows, reds). Games the API never listed get a negative fixture id: played ones from the season files, coming ones
  from fixtures.csv, so boards are still matched and priced. The API wins whenever it has the game.
- Graded from football-data: 1X2, DC, DNB, BTTS, CS, goal totals and team totals (FT, H1, H2), AH, EH, corners, cards,
  shots and shots on target over/under and 1X2, corner handicap (all full time). Cards = yellows + reds; each red widens
  the count by one either way (second-yellow rules) and a line that could flip is held back.
- Not graded from football-data (kept open): player markets, statistics by half, booking points, goal timing.
- soccer_bandit.py download also fetches fixtures.csv and writes data/soccer_fd_fixtures.csv (our six divisions).
- Pricing unchanged (old and new code price Sunday's boards identically): MODEL_VERSION stays soccer-0.2.
- selftest: 58 checks (5 new: blocked ingest, fixtures.csv matching, grading from football-data, player/half rows
  open, card rule).

## 2026-11-22 workflow: ingest log committed
- Ingest and lineups output is written to results/ingest-<date>_<HHMM>.txt and committed every run, failed or not, so a failure can be read from the repo.

## soccer-0.2, 2026-11-21: statistics markets and team profiles
- New markets read, priced and graded: corners 1X2, cards 1X2, shots 1X2 and shots-on-target 1X2 (which team has
  more; "Szöglet 1X2", "Büntetőlapok 1X2", "1X2 kapura / kaput eltaláló lövések száma"), and player booked ("Lapok",
  line 0.5). Still not read: race to N corners (no corner times in the data), player booked incl. substitute.
- Cards 1X2 is graded only when the result is the same under every unconfirmed card rule (second yellow, bench).
- Model: the 1X2 statistics markets and the corner handicap priced from the two teams' negative-binomial counts;
  fouls fitted per team (football-data HF/AF, API team stats); player booked from cards per 90 x expected minutes.
- `teams`: a profile per team this season (corners, cards, shots, on target, fouls; for and against; home, away,
  last 6), written to data/soccer_<season>_team_profiles.csv.
- `statedge` slip strategy: statistics legs only, priced 1.5-2.1, where the team model rates the leg above the book;
  ranked by the model's view of value. Stated chance stays the book's (weight 0 until the record says otherwise).
  Built for every target and for `--legs 7`; graded like all slips; bandit 2 has it as an arm.
- Report: statistics markets book vs model vs happened, team over-lines by team, the statedge legs' record.
- The book's margin on the statistics markets (real boards, 21-22 Nov): 8-10% on totals, 12-16% on the 1X2 ones.
- MODEL_VERSION soccer-0.2: Saturday and Sunday re-priced; Friday stays graded under 0.1 and out of the weights.
- Selftest: 53 checks.

## soccer-0.1, 2026-11-21: first real matchday graded
- Friday 20 Nov: all 5 games fetched by the 23:41 UTC run (API-Football free plan) with events, lineups, team and
  player statistics; the three sanity checks passed on all 5 (goals = score, player goals + own goals = team goals,
  half-time score = goals to minute 45). First proof that the free plan serves our six leagues' match detail.
- 1,597 of 1,647 read rows graded. Not graded: 50 first-half corner lines (no corner counts by half in the data).
- Both Friday 3x paper slips lost.
- Hand check of five results against match reports: not done yet. Report numbers stay hidden until it is.
- Workflow: the repo's `soccer.yml` was overwritten with the NCAAB workflow on 2026-11-21 10:43 UTC by mistake
  and restored the same day.

## 2026-11-20: board reader fitted to real vegas.hu boards
- Fitted on 50 real boards (all six leagues). All 50 games matched to their API fixtures.
- Fixed: every market was listed twice on the page under one id (now read once). The plus sign in "(+0,75)" and
  "2+ gólt szerez" was taken for a combination bet. A board for another sport (NBA) in the folder is skipped.
- Real wording: "Hendikep" (the data's line is the home line on both sides), "Hendikep 1X2" (head start a:b),
  "Szöglet hendikep" (corner handicap), "Pontos végeredmény", "Büntetőlapok száma", shots "(Rendes játékidő)",
  "Gólszerző" with Bármikor / 1. gólszerző / Utolsó per player, "Hány gólt szerez?", "<team> gólt szerez".
- Never read: combination markets (93% of a board's rows), odd/even, ranges, offsides, red cards, last scorer,
  first-or-last scorer, super-sub scorer.
- Team matching: initials ("MK Dons" = Milton Keynes Dons) and "a team plays once a day" (one certain name + date).
- Model history: older API seasons no longer leak in (2,836 stale 2024 games were used twice). The model sits
  0-5 points from the book on average by league; its weight stays 0 until held-out games give it t > 2.
- Slips: `--legs N` builds slips of exactly N legs; legs print their kick-off time. Workflow builds 3x / 6x / 9x
  and 7-leg paper slips.
- Selftest: 50 checks, 43 of them wordings copied from real boards.

## 2026-11-20: free-plan ingest
- Tested with the user's key: the free plan refuses `season=2025/2026` and `ids=`, and allows `date=` only for
  D-1 / D / D+1. `fixtures?id=` for a current-season game in that window works and carries all four detail blocks.
- Ingest per run: 3 date calls, then 1 call per finished game of our six leagues (about 55 on a busy Saturday).
  A game whose day leaves the window unfetched is lost for good and listed as LOST. Two runs a day (05:23 and
  23:41 UTC) so one failure loses nothing.
- History before the first run comes from football-data.co.uk: 877 games this season, all six leagues with shots,
  corners, cards and half-time scores (referee only England). Written to `data/soccer_2026_played_fd.csv` each run.
- football-data team names are mapped to API team ids by results where both hold a game and by name otherwise.
- League ids 40/41/42/79/136/141 confirmed in live responses.

## 2026-11-19: first build from SOCCER_MODEL_SPEC.md
- `soccer_ingest.py` (API-Football), `soccer_model.py` (Dixon-Coles, negative-binomial counts, referee factor,
  player rates), `soccer_system.py` (board reader, odds-ratio de-margining, one-way cap 1/odds/1.03, model weight 0
  until t > 2 over games, settlement incl. quarter lines and void for 0 minutes, slips, four-vote filter, bandit 2,
  report), `soccer_bandit.py` (bandit 1 backtest on football-data closing prices), selftest on a simulated world.
- Work split decided with the user: GitHub fetches data only; boards are sent to the chat as one zip per day and
  pricing, slips and grading run there; the ledger travels as `soccer_handoff.zip`.
