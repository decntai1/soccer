# Changelog: soccer system (decntai1/soccer)

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
