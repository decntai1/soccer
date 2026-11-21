# Changelog

## ncaab-0.10 (2026-11-20)
- `lines`: betting lines (spread, total, moneylines, opening spread and total, by provider) from
  CollegeBasketballData.com into `lines/<YYYY-MM>.csv`, one API call per season month; the current and previous
  month are refetched each run, off-season months skipped. About 6 calls for the 2025-26 backfill, then about 2 a
  day. The key comes only from the CBBD_API_KEY environment variable (a GitHub secret in the workflow) and is never
  printed or written. Tested against a stand-in API (no key, wrong key, first run, rerun); the real API is untested.
  Purpose: test model changes against last season's closing lines instead of waiting weeks for vegas.hu nights.
- Workflow: a `lines` step after the ESPN scrape. Without the secret it prints a note and fetches nothing.

## ncaab-0.9 (2026-11-19) - first real night graded, one settlement bug fixed
- (later the same day) When the board lists home and away the other way round from ESPN on a non-neutral court,
  the model now gives home court to ESPN's home team (none on 2026-11-19: the one flipped game was neutral). Grading is unaffected:
  it always uses the board's own home/away. 9 more names in results_aliases.csv (30 in all).
- **Bug: game-total odd/even was settled on the away team's points.** On pandas 3 a missing `subject` comes back
  from the ledger as NaN, which counts as true, so `if not r.subject` picked the team branch. Hit only ODD_EVEN
  markets for the whole game and the first half (180 bets on 2026-11-18, all regraded). Found by re-settling the
  night by hand. selftest now has four odd/even cases (they fail on 0.8 under pandas 3).
- Race settlement also guards against a NaN line.
- First real night, 2026-11-18: 59 games, 11,654 bets graded, every row re-settled by independent code: 0
  disagreements after the fix; 825 play-by-play race entries checked against the half scores: 0 contradictions.
  Race lines exist for 40 of 59 games; the other 19 have no usable play-by-play and their races stay ungraded.

## ncaab-0.8 (2026-11-18) - review of 0.7, slips merged back
Review of 0.7: every pricing and settlement function is unchanged from 0.5/0.6 (compared function by function);
on the 59 real NCAAB boards all 14,443 rows price identically; selftest PASS on pandas 3.0 and 2.2; a simulated
second capture 7.5 h later dropped the 29 games already tipped and pooled 29 games from the first capture and 30
from the second; a 0.6 ledger opens, grades and reports under the new code. No bug found in 0.7. Added on top:
- `slips` is back (0.6), now picks the latest capture by time, and writes `slips/<night>.csv` next to the ledger;
  a ledger rebuilt from nothing reloads them. `sync` makes the ten slips once per night, from its first priced
  capture. The ten slips of 2026-11-18 come out identical to the ones made under 0.6.
- `pack` / `price` warn when the capture files' own time does not match the HHMM in their names (using the
  browser's `timezoneOffset` stored in the capture): copied files would give a wrong capture time, and with it
  wrongly dropped or wrongly kept games. Zips that carry UTC file times (the 2026-11-18 one does) are fine.
- `picks` no longer depends on a text `idxmax`; report header says "last capture before each game's tip".
- Slips with no model price no longer crash the slips table (found by the new selftest case).
- selftest: 29 checks.
- Still unproven, unchanged: ESPN play-by-play on GitHub, and settlement on real NCAAB boards (no night graded yet).

## ncaab-0.7 (2026-11-18) - review fixes, built on 0.5
(Patched from 0.5; the `slips` command of 0.6 was merged back in 0.8.)
- **Last capture is now chosen per game, not per night.** College games tip from noon to 11pm ET; a game that had
  started before the evening capture was missing from it and was dropped from the pooled ledger entirely (tested).
- **Captures are ordered by when they were taken (`captured_utc`), not by label.** A `0030` capture used to sort
  before `2300`, so the morning prices would have counted as the close (tested). `pack` writes the capture time
  (the .json files' own time, or `--captured-at`); `price --captured-at` overrides. Old ledgers and old packs still
  work and fall back to label order.
- **Each game's tip time is stored** (`start`, from the board). Games that had already tipped at capture time are
  not stored: in-play prices know the score.
- **`sync` grades after pricing and without the 4-day window** for nights whose results file is on disk. Before,
  a ledger rebuilt from nothing graded nothing on the first run and never graded nights older than 4 days (tested).
- **`sync` never overwrites an existing results or slate file.** Its own refetch ran without play-by-play and would
  have dropped the RACE lines wherever ESPN is reachable (read from the code; ESPN is blocked where this was tested).
- Rows priced with an assumed margin (unpaired selections) are now really kept out of the model-vs-book Brier table
  and the weight refit, as the spec always said.
- A team name containing " & " (William & Mary) no longer makes its team totals UNPARSED.
- Bandit table: `clv` and `clv_games` per arm (how far the book moved the arm's bets between a game's first and
  last capture). The gate "positive CLV for that arm" can now be read off the report.
- Checked, NOT changed: "the model underrates blowouts" (0.4) is true against the book's lines only. Against
  results, 2,691 walk-forward games (75 game days): actual margin = 0.96 x predicted; games predicted at 20+ points
  (avg 29.2) came in at 28.7. A slope correction fitted on the first half made the second half worse (margin error
  10.19 -> 10.33), so none was added.
- Tests: selftest 25 checks PASS (7 new). On real boards (NBA 2026-11-16, 6,704 rows) every row prices and grades
  identically to 0.5; a real two-capture night (2026-11-17) fills the open-vs-close table and `clv`.
- Still unproven: ESPN play-by-play on GitHub (no RACE line in any results file yet; first chance is the next
  12:23 UTC run), and settlement on real NCAAB boards (no night graded yet).

## ncaab-0.5 (2026-11-18)
- **Race markets are typed and graded** (`RACE`): "Ki ér el elsőként 20 pontot" (full game, 2-way) and
  "1.félidő - verseny a(z) N pontért" (first half, 3-way with "Egyik sem" = neither). A half race with only two
  live selections is priced with an assumed margin and never pooled.
- Settlement needs the order of scoring, so `results --pbp` reads ESPN's play-by-play for each game and writes a
  `RACE 5:H1 10:A1 ...` line (who reached each multiple of 5 first, and in which period). Written only when the
  play-by-play is monotone and ends on the final score. The workflow uses `--pbp` for the last 3 days only.
  Play-by-play field layout read off the live feed (one game); the fetch itself is untested until the job runs.
- v0 race model (unfitted, weight 0): chance of leading after the share of the game it takes to reach N points;
  half races add the chance that neither gets there. Against the book on 59 boards: correlation 0.93-0.98, mean
  gap 5-6 points.
- Bandit: 14 new arms fixed before any NCAAB game was graded - handicap (line nearest to even: favourite /
  underdog; first half: favourite / underdog; model likes / dislikes by 5+ points) and race (favourite / underdog;
  to 25 or less; to 30 or more; neither; model likes by 5+ points).
- The book's cut, 59 boards: to win 4.3%, first-half draw-no-bet 4.4%, handicap 7.5% over the ladder, totals 7.9%,
  first to 20 7.5%, first-half races **10.2%** (the most expensive market on the board).

## ncaab-0.4 (2026-11-18)
First real NCAAB boards (59 games, capture 2026-11-18 11:31). Found and fixed:
- **Silent wrong type:** `1.félidő - verseny a(z) N pontért` (race to N points) with two selections was read as a
  first-half draw-no-bet and would have been graded wrongly. Now UNPARSED.
- **Duplicate markets:** 48 of 59 boards list every market twice; the doubled rows broke the two-way pairing, so
  only 10 games had a usable to-win price. Markets are now de-duplicated by id (49 games with a to-win price).
- **Names:** 20 of 59 games did not match ESPN (`UConn` vs `Connecticut`, ...). `results_aliases.csv` (espn,board)
  is now applied when results / slate files are read, so already-scraped files are covered. 21 names added.
- **Home/away order:** on neutral courts the board and ESPN can disagree (2 of 59). Results, slate and grading now
  match a game by its two teams, and grading uses the board's home/away.
- `sync` fits once before pricing so the fitted SDs are used; `picks` verdict counts only odds 1.15-6.0.
- Check against the book's own lines, 59 games: model margin vs middle handicap correlation 0.96, mean gap 3.7
  points, same favourite 90%; totals correlation 0.87, mean gap 4.1. The model underrates blowouts (28-point book
  favourites come out at 19-22).
- Still UNPARSED by design: winning-margin bands, double result, races, "will there be overtime", "which half has
  more points", player multis, team stat totals (threes, blocks, assists, steals).

## ncaab-0.3 (2026-11-18)
- First real data: 380 days from ESPN via the GitHub job, 7,205 games (56 skipped by the half-range rule).
- `fit` defaults changed from ridge 3 / half-life 60 to **ridge 0.15 / half-life 150**. The old values shrank every
  team to average in November (last season weighed almost nothing) and pushed the difference into "home court"
  (+14.8 points, nonsense). Walk-forward on real games, old -> new: November 2026 margin error 13.7 -> 10.6 points,
  Jan-Feb 2026 9.9 -> 8.9, home court now 3.1-3.7. Chosen on the same games it is reported on: mildly optimistic.
- Half-range sanity rule widened from 15-80 to 5-95 points: real halves of 10-14 and 60 were being thrown out.
- Workflow is scrape-only (results + slate); pricing and grading run from a clone.

## ncaab-0.2 (2026-11-18)
- `fit`: own team ratings from the results files (ridge, time-weighted, neutral courts, overtime scaled), with a
  walk-forward test; writes ratings/teams.csv and ratings/model.json (fitted home court and SDs replace the guesses).
- `picks`: predicted winner per game with odds, book / model / blended chance, break-even, expected return.
- `slate`: ESPN card of a date with neutral courts; `price --slate` drops home court for those games.
- `results --to`: backfill a date range. A game whose numbers do not add up is written as `# SKIPPED`, so one bad
  box score no longer blocks a night. `NEUTRAL` line added to the results format.
- `pack`: boards folder -> one small .rows.csv.gz; `price --boards` accepts it.
- `sync` + .github/workflows/ncaab.yml: fetch, grade, price, refit, report in one run.
- Teams with fewer than 4 games behind their rating are not priced by the model.
- Fix: the weight of a family not included in the latest refit fell back to its starting value.

## ncaab-0.1 (2026-11-18)
- First version: board reader, de-vig, v0 model, grading, report, bandit, selftest.
