# READING-BAT: what the graded bat addressee test says

Post-hoc reading of `results/bat/bat_results.json` (job `bat-addressee-test-v1`, run 2026-11-22
11:39-12:54 UTC), graded by the rules of `BAT-PREREG.md` (registered 2026-11-20, before any
acoustic feature of this dataset was computed). MV2, Munim, Inc. (mv2@munimtech.com). Made with AI
agents; no human expert has checked it yet. Text CC BY 4.0.

To render the full fixed-format results table with the grader published before the result:
`python test_bat_results.py --render results/bat/bat_results.json RESULTS-BAT.md`.
`test_bat_grade.py` is a shorter regrade written after the result.

## Verdict

**ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING.** All six pre-registered predictions came true
(Brier score 0.225; the score is high because Q2 and Q3, which came true, had low priors of 0.35
and 0.25: we expected the addressee signal to shrink to nothing once days were blocked).

## Data that entered

- Prat et al. 2017 (Scientific Data 4:170143, figshare 3666502, CC0); both label-file MD5s match.
- 91,080 annotations; after the registered selection (treatments 9-10, adult female emitters,
  known addressee, aggressive contexts, one annotation per file) 6,914 calls; the class rule
  (addressees with at least 20 calls, at least 2 per emitter) kept **6,182 calls from 7
  emitters**. Emitters 203 and 223 did not pass the class rule. Every one of the 6,182 calls was
  read from the public archives and yielded features (no energy fallback).
- 211 emitter-days had calls to 2 or more addressees (4,116 calls). Mean chance level 0.386.

## Main numbers (T = mean balanced accuracy over the 7 emitters)

| test | how addressee labels are shuffled | B | T | null mean | p |
|---|---|---|---|---|---|
| naive (as published) | freely within emitter | 99 | 0.5258 | 0.3852 | 0.010 (floor) |
| **day (primary)** | within emitter x recording day | 199 | 0.5258 | 0.4468 | **0.005 (floor)** |
| ctxday | within emitter x context x day | 99 | 0.5258 | 0.4686 | 0.010 (floor) |
| lodo (train without the same day) | within emitter x day | 99 | 0.5044 | 0.4329 | 0.010 (floor) |

- **Day composition alone is worth +0.0616 balanced accuracy** (day-blocked null mean minus naive
  null mean). Of the published-style excess over the free null (0.5258 - 0.3852 = 0.1406), about
  44% can be produced by which addressees each bat called on which day. The rest beats all 199
  day-blocked shuffles.
- Calibration: in 4 worlds where labels follow day composition only, the naive test "detected"
  addressee information in **4 of 4**; the day-blocked test in **0 of 4**.
- Power: at the effect size matched to the observed naive accuracy (d = 0.04), the day-blocked
  test detected 5 of 5 synthetic positive worlds.

## Per emitter

| emitter | calls | addressees | BA (LOCO) | chance | p naive | p day | p ctxday | p lodo |
|---|---|---|---|---|---|---|---|---|
| 201 | 452 | 5 | 0.320 | 0.200 | 0.01 | 0.005 | 0.02 | 0.01 |
| 204 | 839 | 2 | 0.660 | 0.500 | 0.02 | 0.29 | 0.34 | 0.30 |
| 205 | 330 | 2 | 0.792 | 0.500 | 0.01 | 0.005 | 0.09 | 0.01 |
| 213 | 1,042 | 3 | 0.442 | 0.333 | 0.01 | 0.02 | 0.04 | 0.03 |
| 225 | 361 | 2 | 0.663 | 0.500 | 0.01 | 0.005 | 0.02 | 0.01 |
| 226 | 1,953 | 3 | 0.418 | 0.333 | 0.01 | 0.015 | 0.04 | 0.04 |
| 228 | 1,205 | 3 | 0.386 | 0.333 | 0.07 | 0.145 | 0.20 | 0.26 |

Five of seven emitters beat their own day-blocked null at p <= 0.05 (no multiplicity correction;
the pooled test is the registered primary). Emitter 204's naive signal (p 0.02) does not survive
day blocking (p 0.29): for her, the published-style test would have reported addressee
information that day composition explains.

## Reading

1. The 2016 headline that everyday Egyptian fruit bat calls carry information about the addressee
   **survives the confound we expected to kill it**. Within a recording day, calls from the same
   bat to different addressees still sound different enough to be told apart above chance, and
   the result also holds when models are trained without any call from the same day (lodo) and
   when shuffles keep context as well as day (ctxday). To our knowledge this is the first
   day-blocked, calibrated test of the addressee claim.
2. The published-style test is **not** safe on these data in general: it rejects in every
   day-composition-only world we made, and a sizeable part (+0.062 of 0.141) of its excess over
   chance is day composition. The claim holds because the real signal is larger than that bias,
   not because the original test controlled it.
3. Effects are modest: mean balanced accuracy 0.526 against 0.386 chance, and the addressee
   classes are very unbalanced (for example emitter 226: 1,734 of 1,953 calls to one bat).

## Limits

- Day is a coarse block. Within-day bouts (one fight, one position on the perch, one microphone
  geometry) can still differ by addressee; SURVIVES is necessary, not sufficient, for addressee
  coding. Nothing here shows that bats intend or understand an addressee signal.
- One dataset, labelled by its authors (2.1% reported addressee error). Our pipeline (5 ms hop,
  label-free UBM, our selection) differs from 2016, so the naive test is a reproduction in kind.
- p values are permutation p values at their floors (B = 99 or 199).
- Made with AI agents working in MV2's name; not yet checked by a human expert.
