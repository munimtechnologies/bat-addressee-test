# BAT-PREREG: do Egyptian fruit bat calls carry addressee information beyond the recording day?

Registered 2026-11-20, before job `bat-addressee-test-v1` runs and before any acoustic feature of
this dataset has been computed by us. The label-only description job (`bat-addressee-describe-v1`,
`describe_bats.py`) had not returned when this was written, so the design below was fixed without
seeing even the label structure; every data-dependent choice is a rule, not a number.
Code: `bat_addressee.py` (tests: `test_bat_addressee.py`). MV2, Munim, Inc. (mv2@munimtech.com).
Made with AI agents; not yet checked by a human expert. Text CC BY 4.0, code Apache-2.0.

## 1. The claim and why it needs this test

Prat, Taub & Yovel (2016, *Scientific Reports* 6:39419) recorded two colonies of Egyptian fruit
bats continuously for 75 days and reported that everyday aggressive calls carry information about
the emitter, the context and **the addressee**: e.g. emitter F6, balanced accuracy (BA) 50% against
33% chance (p < 0.01), all but one emitter significant (their Fig. S4), addressee sex 64% vs 50%.
Their classifier (64 MFCC per 20 ms window, GMM-UBM with 16 components) was evaluated by
**leave-one-vocalization-out** cross-validation, with significance from 100 label permutations
that are free across the whole data set. Calls from the same day, bout and recording conditions
can therefore sit on both sides of the split, and the null does not preserve which addressees an
emitter called on which days. If addressee composition varies from day to day (it must: bats move,
fight and feed in bouts) and recordings vary from day to day (microphones, positions, the bat's
state), a classifier can score above chance with no call-level addressee information at all.
This is the same confound we test for elephant "names" (`elephant-names-test`) and marmoset
"vocal labels" (`marmoset-labels-test`). We found no published re-analysis of the bat
**addressee** claim that blocks by day (web and Europe PMC searches, 2026-11-20). Closest prior
art: a search snippet describes a GitHub repository, NapolianAdmin/fruit-bat-call-clustering-audit,
as a confound-controlled audit of call *clustering* and *emitter* identification on this data set
(random vs blocked splits, channel x month and session-mean controls). It did not test addressee
information, and it returned 404 when we tried to read it on 2026-11-20. The use of per-session
controls for emitter identity there is credited; the addressee question is not covered by it.

## 2. Data

Prat, Taub, Pratt & Yovel (2017, *Scientific Data* 4:170143), figshare collection 3666502 (CC0):
`Annotations.csv` (MD5 7781a3725d6f2277de74da84b1383ce0), `FileInfo.csv`
(MD5 b27252490ac5618bd55a9038110809d5) and 30 zip archives of 250 kHz, 16-bit WAV files. Single
recordings are read by HTTP Range requests from the public archives; only features are kept.

## 3. Selection (rules fixed now)

- Treatments 9 and 10 (the two colonies recorded 2012-10-01 to 2012-12-15, Table 1 of the 2017
  paper; 22 bats over about 75 days, as in the 2016 paper).
- Emitters: the wild-caught adult females of those treatments (Table 2 of the 2017 paper):
  201, 203, 204, 205, 213, 223, 225, 226, 228. (The 2016 paper names its 7 females F1-F7 without
  IDs; we take every adult female and let the class rule decide.)
- Addressee: a known bat (ID > 0, not the emitter). Negative IDs (roles in doubt) and 0 (unknown)
  are excluded.
- Contexts: 3 feeding, 4 fighting, 10 threat-like (perch aggression), 11 general, 12 sleeping:
  the aggressive contexts of the 2016 addressee task. Mating protest (9) is excluded, as in 2016.
- Files carrying exactly one annotation (so file = call; multi-annotation files are dropped).
- Recording day: the calendar day of the file's recording time in FileInfo.csv.
- Class rule (applied to calls that have features): per emitter, addressees with >= 20 calls; an
  emitter enters with >= 2 such addressees.

## 4. Features

Voiced segments from FileInfo.csv (files without parsable segments: frames within 20 dB of the
file's loudest frame; their number is reported). 20 ms Hann frames, **5 ms hop** (the 2016 paper
used 1 ms; 5 ms keeps the job within budget), at most 64 frames per call (evenly spaced), 8192-point
FFT, 96 mel filters 1-125 kHz, log, DCT-II, **64 coefficients**. Per-recording-channel mean
subtraction, as in 2016 (label-free).

## 5. Classifier

GMM-UBM (Reynolds et al. 2000), as in 2016: a 16-component diagonal UBM (scikit-learn, seed fixed)
fitted on up to 100,000 frames sampled from all selected calls (label-free; 2016 used calls of
uncertain role, a documented deviation), MAP adaptation of the means with relevance factor 16,
score = mean frame log-likelihood under each addressee model, prediction = best score. Metric:
balanced accuracy per emitter; pooled statistic **T = mean BA over emitters**.
Cross-validation: **LOCO** (each call scored by models trained without it; as published) and
**LODO** (each call scored by models trained without any call of its emitter's same day).

## 6. Tests

| name | CV | null: addressee labels permuted | B | role |
|---|---|---|---|---|
| naive | LOCO | freely within emitter (as published) | 99 | reproduction |
| **day** | LOCO | **within emitter x recording day** | 199 | **primary** |
| ctxday | LOCO | within emitter x context x day | 99 | secondary |
| lodo | LODO | within emitter x day | 99 | secondary |

p = (1 + #{null T >= observed T}) / (B + 1). The day-blocked null keeps exactly which addressees
each emitter called on each day, so any accuracy that day composition plus day-level acoustics can
produce is in the null; the observed T beats it only if, **within a day**, calls to different
addressees sound different. Per-emitter p values are reported from the same permutations.

## 7. Calibration and power

- **Confound worlds (4):** per emitter and day, class probabilities = base rates x exp(1.5 z),
  z ~ N(0,1); labels drawn independently per call. Within a day these labels carry no information
  about the sound, so the day-blocked test must not detect them: CALIBRATION_FAILED if it does in
  2 or more of 4 (B = 49 each). How often the naive test "detects" them is reported: it measures
  how much a pure day-composition label can be decoded from these recordings.
- **Positive worlds (power):** real labels shuffled within day (real addressee information
  removed, day composition kept) plus a synthetic addressee effect: every frame of a call to class
  k shifted by d x SD x z_k, z_k ~ N(0, I), d in {0.005, 0.01, 0.02, 0.04, 0.08}. The matched d
  is the one whose naive LOCO T is closest to the observed naive T (ties to the smaller d); at the
  matched d, 5 fresh worlds are tested with the day-blocked test (B = 49). Power = detections of 5.

## 8. Decision rules (applied in this order)

Gates: DATA_MISMATCH (label MD5), LABELS_UNPARSED (a required column missing, or < 90% of
candidate calls with a parsable day), TOO_FEW_EMITTERS (< 3 emitters pass the class rule, before
or after audio), MAPPING_FAILED (< 80% of candidate calls yield features), NO_INFORMATIVE_BLOCKS
(< 10 emitter-day blocks with calls to >= 2 addressees). Then:

1. CALIBRATION_FAILED if the day-blocked test detects >= 2 of 4 confound worlds.
2. NO_NAIVE_SIGNAL if the naive test has p > 0.05 (the published-style signal is not reproduced
   on this pipeline; nothing further is claimed).
3. **ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING** if the primary day-blocked p <= 0.05.
4. **NOT_DETECTED_WITH_POWER** if not, and power >= 4 of 5.
5. UNDERPOWERED otherwise.

Secondary readings (no verdict change): ctxday and lodo p <= 0.05 or not; the day-composition gap
= mean null T (day) - mean null T (naive).

## 9. Predictions (priors fixed now; Brier score reported)

| id | prediction | prior |
|---|---|---|
| Q1 | naive test reproduces addressee information (p <= 0.05) | 0.85 |
| Q2 | primary day-blocked test detects it (p <= 0.05) | 0.35 |
| Q3 | leave-one-day-out test detects it (p <= 0.05) | 0.25 |
| Q4 | the naive test "detects" >= 2 of 4 confound worlds | 0.70 |
| Q5 | day composition alone is worth >= 0.02 BA (null-mean gap) | 0.70 |
| Q6 | the blocked test detects >= 4 of 5 effect-matched positive worlds | 0.60 |

## 10. What each outcome would mean

- SURVIVES: the first evidence that bat calls differ by addressee within a day, which no day-level
  confound can explain; the claim gains a confound-robust replication.
- NOT_DETECTED_WITH_POWER: the published addressee accuracy is explained by which addressees were
  called on which days, with a test shown able to detect an effect of the reported size.
- UNDERPOWERED: the data cannot decide at this size; reported as such.

## 11. Limits known in advance

One data set, annotated by its authors (reported 2.1% identity error). Our selection and 5 ms hop
differ from the 2016 pipeline, so the naive test is a reproduction in kind, not to the digit. Day
is a coarse block: within-day bouts remain a possible confound, so a SURVIVES verdict is a
necessary, not sufficient, condition for addressee coding. Permutation p values with B = 199 have
a floor of 0.005.

Amendments, if any, will be added as AMENDMENT sections below, dated, without editing the above.
