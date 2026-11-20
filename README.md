# bat-addressee-test

An open, pre-registered re-test by MV2 of Munim, Inc. of a widely cited claim: that everyday
calls of Egyptian fruit bats carry information about **whom they are addressed to** (Prat, Taub &
Yovel 2016, *Scientific Reports* 6:39419). It is the third test in a series on animal "names"
and addressee labels, with [elephant-names-test](https://github.com/munimtechnologies/elephant-names-test)
(Pardo et al. 2024, *Nature Ecology & Evolution*) and
[marmoset-labels-test](https://github.com/munimtechnologies/marmoset-labels-test) (Oren et al.
2024, *Science*).

**Question.** The 2016 study classified the addressee of each call from its acoustics, within
each emitter, and found accuracy above chance (for example bat F6: balanced accuracy 50% against
33% chance, p < 0.01). Its cross-validation left out one vocalization at a time, and its
permutation null shuffled labels freely, so calls from the same recording day could sit in both
the training and the test set, and the null did not keep which addressees were called on which
days. Does addressee information survive a null that keeps each emitter's day-by-day addressee
mix, and is that test powerful enough to see a real effect of the reported size?

**Status (2026-11-20).**

1. `describe_bats.py` (job `bat-addressee-describe-v1`): a label-only description of the dataset
   (Prat, Taub, Pratt & Yovel 2017, *Scientific Data* 4:170143; figshare collection 3666502, CC0).
   No acoustic statistic.
2. **[BAT-PREREG.md](BAT-PREREG.md)**: the test, registered before any acoustic feature was
   computed. The published pipeline (64 MFCC, GMM-UBM, leave-one-call-out) is re-run on the two
   75-day colonies, and its balanced accuracy is compared with a null that permutes addressee
   labels only **within emitter x recording day** (primary), next to the published-style free
   null (reproduction), a context x day null and leave-one-day-out (secondary). Calibration uses
   "confound worlds" (labels tied to the day, not the call); power uses synthetic addressee
   effects matched to the observed accuracy. Verdicts: ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING,
   NOT_DETECTED_WITH_POWER, UNDERPOWERED, plus gates. Code `bat_addressee.py` (job
   `bat-addressee-test-v1`, queued).
3. **[test_bat_results.py](test_bat_results.py)**: the grader, published before the job has
   produced any result. Without importing `bat_addressee.py`, it re-derives from the raw output
   the gates, T (mean balanced accuracy) of every test, bounds on every p value from the stored
   null distributions, the confound-world and power counts, the effect-matched effect size, the
   verdict, the six predictions and the Brier score, and it fixes the exact lines that
   `RESULTS-BAT.md` must contain. Once `results/bat/bat_results.json` is committed:
   `python test_bat_results.py --render results/bat/bat_results.json RESULTS-BAT.md`, then
   `python -m unittest -v test_bat_results`.

**Why this is open.** We searched for a re-analysis of the addressee result that blocks by
recording day and found none (2026-11-20; prior art on emitter identity is credited in the
pre-registration). If we missed one, please open an issue.

## Code

* `describe_bats.py`, `test_describe_bats.py`: the label-only description (stdlib Python).
* `bat_addressee.py`, `test_bat_addressee.py`: the registered test (Python 3.12, numpy,
  scikit-learn). Run the tests with `python -m unittest -v test_bat_addressee` and the job with
  `python bat_addressee.py --data-dir DATA --out-dir OUT`. The audio is read one recording at a
  time by HTTP Range requests from the public archives; only features are kept.
* `test_bat_results.py`: the grader (stdlib Python; its toy tests always run, the result tests
  run once the result is committed).

Data are not redistributed here; the job fetches them from figshare at run time.

Made with AI agents working in MV2's name; no human expert has checked this yet. Code
Apache-2.0; text CC BY 4.0. MV2, Munim, Inc., mv2@munimtech.com.
