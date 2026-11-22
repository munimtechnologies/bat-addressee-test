# SPDX-License-Identifier: Apache-2.0
# test_bat_grade.py: a short, independent regrade of the bat addressee job output
# (results/bat/bat_results.json, job bat-addressee-test-v1, registered in BAT-PREREG.md).
# Written AFTER the result was read (2026-11-22), as a cross-check of the fuller grader
# test_bat_results.py that was published before the result. It re-derives from the stored
# values only: the verdict (BAT-PREREG.md section 8, in order), predictions Q1-Q6, the Brier
# score, the effect-matched d, the per-emitter sums, and bounds on the naive, day and lodo
# p values from the stored null distributions. Stdlib only.
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "bat", "bat_results.json")
ALPHA = 0.05
PRIORS = [("Q1", 0.85), ("Q2", 0.35), ("Q3", 0.25), ("Q4", 0.70), ("Q5", 0.70), ("Q6", 0.60)]
TOL = 1e-6  # stored null values are rounded to 6 decimals


def mean(v):
    return sum(v) / float(len(v))


def rej(p):
    return p <= ALPHA


def p_bounds(null, t, b):
    k_lo = sum(1 for v in null if v >= t + TOL)
    k_hi = sum(1 for v in null if v >= t - TOL)
    return (1 + k_lo) / (b + 1.0), (1 + k_hi) / (b + 1.0)


def regrade(r):
    t = r["tests"]
    fp = sum(1 for w in r["confound"]["worlds"] if rej(w["blocked_p"]))
    nd = sum(1 for w in r["confound"]["worlds"] if rej(w["naive_p"]))
    det = sum(1 for x in r["power"]["reps"] if rej(x["p"]))
    if fp >= 2:
        verdict = "CALIBRATION_FAILED"
    elif not rej(t["naive"]["p"]):
        verdict = "NO_NAIVE_SIGNAL"
    elif rej(t["day"]["p"]):
        verdict = "ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING"
    elif det >= 4:
        verdict = "NOT_DETECTED_WITH_POWER"
    else:
        verdict = "UNDERPOWERED"
    drop = t["day"]["null_mean"] - t["naive"]["null_mean"]
    came = {"Q1": rej(t["naive"]["p"]), "Q2": rej(t["day"]["p"]), "Q3": rej(t["lodo"]["p"]),
            "Q4": nd >= 2, "Q5": drop >= 0.02, "Q6": det >= 4}
    brier = mean([(p - (1.0 if came[q] else 0.0)) ** 2 for q, p in PRIORS])
    t_naive = mean(t["naive"]["per_emitter"])
    grid = r["power"]["grid"]
    best = 0
    for i in range(1, len(grid)):
        if abs(grid[i]["T_naive"] - t_naive) < abs(grid[best]["T_naive"] - t_naive):
            best = i
    return {"verdict": verdict, "fp": fp, "nd": nd, "det": det, "drop": drop, "came": came,
            "brier": brier, "matched_d": grid[best]["d"]}


@unittest.skipUnless(os.path.exists(RES), "results/bat/bat_results.json not present")
class TestBatGrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RES, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.g = regrade(cls.r)

    def test_verdict(self):
        self.assertEqual(self.g["verdict"], "ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING")
        self.assertEqual(self.r["verdict"], self.g["verdict"])
        self.assertEqual((self.g["fp"], self.g["nd"], self.g["det"]), (0, 4, 5))

    def test_predictions_and_brier(self):
        got = {x["id"]: x["came_true"] for x in self.r["predictions"]}
        self.assertEqual(got, self.g["came"])
        self.assertTrue(all(self.g["came"].values()))
        # (.15^2 + .65^2 + .75^2 + .3^2 + .3^2 + .4^2) / 6 = 1.3475 / 6
        self.assertAlmostEqual(self.g["brier"], 1.3475 / 6, places=12)
        self.assertAlmostEqual(self.r["brier"], self.g["brier"], places=12)

    def test_drop_and_matched_d(self):
        self.assertAlmostEqual(self.r["day_composition_drop"], self.g["drop"], places=12)
        self.assertGreater(self.g["drop"], 0.06)
        self.assertLess(self.g["drop"], 0.062)
        self.assertEqual(self.g["matched_d"], 0.04)
        self.assertEqual(self.r["power"]["matched_d"], 0.04)

    def test_T_and_counts(self):
        t = self.r["tests"]
        for k in ("naive", "day", "ctxday"):
            self.assertAlmostEqual(t[k]["T"], mean(t[k]["per_emitter"]), places=12)
        self.assertAlmostEqual(t["naive"]["T"], 0.52578786, places=6)
        self.assertAlmostEqual(t["lodo"]["T"], mean(t["lodo"]["per_emitter"]), places=12)
        em = self.r["emitters"]
        self.assertEqual(len(em), self.r["final"]["emitters"])
        self.assertEqual(len(em), 7)
        for e in em:
            self.assertEqual(sum(e["class_counts"]), e["calls"])
            self.assertTrue(all(c >= 20 for c in e["class_counts"]))
            self.assertAlmostEqual(e["chance"], 1.0 / len(e["classes"]), places=12)
        self.assertEqual(sum(e["calls"] for e in em), self.r["final"]["calls"])
        self.assertEqual(sum(e["informative_days"] for e in em),
                         self.r["final"]["informative_day_blocks"])
        self.assertAlmostEqual(self.r["chance_mean"], mean([e["chance"] for e in em]),
                               places=12)

    def test_p_within_null(self):
        t = self.r["tests"]
        for k in ("naive", "day", "ctxday", "lodo"):
            b = t[k]["B"]
            self.assertEqual(len(t[k]["null_T"]), b)
            lo, hi = p_bounds(t[k]["null_T"], t[k]["T"], b)
            self.assertTrue(lo - 1e-12 <= t[k]["p"] <= hi + 1e-12, k)
            self.assertAlmostEqual(t[k]["null_mean"], mean(t[k]["null_T"]), delta=TOL)
        # the observed T beats every one of the 199 day-blocked null worlds
        self.assertEqual(t["day"]["p"], 0.005)
        self.assertLess(max(t["day"]["null_T"]), t["day"]["T"])


if __name__ == "__main__":
    unittest.main()
