# SPDX-License-Identifier: Apache-2.0
# Unit tests for bat_addressee.py. The label, zip and verdict tests need only the standard
# library; the audio and classifier tests run when numpy is installed (inside the job).
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import io
import struct
import unittest
import wave
import zipfile

import bat_addressee as ba

try:
    import numpy as np
except ImportError:
    np = None

AH = ["File ID", "Emitter", "Addressee", "Context", "Emitter pre-vocalization action",
      "Addressee pre-vocalization action", "Emitter post-vocalization action",
      "Addressee post-vocalization action", "Start sample", "End sample"]
FH = ["File ID", "Treatment ID", "File name", "File folder", "Recording channel",
      "Recording time", "Voice start samples", "Voice end samples"]


def arow(k, e, a, c):
    return [k, e, a, c, "0", "0", "0", "0", "0", "100"]


def frow(k, t, name, ch, tm, vs, ve):
    return [k, t, name, "files101", ch, tm, vs, ve]


class TestLabels(unittest.TestCase):
    def test_columns(self):
        c = ba.label_columns(AH, FH)
        self.assertEqual((c["key_a"], c["key_f"]), ("File ID", "File ID"))
        self.assertEqual(c["emitter"], "Emitter")
        self.assertEqual(c["addressee"], "Addressee")
        self.assertEqual(c["context"], "Context")
        self.assertEqual(c["treatment"], "Treatment ID")
        self.assertEqual(c["name"], "File name")
        self.assertEqual(c["channel"], "Recording channel")
        self.assertEqual(c["time"], "Recording time")
        self.assertEqual(c["vstart"], "Voice start samples")
        self.assertEqual(c["vend"], "Voice end samples")

    def test_parse_day(self):
        self.assertEqual(ba.parse_day("2012-10-05 01:02:03"), "2012-10-05")
        self.assertEqual(ba.parse_day("05-Oct-2012 10:00:00"), "2012-10-05")
        self.assertEqual(ba.parse_day("735147.25"), "dn735147")
        self.assertEqual(ba.parse_day("5/10/2012 10:00"), "05/10/2012")
        self.assertIsNone(ba.parse_day(""))
        self.assertIsNone(ba.parse_day("abc"))

    def test_numbers(self):
        self.assertEqual(ba.to_int("201"), 201)
        self.assertEqual(ba.to_int("-201"), -201)
        self.assertEqual(ba.to_int("201.0"), 201)
        self.assertEqual(ba.to_int("Treatment 9"), 9)
        self.assertIsNone(ba.to_int(""))
        self.assertIsNone(ba.to_int(None))
        self.assertEqual(ba.parse_nums("[100 2000]"), [100.0, 2000.0])
        self.assertEqual(ba.stem("files102\\F0007.wav"), "f0007")

    def test_select(self):
        arows = [arow("1", "201", "203", "12"), arow("2", "201", "0", "12"),
                 arow("3", "999", "203", "12"), arow("4", "201", "203", "9"),
                 arow("5", "201", "203", "11"), arow("5", "201", "204", "11"),
                 arow("6", "201", "204", "3"), arow("7", "203", "201", "4"),
                 arow("8", "205", "-201", "4")]
        frows = [frow("1", "10", "f0001.WAV", "1", "2012-10-05 01:02:03", "[100 2000]",
                      "[1500 4000]"),
                 frow("2", "10", "f0002.WAV", "1", "2012-10-05 01:02:03", "", ""),
                 frow("3", "10", "f0003.WAV", "1", "2012-10-05 01:02:03", "", ""),
                 frow("4", "10", "f0004.WAV", "1", "2012-10-05 01:02:03", "", ""),
                 frow("5", "10", "f0005.WAV", "1", "2012-10-05 01:02:03", "", ""),
                 frow("6", "3", "f0006.WAV", "1", "2012-10-05 01:02:03", "", ""),
                 frow("7", "9", "files102\\F0007.wav", "3", "05-Oct-2012 10:00:00", "", ""),
                 frow("8", "10", "f0008.WAV", "1", "2012-10-05 01:02:03", "", "")]
        cols = ba.label_columns(AH, FH)
        out, counts = ba.select_calls(AH, arows, FH, frows, cols)
        self.assertEqual([c["key"] for c in out], ["1", "7"])
        self.assertEqual(out[0]["segments"], [(100, 1500), (2000, 4000)])
        self.assertEqual(out[0]["day"], "2012-10-05")
        self.assertEqual(out[0]["stem"], "f0001")
        self.assertEqual(out[1]["segments"], [])
        self.assertEqual(out[1]["day"], "2012-10-05")
        self.assertEqual(out[1]["channel"], "3")
        self.assertEqual((out[1]["emitter"], out[1]["addressee"], out[1]["context"]),
                         (203, 201, 4))
        self.assertEqual(counts, {"annotations": 9, "emitter_ok": 8, "addressee_ok": 6,
                                  "context_ok": 5, "single_annotation": 3, "joined": 3,
                                  "treatment_ok": 2, "day_parsed": 2})

    def test_class_filter_and_blocks(self):
        calls = ([{"emitter": 1, "addressee": "A", "day": "d1"}] * 20
                 + [{"emitter": 1, "addressee": "B", "day": "d1"}] * 20
                 + [{"emitter": 1, "addressee": "C", "day": "d2"}] * 19
                 + [{"emitter": 2, "addressee": "A", "day": "d1"}] * 25
                 + [{"emitter": 2, "addressee": "B", "day": "d1"}] * 5)
        kept, classes = ba.class_filter(calls, 20)
        self.assertEqual(classes, {1: ["A", "B"]})
        self.assertEqual(len(kept), 40)
        calls2 = [{"emitter": 1, "addressee": "A", "day": "d1"},
                  {"emitter": 1, "addressee": "B", "day": "d1"},
                  {"emitter": 1, "addressee": "A", "day": "d2"},
                  {"emitter": 2, "addressee": "A", "day": "d1"}]
        self.assertEqual(ba.informative_blocks(calls2), (1, 2))


class TestZip(unittest.TestCase):
    def test_round_trip(self):
        payload = {"files101/a.WAV": b"abc" * 1000, "b.wav": bytes(range(256)) * 10,
                   "c.wav": b"xyz" * 777, "d/e.wav": b"\x00\x01" * 3000}
        methods = {"files101/a.WAV": zipfile.ZIP_STORED, "b.wav": zipfile.ZIP_DEFLATED,
                   "c.wav": zipfile.ZIP_BZIP2, "d/e.wav": zipfile.ZIP_LZMA}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, data in payload.items():
                z.writestr(name, data, compress_type=methods[name])
        blob = buf.getvalue()
        info = ba.find_eocd(blob[-ba.EOCD_SPAN:])
        self.assertEqual(info["n"], 4)
        cd = blob[info["cd_offset"]:info["cd_offset"] + info["cd_size"]]
        entries = ba.parse_central_directory(cd)
        self.assertEqual(sorted(e["name"] for e in entries), sorted(payload))
        self.assertEqual(sorted(e["method"] for e in entries), [0, 8, 12, 14])
        for e in entries:
            self.assertEqual(ba.member_bytes(blob[e["offset"]:], e), payload[e["name"]])
        e = entries[0]
        with self.assertRaises(ba.ShortRead) as cm:
            ba.member_bytes(blob[e["offset"]:e["offset"] + 40], e)
        self.assertEqual(cm.exception.need,
                         30 + len(e["name"].encode()) + e["csize"])


def base_result():
    return {"confound": {"blocked_fp": 0, "naive_detected": 3},
            "tests": {"naive": {"p": 0.01, "null_mean": 0.34},
                      "day": {"p": 0.2, "null_mean": 0.40},
                      "lodo": {"p": 0.5}},
            "power": {"detected": 5}}


class TestVerdict(unittest.TestCase):
    def test_rules(self):
        r = base_result()
        self.assertEqual(ba.verdict(r), "NOT_DETECTED_WITH_POWER")
        r["power"]["detected"] = 3
        self.assertEqual(ba.verdict(r), "UNDERPOWERED")
        r["tests"]["day"]["p"] = 0.03
        self.assertEqual(ba.verdict(r), "ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING")
        r["tests"]["naive"]["p"] = 0.2
        self.assertEqual(ba.verdict(r), "NO_NAIVE_SIGNAL")
        r["confound"]["blocked_fp"] = 2
        self.assertEqual(ba.verdict(r), "CALIBRATION_FAILED")
        r["gate"] = "MAPPING_FAILED"
        self.assertEqual(ba.verdict(r), "MAPPING_FAILED")

    def test_outcomes(self):
        truth, brier, drop = ba.outcomes(base_result())
        self.assertEqual(truth, {"Q1": True, "Q2": False, "Q3": False, "Q4": True, "Q5": True,
                                 "Q6": True})
        self.assertAlmostEqual(brier, 0.09125, places=9)
        self.assertAlmostEqual(drop, 0.06, places=9)
        truth, brier, drop = ba.outcomes({"gate": "TOO_FEW_EMITTERS"})
        self.assertIsNone(brier)
        self.assertIsNone(drop)


@unittest.skipIf(np is None, "numpy not installed")
class TestAudio(unittest.TestCase):
    def test_wav(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(250000)
            w.writeframes(struct.pack("<4h", 1000, -5, -2000, 7))
        x, sr = ba.wav_samples(buf.getvalue())
        self.assertEqual(sr, 250000)
        self.assertTrue(np.allclose(x, [1000 / 32768.0, -2000 / 32768.0]))

    def test_frame_starts(self):
        self.assertEqual(ba.frame_starts(20000, [(0, 10000)], 5000, 2500), [0, 2500, 5000])
        self.assertEqual(ba.frame_starts(20000, [(0, 10000), (12000, 13000)], 5000, 2500),
                         [0, 2500, 5000, 10000])
        self.assertEqual(ba.frame_starts(20000, [(30000, 40000)], 5000, 2500), [])

    def test_features(self):
        rng = np.random.default_rng(0)
        f, how = ba.call_features(rng.normal(size=50000), 250000, [(0, 50000)])
        self.assertEqual((f.shape, how), ((37, 64), "segments"))
        self.assertTrue(np.all(np.isfinite(f)))
        f, how = ba.call_features(rng.normal(size=250000), 250000, [(0, 250000)])
        self.assertEqual(f.shape, (64, 64))
        f, how = ba.call_features(rng.normal(size=50000), 250000, [])
        self.assertEqual(how, "energy")
        self.assertTrue(1 <= f.shape[0] <= 64)
        f, how = ba.call_features(rng.normal(size=3000), 250000, [(0, 3000)])
        self.assertEqual(f.shape, (1, 64))
        fb = ba.mel_filterbank(250000, 8192)
        self.assertEqual(fb.shape, (96, 4097))
        self.assertTrue(np.all(fb.sum(axis=1) > 0))


def toy_emitter(seed, n_calls=12, d=3):
    rng = np.random.default_rng(seed)
    ubm = ba.Ubm([0.4, 0.6], rng.normal(size=(2, d)), 0.5 + rng.random((2, d)))
    lens = rng.integers(2, 6, size=n_calls)
    x = rng.normal(size=(int(lens.sum()), d))
    labels = np.array([0, 1, 2] * (n_calls // 3))
    day = [i * 3 // n_calls for i in range(n_calls)]
    return ba.Emitter(7, ["a", "b", "c"], labels, x, lens, day, day, ubm), ubm


@unittest.skipIf(np is None, "numpy not installed")
class TestClassifier(unittest.TestCase):
    def test_comp_ll(self):
        e, ubm = toy_emitter(1)
        x = e.X[:5]
        ll = ubm.comp_ll(x, (x ** 2) @ ubm.ivar.T, ubm.mu)
        for k in range(2):
            direct = np.log(ubm.w[k]) - 0.5 * np.sum(
                np.log(2 * np.pi * ubm.var[k]) + (x - ubm.mu[k]) ** 2 / ubm.var[k], axis=1)
            self.assertTrue(np.allclose(ll[:, k], direct))
        self.assertTrue(np.allclose(e.n.sum(axis=1), e.lens))

    def brute(self, e, ubm, scheme):
        c_n = len(e.lens)
        s = np.zeros((c_n, 3))
        for c in range(c_n):
            a, b = e.off[c], e.off[c + 1]
            xc = e.X[a:b]
            for j in range(3):
                if scheme == "loco":
                    idx = [i for i in range(c_n) if e.labels[i] == j and i != c]
                else:
                    idx = [i for i in range(c_n) if e.labels[i] == j and e.day[i] != e.day[c]]
                n = e.n[idx].sum(axis=0) if idx else np.zeros(2)
                f = e.F[idx].sum(axis=0) if idx else np.zeros((2, e.X.shape[1]))
                m = (f + ba.RELEVANCE * ubm.mu) / (n + ba.RELEVANCE)[:, None]
                s[c, j] = ba.lse(ubm.comp_ll(xc, (xc ** 2) @ ubm.ivar.T, m)).mean()
        return s

    def test_loco_matches_brute_force(self):
        e, ubm = toy_emitter(2)
        s = ba.class_scores(e, ubm, e.labels, "loco")
        self.assertTrue(np.allclose(s, self.brute(e, ubm, "loco"), rtol=1e-9, atol=1e-9))

    def test_lodo_matches_brute_force(self):
        e, ubm = toy_emitter(3)
        self.assertEqual(e.day_slices, [(0, 4), (4, 8), (8, 12)])
        s = ba.class_scores(e, ubm, e.labels, "lodo")
        self.assertTrue(np.allclose(s, self.brute(e, ubm, "lodo"), rtol=1e-9, atol=1e-9))

    def test_separable_and_random(self):
        rng = np.random.default_rng(4)
        n_calls = 40
        labels = np.array([i % 2 for i in range(n_calls)])
        lens = np.full(n_calls, 10)
        x = np.concatenate([rng.normal(loc=(1.0 if k else -1.0), size=(10, 4)) for k in labels])
        day = [i // 10 for i in range(n_calls)]
        ubm = ba.Ubm([1.0], np.zeros((1, 4)), np.full((1, 4), 2.0))
        e = ba.Emitter(1, ["A", "B"], labels, x, lens, day, day, ubm)
        self.assertGreater(ba.ba_of(e, ubm, e.labels, "loco"), 0.9)
        self.assertGreater(ba.ba_of(e, ubm, e.labels, "lodo"), 0.9)
        self.assertLess(ba.ba_of(e, ubm, rng.permutation(labels), "loco"), 0.8)
        res = ba.run_test([e], ubm, "loco", "day", 19, np.random.default_rng(5))
        self.assertAlmostEqual(res["p"], 0.05)
        self.assertEqual(len(res["null_T"]), 19)

    def test_permutations_and_worlds(self):
        rng = np.random.default_rng(6)
        labels = np.arange(10) % 3
        blocks = [np.arange(0, 5), np.arange(5, 10)]
        p = ba.permute_within(labels, blocks, rng)
        for b in blocks:
            self.assertEqual(sorted(p[b].tolist()), sorted(labels[b].tolist()))
        e, ubm = toy_emitter(7)
        cl = ba.confound_labels(e, rng)
        self.assertEqual(cl.shape, e.labels.shape)
        self.assertTrue(cl.min() >= 0 and cl.max() <= 2)
        w, base = ba.pos_world(e, ubm, 0.1, np.ones(3), rng)
        self.assertEqual(w.X.shape, e.X.shape)
        for i0, i1 in e.day_slices:
            self.assertEqual(sorted(base[i0:i1].tolist()), sorted(e.labels[i0:i1].tolist()))


@unittest.skipIf(np is None, "numpy not installed")
class TestUbmFit(unittest.TestCase):
    def test_fit(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        x = np.random.default_rng(8).normal(size=(500, 3))
        u = ba.fit_ubm(x, k=2)
        self.assertAlmostEqual(float(u.w.sum()), 1.0, places=6)
        self.assertEqual(u.mu.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
