# SPDX-License-Identifier: Apache-2.0
# Tests for describe_bats.py (stdlib only; every expected value worked out by hand).
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import io
import struct
import unittest
import zipfile

import describe_bats as db


class TestHelpers(unittest.TestCase):
    def test_find(self):
        h = ["FileID", "Emitter", "Addressee", "Context", "Start sample"]
        self.assertEqual(db.find_col(h, "emitter"), "Emitter")
        self.assertEqual(db.find_col(h, "start", "sample"), "Start sample")
        self.assertIsNone(db.find_col(h, "time"))
        self.assertEqual(db.find_key(["FileID", "Emitter"],
                                     ["File folder", "File ID", "Recording time"]),
                         ("FileID", "File ID"))
        self.assertEqual(db.find_key(["File", "x"], ["y", "file"]), ("File", "file"))
        self.assertEqual(db.find_key(["a", "b"], ["c"]), (None, None))
        self.assertEqual(db.find_key(["b", "a"], ["a", "b"]), ("b", "b"))

    def test_parse_day(self):
        cases = {"735398.5": "dn735398", "2013-06-12 14:00:01": "2013-06-12",
                 "2013/6/2": "2013-06-02", "12/06/2013 14:00": "12/06/2013",
                 "20130612143000": "20130612", "": None, "abc": None, "12": None,
                 "1.5": None, None: None}
        for v, want in cases.items():
            self.assertEqual(db.parse_day(v), want, v)

    def test_describe_rows(self):
        d = db.describe_rows(["a", "b"], [["1", "x"], ["2", ""], ["3", "x", "extra"], ["4"]])
        self.assertEqual(d["n_rows"], 4)
        self.assertEqual(d["ragged_rows"], 2)
        a, b = d["columns"]["a"], d["columns"]["b"]
        self.assertEqual((a["n"], a["empty"], a["distinct"], a["all_numeric"], a["min"],
                          a["max"]), (4, 0, 4, True, 1.0, 4.0))
        self.assertEqual((b["n"], b["empty"], b["distinct"], b["all_numeric"],
                          b["n_numeric"], b["min"], b["top"]),
                         (4, 2, 1, False, 0, None, [["x", 2]]))


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.old = db.MIN_EMITTER
        db.MIN_EMITTER = 1

    def tearDown(self):
        db.MIN_EMITTER = self.old

    def test_structure(self):
        def r(e, a, c, f, d, ch, t):
            return {"emitter": e, "addressee": a, "context": c, "file": f, "day": d,
                    "channel": ch, "treatment": t}
        recs = [r("E1", "A", "c1", "f1", "d1", "1", "T"),
                r("E1", "B", "c1", "f2", "d1", "1", "T"),
                r("E1", "A", "c2", "f3", "d2", "2", "T"),
                r("E1", "B", "c2", "f3", "d2", "2", "T"),
                r("E2", "A", "c1", "f4", None, "1", "U")]
        s = db.structure(recs)
        self.assertEqual((s["n"], s["n_with_day"], s["n_emitters"], s["n_files"]), (5, 4, 2, 4))
        self.assertEqual(s["files_with_2plus_annotations"], 1)
        self.assertEqual(s["files_where_an_emitter_has_2plus_addressees"], 1)
        self.assertEqual(s["totals"], {"informative_days": 2, "records_in_informative_days": 4,
                                       "informative_context_days": 2,
                                       "records_in_informative_context_days": 4})
        e1, e2 = s["emitters"]
        self.assertEqual(e1["emitter"], "E1")
        self.assertEqual((e1["n"], e1["n_addressees"], e1["files"], e1["channels"], e1["days"],
                          e1["first_day"], e1["last_day"]), (4, 2, 3, 2, 2, "d1", "d2"))
        self.assertEqual(e1["addressees"], [["A", 2], ["B", 2]])
        self.assertEqual(e1["contexts"], [["c1", 2], ["c2", 2]])
        self.assertEqual((e2["n"], e2["days"], e2["first_day"], e2["informative_days"]),
                         (1, 0, None, 0))
        t1, t2 = s["treatments"]
        self.assertEqual((t1["treatment"], t1["n"], t1["emitters"], t1["addressees"],
                          t1["days"]), ("T", 4, 1, 2, 2))
        self.assertEqual((t2["treatment"], t2["n"], t2["days"], t2["first_day"]),
                         ("U", 1, 0, None))

    def test_min_emitter(self):
        db.MIN_EMITTER = 2
        recs = [{"emitter": "E", "addressee": "A", "context": "c", "file": "f", "day": "d",
                 "channel": "1", "treatment": "T"}]
        s = db.structure(recs)
        self.assertEqual(s["emitters"], [])
        self.assertEqual(s["n_emitters"], 1)


class TestZip(unittest.TestCase):
    def test_central_directory(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("rec/a1.WAV", b"abc" * 400, compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("rec/a22.wav", b"q", compress_type=zipfile.ZIP_STORED)
            zf.writestr("b.txt", b"hello", compress_type=zipfile.ZIP_STORED)
        data = buf.getvalue()
        tail = data[max(0, len(data) - db.EOCD_SPAN):]
        info = db.find_eocd(tail)
        self.assertEqual(info["n"], 3)
        self.assertFalse(info["zip64"])
        cd = data[info["cd_offset"]:info["cd_offset"] + info["cd_size"]]
        ent = db.parse_central_directory(cd)
        self.assertEqual([e["name"] for e in ent], ["rec/a1.WAV", "rec/a22.wav", "b.txt"])
        self.assertEqual([e["method"] for e in ent], [8, 0, 0])
        self.assertEqual([e["usize"] for e in ent], [1200, 1, 5])
        self.assertLess(ent[0]["csize"], 1200)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            self.assertEqual([e["offset"] for e in ent], [i.header_offset for i in infos])
        s = db.summarise_members(ent)
        self.assertEqual(s["members"], 3)
        self.assertEqual(s["extensions"], {".wav": 2, ".txt": 1})
        self.assertEqual(s["dirs"], [["rec", 2], ["", 1]])
        self.assertEqual((s["min_number_in_name"], s["max_number_in_name"]), (1, 22))
        self.assertEqual(s["methods"], {"8": 1, "0": 2})
        self.assertEqual(s["uncompressed_bytes"], 1206)

    def test_zip64_records(self):
        loc = struct.pack("<IIQI", 0x07064B50, 0, 12345, 1)
        eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF,
                           0xFFFFFFFF, 0)
        info = db.find_eocd(b"xx" + loc + eocd)
        self.assertTrue(info["zip64"])
        self.assertEqual(info["z64_eocd_offset"], 12345)
        rec = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, 3, 3, 150, 1000)
        self.assertEqual(db.parse_z64_eocd(rec), {"n": 3, "cd_size": 150, "cd_offset": 1000})
        with self.assertRaises(ValueError):
            db.parse_z64_eocd(b"\x00" * 56)
        with self.assertRaises(ValueError):
            db.find_eocd(b"no zip here")


if __name__ == "__main__":
    unittest.main()
