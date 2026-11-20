# SPDX-License-Identifier: Apache-2.0
# describe_bats.py: a label-and-structure description of the public Egyptian fruit bat dataset of
# Prat, Taub, Pratt & Yovel (2017, Scientific Data 4:170143; figshare collection 3666502, CC0),
# the recordings behind Prat, Taub & Yovel (2016, Scientific Reports 6:39419), which reported that
# everyday bat calls carry information about their addressee. It downloads only the two label
# files (Annotations.csv and FileInfo.csv, checked by MD5), describes every column, joins them,
# and counts how emitters, addressees, contexts, files, recording days and channels are arranged:
# in particular, how many emitter-by-day blocks contain calls to two or more addressees (the
# blocks a day-blocked test of addressee information can use). It also reads only the central
# directory of each audio archive (by HTTP Range requests; no audio is downloaded) to list which
# recordings each archive holds. It computes no acoustic statistic and nothing about addressee
# information: it exists so that a later test can be designed, and pre-registered, knowing only
# the label structure.
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import time
import urllib.request
from collections import Counter, defaultdict

VERSION = "bats-describe-1"
COLLECTION = "3666502"
API = "https://api.figshare.com/v2"
LABEL_FILES = {
    "Annotations.csv": ("https://ndownloader.figshare.com/files/7379008",
                        "7781a3725d6f2277de74da84b1383ce0", 3286405),
    "FileInfo.csv": ("https://ndownloader.figshare.com/files/8900695",
                     "b27252490ac5618bd55a9038110809d5", 31574701),
}
UA = "MV2-Munim-research/1.0 (mv2@munimtech.com)"
MAX_DISTINCT = 5000     # per-column value counting stops adding new values after this many
TOP = 20                # values listed per column
MIN_EMITTER = 20        # emitters with fewer annotations are only counted, not detailed
MAX_EMITTERS = 80
EOCD_SPAN = 65557       # 22-byte end record + the longest possible zip comment


# ---------------------------------------------------------------- small pure helpers
def norm(h):
    """Header name normalised for matching: lower case, no spaces, underscores or dashes."""
    return re.sub(r"[\s_\-]+", "", (h or "").strip().lower())


def find_col(header, *subs):
    """The first header name whose normalised form contains every given substring, else None."""
    for h in header:
        n = norm(h)
        if all(s in n for s in subs):
            return h
    return None


def find_key(h1, h2):
    """The join column shared by two headers (normalised match): prefer one naming a file id,
    then one naming a file, then the first shared column. Returns (name in h1, name in h2)."""
    m2 = {norm(h): h for h in h2}
    common = [h for h in h1 if norm(h) in m2]
    if not common:
        return None, None
    for pick in (lambda n: "file" in n and "id" in n, lambda n: "file" in n, lambda n: True):
        for h in common:
            if pick(norm(h)):
                return h, m2[norm(h)]
    return None, None


def as_float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_day(v):
    """A day key from a recording-time value whose format is not known in advance:
    a MATLAB datenum (days since year 0; 1900-2100) -> 'dn<integer day>';
    'YYYY-MM-DD...' (or / or . separators) -> 'YYYY-MM-DD';
    'D/M/YYYY...' -> 'DD/MM/YYYY' (day-month order kept as written);
    a run of digits starting with 19 or 20 and at least 8 long -> its first 8 digits;
    anything else -> None."""
    s = (v or "").strip()
    if not s:
        return None
    x = as_float(s)
    if x is not None and 693962 <= x < 767011:
        return "dn%d" % int(math.floor(x))
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return "%s-%02d-%02d" % (m.group(1), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", s)
    if m:
        return "%02d/%02d/%s" % (int(m.group(1)), int(m.group(2)), m.group(3))
    m = re.match(r"((?:19|20)\d{6})\d*", s)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------- column description
class ColumnStats:
    def __init__(self):
        self.counter = Counter()
        self.overflow = False
        self.numeric = True
        self.n_numeric = 0
        self.min = None
        self.max = None
        self.empty = 0
        self.n = 0

    def add(self, v):
        self.n += 1
        v = (v or "").strip()
        if v == "":
            self.empty += 1
            return
        if v in self.counter or len(self.counter) < MAX_DISTINCT:
            self.counter[v] += 1
        else:
            self.overflow = True
        x = as_float(v)
        if x is None:
            self.numeric = False
        else:
            self.n_numeric += 1
            self.min = x if self.min is None else min(self.min, x)
            self.max = x if self.max is None else max(self.max, x)

    def summary(self):
        top = [[k[:60], c] for k, c in self.counter.most_common(TOP)]
        return {"n": self.n, "empty": self.empty,
                "distinct": (">%d" % MAX_DISTINCT) if self.overflow else len(self.counter),
                "all_numeric": bool(self.numeric and self.n_numeric > 0),
                "n_numeric": self.n_numeric, "min": self.min, "max": self.max, "top": top}


def describe_rows(header, rows):
    """Per-column summaries of an iterable of rows (lists; short rows are padded with '')."""
    stats = [ColumnStats() for _ in header]
    n = 0
    ragged = 0
    for row in rows:
        n += 1
        if len(row) != len(header):
            ragged += 1
        for i, st in enumerate(stats):
            st.add(row[i] if i < len(row) else "")
    return {"n_rows": n, "ragged_rows": ragged,
            "columns": {h: st.summary() for h, st in zip(header, stats)}}


# ---------------------------------------------------------------- label structure
def structure(records):
    """records: dicts with keys emitter, addressee, context, file, day, channel, treatment
    (strings or None). Counts how addressees are spread over emitters, files and days."""
    n = len(records)
    by_e = defaultdict(list)
    for r in records:
        by_e[r["emitter"]].append(r)
    files = defaultdict(set)          # file -> {(emitter, addressee)}
    for r in records:
        files[r["file"]].add((r["emitter"], r["addressee"]))
    multi = 0
    mixed = 0
    for f, pairs in files.items():
        if len(pairs) > 1:
            multi += 1
        per_e = defaultdict(set)
        for e, a in pairs:
            per_e[e].add(a)
        if any(len(s) > 1 for s in per_e.values()):
            mixed += 1
    emitters = []
    tot = Counter()
    for e, rs in sorted(by_e.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        days = defaultdict(set)
        cdays = defaultdict(set)
        for r in rs:
            if r["day"] is not None:
                days[r["day"]].add(r["addressee"])
                cdays[(r["context"], r["day"])].add(r["addressee"])
        inf_days = {d for d, s in days.items() if len(s) > 1}
        inf_cdays = {k for k, s in cdays.items() if len(s) > 1}
        rec_inf = sum(1 for r in rs if r["day"] in inf_days)
        rec_cinf = sum(1 for r in rs if (r["context"], r["day"]) in inf_cdays)
        tot["informative_days"] += len(inf_days)
        tot["records_in_informative_days"] += rec_inf
        tot["informative_context_days"] += len(inf_cdays)
        tot["records_in_informative_context_days"] += rec_cinf
        if len(rs) < MIN_EMITTER or len(emitters) >= MAX_EMITTERS:
            continue
        ac = Counter(r["addressee"] for r in rs)
        cc = Counter(r["context"] for r in rs)
        tc = Counter(r["treatment"] for r in rs)
        dl = sorted(d for d in days)
        emitters.append({
            "emitter": e, "n": len(rs),
            "n_addressees": len(ac),
            "addressees": [[a, c] for a, c in ac.most_common(15)],
            "contexts": [[k, c] for k, c in cc.most_common(10)],
            "treatments": [[k, c] for k, c in tc.most_common(5)],
            "files": len({r["file"] for r in rs}),
            "channels": len({r["channel"] for r in rs}),
            "days": len(days), "first_day": dl[0] if dl else None,
            "last_day": dl[-1] if dl else None,
            "informative_days": len(inf_days), "records_in_informative_days": rec_inf,
            "informative_context_days": len(inf_cdays),
            "records_in_informative_context_days": rec_cinf})
    by_t = defaultdict(list)
    for r in records:
        by_t[r["treatment"]].append(r)
    treatments = []
    for t, rs in sorted(by_t.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))[:40]:
        dl = sorted({r["day"] for r in rs if r["day"] is not None})
        treatments.append({"treatment": t, "n": len(rs),
                           "emitters": len({r["emitter"] for r in rs}),
                           "addressees": len({r["addressee"] for r in rs}),
                           "days": len(dl), "first_day": dl[0] if dl else None,
                           "last_day": dl[-1] if dl else None})
    return {"n": n, "n_with_day": sum(1 for r in records if r["day"] is not None),
            "n_emitters": len(by_e), "n_files": len(files),
            "files_with_2plus_annotations": multi,
            "files_where_an_emitter_has_2plus_addressees": mixed,
            "totals": dict(tot), "emitters": emitters, "treatments": treatments}


# ---------------------------------------------------------------- zip central directory
def find_eocd(tail):
    """Parse the end-of-central-directory record at the end of a zip (tail = last bytes)."""
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or i + 22 > len(tail):
        raise ValueError("no end-of-central-directory record")
    _, _, _, _, n_total, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    info = {"n": n_total, "cd_size": cd_size, "cd_offset": cd_off, "zip64": False}
    if n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x07", 0, i)
        if j < 0:
            raise ValueError("zip64 locator missing")
        _, _, z64off, _ = struct.unpack("<IIQI", tail[j:j + 20])
        info["zip64"] = True
        info["z64_eocd_offset"] = z64off
    return info


def parse_z64_eocd(b):
    f = struct.unpack("<IQHHIIQQQQ", b[:56])
    if f[0] != 0x06064B50:
        raise ValueError("bad zip64 end record")
    return {"n": f[7], "cd_size": f[8], "cd_offset": f[9]}


def parse_central_directory(cd):
    out = []
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        nlen, xlen, clen = f[10], f[11], f[12]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        out.append({"name": name, "method": f[4], "csize": f[8], "usize": f[9],
                    "offset": f[16]})
        p += 46 + nlen + xlen + clen
    return out


def summarise_members(entries):
    base = [e["name"].replace("\\", "/").split("/")[-1] for e in entries]
    ext = Counter(os.path.splitext(b)[1].lower() for b in base if b)
    dirs = Counter("/".join(e["name"].replace("\\", "/").split("/")[:-1]) for e in entries)
    nums = [int(m.group(0)) for m in (re.search(r"\d+", b) for b in base) if m]
    return {"members": len(entries),
            "files": sum(1 for b in base if b),
            "methods": dict(Counter(str(e["method"]) for e in entries)),
            "uncompressed_bytes": sum(e["usize"] for e in entries),
            "compressed_bytes": sum(e["csize"] for e in entries),
            "extensions": dict(ext.most_common(10)),
            "dirs": [[d, c] for d, c in dirs.most_common(10)],
            "first_names": [e["name"] for e in entries[:5]],
            "last_names": [e["name"] for e in entries[-5:]],
            "min_number_in_name": min(nums) if nums else None,
            "max_number_in_name": max(nums) if nums else None}


# ---------------------------------------------------------------- network
def http_get(url, rng=None, tries=4, timeout=120):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            if rng is not None:
                req.add_header("Range", "bytes=%d-%d" % rng)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if rng is not None and r.status != 206:
                    raise IOError("server ignored the Range header (status %s)" % r.status)
                return r.read()
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            last = e
            time.sleep(5 * (i + 1))
    raise last


def download(name, url, md5, size, data_dir):
    path = os.path.join(data_dir, name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        if h == md5:
            return path, h
    for i in range(4):
        try:
            h = hashlib.md5()
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    f.write(b)
            os.replace(path + ".part", path)
            return path, h.hexdigest()
        except Exception:  # noqa: BLE001
            time.sleep(10 * (i + 1))
    raise IOError("download failed: " + name)


def read_csv(path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return [h.strip() for h in rows[0]], rows[1:]


def archives(deadline):
    lst = json.loads(http_get("%s/collections/%s/articles?page_size=1000" % (API, COLLECTION)))
    out = []
    for a in lst:
        m = re.search(r"files\s+(\d+)\s*$", a.get("title") or "")
        if m:
            out.append((int(m.group(1)), a["id"], a["title"]))
    out.sort()
    res = []
    for num, aid, title in out:
        if time.time() > deadline:
            res.append({"archive": num, "article": aid, "error": "time budget reached"})
            continue
        try:
            art = json.loads(http_get("%s/articles/%s" % (API, aid)))
            f = art["files"][0]
            size, url = f["size"], f["download_url"]
            tail_start = max(0, size - EOCD_SPAN)
            info = find_eocd(http_get(url, (tail_start, size - 1)))
            if info["zip64"]:
                z = info["z64_eocd_offset"]
                info.update(parse_z64_eocd(http_get(url, (z, z + 55))))
            cd = http_get(url, (info["cd_offset"], info["cd_offset"] + info["cd_size"] - 1),
                          timeout=300)
            entries = parse_central_directory(cd)
            s = summarise_members(entries)
            s.update({"archive": num, "article": aid, "title": title, "zip_name": f["name"],
                      "zip_bytes": size, "zip_md5": f.get("computed_md5"),
                      "eocd_count": info["n"], "zip64": info["zip64"]})
            s["names"] = [e["name"] for e in entries]
            res.append(s)
        except Exception as e:  # noqa: BLE001 - reported, not fatal
            res.append({"archive": num, "article": aid, "error": repr(e)[:300]})
    return res


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-zips", action="store_true")
    ap.add_argument("--zip-minutes", type=float, default=40.0)
    a = ap.parse_args(argv)
    os.makedirs(a.data_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    out = {"version": VERSION, "python": sys.version.split()[0], "collection": COLLECTION,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    paths = {}
    for name, (url, md5, size) in LABEL_FILES.items():
        p, h = download(name, url, md5, size, a.data_dir)
        paths[name] = p
        out["files"][name] = {"bytes": os.path.getsize(p), "md5": h, "md5_ok": h == md5}
    ah, arows = read_csv(paths["Annotations.csv"])
    fh, frows = read_csv(paths["FileInfo.csv"])
    out["annotations"] = {"header": ah, "first_rows": arows[:3]}
    out["annotations"].update(describe_rows(ah, arows))
    out["fileinfo"] = {"header": fh, "first_rows": frows[:3]}
    out["fileinfo"].update(describe_rows(fh, frows))
    ka, kf = find_key(ah, fh)
    cols = {"key_annotations": ka, "key_fileinfo": kf,
            "emitter": find_col(ah, "emitter"), "addressee": find_col(ah, "addressee"),
            "context": find_col(ah, "context"),
            "time": find_col(fh, "time") or find_col(fh, "date"),
            "channel": find_col(fh, "channel"), "treatment": find_col(fh, "treatment"),
            "folder": find_col(fh, "folder")}
    out["columns_used"] = cols
    if ka and kf and cols["emitter"] and cols["addressee"]:
        fi = {}
        idx = {h: i for i, h in enumerate(fh)}

        def g(row, col):
            if not col or idx[col] >= len(row):
                return None
            return row[idx[col]].strip()
        for row in frows:
            k = g(row, kf)
            if k is not None:
                fi[k] = (g(row, cols["time"]), g(row, cols["channel"]),
                         g(row, cols["treatment"]), g(row, cols["folder"]))
        ai = {h: i for i, h in enumerate(ah)}

        def ga(row, col):
            if not col or ai[col] >= len(row):
                return None
            return row[ai[col]].strip()
        recs = []
        joined = 0
        raw_times = []
        for row in arows:
            k = ga(row, ka)
            t = fi.get(k)
            if t is not None:
                joined += 1
                if len(raw_times) < 10:
                    raw_times.append(t[0])
            tm, ch, tr, fo = t if t is not None else (None, None, None, None)
            recs.append({"emitter": ga(row, cols["emitter"]),
                         "addressee": ga(row, cols["addressee"]),
                         "context": ga(row, cols["context"]), "file": k,
                         "day": parse_day(tm), "channel": ch, "treatment": tr,
                         "folder": fo})
        out["join"] = {"annotations": len(arows), "joined": joined,
                       "fileinfo_keys": len(fi), "raw_time_samples": raw_times,
                       "day_samples": [parse_day(x) for x in raw_times]}
        out["folders"] = [[k, c] for k, c in
                          Counter(r["folder"] for r in recs).most_common(40)]
        out["structure"] = structure(recs)
    else:
        out["structure"] = None
    out["seconds_labels"] = round(time.time() - t0, 1)
    print(json.dumps({k: out.get(k) for k in ("files", "columns_used", "join")}, indent=1))
    names = {}
    if not a.skip_zips:
        res = archives(time.time() + 60 * a.zip_minutes)
        for s in res:
            if "names" in s:
                names[str(s["archive"])] = s.pop("names")
        out["archives"] = res
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out_dir, "bats_describe.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.out_dir, "zip_members.json"), "w") as f:
        json.dump(names, f)
    print("archives read: %d; seconds %.1f" % (len(names), out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
