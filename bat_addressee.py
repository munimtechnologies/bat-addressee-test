# SPDX-License-Identifier: Apache-2.0
# bat_addressee.py: the pre-registered, day-blocked re-test of the addressee claim of Prat, Taub &
# Yovel (2016, Scientific Reports 6:39419: everyday Egyptian fruit bat calls carry information about
# which bat they are addressed to), run on the public dataset of Prat, Taub, Pratt & Yovel (2017,
# Scientific Data 4:170143; figshare collection 3666502, CC0). Design, decision rules, predictions
# and gates are fixed in BAT-PREREG.md, which was published before this code first ran.
#
# Pipeline: labels (MD5-checked) -> selection -> audio of single recordings read by HTTP Range from
# the public zip archives (nothing is stored but features) -> 64 MFCC per 20 ms frame of the voiced
# segments -> per-channel mean subtraction -> GMM-UBM (16 diagonal components, MAP mean adaptation,
# relevance 16) addressee classifier per emitter -> balanced accuracy (BA) -> permutation tests in
# which addressee labels are shuffled freely within emitter (naive, as published) or only within
# emitter x recording day (primary) -> confound worlds (calibration) and effect-matched synthetic
# positive worlds (power) -> verdict.
# MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import argparse
import bz2
import csv
import hashlib
import json
import lzma
import math
import os
import re
import struct
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ImportError:  # labels, zip parsing and the verdict rules work without numpy
    np = None

VERSION = "bat-addressee-1"
SEED = 20261120
COLLECTION = "3666502"
API = "https://api.figshare.com/v2"
UA = "MV2-Munim-research/1.0 (mv2@munimtech.com)"
LABEL_FILES = {
    "Annotations.csv": ("https://ndownloader.figshare.com/files/7379008",
                        "7781a3725d6f2277de74da84b1383ce0", 3286405),
    "FileInfo.csv": ("https://ndownloader.figshare.com/files/8900695",
                     "b27252490ac5618bd55a9038110809d5", 31574701),
}
EOCD_SPAN = 65557

# ---- selection (BAT-PREREG.md section 3)
EMITTERS = (201, 203, 204, 205, 213, 223, 225, 226, 228)   # wild-caught adult females, T9/T10
TREATMENTS = (9, 10)                                         # the two 75-day colony treatments
CONTEXTS = (3, 4, 10, 11, 12)    # feeding, fighting, threat-like (perch), general, sleeping
MIN_CLASS = 20
MIN_EMITTERS = 3
MIN_BLOCKS = 10
MIN_MAPPED = 0.80
MIN_DAY_PARSED = 0.90

# ---- features (section 4)
WIN_S = 0.020
HOP_S = 0.005
NFFT = 8192
N_MEL = 96
N_MFCC = 64
FMIN = 1000.0
FMAX = 125000.0
MAX_FRAMES = 64
FALLBACK_DB = 20.0

# ---- classifier (section 5)
UBM_K = 16
RELEVANCE = 16.0
UBM_FRAMES = 100000

# ---- tests (sections 6-8)
ALPHA = 0.05
B_NAIVE = 99
B_DAY = 199
B_CTXDAY = 99
B_LODO = 99
B_WORLD = 49
CONFOUND_WORLDS = 4
CONFOUND_STRENGTH = 1.5
CONFOUND_MAX_FP = 1
POS_GRID = (0.005, 0.01, 0.02, 0.04, 0.08)
POS_REPS = 5
POWER_MIN = 4
DROP_MIN = 0.02

PREDICTIONS = [  # id, text, prior probability that it comes true (section 9)
    ("Q1", "naive (as published) test reproduces addressee information, p <= 0.05", 0.85),
    ("Q2", "primary day-blocked test detects addressee information, p <= 0.05", 0.35),
    ("Q3", "leave-one-day-out test detects addressee information, p <= 0.05", 0.25),
    ("Q4", "the naive test 'detects' at least 2 of 4 confound worlds", 0.70),
    ("Q5", "day composition alone is worth >= 0.02 balanced accuracy (null-mean gap)", 0.70),
    ("Q6", "the blocked test detects at least 4 of 5 effect-matched positive worlds", 0.60),
]


# ======================================================================= small pure helpers
def norm(h):
    return re.sub(r"[\s_\-]+", "", (h or "").strip().lower())


def find_col(header, subs, avoid=()):
    """The shortest header whose normalised name contains every substring in subs and none in
    avoid, else None."""
    best = None
    for h in header:
        n = norm(h)
        if all(s in n for s in subs) and not any(a in n for a in avoid):
            if best is None or len(n) < len(norm(best)):
                best = h
    return best


def find_key(h1, h2):
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


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_nums(s):
    return [float(t) for t in NUM_RE.findall(s or "")]


def to_int(v):
    x = as_float((v or "").strip()) if isinstance(v, str) else as_float(v)
    if x is None:
        nums = parse_nums(v if isinstance(v, str) else "")
        if not nums:
            return None
        x = nums[0]
    return int(x) if x == int(x) else None


MONTHS = {m: i + 1 for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}


def parse_day(v):
    """A recording-day key from a time value whose format is not known in advance."""
    s = (v or "").strip()
    if not s:
        return None
    x = as_float(s)
    if x is not None and 693962 <= x < 767011:          # MATLAB datenum
        return "dn%d" % int(math.floor(x))
    if x is not None and 1.0e9 <= x < 2.0e9:            # Unix seconds (UTC day)
        return time.strftime("%Y-%m-%d", time.gmtime(x))
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return "%s-%02d-%02d" % (m.group(1), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})[-/. ]([A-Za-z]{3})[A-Za-z]*[-/. ](\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return "%s-%02d-%02d" % (m.group(3), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", s)
    if m:
        return "%02d/%02d/%s" % (int(m.group(1)), int(m.group(2)), m.group(3))
    m = re.match(r"((?:19|20)\d{6})\d*", s)
    if m:
        return m.group(1)
    return None


def stem(name):
    base = (name or "").replace("\\", "/").split("/")[-1]
    return os.path.splitext(base)[0].strip().lower()


# ======================================================================= labels and selection
def label_columns(ah, fh):
    ka, kf = find_key(ah, fh)
    act = ("action", "pre", "post")
    return {"key_a": ka, "key_f": kf,
            "emitter": find_col(ah, ("emitter",), act),
            "addressee": find_col(ah, ("addressee",), act),
            "context": find_col(ah, ("context",)),
            "treatment": find_col(fh, ("treatment",)),
            "name": find_col(fh, ("name",)),
            "folder": find_col(fh, ("folder",)),
            "channel": find_col(fh, ("channel",)),
            "time": find_col(fh, ("time",), ("voice", "sample")) or find_col(fh, ("date",)),
            "vstart": find_col(fh, ("start",), ("time", "date")),
            "vend": find_col(fh, ("end",), ("time", "date")),
            "voice": find_col(fh, ("voice",)) or find_col(fh, ("segment",))}


def segments_from(get, cols):
    vs, ve = cols.get("vstart"), cols.get("vend")
    if vs and ve and vs != ve:
        s, e = parse_nums(get(vs)), parse_nums(get(ve))
        if s and len(s) == len(e):
            return [(int(a), int(b)) for a, b in zip(s, e) if b > a]
    if cols.get("voice"):
        v = parse_nums(get(cols["voice"]))
        if len(v) >= 2 and len(v) % 2 == 0:
            return [(int(v[i]), int(v[i + 1])) for i in range(0, len(v), 2) if v[i + 1] > v[i]]
    return []


def select_calls(ah, arows, fh, frows, cols):
    """Candidate calls (before the class rule): emitter in EMITTERS, addressee a known other bat,
    context in CONTEXTS, file carrying exactly one annotation, treatment in TREATMENTS, day parsed."""
    ai = {h: i for i, h in enumerate(ah)}
    fi = {h: i for i, h in enumerate(fh)}

    def ga(row, c):
        if not c or ai[c] >= len(row):
            return ""
        return row[ai[c]].strip()

    def gf(row, c):
        if not c or fi[c] >= len(row):
            return ""
        return row[fi[c]].strip()

    n_ann = Counter(ga(r, cols["key_a"]) for r in arows)
    fmeta = {}
    for row in frows:
        k = gf(row, cols["key_f"])
        if k:
            fmeta[k] = row
    counts = Counter()
    out = []
    for row in arows:
        counts["annotations"] += 1
        k = ga(row, cols["key_a"])
        e = to_int(ga(row, cols["emitter"]))
        if e not in EMITTERS:
            continue
        counts["emitter_ok"] += 1
        a = to_int(ga(row, cols["addressee"]))
        if a is None or a <= 0 or a == e:
            continue
        counts["addressee_ok"] += 1
        c = to_int(ga(row, cols["context"]))
        if c not in CONTEXTS:
            continue
        counts["context_ok"] += 1
        if n_ann[k] != 1:
            continue
        counts["single_annotation"] += 1
        fr = fmeta.get(k)
        if fr is None:
            continue
        counts["joined"] += 1
        if cols["treatment"]:
            if to_int(gf(fr, cols["treatment"])) not in TREATMENTS:
                continue
        counts["treatment_ok"] += 1
        name = gf(fr, cols["name"]) if cols["name"] else k
        day = parse_day(gf(fr, cols["time"])) if cols["time"] else None
        if day is None:
            day = parse_day(stem(name))
        if day is None:
            counts["day_missing"] += 1
            continue
        counts["day_parsed"] += 1
        out.append({"key": k, "emitter": e, "addressee": a, "context": c, "day": day,
                    "name": name, "stem": stem(name),
                    "channel": gf(fr, cols["channel"]) if cols["channel"] else "",
                    "segments": segments_from(lambda col: gf(fr, col), cols)})
    return out, dict(counts)


def class_filter(calls, min_class=MIN_CLASS):
    """Keep, per emitter, the addressees with >= min_class calls; keep emitters with >= 2."""
    cnt = defaultdict(Counter)
    for r in calls:
        cnt[r["emitter"]][r["addressee"]] += 1
    classes = {}
    for e, c in cnt.items():
        keep = sorted(a for a, n in c.items() if n >= min_class)
        if len(keep) >= 2:
            classes[e] = keep
    kept = [r for r in calls if r["emitter"] in classes and r["addressee"] in classes[r["emitter"]]]
    return kept, classes


def informative_blocks(calls):
    """(emitter, day) blocks holding calls to two or more addressees, and the calls in them."""
    s = defaultdict(set)
    n = Counter()
    for r in calls:
        s[(r["emitter"], r["day"])].add(r["addressee"])
        n[(r["emitter"], r["day"])] += 1
    inf = [k for k, v in s.items() if len(v) > 1]
    return len(inf), sum(n[k] for k in inf)


# ======================================================================= zip archives
def find_eocd(tail):
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
        method, csize, usize = f[4], f[8], f[9]
        nlen, xlen, clen, off = f[10], f[11], f[12], f[16]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, off):
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = extra[q + 4:q + 4 + hsz]
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[k:k + 8])[0]
                        k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[k:k + 8])[0]
                        k += 8
                    if off == 0xFFFFFFFF:
                        off = struct.unpack("<Q", vals[k:k + 8])[0]
                    break
                q += 4 + hsz
        out.append({"name": name, "method": method, "csize": csize, "usize": usize,
                    "offset": off})
        p += 46 + nlen + xlen + clen
    return out


class ShortRead(Exception):
    def __init__(self, need):
        super().__init__("short read; need %d bytes" % need)
        self.need = need


def decompress(data, method):
    if method == 0:
        return bytes(data)
    if method == 8:
        return zlib.decompress(data, -15)
    if method == 12:
        return bz2.decompress(data)
    if method == 14:
        psize = struct.unpack("<H", data[2:4])[0]
        filt = lzma._decode_filter_properties(lzma.FILTER_LZMA1, data[4:4 + psize])
        d = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[filt])
        return d.decompress(data[4 + psize:])
    raise ValueError("unsupported zip method %d" % method)


def member_bytes(blob, ent):
    """The uncompressed bytes of one member, from a blob starting at its local header."""
    if len(blob) < 30:
        raise ShortRead(30 + 1024 + ent["csize"])
    if blob[:4] != b"PK\x03\x04":
        raise ValueError("bad local header")
    nlen, xlen = struct.unpack("<HH", blob[26:30])
    start = 30 + nlen + xlen
    if len(blob) < start + ent["csize"]:
        raise ShortRead(start + ent["csize"])
    return decompress(blob[start:start + ent["csize"]], ent["method"])


# ======================================================================= network
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
            time.sleep(3 * (i + 1))
    raise last


def download(name, url, md5, size, data_dir):
    path = os.path.join(data_dir, name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
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


def read_central_directory(url, size):
    tail_start = max(0, size - EOCD_SPAN)
    info = find_eocd(http_get(url, (tail_start, size - 1)))
    if info["zip64"]:
        z = info["z64_eocd_offset"]
        info.update(parse_z64_eocd(http_get(url, (z, z + 55))))
    cd = http_get(url, (info["cd_offset"], info["cd_offset"] + info["cd_size"] - 1), timeout=300)
    return parse_central_directory(cd)


def archive_index(needed):
    lst = json.loads(http_get("%s/collections/%s/articles?page_size=1000" % (API, COLLECTION)))
    index = {}
    info = []
    collisions = 0
    for a in lst:
        m = re.search(r"files\s+(\d+)\s*$", a.get("title") or "")
        if not m:
            continue
        try:
            art = json.loads(http_get("%s/articles/%s" % (API, a["id"])))
            f = art["files"][0]
            entries = read_central_directory(f["download_url"], f["size"])
        except Exception as e:  # noqa: BLE001 - reported
            info.append({"archive": int(m.group(1)), "error": repr(e)[:300]})
            continue
        hit = 0
        for ent in entries:
            s = stem(ent["name"])
            if s in needed:
                if s in index:
                    collisions += 1
                index[s] = (f["download_url"], ent)
                hit += 1
        info.append({"archive": int(m.group(1)), "members": len(entries), "needed_here": hit,
                     "methods": dict(Counter(str(e["method"]) for e in entries)),
                     "first_names": [e["name"] for e in entries[:3]]})
    info.sort(key=lambda d: d["archive"])
    return index, info, collisions


def fetch_member(url, ent):
    need = 30 + len(ent["name"].encode("utf-8")) + 256 + ent["csize"]
    off = ent["offset"]
    blob = http_get(url, (off, off + need - 1))
    try:
        return member_bytes(blob, ent)
    except ShortRead as e:
        blob = http_get(url, (off, off + e.need - 1))
        return member_bytes(blob, ent)


# ======================================================================= audio and features
def wav_samples(b):
    """Mono float samples (first channel) and sample rate of a 16-bit PCM RIFF/WAVE file."""
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    p = 12
    fmt = None
    data = None
    while p + 8 <= len(b):
        cid = b[p:p + 4]
        size = struct.unpack("<I", b[p + 4:p + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", b[p + 8:p + 24])
        elif cid == b"data":
            data = b[p + 8:] if size in (0, 0xFFFFFFFF) else b[p + 8:p + 8 + size]
            break
        p += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    tag, ch, sr, _, _, bits = fmt
    if bits != 16 or tag not in (1, 0xFFFE):
        raise ValueError("unsupported wav format tag %d, %d bits" % (tag, bits))
    ch = max(1, ch)
    n = len(data) // (2 * ch)
    x = np.frombuffer(data[:2 * n * ch], dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return x, sr


def mel_filterbank(sr, nfft, n_mel=N_MEL, fmin=FMIN, fmax=FMAX):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    fmax = min(fmax, sr / 2.0)
    hz = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2))
    freqs = np.arange(nfft // 2 + 1) * float(sr) / nfft
    fb = np.zeros((n_mel, len(freqs)))
    for i in range(n_mel):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)))
    return fb


def dct_matrix(n_out=N_MFCC, n_in=N_MEL):
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    d = np.sqrt(2.0 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_in))
    d[0] /= np.sqrt(2.0)
    return d


def frame_starts(n, segments, win, hop):
    starts = []
    for s, e in segments:
        s = max(0, int(s))
        e = min(n, int(e))
        if e <= s:
            continue
        if e - s >= win:
            starts.extend(range(s, e - win + 1, hop))
        else:
            starts.append(max(0, min(n - win, (s + e) // 2 - win // 2)))
    return sorted(set(st for st in starts if st >= 0 and st + win <= n))


_FB = {}
_FB_LOCK = threading.Lock()


def _bank(sr, nfft):
    with _FB_LOCK:
        if (sr, nfft) not in _FB:
            _FB[(sr, nfft)] = mel_filterbank(sr, nfft)
        return _FB[(sr, nfft)]


_DCT = None


def call_features(x, sr, segments):
    """(frames x 64) MFCC matrix of one call and how frames were chosen ('segments' or 'energy')."""
    global _DCT
    if _DCT is None:
        _DCT = dct_matrix()
    win = int(round(WIN_S * sr))
    hop = max(1, int(round(HOP_S * sr)))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = len(x)
    starts = frame_starts(n, segments, win, hop)
    how = "segments"
    if not starts:
        how = "energy"
        starts = list(range(0, n - win + 1, hop))
        en = np.array([float(np.sum(x[s:s + win] ** 2)) for s in starts]) + 1e-20
        db = 10.0 * np.log10(en)
        starts = [s for s, v in zip(starts, db) if v >= db.max() - FALLBACK_DB]
    if len(starts) > MAX_FRAMES:
        idx = np.unique(np.round(np.linspace(0, len(starts) - 1, MAX_FRAMES)).astype(int))
        starts = [starts[i] for i in idx]
    nfft = max(NFFT, 1 << int(math.ceil(math.log2(win))))
    frames = np.stack([x[s:s + win] for s in starts]) * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=nfft, axis=1)) ** 2
    mel = spec @ _bank(sr, nfft).T
    return (np.log(mel + 1e-12) @ _DCT.T).astype(np.float32), how


# ======================================================================= GMM-UBM
def lse(a):
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True)))[:, 0]


class Ubm:
    def __init__(self, w, mu, var):
        self.w = np.asarray(w, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)
        self.ivar = 1.0 / self.var
        self.logw = np.log(self.w)
        self.cst = -0.5 * np.log(2.0 * np.pi * self.var).sum(axis=1)

    def comp_ll(self, x, x2iv, means):
        """(frames x K) log w_k + log N(x | means_k, var_k); x2iv = (x**2) @ ivar.T."""
        mi = means * self.ivar
        return (self.cst + self.logw - 0.5 * (means * mi).sum(axis=1))[None, :] \
            - 0.5 * x2iv + x @ mi.T


def fit_ubm(x, k=UBM_K, seed=SEED):
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(n_components=k, covariance_type="diag", max_iter=200, reg_covar=1e-3,
                        random_state=seed % (2 ** 31))
    g.fit(x)
    return Ubm(g.weights_, g.means_, g.covariances_)


def slices_of(day):
    out = []
    i0 = 0
    for i in range(1, len(day) + 1):
        if i == len(day) or day[i] != day[i0]:
            out.append((i0, i))
            i0 = i
    return out


def groups_of(keys):
    g = defaultdict(list)
    for i, k in enumerate(keys):
        g[k].append(i)
    return [np.array(v, dtype=np.int64) for _, v in sorted(g.items())]


class Emitter:
    """All calls of one emitter, sorted by day so that each day's calls (and frames) are
    contiguous. labels: 0..K-1 index into classes; day, ctxday: integer codes."""

    def __init__(self, eid, classes, labels, x, lens, day, ctxday, ubm):
        self.eid = eid
        self.classes = list(classes)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.X = np.asarray(x, dtype=np.float64)
        self.lens = np.asarray(lens, dtype=np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.lens)]).astype(np.int64)
        self.day = np.asarray(day, dtype=np.int64)
        self.ctxday = np.asarray(ctxday, dtype=np.int64)
        self.day_slices = slices_of(list(self.day))
        self.blocks = {"emitter": [np.arange(len(self.lens), dtype=np.int64)],
                       "day": [np.arange(a, b, dtype=np.int64) for a, b in self.day_slices],
                       "ctxday": groups_of(list(self.ctxday))}
        self.stats(ubm)

    def stats(self, ubm):
        self.X2iv = (self.X ** 2) @ ubm.ivar.T
        ll = ubm.comp_ll(self.X, self.X2iv, ubm.mu)
        g = np.exp(ll - lse(ll)[:, None])
        self.n = np.add.reduceat(g, self.off[:-1], axis=0)
        self.F = np.stack([g[a:b].T @ self.X[a:b]
                           for a, b in zip(self.off[:-1], self.off[1:])])

    def with_x(self, x, ubm):
        return Emitter(self.eid, self.classes, self.labels, x, self.lens, self.day, self.ctxday,
                       ubm)


def frame_means(fl, off):
    return np.add.reduceat(fl, off[:-1]) / np.diff(off)


def class_scores(e, ubm, labels, scheme):
    """(calls x classes) mean frame log-likelihood under MAP-adapted class models. 'loco': every
    call is scored by models trained without it; 'lodo': without any call of its day."""
    c_n = len(e.lens)
    k_n = len(e.classes)
    oh = np.zeros((c_n, k_n))
    oh[np.arange(c_n), labels] = 1.0
    nk = oh.T @ e.n
    fk = np.tensordot(oh.T, e.F, axes=1)
    rmu = RELEVANCE * ubm.mu
    s = np.empty((c_n, k_n))
    if scheme == "loco":
        for j in range(k_n):
            m = (fk[j] + rmu) / (nk[j] + RELEVANCE)[:, None]
            s[:, j] = frame_means(lse(ubm.comp_ll(e.X, e.X2iv, m)), e.off)
        for c in range(c_n):
            j = labels[c]
            a, b = e.off[c], e.off[c + 1]
            m = (fk[j] - e.F[c] + rmu) / (nk[j] - e.n[c] + RELEVANCE)[:, None]
            s[c, j] = lse(ubm.comp_ll(e.X[a:b], e.X2iv[a:b], m)).mean()
    elif scheme == "lodo":
        for i0, i1 in e.day_slices:
            nd = nk - oh[i0:i1].T @ e.n[i0:i1]
            fd = fk - np.tensordot(oh[i0:i1].T, e.F[i0:i1], axes=1)
            a, b = e.off[i0], e.off[i1]
            sub = e.off[i0:i1 + 1] - a
            for j in range(k_n):
                m = (fd[j] + rmu) / (nd[j] + RELEVANCE)[:, None]
                s[i0:i1, j] = frame_means(lse(ubm.comp_ll(e.X[a:b], e.X2iv[a:b], m)), sub)
    else:
        raise ValueError(scheme)
    return s


def balanced_accuracy(labels, pred, k):
    rec = []
    for j in range(k):
        m = labels == j
        if m.any():
            rec.append(float((pred[m] == j).mean()))
    return sum(rec) / len(rec)


def ba_of(e, ubm, labels, scheme):
    return balanced_accuracy(labels, class_scores(e, ubm, labels, scheme).argmax(axis=1),
                             len(e.classes))


def permute_within(labels, blocks, rng):
    out = labels.copy()
    for idx in blocks:
        if len(idx) > 1:
            out[idx] = labels[rng.permutation(idx)]
    return out


def run_test(es, ubm, scheme, block, b_n, rng, labels_list=None, keep_null=True):
    labels_list = labels_list if labels_list is not None else [e.labels for e in es]
    obs = [ba_of(e, ubm, lab, scheme) for e, lab in zip(es, labels_list)]
    t_obs = sum(obs) / len(obs)
    null = []
    for _ in range(b_n):
        row = []
        for e, lab in zip(es, labels_list):
            row.append(ba_of(e, ubm, permute_within(lab, e.blocks[block], rng), scheme))
        null.append(row)
    t_null = [sum(r) / len(r) for r in null]
    p = (1 + sum(1 for t in t_null if t >= t_obs - 1e-12)) / (b_n + 1.0)
    p_e = [(1 + sum(1 for r in null if r[i] >= obs[i] - 1e-12)) / (b_n + 1.0)
           for i in range(len(es))]
    res = {"scheme": scheme, "block": block, "B": b_n, "T": t_obs, "per_emitter": obs,
           "p": p, "p_emitter": p_e, "null_mean": sum(t_null) / len(t_null)}
    if keep_null:
        res["null_T"] = [round(t, 6) for t in t_null]
        res["null_per_emitter"] = [[round(v, 6) for v in r] for r in null]
    return res


def confound_labels(e, rng):
    """Labels drawn per call from day-specific class probabilities: day composition varies, but
    within a day labels are independent of the sound."""
    k_n = len(e.classes)
    base = np.bincount(e.labels, minlength=k_n) / float(len(e.labels))
    out = np.empty_like(e.labels)
    for i0, i1 in e.day_slices:
        w = base * np.exp(CONFOUND_STRENGTH * rng.standard_normal(k_n))
        w = w / w.sum()
        out[i0:i1] = rng.choice(k_n, size=i1 - i0, p=w)
    return out


def pos_world(e, ubm, d, sd, rng):
    """Real labels shuffled within day (real addressee information removed, day composition
    kept) plus a synthetic addressee effect: every frame of a call to class k is shifted by
    d * sd * z_k, z_k ~ N(0, I)."""
    base = permute_within(e.labels, e.blocks["day"], rng)
    z = rng.standard_normal((len(e.classes), e.X.shape[1]))
    shift = (d * sd)[None, :] * z
    return e.with_x(e.X + shift[np.repeat(base, e.lens)], ubm), base


# ======================================================================= verdict
def verdict(r):
    if r.get("gate"):
        return r["gate"]
    if r["confound"]["blocked_fp"] > CONFOUND_MAX_FP:
        return "CALIBRATION_FAILED"
    if r["tests"]["naive"]["p"] > ALPHA:
        return "NO_NAIVE_SIGNAL"
    if r["tests"]["day"]["p"] <= ALPHA:
        return "ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING"
    if r["power"]["detected"] >= POWER_MIN:
        return "NOT_DETECTED_WITH_POWER"
    return "UNDERPOWERED"


def outcomes(r):
    """Truth of Q1-Q6 (None when the result needed is missing) and the Brier score."""
    t = r.get("tests") or {}
    conf = r.get("confound") or {}
    pw = r.get("power") or {}

    def p_ok(name):
        return None if name not in t else t[name]["p"] <= ALPHA
    drop = None
    if "naive" in t and "day" in t:
        drop = t["day"]["null_mean"] - t["naive"]["null_mean"]
    truth = {"Q1": p_ok("naive"), "Q2": p_ok("day"), "Q3": p_ok("lodo"),
             "Q4": None if "naive_detected" not in conf else conf["naive_detected"] >= 2,
             "Q5": None if drop is None else drop >= DROP_MIN,
             "Q6": None if "detected" not in pw else pw["detected"] >= POWER_MIN}
    sq = [(prior - (1.0 if truth[q] else 0.0)) ** 2
          for q, _, prior in PREDICTIONS if truth[q] is not None]
    return truth, (sum(sq) / len(sq) if sq else None), drop


# ======================================================================= main
def save(out_dir, r):
    with open(os.path.join(out_dir, "bat_results.json"), "w") as f:
        json.dump(r, f, indent=1)


def finish(out_dir, r, t0):
    r["verdict"] = verdict(r)
    truth, brier, drop = outcomes(r)
    r["predictions"] = [{"id": q, "text": txt, "prior": pr, "came_true": truth[q]}
                        for q, txt, pr in PREDICTIONS]
    r["brier"] = brier
    r["day_composition_drop"] = drop
    r["seconds"] = round(time.time() - t0, 1)
    save(out_dir, r)
    print("VERDICT", r["verdict"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fetch-minutes", type=float, default=150.0)
    a = ap.parse_args(argv)
    os.makedirs(a.data_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    r = {"version": VERSION, "seed": SEED, "python": sys.version.split()[0],
         "numpy": np.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "config": {"EMITTERS": EMITTERS, "TREATMENTS": TREATMENTS, "CONTEXTS": CONTEXTS,
                    "MIN_CLASS": MIN_CLASS, "WIN_S": WIN_S, "HOP_S": HOP_S, "N_MFCC": N_MFCC,
                    "MAX_FRAMES": MAX_FRAMES, "UBM_K": UBM_K, "RELEVANCE": RELEVANCE,
                    "B": [B_NAIVE, B_DAY, B_CTXDAY, B_LODO, B_WORLD], "POS_GRID": POS_GRID},
         "files": {}}
    paths = {}
    for name, (url, md5, size) in LABEL_FILES.items():
        p, h = download(name, url, md5, size, a.data_dir)
        paths[name] = p
        r["files"][name] = {"md5": h, "md5_ok": h == md5}
    if not all(v["md5_ok"] for v in r["files"].values()):
        r["gate"] = "DATA_MISMATCH"
        return finish(a.out_dir, r, t0)
    ah, arows = read_csv(paths["Annotations.csv"])
    fh, frows = read_csv(paths["FileInfo.csv"])
    cols = label_columns(ah, fh)
    r["headers"] = {"annotations": ah, "fileinfo": fh}
    r["columns_used"] = cols
    if not all(cols[k] for k in ("key_a", "key_f", "emitter", "addressee", "context")):
        r["gate"] = "LABELS_UNPARSED"
        r["gate_reason"] = "a required column was not found"
        return finish(a.out_dir, r, t0)
    cand, counts = select_calls(ah, arows, fh, frows, cols)
    r["selection_counts"] = counts
    tried = counts.get("day_parsed", 0) + counts.get("day_missing", 0)
    if tried == 0 or counts.get("day_parsed", 0) < MIN_DAY_PARSED * tried:
        r["gate"] = "LABELS_UNPARSED"
        r["gate_reason"] = "recording days could not be parsed"
        ti = fh.index(cols["time"]) if cols["time"] else None
        r["time_samples"] = [row[ti] for row in frows[:5] if ti is not None and ti < len(row)]
        return finish(a.out_dir, r, t0)
    cand, classes0 = class_filter(cand)
    r["label_level"] = {"calls": len(cand), "emitters": len(classes0),
                        "segments_missing": sum(1 for c in cand if not c["segments"])}
    if len(classes0) < MIN_EMITTERS:
        r["gate"] = "TOO_FEW_EMITTERS"
        return finish(a.out_dir, r, t0)
    save(a.out_dir, r)

    # ---- audio -> features (cached in the data volume by file key)
    cache_path = os.path.join(a.data_dir, "features-%s.npz" % VERSION)
    feats = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        z_off, z_keys, z_how, z_x = z["off"], z["keys"], z["how"], z["X"]
        for i, k in enumerate(z_keys):
            feats[str(k)] = (z_x[z_off[i]:z_off[i + 1]], str(z_how[i]))
        del z
    need = [c for c in cand if c["key"] not in feats]
    index, ainfo, collisions = archive_index({c["stem"] for c in need}) if need else ({}, [], 0)
    r["archives"] = ainfo
    r["stem_collisions"] = collisions
    r["stem_examples"] = sorted({c["stem"] for c in need})[:5]
    errors = Counter()
    err_examples = []
    srs = Counter()
    deadline = time.time() + 60.0 * a.fetch_minutes

    def work(i):
        c = need[i]
        if time.time() > deadline:
            return i, None, "deadline", None
        hit = index.get(c["stem"])
        if hit is None:
            return i, None, "not_in_archives", None
        try:
            x, sr = wav_samples(fetch_member(hit[0], hit[1]))
            f, how = call_features(x, sr, c["segments"])
            return i, (f, how), None, sr
        except Exception as e:  # noqa: BLE001 - counted and reported
            return i, None, "error:" + type(e).__name__, repr(e)[:200]
    with ThreadPoolExecutor(max(1, a.threads)) as ex:
        for i, res, err, extra in ex.map(work, range(len(need))):
            if err:
                errors[err] += 1
                if extra and len(err_examples) < 10:
                    err_examples.append(extra)
            else:
                feats[need[i]["key"]] = res
                srs[str(extra)] += 1
    r["fetch"] = {"needed": len(need), "errors": dict(errors), "error_examples": err_examples,
                  "sample_rates": dict(srs), "seconds": round(time.time() - t0, 1)}
    if feats:
        keys = sorted(feats)
        lens = [len(feats[k][0]) for k in keys]
        np.savez(cache_path, keys=np.array(keys), how=np.array([feats[k][1] for k in keys]),
                 off=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                 X=np.concatenate([feats[k][0] for k in keys]).astype(np.float32))
    mapped = sum(1 for c in cand if c["key"] in feats)
    r["mapped"] = {"calls": len(cand), "with_features": mapped,
                   "fraction": mapped / float(len(cand)),
                   "energy_fallback": sum(1 for c in cand
                                          if c["key"] in feats and feats[c["key"]][1] == "energy")}
    if mapped < MIN_MAPPED * len(cand):
        r["gate"] = "MAPPING_FAILED"
        return finish(a.out_dir, r, t0)
    calls, classes = class_filter([c for c in cand if c["key"] in feats])
    n_inf, n_inf_calls = informative_blocks(calls)
    r["final"] = {"calls": len(calls), "emitters": len(classes), "informative_day_blocks": n_inf,
                  "calls_in_informative_blocks": n_inf_calls}
    if len(classes) < MIN_EMITTERS:
        r["gate"] = "TOO_FEW_EMITTERS"
        return finish(a.out_dir, r, t0)
    if n_inf < MIN_BLOCKS:
        r["gate"] = "NO_INFORMATIVE_BLOCKS"
        return finish(a.out_dir, r, t0)

    # ---- per-channel mean subtraction (label-free)
    ch_sum = defaultdict(lambda: np.zeros(N_MFCC))
    ch_n = Counter()
    for c in calls:
        f = feats[c["key"]][0].astype(np.float64)
        ch_sum[c["channel"]] += f.sum(axis=0)
        ch_n[c["channel"]] += len(f)
    ch_mean = {k: ch_sum[k] / ch_n[k] for k in ch_n}
    norm_f = {c["key"]: feats[c["key"]][0].astype(np.float64) - ch_mean[c["channel"]]
              for c in calls}
    feats = None
    allx = np.concatenate([norm_f[c["key"]] for c in calls])
    sd = allx.std(axis=0)
    rng = np.random.default_rng([SEED, 0, 0])
    take = rng.choice(len(allx), size=min(UBM_FRAMES, len(allx)), replace=False)
    ubm = fit_ubm(allx[np.sort(take)])
    del allx

    es = []
    emit_info = []
    for eid in sorted(classes):
        cs = sorted([c for c in calls if c["emitter"] == eid], key=lambda c: (c["day"], c["key"]))
        days = sorted({c["day"] for c in cs})
        cds = sorted({(c["context"], c["day"]) for c in cs})
        e = Emitter(eid, classes[eid], [classes[eid].index(c["addressee"]) for c in cs],
                    np.concatenate([norm_f[c["key"]] for c in cs]),
                    [len(norm_f[c["key"]]) for c in cs],
                    [days.index(c["day"]) for c in cs],
                    [cds.index((c["context"], c["day"])) for c in cs], ubm)
        es.append(e)
        inf = sum(1 for blk in e.blocks["day"] if len(set(e.labels[blk].tolist())) > 1)
        emit_info.append({"emitter": eid, "calls": len(cs), "classes": classes[eid],
                          "class_counts": np.bincount(e.labels, minlength=len(classes[eid]))
                          .tolist(), "days": len(days), "informative_days": inf,
                          "chance": 1.0 / len(classes[eid])})
    norm_f = None
    r["emitters"] = emit_info
    r["chance_mean"] = sum(d["chance"] for d in emit_info) / len(emit_info)
    r["tests"] = {}
    save(a.out_dir, r)

    # ---- the naive (published) test and the primary day-blocked test
    r["tests"]["naive"] = run_test(es, ubm, "loco", "emitter", B_NAIVE,
                                   np.random.default_rng([SEED, 1, 0]))
    save(a.out_dir, r)
    r["tests"]["day"] = run_test(es, ubm, "loco", "day", B_DAY,
                                 np.random.default_rng([SEED, 1, 1]))
    save(a.out_dir, r)

    # ---- calibration: confound worlds
    worlds = []
    for w in range(CONFOUND_WORLDS):
        g = np.random.default_rng([SEED, 2, w])
        labs = [confound_labels(e, g) for e in es]
        nv = run_test(es, ubm, "loco", "emitter", B_WORLD, g, labs, keep_null=False)
        bl = run_test(es, ubm, "loco", "day", B_WORLD, g, labs, keep_null=False)
        worlds.append({"world": w, "naive_T": nv["T"], "naive_p": nv["p"],
                       "naive_null_mean": nv["null_mean"], "blocked_p": bl["p"]})
        r["confound"] = {"worlds": worlds}
        save(a.out_dir, r)
    r["confound"] = {"worlds": worlds,
                     "naive_detected": sum(1 for w in worlds if w["naive_p"] <= ALPHA),
                     "blocked_fp": sum(1 for w in worlds if w["blocked_p"] <= ALPHA)}
    save(a.out_dir, r)

    # ---- power: effect-matched synthetic positive worlds
    grid = []
    for gi, d in enumerate(POS_GRID):
        g = np.random.default_rng([SEED, 3, gi])
        ws = [pos_world(e, ubm, d, sd, g) for e in es]
        t = sum(ba_of(w, ubm, b, "loco") for w, b in ws) / len(ws)
        grid.append({"d": d, "T_naive": t})
    t_obs = r["tests"]["naive"]["T"]
    best = min(range(len(grid)), key=lambda i: (abs(grid[i]["T_naive"] - t_obs), i))
    dstar = grid[best]["d"]
    reps = []
    for rep in range(POS_REPS):
        g = np.random.default_rng([SEED, 4, rep])
        ws = [pos_world(e, ubm, dstar, sd, g) for e in es]
        res = run_test([w for w, _ in ws], ubm, "loco", "day", B_WORLD, g, [b for _, b in ws],
                       keep_null=False)
        reps.append({"rep": rep, "T": res["T"], "p": res["p"]})
    r["power"] = {"grid": grid, "matched_d": dstar, "reps": reps,
                  "detected": sum(1 for x in reps if x["p"] <= ALPHA), "of": POS_REPS}
    save(a.out_dir, r)

    # ---- secondary tests
    r["tests"]["ctxday"] = run_test(es, ubm, "loco", "ctxday", B_CTXDAY,
                                    np.random.default_rng([SEED, 1, 2]))
    save(a.out_dir, r)
    r["tests"]["lodo"] = run_test(es, ubm, "lodo", "day", B_LODO,
                                  np.random.default_rng([SEED, 1, 3]))
    return finish(a.out_dir, r, t0)


if __name__ == "__main__":
    sys.exit(main())
