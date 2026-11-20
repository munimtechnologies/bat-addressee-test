# SPDX-License-Identifier: Apache-2.0
# test_bat_results.py: the grader for the pre-registered bat addressee test (BAT-PREREG.md, job
# bat-addressee-test-v1), written and published BEFORE the job has produced any result. With its
# own code (it does not import bat_addressee.py) it re-derives, from the raw job output
# (results/bat/bat_results.json): the gates of BAT-PREREG.md section 8 from the stored counts; T
# (mean balanced accuracy over emitters) of every test from the per-emitter values; bounds on every
# p value from the stored null distributions (null values are stored rounded to 6 decimals, so
# values within 1e-6 of T count as ties either way); the confound-world and power counts from the
# stored p values; the effect-matched d (grid value whose naive T is closest to the observed naive
# T, ties to the smaller); the verdict (section 8, in order); predictions Q1-Q6 and the Brier
# score. It checks every p value against its permutation lattice (1 + k) / (B + 1), the selection
# counts for monotonicity, the per-emitter table against the final counts, the settings against
# the registration, and everything against the job's own verdict, predictions and Brier score.
# It fixes now the exact lines RESULTS-BAT.md must contain (render_md_lines) and can write the
# whole document: python test_bat_results.py --render <bat_results.json> <RESULTS-BAT.md>.
# Until results/bat/bat_results.json exists the result tests are skipped; the toy tests always
# run. Stdlib only. MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents. Apache-2.0.
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "bat", "bat_results.json")
MD = os.path.join(HERE, "RESULTS-BAT.md")
HAVE = os.path.exists(RES)

VERSION = "bat-addressee-1"
SEED = 20261120
LABEL_MD5 = {"Annotations.csv": "7781a3725d6f2277de74da84b1383ce0",
             "FileInfo.csv": "b27252490ac5618bd55a9038110809d5"}
EMITTERS = [201, 203, 204, 205, 213, 223, 225, 226, 228]
POS_GRID = [0.005, 0.01, 0.02, 0.04, 0.08]
EXPECTED_CONFIG = {"EMITTERS": EMITTERS, "TREATMENTS": [9, 10], "CONTEXTS": [3, 4, 10, 11, 12],
                   "MIN_CLASS": 20, "WIN_S": 0.02, "HOP_S": 0.005, "N_MFCC": 64,
                   "MAX_FRAMES": 64, "UBM_K": 16, "RELEVANCE": 16.0,
                   "B": [99, 199, 99, 99, 49], "POS_GRID": POS_GRID}
ALPHA = 0.05
MIN_CLASS = 20
MIN_EMITTERS = 3
MIN_BLOCKS = 10
MIN_MAPPED = 0.80
MIN_DAY_PARSED = 0.90
B_WORLD = 49
CONFOUND_WORLDS = 4
CONFOUND_MAX_FP = 1
POS_REPS = 5
POWER_MIN = 4
DROP_MIN = 0.02
TOL_R = 1e-6        # stored null values are rounded to 6 decimals
REQUIRED_COLS = ("key_a", "key_f", "emitter", "addressee", "context")
SEL_KEYS = ("annotations", "emitter_ok", "addressee_ok", "context_ok", "single_annotation",
            "joined", "treatment_ok")
TESTS = ("naive", "day", "ctxday", "lodo")
TEST_SPEC = {"naive": ("loco", "emitter", 99), "day": ("loco", "day", 199),
             "ctxday": ("loco", "ctxday", 99), "lodo": ("lodo", "day", 99)}
TEST_TEXT = {"naive": "naive: free within emitter (as published)",
             "day": "day: within emitter x day (primary)",
             "ctxday": "ctxday: within emitter x context x day",
             "lodo": "lodo: within emitter x day"}
ORDER = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
PRIORS = {"Q1": 0.85, "Q2": 0.35, "Q3": 0.25, "Q4": 0.70, "Q5": 0.70, "Q6": 0.60}
PRED_TEXT = {  # exactly as in bat_addressee.PREDICTIONS
    "Q1": "naive (as published) test reproduces addressee information, p <= 0.05",
    "Q2": "primary day-blocked test detects addressee information, p <= 0.05",
    "Q3": "leave-one-day-out test detects addressee information, p <= 0.05",
    "Q4": "the naive test 'detects' at least 2 of 4 confound worlds",
    "Q5": "day composition alone is worth >= 0.02 balanced accuracy (null-mean gap)",
    "Q6": "the blocked test detects at least 4 of 5 effect-matched positive worlds"}
DM = "DATA_MISMATCH"
LU = "LABELS_UNPARSED"
TFE = "TOO_FEW_EMITTERS"
MF = "MAPPING_FAILED"
NIB = "NO_INFORMATIVE_BLOCKS"
INC = "INCOMPLETE"
STAT_KEYS = ("emitters", "chance_mean", "tests", "confound", "power")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def finite(x):
    return is_num(x) and x == x and abs(x) != float("inf")


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if not (is_num(a) and is_num(b)):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def p_ok(p, n):
    """p = (1 + k) / (1 + n) for an integer 0 <= k <= n."""
    if not finite(p):
        return False
    k = p * (n + 1) - 1
    return abs(k - round(k)) < 1e-6 and 0 <= round(k) <= n


def le_alpha(p):
    return finite(p) and p <= ALPHA


def n_rej(ps):
    return sum(1 for p in ps if le_alpha(p))


def mean(v):
    return sum(v) / len(v)


def argmin(v):
    best = 0
    for i in range(1, len(v)):
        if v[i] < v[best]:
            best = i
    return best


def p_bounds(null, t, b):
    """Smallest and largest p the job can have computed from these rounded null values."""
    k_lo = sum(1 for v in null if v >= t + TOL_R)
    k_hi = sum(1 for v in null if v >= t - TOL_R)
    return (1 + k_lo) / (b + 1.0), (1 + k_hi) / (b + 1.0)


def in_bounds(p, lohi):
    return lohi[0] - 1e-12 <= p <= lohi[1] + 1e-12


def unit(x):
    return finite(x) and 0.0 <= x <= 1.0


# --------------------------------------------------------------------------
# independent re-implementation of BAT-PREREG.md section 8
# --------------------------------------------------------------------------
def gate_of(r):
    files = r.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(LABEL_MD5):
        return INC
    if not all(isinstance(files[k], dict) and files[k].get("md5_ok") is True for k in files):
        return DM
    cols = r.get("columns_used")
    if not isinstance(cols, dict):
        return INC
    if not all(cols.get(k) for k in REQUIRED_COLS):
        return LU
    c = r.get("selection_counts")
    if not isinstance(c, dict):
        return INC
    dp, dmiss = c.get("day_parsed", 0), c.get("day_missing", 0)
    tried = dp + dmiss
    if tried == 0 or dp < MIN_DAY_PARSED * tried:
        return LU
    ll = r.get("label_level")
    if not isinstance(ll, dict):
        return INC
    if ll.get("emitters", 0) < MIN_EMITTERS:
        return TFE
    m = r.get("mapped")
    if not isinstance(m, dict):
        return INC
    if m.get("with_features", 0) < MIN_MAPPED * m.get("calls", 0):
        return MF
    f = r.get("final")
    if not isinstance(f, dict):
        return INC
    if f.get("emitters", 0) < MIN_EMITTERS:
        return TFE
    if f.get("informative_day_blocks", 0) < MIN_BLOCKS:
        return NIB
    return None


def complete(r):
    t = r.get("tests")
    c = r.get("confound")
    pw = r.get("power")
    return (isinstance(t, dict) and all(isinstance(t.get(k), dict) for k in TESTS)
            and isinstance(c, dict) and len(c.get("worlds") or []) == CONFOUND_WORLDS
            and isinstance(pw, dict) and len(pw.get("reps") or []) == POS_REPS
            and len(pw.get("grid") or []) == len(POS_GRID))


def verdict_of(blocked_fp, p_naive, p_day, detected):
    if blocked_fp > CONFOUND_MAX_FP:
        return "CALIBRATION_FAILED"
    if not le_alpha(p_naive):
        return "NO_NAIVE_SIGNAL"
    if le_alpha(p_day):
        return "ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING"
    if detected >= POWER_MIN:
        return "NOT_DETECTED_WITH_POWER"
    return "UNDERPOWERED"


def regrade(r):
    out = {"verdict": gate_of(r), "T": None, "p": None, "naive_detected": None,
           "blocked_fp": None, "matched_index": None, "matched_d": None, "detected": None,
           "drop": None, "predictions": None, "brier": None, "n_true": None}
    if out["verdict"]:
        return out
    if not complete(r):
        out["verdict"] = INC
        return out
    t = r["tests"]
    T = {k: mean(t[k]["per_emitter"]) for k in TESTS}
    p = {k: t[k]["p"] for k in TESTS}
    worlds = r["confound"]["worlds"]
    nd = n_rej([w["naive_p"] for w in worlds])
    fp = n_rej([w["blocked_p"] for w in worlds])
    grid = r["power"]["grid"]
    gi = argmin([abs(g["T_naive"] - T["naive"]) for g in grid])
    det = n_rej([x["p"] for x in r["power"]["reps"]])
    drop = t["day"]["null_mean"] - t["naive"]["null_mean"]
    pred = {"Q1": le_alpha(p["naive"]), "Q2": le_alpha(p["day"]), "Q3": le_alpha(p["lodo"]),
            "Q4": nd >= 2, "Q5": drop >= DROP_MIN, "Q6": det >= POWER_MIN}
    brier = sum((PRIORS[k] - (1.0 if pred[k] else 0.0)) ** 2 for k in ORDER) / len(ORDER)
    out.update(verdict=verdict_of(fp, p["naive"], p["day"], det), T=T, p=p, naive_detected=nd,
               blocked_fp=fp, matched_index=gi, matched_d=grid[gi]["d"], detected=det,
               drop=drop, predictions=pred, brier=brier,
               n_true=sum(1 for k in ORDER if pred[k]))
    return out


# --------------------------------------------------------------------------
# internal consistency of the raw output
# --------------------------------------------------------------------------
def check_stages(r):
    bad = []
    c = r.get("selection_counts")
    if isinstance(c, dict):
        seq = [c.get(k, 0) for k in SEL_KEYS]
        dp, dmiss = c.get("day_parsed", 0), c.get("day_missing", 0)
        if (not all(is_int(v) and v >= 0 for v in seq + [dp, dmiss])
                or any(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
                or seq[-1] != dp + dmiss):
            bad.append("selection counts")
    ll = r.get("label_level")
    if isinstance(ll, dict):
        v = [ll.get(k) for k in ("calls", "emitters", "segments_missing")]
        if (not isinstance(c, dict) or not all(is_int(x) and x >= 0 for x in v)
                or v[0] > c.get("day_parsed", 0) or v[1] > len(EMITTERS) or v[2] > v[0]):
            bad.append("label level")
    m = r.get("mapped")
    if isinstance(m, dict):
        v = [m.get(k) for k in ("calls", "with_features", "energy_fallback")]
        if (not isinstance(ll, dict) or not all(is_int(x) and x >= 0 for x in v)
                or v[0] != ll.get("calls") or v[1] > v[0] or v[2] > v[1] or v[0] == 0
                or not close(m.get("fraction"), v[1] / float(v[0]))):
            bad.append("mapped")
    f = r.get("final")
    if isinstance(f, dict):
        v = [f.get(k) for k in ("calls", "emitters", "informative_day_blocks",
                                "calls_in_informative_blocks")]
        if (not isinstance(m, dict) or not isinstance(ll, dict)
                or not all(is_int(x) and x >= 0 for x in v)
                or v[0] > m.get("with_features", 0) or v[1] > ll.get("emitters", 0)
                or not (2 * v[2] <= v[3] <= v[0])):
            bad.append("final")
    return bad


def check_emitters(r):
    em = r.get("emitters")
    f = r["final"]
    if not isinstance(em, list) or len(em) != f["emitters"]:
        return ["emitters"]
    bad = []
    ids = [e.get("emitter") for e in em]
    if ids != sorted(set(ids)) or not all(i in EMITTERS for i in ids):
        bad.append("emitter ids")
    for e in em:
        cl, cc = e.get("classes") or [], e.get("class_counts") or []
        ok = (is_int(e.get("calls")) and is_int(e.get("days")) and is_int(e.get("informative_days"))
              and len(cl) >= 2 and len(cl) == len(cc) and cl == sorted(set(cl))
              and e.get("emitter") not in cl and all(is_int(x) and x >= MIN_CLASS for x in cc)
              and sum(cc) == e.get("calls") and 1 <= e.get("days", 0) <= e.get("calls", 0)
              and 0 <= e.get("informative_days", -1) <= e.get("days", 0)
              and close(e.get("chance"), 1.0 / len(cl)))
        if not ok:
            bad.append("emitter %s" % e.get("emitter"))
    if bad:
        return bad
    if sum(e["calls"] for e in em) != f["calls"]:
        bad.append("emitter calls do not add up")
    if sum(e["informative_days"] for e in em) != f["informative_day_blocks"]:
        bad.append("informative days do not add up")
    if not close(r.get("chance_mean"), mean([e["chance"] for e in em])):
        bad.append("chance mean")
    return bad


def check_test(r, name, n_em):
    t = r["tests"][name]
    tag = "test " + name
    sch, blk, b = TEST_SPEC[name]
    if t.get("scheme") != sch or t.get("block") != blk or t.get("B") != b:
        return [tag + ": settings"]
    pe = t.get("per_emitter")
    if not isinstance(pe, list) or len(pe) != n_em or not all(unit(x) for x in pe):
        return [tag + ": per_emitter"]
    nt, npe = t.get("null_T"), t.get("null_per_emitter")
    if (not isinstance(nt, list) or len(nt) != b or not isinstance(npe, list) or len(npe) != b
            or not all(isinstance(row, list) and len(row) == n_em for row in npe)
            or not all(unit(x) for x in nt) or not all(unit(x) for row in npe for x in row)):
        return [tag + ": null"]
    bad = []
    T = mean(pe)
    if not close(t.get("T"), T, 1e-12):
        bad.append(tag + ": T")
    if any(abs(nt[i] - mean(npe[i])) > 1.5 * TOL_R for i in range(b)):
        bad.append(tag + ": null_T rows")
    if not (finite(t.get("null_mean")) and abs(t["null_mean"] - mean(nt)) <= TOL_R):
        bad.append(tag + ": null mean")
    p = t.get("p")
    if not p_ok(p, b):
        bad.append(tag + ": p lattice")
    elif not in_bounds(p, p_bounds(nt, T, b)):
        bad.append(tag + ": p outside its null")
    pes = t.get("p_emitter")
    if not isinstance(pes, list) or len(pes) != n_em:
        bad.append(tag + ": p_emitter")
    else:
        for i in range(n_em):
            col = [row[i] for row in npe]
            if not p_ok(pes[i], b) or not in_bounds(pes[i], p_bounds(col, pe[i], b)):
                bad.append("%s: p_emitter %d" % (tag, i))
    return bad


def check_cal(r):
    bad = []
    ws = r["confound"]["worlds"]
    for i, w in enumerate(ws):
        if (w.get("world") != i or not unit(w.get("naive_T")) or not unit(w.get("naive_null_mean"))
                or not p_ok(w.get("naive_p"), B_WORLD) or not p_ok(w.get("blocked_p"), B_WORLD)):
            bad.append("confound world %d" % i)
    pw = r["power"]
    for i, g in enumerate(pw["grid"]):
        if not close(g.get("d"), POS_GRID[i]) or not unit(g.get("T_naive")):
            bad.append("power grid %d" % i)
    for i, x in enumerate(pw["reps"]):
        if x.get("rep") != i or not unit(x.get("T")) or not p_ok(x.get("p"), B_WORLD):
            bad.append("power rep %d" % i)
    return bad


def check_result(r):
    bad = []
    if r.get("version") != VERSION:
        bad.append("version")
    if r.get("seed") != SEED:
        bad.append("seed")
    if r.get("config") != EXPECTED_CONFIG:
        bad.append("config")
    files = r.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(LABEL_MD5):
        return bad + ["files"]
    for k, v in files.items():
        if not isinstance(v, dict) or v.get("md5_ok") is not (v.get("md5") == LABEL_MD5[k]):
            bad.append("md5 flag " + k)
    g = gate_of(r)
    if g == DM:
        if any(k in r for k in ("columns_used", "selection_counts", "label_level") + STAT_KEYS):
            bad.append("results despite a data mismatch")
        return bad
    if g == INC:
        return bad + ["incomplete"]
    stage_bad = check_stages(r)
    bad += stage_bad
    if g:
        if r.get("gate") != g:
            bad.append("gate")
        if any(k in r for k in STAT_KEYS):
            bad.append("statistics despite gate " + g)
        return bad
    if r.get("gate"):
        bad.append("gate")
    if not complete(r):
        return bad + ["incomplete"]
    if stage_bad:
        return bad
    em = check_emitters(r)
    if em:
        return bad + em
    n_em = len(r["emitters"])
    for k in TESTS:
        bad += check_test(r, k, n_em)
    if not any(m.endswith("per_emitter") for m in bad):
        t = r["tests"]
        if not (t["naive"]["per_emitter"] == t["day"]["per_emitter"]
                == t["ctxday"]["per_emitter"]):
            bad.append("LOCO observed values differ between tests")
    return bad + check_cal(r)


def expected_predictions(rg):
    pr = rg["predictions"]
    return [{"id": q, "text": PRED_TEXT[q], "prior": PRIORS[q],
             "came_true": None if pr is None else pr[q]} for q in ORDER]


def compare_job(r, rg):
    """Mismatches between the job's own verdict and grading and the regrade."""
    bad = []
    if r.get("verdict") != rg["verdict"]:
        bad.append("verdict")
    if r.get("predictions") != expected_predictions(rg):
        bad.append("predictions")
    if not close(r.get("brier"), rg["brier"]):
        bad.append("brier")
    if not close(r.get("day_composition_drop"), rg["drop"]):
        bad.append("day_composition_drop")
    if rg["predictions"] is not None:
        c, pw = r["confound"], r["power"]
        if c.get("naive_detected") != rg["naive_detected"] or c.get("blocked_fp") != rg["blocked_fp"]:
            bad.append("confound counts")
        if pw.get("detected") != rg["detected"] or pw.get("of") != POS_REPS:
            bad.append("power count")
        if not close(pw.get("matched_d"), rg["matched_d"]):
            bad.append("matched d")
    return bad


# --------------------------------------------------------------------------
# formats fixed before the result exists (RESULTS-BAT.md must contain these lines)
# --------------------------------------------------------------------------
def yes_no(v):
    return "not run" if v is None else ("yes" if v else "no")


def md5_line(r):
    f = r.get("files") or {}
    return "Code version `{}`; label files md5 match: Annotations.csv {}, FileInfo.csv {}".format(
        r.get("version"), yes_no((f.get("Annotations.csv") or {}).get("md5_ok")),
        yes_no((f.get("FileInfo.csv") or {}).get("md5_ok")))


def selection_line(r):
    c = r["selection_counts"]
    g = [c.get(k, 0) for k in SEL_KEYS] + [c.get("day_parsed", 0), c.get("day_missing", 0)]
    return ("Selection: {:,} annotations; emitter one of the 9 adult females {:,}; addressee a "
            "known other bat {:,}; aggressive context {:,}; one annotation per file {:,}; found "
            "in FileInfo {:,}; treatment 9 or 10 {:,}; recording day parsed {:,}, not parsed "
            "{:,}").format(*g)


def label_line(r):
    ll = r["label_level"]
    return "Class rule on labels: {:,} calls from {} emitters ({:,} without voiced segments)".format(
        ll["calls"], ll["emitters"], ll["segments_missing"])


def mapped_line(r):
    m = r["mapped"]
    return ("Audio features for **{:,} of {:,}** candidate calls ({:.1f}%; {:,} by the 20 dB "
            "energy fallback)").format(m["with_features"], m["calls"],
                                       100.0 * m["with_features"] / m["calls"],
                                       m["energy_fallback"])


def final_line(r):
    f = r["final"]
    return ("Final data: **{:,}** calls from **{}** emitters; **{}** emitter-day blocks with calls "
            "to 2 or more addressees, holding {:,} calls").format(
        f["calls"], f["emitters"], f["informative_day_blocks"], f["calls_in_informative_blocks"])


def chance_line(r):
    return "Mean chance level over emitters: {:.4f}".format(r["chance_mean"])


def emitter_row(r, i):
    """| emitter | calls | addressees | calls per addressee | days | informative days | chance |
    BA (LOCO) | p naive | p day |"""
    e = r["emitters"][i]
    t = r["tests"]
    return "| {} | {:,} | {} | {} | {} | {} | {:.3f} | {:.4f} | {:.4f} | {:.4f} |".format(
        e["emitter"], e["calls"], ", ".join(str(x) for x in e["classes"]),
        ", ".join(str(x) for x in e["class_counts"]), e["days"], e["informative_days"],
        e["chance"], t["naive"]["per_emitter"][i], t["naive"]["p_emitter"][i],
        t["day"]["p_emitter"][i])


def test_row(r, name):
    """| test (null) | CV | B | T = mean BA | null mean | null max | p |"""
    t = r["tests"][name]
    return "| {} | {} | {} | **{:.4f}** | {:.4f} | {:.4f} | **{:.4f}** |".format(
        TEST_TEXT[name], TEST_SPEC[name][0].upper(), TEST_SPEC[name][2], mean(t["per_emitter"]),
        t["null_mean"], max(t["null_T"]), t["p"])


def drop_line(rg):
    return ("Day-composition gap (mean day-blocked null T minus mean naive null T): **{:+.4f}** "
            "balanced accuracy").format(rg["drop"])


def confound_row(r, i):
    """| world | naive T | naive null mean | naive p | day-blocked p |"""
    w = r["confound"]["worlds"][i]
    return "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
        w["world"], w["naive_T"], w["naive_null_mean"], w["naive_p"], w["blocked_p"])


def confound_line(rg):
    return ("Confound worlds (day composition only, B = 49): naive test detects **{} of 4**; "
            "day-blocked test detects **{} of 4** (calibration fails at 2 or more)").format(
        rg["naive_detected"], rg["blocked_fp"])


def grid_row(r, i):
    """| d | naive T |"""
    g = r["power"]["grid"][i]
    return "| {:.3f} | {:.4f} |".format(g["d"], g["T_naive"])


def rep_row(r, i):
    """| rep | T | day-blocked p |"""
    x = r["power"]["reps"][i]
    return "| {} | {:.4f} | {:.4f} |".format(x["rep"], x["T"], x["p"])


def power_line(r, rg):
    g = r["power"]["grid"][rg["matched_index"]]
    return ("Effect-matched d = **{:.3f}** (naive T there {:.4f}; observed naive T {:.4f}); "
            "day-blocked test detects **{} of 5** positive worlds (B = 49; 4 needed)").format(
        g["d"], g["T_naive"], rg["T"]["naive"], rg["detected"])


def verdict_line(rg):
    return "Verdict: **`%s`**" % rg["verdict"]


def pred_end(k, v):
    return "| {:.2f} | **{}** |".format(
        PRIORS[k], "not graded" if v is None else ("true" if v else "false"))


def pred_row(k, rg):
    pr = rg["predictions"] or {}
    return "| %s | %s %s" % (k, PRED_TEXT[k], pred_end(k, pr.get(k)))


def data_lines(r):
    out = [md5_line(r)]
    for key, fn in (("selection_counts", selection_line), ("label_level", label_line),
                    ("mapped", mapped_line), ("final", final_line)):
        if isinstance(r.get(key), dict):
            out.append(fn(r))
    return out


def score_lines(rg):
    if rg["predictions"] is None:
        return ["0 of 0 graded predictions true"]
    return ["%d of %d graded predictions true" % (rg["n_true"], len(ORDER)),
            "Brier score **%.3f**" % rg["brier"]]


def required_strings(r, rg):
    need = data_lines(r)
    if rg["predictions"] is not None:
        n_em = len(r["emitters"])
        need.append(chance_line(r))
        need += [emitter_row(r, i) for i in range(n_em)]
        need += [test_row(r, k) for k in TESTS] + [drop_line(rg)]
        need += [confound_row(r, i) for i in range(CONFOUND_WORLDS)] + [confound_line(rg)]
        need += [grid_row(r, i) for i in range(len(POS_GRID))]
        need += [rep_row(r, i) for i in range(POS_REPS)] + [power_line(r, rg)]
    need.append(verdict_line(rg))
    return need + score_lines(rg)


def render_md_lines(r, rg):
    """Every fixed line, in one list (prediction rows included)."""
    return required_strings(r, rg) + [pred_row(k, rg) for k in ORDER]


def check_md(md, r, rg):
    """List of required lines missing from md (empty list = all present)."""
    missing = [s for s in required_strings(r, rg) if s not in md]
    pr = rg["predictions"] or {}
    lines = md.splitlines()
    for k in ORDER:
        hits = [ln for ln in lines if ln.startswith("| %s |" % k)]
        if len(hits) != 1 or not hits[0].rstrip().endswith(pred_end(k, pr.get(k))):
            missing.append("prediction row " + k)
    return missing


def render_markdown(r, rg):
    """The whole RESULTS-BAT.md (every required line, inside headings and table headers)."""
    L = ["# RESULTS-BAT: addressee information in Egyptian fruit bat calls, blocked by "
         "recording day", "",
         "Graded by `test_bat_results.py` (published before the result was read) from "
         "`results/bat/bat_results.json`, the output of `bat_addressee.py` registered in "
         "`BAT-PREREG.md`. MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents; not yet "
         "checked by a human expert. Text CC BY 4.0.", "", "## Verdict", "", verdict_line(rg), ""]
    for s in score_lines(rg):
        L += [s, ""]
    L += ["## Data", ""]
    for s in data_lines(r):
        L += [s, ""]
    if rg["predictions"] is not None:
        n_em = len(r["emitters"])
        L += [chance_line(r), "", "## Per emitter", "",
              "| emitter | calls | addressees | calls per addressee | days | informative days "
              "| chance | BA (LOCO) | p naive | p day |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        L += [emitter_row(r, i) for i in range(n_em)]
        L += ["", "## Tests", "", "| test (null) | CV | B | T = mean BA | null mean | null max "
              "| p |", "|---|---|---|---|---|---|---|"]
        L += [test_row(r, k) for k in TESTS] + ["", drop_line(rg), ""]
        L += ["## Calibration: confound worlds", "",
              "| world | naive T | naive null mean | naive p | day-blocked p |",
              "|---|---|---|---|---|"]
        L += [confound_row(r, i) for i in range(CONFOUND_WORLDS)] + ["", confound_line(rg), ""]
        L += ["## Power: effect-matched positive worlds", "", "| d | naive T |", "|---|---|"]
        L += [grid_row(r, i) for i in range(len(POS_GRID))]
        L += ["", "| rep | T | day-blocked p |", "|---|---|---|"]
        L += [rep_row(r, i) for i in range(POS_REPS)] + ["", power_line(r, rg), ""]
    L += ["## Predictions (priors fixed in BAT-PREREG.md)", "",
          "| id | prediction | prior | came true |", "|---|---|---|---|"]
    L += [pred_row(k, rg) for k in ORDER]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# result tests (skipped until the job's output is published)
# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE, "results/bat/bat_results.json not published yet")
class TestBatResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RES, encoding="utf-8") as f:
            cls.r = json.load(f)
        cls.rg = regrade(cls.r)
        cls.md = open(MD, encoding="utf-8").read() if os.path.exists(MD) else ""

    def test_consistency(self):
        self.assertEqual(check_result(self.r), [])

    def test_job_matches_regrade(self):
        self.assertEqual(compare_job(self.r, self.rg), [])

    @unittest.skipUnless(os.path.exists(MD), "RESULTS-BAT.md not written yet")
    def test_md_lines(self):
        self.assertEqual(check_md(self.md, self.r, self.rg), [])


# --------------------------------------------------------------------------
# toy tests (always run; every expected value worked out by hand)
# --------------------------------------------------------------------------
OBS = [0.625, 0.5, 0.5625]          # mean 0.5625, exact in binary
OBS_LODO = [0.5, 0.5, 0.5625]       # mean 0.520833...


def mk_test(name, obs, spec):
    """A test block built the way bat_addressee.run_test builds it (null rows given as
    (count, row) pairs)."""
    sch, blk, b = TEST_SPEC[name]
    rows = []
    for n, row in spec:
        rows += [list(row) for _ in range(n)]
    assert len(rows) == b
    means = [sum(rw) / len(rw) for rw in rows]
    t_obs = sum(obs) / len(obs)
    p = (1 + sum(1 for m in means if m >= t_obs - 1e-12)) / (b + 1.0)
    pe = [(1 + sum(1 for rw in rows if rw[i] >= obs[i] - 1e-12)) / (b + 1.0)
          for i in range(len(obs))]
    return {"scheme": sch, "block": blk, "B": b, "T": t_obs, "per_emitter": list(obs), "p": p,
            "p_emitter": pe, "null_mean": sum(means) / len(means),
            "null_T": [round(m, 6) for m in means],
            "null_per_emitter": [[round(v, 6) for v in rw] for rw in rows]}


def toy_result(naive_hits=2, day_hits=79, blocked=None, reps_p=None):
    """Default: verdict NOT_DETECTED_WITH_POWER (naive p 0.03, day p 0.40, ctxday p 0.10,
    lodo p 0.50; confound 3 of 4 naive, 1 of 4 blocked; power 4 of 5 at d = 0.02)."""
    blocked = list(blocked or [0.5, 0.06, 0.8, 0.04])
    reps_p = list(reps_p or [0.02, 0.02, 0.04, 0.1, 0.02])
    naive_p = [0.02, 0.04, 0.3, 0.02]
    hi = [0.625, 0.5, 0.5625]
    tests = {"naive": mk_test("naive", OBS, [(naive_hits, hi), (99 - naive_hits, [0.5] * 3)]),
             "day": mk_test("day", OBS, [(day_hits, hi), (199 - day_hits, [0.5, 0.5625, 0.5])]),
             "ctxday": mk_test("ctxday", OBS, [(9, hi), (90, [0.5] * 3)]),
             "lodo": mk_test("lodo", OBS_LODO, [(49, [0.5, 0.5, 0.5625]), (50, [0.5] * 3)])}
    worlds = [{"world": i, "naive_T": t, "naive_null_mean": 0.5, "naive_p": a, "blocked_p": b}
              for i, (t, a, b) in enumerate(zip([0.58, 0.57, 0.52, 0.6], naive_p, blocked))]
    grid = [{"d": d, "T_naive": t} for d, t in zip(POS_GRID, [0.51, 0.53, 0.56, 0.62, 0.75])]
    reps = [{"rep": i, "T": t, "p": p} for i, (t, p) in
            enumerate(zip([0.57, 0.56, 0.55, 0.54, 0.58], reps_p))]
    em = [{"emitter": 201, "calls": 55, "classes": [203, 205], "class_counts": [30, 25],
           "days": 10, "informative_days": 4, "chance": 0.5},
          {"emitter": 203, "calls": 82, "classes": [201, 204, 213], "class_counts": [20, 22, 40],
           "days": 12, "informative_days": 5, "chance": 1.0 / 3},
          {"emitter": 205, "calls": 42, "classes": [201, 228], "class_counts": [21, 21],
           "days": 8, "informative_days": 3, "chance": 0.5}]
    return {"version": VERSION, "seed": SEED, "python": "3.12.8", "numpy": "2.1.3",
            "utc": "2026-11-21T00:00:00Z", "config": copy.deepcopy(EXPECTED_CONFIG),
            "files": {k: {"md5": v, "md5_ok": True} for k, v in LABEL_MD5.items()},
            "headers": {"annotations": ["FileID", "Emitter", "Addressee", "Context"],
                        "fileinfo": ["FileID", "Treatment", "File name", "Recording time"]},
            "columns_used": {"key_a": "FileID", "key_f": "FileID", "emitter": "Emitter",
                             "addressee": "Addressee", "context": "Context",
                             "treatment": "Treatment", "name": "File name",
                             "time": "Recording time"},
            "selection_counts": {"annotations": 91000, "emitter_ok": 20000,
                                 "addressee_ok": 9000, "context_ok": 7000,
                                 "single_annotation": 5000, "joined": 5000,
                                 "treatment_ok": 3000, "day_parsed": 3000},
            "label_level": {"calls": 250, "emitters": 3, "segments_missing": 10},
            "archives": [], "stem_collisions": 0, "stem_examples": [],
            "fetch": {"needed": 250, "errors": {"not_in_archives": 10}},
            "mapped": {"calls": 250, "with_features": 240, "fraction": 0.96,
                       "energy_fallback": 12},
            "final": {"calls": 179, "emitters": 3, "informative_day_blocks": 12,
                      "calls_in_informative_blocks": 60},
            "emitters": em, "chance_mean": (0.5 + 1.0 / 3 + 0.5) / 3, "tests": tests,
            "confound": {"worlds": worlds, "naive_detected": n_rej(naive_p),
                         "blocked_fp": n_rej(blocked)},
            "power": {"grid": grid, "matched_d": 0.02, "reps": reps,
                      "detected": n_rej(reps_p), "of": 5}}


def attach(r):
    """Add the job's own grading fields, as bat_addressee.finish writes them."""
    rg = regrade(r)
    r["verdict"] = rg["verdict"]
    r["predictions"] = expected_predictions(rg)
    r["brier"] = rg["brier"]
    r["day_composition_drop"] = rg["drop"]
    return r


LATER = ("archives", "stem_collisions", "stem_examples", "fetch", "mapped", "final")


def toy_gate(kind):
    r = toy_result()
    for k in STAT_KEYS:
        del r[k]
    if kind == DM:
        r["files"]["FileInfo.csv"] = {"md5": "0" * 32, "md5_ok": False}
        for k in ("headers", "columns_used", "selection_counts", "label_level") + LATER:
            del r[k]
    elif kind == LU:
        r["selection_counts"].update(day_parsed=2600, day_missing=400)
        del r["label_level"]
        for k in LATER:
            del r[k]
        r["time_samples"] = ["x"]
    elif kind == TFE:
        r["label_level"]["emitters"] = 2
        for k in LATER:
            del r[k]
    elif kind == MF:
        r["mapped"].update(with_features=199, fraction=0.796)
        del r["final"]
    elif kind == NIB:
        r["final"]["informative_day_blocks"] = 9
    r["gate"] = kind
    return attach(r)


class TestToy(unittest.TestCase):
    def test_default(self):
        r = toy_result()
        t = r["tests"]
        self.assertEqual([t[k]["p"] for k in TESTS], [0.03, 0.4, 0.1, 0.5])
        self.assertEqual(t["naive"]["p_emitter"], [0.03, 1.0, 0.03])
        self.assertEqual(t["day"]["p_emitter"], [0.4, 1.0, 0.4])
        self.assertEqual(t["lodo"]["p_emitter"], [1.0, 1.0, 0.5])
        rg = regrade(r)
        self.assertEqual(rg["verdict"], "NOT_DETECTED_WITH_POWER")
        self.assertEqual((rg["naive_detected"], rg["blocked_fp"], rg["detected"]), (3, 1, 4))
        self.assertEqual((rg["matched_index"], rg["matched_d"]), (2, 0.02))
        # day null mean (79 x 0.5625 + 120 x 0.520833..) / 199 = 0.537374..;
        # naive null mean (2 x 0.5625 + 97 x 0.5) / 99 = 0.501263..
        self.assertAlmostEqual(rg["drop"], 106.9375 / 199 - 49.625 / 99, places=12)
        self.assertEqual([rg["predictions"][k] for k in ORDER],
                         [True, False, False, True, True, True])
        # .0225 + .1225 + .0625 + .09 + .09 + .16 = .5475
        self.assertAlmostEqual(rg["brier"], 0.09125, places=12)
        self.assertEqual(rg["n_true"], 4)
        self.assertEqual(check_result(r), [])
        self.assertEqual(compare_job(r, rg),
                         ["verdict", "predictions", "brier", "day_composition_drop"])
        attach(r)
        r2 = json.loads(json.dumps(r))
        self.assertEqual(check_result(r2), [])
        self.assertEqual(compare_job(r2, regrade(r2)), [])
        r2["brier"] = 0.1
        r2["power"]["detected"] = 5
        r2["predictions"][1]["came_true"] = True
        self.assertEqual(compare_job(r2, regrade(r2)), ["predictions", "brier", "power count"])

    def test_verdicts(self):
        r = toy_result(day_hits=7)                                  # day p = 8/200 = 0.04
        rg = regrade(r)
        self.assertEqual(r["tests"]["day"]["p"], 0.04)
        self.assertEqual(rg["verdict"], "ADDRESSEE_SIGNAL_SURVIVES_DAY_BLOCKING")
        self.assertTrue(rg["predictions"]["Q2"])
        self.assertAlmostEqual(rg["brier"], 0.14125, places=12)
        self.assertEqual(check_result(r), [])
        rg = regrade(toy_result(reps_p=[0.02, 0.02, 0.04, 0.1, 0.5]))
        self.assertEqual(rg["verdict"], "UNDERPOWERED")
        self.assertFalse(rg["predictions"]["Q6"])
        self.assertAlmostEqual(rg["brier"], 0.7475 / 6, places=12)
        rg = regrade(toy_result(blocked=[0.5, 0.04, 0.8, 0.04]))
        self.assertEqual(rg["verdict"], "CALIBRATION_FAILED")
        r = toy_result(naive_hits=5)                                # naive p = 0.06
        self.assertEqual(regrade(r)["verdict"], "NO_NAIVE_SIGNAL")
        self.assertEqual(check_result(r), [])
        rg = regrade(toy_result(naive_hits=4))                      # p = 0.05 rejects
        self.assertEqual(rg["verdict"], "NOT_DETECTED_WITH_POWER")
        self.assertTrue(rg["predictions"]["Q1"])
        # calibration is checked before the naive signal
        rg = regrade(toy_result(naive_hits=5, blocked=[0.02, 0.04, 0.8, 0.5]))
        self.assertEqual(rg["verdict"], "CALIBRATION_FAILED")
        self.assertEqual(verdict_of(1, 0.01, 0.2, 3), "UNDERPOWERED")

    def test_gates(self):
        # data lines present + verdict + "0 of 0" + 6 prediction rows
        for kind, n_lines in ((DM, 9), (LU, 10), (TFE, 11), (MF, 12), (NIB, 13)):
            r = toy_gate(kind)
            rg = regrade(r)
            self.assertEqual(rg["verdict"], kind)
            self.assertIsNone(rg["predictions"])
            self.assertEqual(check_result(r), [])
            self.assertEqual(compare_job(r, rg), [])
            lines = render_md_lines(r, rg)
            self.assertEqual(len(lines), n_lines)
            self.assertEqual(check_md(render_markdown(r, rg), r, rg), [])
        r = toy_gate(MF)
        r["mapped"].update(with_features=200, fraction=0.8)
        r["final"] = {"calls": 179, "emitters": 3, "informative_day_blocks": 12,
                      "calls_in_informative_blocks": 60}
        self.assertEqual(gate_of(r), None)
        self.assertEqual(regrade(r)["verdict"], INC)
        self.assertIn("incomplete", check_result(r))
        r = toy_gate(NIB)
        r["tests"] = {}
        self.assertEqual(check_result(r), ["statistics despite gate " + NIB])
        r = toy_gate(DM)
        r["label_level"] = {}
        self.assertEqual(check_result(r), ["results despite a data mismatch"])
        r = toy_result()
        r["columns_used"]["addressee"] = None
        self.assertEqual(gate_of(r), LU)
        r = toy_result()
        r["selection_counts"].update(day_parsed=2700, day_missing=300)   # exactly 90% passes
        self.assertEqual(gate_of(r), None)
        self.assertEqual(check_result(r), [])
        r = toy_result()
        del r["power"]
        self.assertEqual(regrade(r)["verdict"], INC)

    def test_consistency_catches_errors(self):
        r = toy_result()
        r["tests"]["day"]["p"] = 0.45                  # on the 1/200 lattice, above its null
        r["tests"]["naive"]["p"] = 0.0123
        r["tests"]["lodo"]["T"] = 0.6
        r["confound"]["worlds"][1]["blocked_p"] = 0.01
        r["power"]["reps"][3]["rep"] = 7
        r["config"]["HOP_S"] = 0.001
        self.assertEqual(sorted(check_result(r)), sorted([
            "config", "test day: p outside its null", "test naive: p lattice", "test lodo: T",
            "confound world 1", "power rep 3"]))
        r = toy_result()
        r["emitters"][2]["class_counts"] = [19, 23]
        self.assertEqual(check_result(r), ["emitter 205"])
        r = toy_result()
        r["emitters"][2]["informative_days"] = 2
        self.assertEqual(check_result(r), ["informative days do not add up"])
        r = toy_result()
        r["selection_counts"]["joined"] = 5001
        r["mapped"]["fraction"] = 0.9
        self.assertEqual(check_result(r), ["selection counts", "mapped"])
        r = toy_result()
        r["tests"]["ctxday"]["per_emitter"] = [0.625, 0.5, 0.5]
        r["tests"]["ctxday"]["T"] = 1.625 / 3
        bad = check_result(r)
        self.assertIn("LOCO observed values differ between tests", bad)
        r = toy_result()
        r["tests"]["day"]["null_T"][0] = 0.7
        self.assertEqual(check_result(r), ["test day: null_T rows", "test day: null mean"])
        r = toy_result()
        r["tests"]["naive"]["p_emitter"][0] = 0.5      # its null allows 0.01 to 0.03 only
        self.assertEqual(check_result(r), ["test naive: p_emitter 0"])

    def test_helpers(self):
        self.assertTrue(p_ok(0.005, 199))
        self.assertTrue(p_ok(0.06, 49))
        self.assertFalse(p_ok(0.01, 49))
        self.assertFalse(p_ok(0.0, 99))
        # a null value 3.3e-7 below T (rounding) may or may not have counted
        self.assertEqual(p_bounds([0.520833, 0.5], 1.5625 / 3, 99), (0.01, 0.02))
        self.assertEqual(p_bounds([0.6, 0.5], 0.55, 99), (0.02, 0.02))
        self.assertEqual(argmin([0.3, 0.1, 0.1]), 1)
        self.assertTrue(close(None, None))
        self.assertFalse(close(None, 0.0))

    def test_formats(self):
        r = attach(toy_result())
        rg = regrade(r)
        self.assertEqual(md5_line(r), "Code version `bat-addressee-1`; label files md5 match: "
                                      "Annotations.csv yes, FileInfo.csv yes")
        self.assertEqual(selection_line(r),
                         "Selection: 91,000 annotations; emitter one of the 9 adult females "
                         "20,000; addressee a known other bat 9,000; aggressive context 7,000; "
                         "one annotation per file 5,000; found in FileInfo 5,000; treatment 9 or "
                         "10 3,000; recording day parsed 3,000, not parsed 0")
        self.assertEqual(label_line(r),
                         "Class rule on labels: 250 calls from 3 emitters (10 without voiced "
                         "segments)")
        self.assertEqual(mapped_line(r),
                         "Audio features for **240 of 250** candidate calls (96.0%; 12 by the "
                         "20 dB energy fallback)")
        self.assertEqual(final_line(r),
                         "Final data: **179** calls from **3** emitters; **12** emitter-day "
                         "blocks with calls to 2 or more addressees, holding 60 calls")
        self.assertEqual(chance_line(r), "Mean chance level over emitters: 0.4444")
        self.assertEqual(emitter_row(r, 0),
                         "| 201 | 55 | 203, 205 | 30, 25 | 10 | 4 | 0.500 | 0.6250 | 0.0300 | "
                         "0.4000 |")
        self.assertEqual(emitter_row(r, 1),
                         "| 203 | 82 | 201, 204, 213 | 20, 22, 40 | 12 | 5 | 0.333 | 0.5000 | "
                         "1.0000 | 1.0000 |")
        self.assertEqual(test_row(r, "naive"),
                         "| naive: free within emitter (as published) | LOCO | 99 | **0.5625** | "
                         "0.5013 | 0.5625 | **0.0300** |")
        self.assertEqual(test_row(r, "day"),
                         "| day: within emitter x day (primary) | LOCO | 199 | **0.5625** | "
                         "0.5374 | 0.5625 | **0.4000** |")
        self.assertEqual(test_row(r, "ctxday"),
                         "| ctxday: within emitter x context x day | LOCO | 99 | **0.5625** | "
                         "0.5057 | 0.5625 | **0.1000** |")
        self.assertEqual(test_row(r, "lodo"),
                         "| lodo: within emitter x day | LODO | 99 | **0.5208** | 0.5103 | "
                         "0.5208 | **0.5000** |")
        self.assertEqual(drop_line(rg),
                         "Day-composition gap (mean day-blocked null T minus mean naive null "
                         "T): **+0.0361** balanced accuracy")
        self.assertEqual(confound_row(r, 0), "| 0 | 0.5800 | 0.5000 | 0.0200 | 0.5000 |")
        self.assertEqual(confound_line(rg),
                         "Confound worlds (day composition only, B = 49): naive test detects "
                         "**3 of 4**; day-blocked test detects **1 of 4** (calibration fails at "
                         "2 or more)")
        self.assertEqual(grid_row(r, 0), "| 0.005 | 0.5100 |")
        self.assertEqual(grid_row(r, 2), "| 0.020 | 0.5600 |")
        self.assertEqual(rep_row(r, 0), "| 0 | 0.5700 | 0.0200 |")
        self.assertEqual(power_line(r, rg),
                         "Effect-matched d = **0.020** (naive T there 0.5600; observed naive T "
                         "0.5625); day-blocked test detects **4 of 5** positive worlds (B = 49; "
                         "4 needed)")
        self.assertEqual(verdict_line(rg), "Verdict: **`NOT_DETECTED_WITH_POWER`**")
        self.assertEqual(score_lines(rg), ["4 of 6 graded predictions true",
                                           "Brier score **0.091**"])
        self.assertEqual(pred_row("Q2", rg),
                         "| Q2 | primary day-blocked test detects addressee information, p <= "
                         "0.05 | 0.35 | **false** |")
        self.assertEqual(pred_end("Q6", None), "| 0.60 | **not graded** |")

    def test_md_check(self):
        r = attach(toy_result())
        rg = regrade(r)
        lines = render_md_lines(r, rg)
        # 5 data + chance + 3 emitters + 4 tests + drop + 4 worlds + confound + 5 grid
        # + 5 reps + power + verdict + 2 scores + 6 predictions
        self.assertEqual(len(lines), 39)
        md = render_markdown(r, rg)
        self.assertEqual(check_md(md, r, rg), [])
        v = verdict_line(rg)
        self.assertEqual(check_md(md.replace(v, "Verdict: none"), r, rg), [v])
        no_q3 = "\n".join(ln for ln in md.splitlines() if not ln.startswith("| Q3 |"))
        self.assertEqual(check_md(no_q3, r, rg), ["prediction row Q3"])
        self.assertTrue(all(ord(ch) < 128 for ch in md))
        r = toy_gate(DM)
        lines = render_md_lines(r, regrade(r))
        self.assertEqual(lines[0], "Code version `bat-addressee-1`; label files md5 match: "
                                   "Annotations.csv yes, FileInfo.csv no")
        self.assertEqual(lines[1], "Verdict: **`DATA_MISMATCH`**")
        self.assertEqual(lines[-1], "| Q6 | the blocked test detects at least 4 of 5 "
                                    "effect-matched positive worlds | 0.60 | **not graded** |")


def render_cli(src, dst):
    with open(src, encoding="utf-8") as f:
        r = json.load(f)
    rg = regrade(r)
    md = render_markdown(r, rg)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(json.dumps({"verdict": rg["verdict"], "check_result": check_result(r),
                      "compare_job": compare_job(r, rg), "missing": check_md(md, r, rg)}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--render":
        sys.exit(render_cli(sys.argv[2], sys.argv[3]))
    unittest.main()
