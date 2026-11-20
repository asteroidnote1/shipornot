"""verify_article02.py -- computes every number quoted in
articles/02_how_many_judge_labels.md.

All data is SYNTHETIC. The 50- and 200-label confusion tables are scaled
from the walkthrough's 100-label judge validation (73 human passes, 67 of
them judged pass; 27 human fails, 24 of them judged fail), rounding half up.


Usage: python verify_article02.py > article02_numbers.json
"""
import json
import math
import sys
from statistics import NormalDist

import shipornot as so

CONF = 0.95
SEED = 20261118          # numbers.json _meta.seed
N_BOOT = 10000           # bootstrap replicates for the kappa interval
LABEL_COUNTS = (50, 100, 200)
BASE = {"n": 100, "pos": 73, "tp": 67, "neg": 27, "tn": 24}
MODEL_A = {"k_obs": 192, "n_obs": 300}
WALKTHROUGH_100 = {
    "sens_ci": (0.832, 0.962), "spec_ci": (0.719, 0.961),
    "kappa": 0.779, "kappa_ci": (0.632, 0.901),
    "rg": 0.656, "rg_ci": (0.541, 0.754),
}
Z = NormalDist().inv_cdf(1 - (1 - CONF) / 2)


# ---- thin API layer: the only place that calls shipornot -------------------
def api_wilson(k, n):
    return so.wilson_interval(k, n, conf=CONF)


def api_kappa(judge, human):
    return so.cohen_kappa(judge, human, conf=CONF, n_boot=N_BOOT, seed=SEED)


def api_rogan_gladen(t):
    return so.rogan_gladen_interval(k_obs=MODEL_A["k_obs"], n_obs=MODEL_A["n_obs"],
                           tp=t["tp"], fn=t["fn"], tn=t["tn"], fp=t["fp"],
                           method="jeffreys", conf=CONF, seed=SEED)


# ---- result readers (accept dicts, objects, or tuples) ---------------------
def _pick(res, names):
    for name in names:
        if isinstance(res, dict) and name in res:
            return res[name]
        if not isinstance(res, dict) and hasattr(res, name):
            return getattr(res, name)
    return None


def interval(res):
    ci = _pick(res, ("ci", "interval"))
    if ci is not None:
        return float(ci[0]), float(ci[1])
    lo = _pick(res, ("lo", "lower", "ci_lo", "low"))
    hi = _pick(res, ("hi", "upper", "ci_hi", "high"))
    if lo is not None and hi is not None:
        return float(lo), float(hi)
    if isinstance(res, (tuple, list)) and len(res) in (2, 3):
        return float(res[-2]), float(res[-1])
    raise TypeError(f"cannot read an interval from {res!r}")


def point(res):
    est = _pick(res, ("estimate", "est", "kappa", "corrected", "rate", "p_hat"))
    if est is None and isinstance(res, (tuple, list)) and len(res) == 3:
        est = res[0]
    if est is None:
        raise TypeError(f"cannot read a point estimate from {res!r}")
    return float(est)


# ---- independent closed-form cross-checks ----------------------------------
def wilson_local(k, n):
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kappa_local(t):
    n = sum(t.values())
    po = (t["tp"] + t["tn"]) / n
    judge_pass = (t["tp"] + t["fp"]) / n
    human_pass = (t["tp"] + t["fn"]) / n
    pe = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
    return (po - pe) / (1 - pe)


def rogan_gladen_local(t):
    sens = t["tp"] / (t["tp"] + t["fn"])
    spec = t["tn"] / (t["tn"] + t["fp"])
    obs = MODEL_A["k_obs"] / MODEL_A["n_obs"]
    return (obs + spec - 1) / (sens + spec - 1)


# ---- synthetic tables ------------------------------------------------------
def table_for(n_labels):
    f = n_labels / BASE["n"]
    pos = math.floor(BASE["pos"] * f + 0.5)
    neg = n_labels - pos
    tp = math.floor(pos * BASE["tp"] / BASE["pos"] + 0.5)
    tn = math.floor(neg * BASE["tn"] / BASE["neg"] + 0.5)
    return {"tp": tp, "fn": pos - tp, "tn": tn, "fp": neg - tn}


def label_lists(t):
    judge = (["pass"] * t["tp"] + ["fail"] * t["fn"]
             + ["fail"] * t["tn"] + ["pass"] * t["fp"])
    human = ["pass"] * (t["tp"] + t["fn"]) + ["fail"] * (t["tn"] + t["fp"])
    return judge, human


def row(n_labels):
    t = table_for(n_labels)
    out = {"counts": t}
    for name, k, n in (("sens", t["tp"], t["tp"] + t["fn"]),
                       ("spec", t["tn"], t["tn"] + t["fp"])):
        lo, hi = interval(api_wilson(k, n))
        llo, lhi = wilson_local(k, n)
        assert abs(lo - llo) < 1e-9 and abs(hi - lhi) < 1e-9, \
            f"{name} Wilson mismatch at {n_labels} labels"
        out.update({name: k / n, f"{name}_lo": lo, f"{name}_hi": hi,
                    f"{name}_width": hi - lo})
    kres = api_kappa(*label_lists(t))
    assert abs(point(kres) - kappa_local(t)) < 1e-9, "kappa point mismatch"
    lo, hi = interval(kres)
    out.update({"kappa": point(kres), "kappa_lo": lo, "kappa_hi": hi,
                "kappa_width": hi - lo})
    rres = api_rogan_gladen(t)
    assert abs(point(rres) - rogan_gladen_local(t)) < 1e-9, "Rogan-Gladen mismatch"
    lo, hi = interval(rres)
    out.update({"rg": point(rres), "rg_lo": lo, "rg_hi": hi, "rg_width": hi - lo})
    return out


def check_walkthrough(r):
    """The 100-label row must reproduce examples/walkthrough.py."""
    w = WALKTHROUGH_100
    assert (round(r["sens_lo"], 3), round(r["sens_hi"], 3)) == w["sens_ci"]
    assert (round(r["spec_lo"], 3), round(r["spec_hi"], 3)) == w["spec_ci"]
    assert round(r["kappa"], 3) == w["kappa"]
    assert round(r["rg"], 3) == w["rg"]
    # Bootstrap / posterior-draw intervals depend on the seed: allow small drift.
    for key, (lo, hi) in (("kappa", w["kappa_ci"]), ("rg", w["rg_ci"])):
        assert abs(r[f"{key}_lo"] - lo) < 0.01 and abs(r[f"{key}_hi"] - hi) < 0.01, \
            f"{key} interval drifted from walkthrough: rerun with more draws"


def main():
    rows = {n: row(n) for n in LABEL_COUNTS}
    check_walkthrough(rows[100])
    flat = {"_meta": {"shipornot_version": getattr(so, "__version__", "unknown"),
                      "seed": SEED, "n_boot": N_BOOT, "conf": CONF,
                      "data": "SYNTHETIC"}}
    for n, r in rows.items():
        for key, val in r.items():
            flat[f"art02.n{n}.{key}"] = val
        for m in ("sens", "spec", "kappa", "rg"):
            flat[f"art02.n{n}.{m}_ci_and_width"] = (
                f"[{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}] "
                f"(width {r[m + '_width']:.3f})")
    for m in ("sens", "spec", "kappa", "rg"):
        flat[f"art02.ratio.{m}_width_200_over_100"] = (
            rows[200][f"{m}_width"] / rows[100][f"{m}_width"])
        flat[f"art02.ratio.{m}_width_100_over_50"] = (
            rows[100][f"{m}_width"] / rows[50][f"{m}_width"])
    json.dump(flat, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()