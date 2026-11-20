"""pytest suite for kit/shipornot.py.

Run from the repository's product/ directory (or from anywhere):

    python -m pytest kit/tests -q

statsmodels is used only as an independent reference. Tests that need it are
skipped when it is not installed. scikit-learn is not used: Cohen's kappa is
checked against a hand computation. Target runtime: well under 30 s.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

KIT_DIR = Path(__file__).resolve().parents[1]
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

import shipornot as es  # noqa: E402


# =============================================================================
# Independent reference implementations (written from the formulas, not the
# library code paths)
# =============================================================================


def wilson_ref(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    z = float(stats.norm.ppf(0.5 + conf / 2.0))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z / den * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def newcombe10_ref(n11: int, n10: int, n01: int, n00: int,
                   conf: float = 0.95) -> tuple[float, float, float]:
    n = n11 + n10 + n01 + n00
    p_b = (n11 + n01) / n
    p_a = (n11 + n10) / n
    l_b, u_b = wilson_ref(n11 + n01, n, conf)
    l_a, u_a = wilson_ref(n11 + n10, n, conf)
    prod = (n11 + n01) * (n10 + n00) * (n11 + n10) * (n01 + n00)
    phi = (n11 * n00 - n10 * n01) / math.sqrt(prod) if prod > 0 else 0.0
    d = p_b - p_a
    dl = math.sqrt(max((p_b - l_b) ** 2 - 2 * phi * (p_b - l_b) * (u_a - p_a) + (u_a - p_a) ** 2, 0.0))
    du = math.sqrt(max((u_b - p_b) ** 2 - 2 * phi * (u_b - p_b) * (p_a - l_a) + (p_a - l_a) ** 2, 0.0))
    return d, max(d - dl, -1.0), min(d + du, 1.0)


def kappa_ref(a: list, b: list) -> float:
    n = len(a)
    labels = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    return (po - pe) / (1.0 - pe)


def z_quantiles(alpha: float = 0.05, power: float = 0.80) -> tuple[float, float]:
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def ceil_ref(x: float) -> int:
    return int(math.ceil(x - 1e-9))


# =============================================================================
# 1. One proportion
# =============================================================================


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_wilson_matches_statsmodels(conf):
    smp = pytest.importorskip("statsmodels.stats.proportion")
    for n in (1, 5, 20, 40, 137):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            lo, hi = es.wilson_interval(k, n, conf)
            ref_lo, ref_hi = smp.proportion_confint(k, n, alpha=1 - conf, method="wilson")
            assert lo == pytest.approx(ref_lo, abs=1e-10)
            assert hi == pytest.approx(ref_hi, abs=1e-10)


def test_wilson_matches_hand_formula_32_of_40():
    r = es.wilson_interval(32, 40)
    lo, hi = wilson_ref(32, 40)
    assert r.lo == pytest.approx(lo, abs=1e-12)
    assert r.hi == pytest.approx(hi, abs=1e-12)
    assert r.lo < 0.8 < r.hi
    assert r._asdict() == {"lo": r.lo, "hi": r.hi}


@pytest.mark.parametrize("n", [1, 7, 30, 500])
def test_wilson_edges(n):
    z = float(stats.norm.ppf(0.975))
    r0 = es.wilson_interval(0, n)
    assert r0.lo == 0.0
    # textbook identity: Wilson upper bound after 0/n is z^2 / (n + z^2)
    assert r0.hi == pytest.approx(z * z / (n + z * z), rel=1e-12)
    rn = es.wilson_interval(n, n)
    assert rn.hi == 1.0
    assert rn.lo == pytest.approx(n / (n + z * z), rel=1e-12)


@pytest.mark.parametrize("conf", [0.90, 0.95])
def test_clopper_pearson_matches_statsmodels_interior(conf):
    smp = pytest.importorskip("statsmodels.stats.proportion")
    for n in (5, 20, 40, 137):
        for k in (1, n // 3, n // 2, n - 1):
            lo, hi = es.clopper_pearson_interval(k, n, conf)
            ref_lo, ref_hi = smp.proportion_confint(k, n, alpha=1 - conf, method="beta")
            assert lo == pytest.approx(ref_lo, abs=1e-10)
            assert hi == pytest.approx(ref_hi, abs=1e-10)


@pytest.mark.parametrize("n", [1, 10, 50])
def test_clopper_pearson_edges_closed_form(n):
    a = 0.05
    r0 = es.clopper_pearson_interval(0, n)
    assert r0.lo == 0.0
    assert r0.hi == pytest.approx(1 - (a / 2) ** (1 / n), rel=1e-10)
    rn = es.clopper_pearson_interval(n, n)
    assert rn.hi == 1.0
    assert rn.lo == pytest.approx((a / 2) ** (1 / n), rel=1e-10)


def test_rule_of_three_textbook():
    r = es.rule_of_three(100)
    assert r.rule_of_three_upper == pytest.approx(-math.log(0.05) / 100, rel=1e-12)
    assert r.rule_of_three_upper == pytest.approx(0.029957, abs=1e-6)
    assert r.exact_upper == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-12)
    # exact_upper is the one-sided Clopper-Pearson bound Beta.ppf(conf; 1, n)
    assert r.exact_upper == pytest.approx(float(stats.beta.ppf(0.95, 1, 100)), rel=1e-9)
    assert r.conf == 0.95


@pytest.mark.parametrize("p,h", [(0.5, 0.05), (0.8, 0.05), (0.9, 0.03), (0.0, 0.1)])
def test_n_for_wilson_halfwidth_is_minimal(p, h):
    n = es.n_for_wilson_halfwidth(p, h)

    def hw(m: int) -> float:
        lo, hi = wilson_ref(p * m, m)
        return (hi - lo) / 2.0

    assert hw(n) <= h + 1e-12
    if n > 1:
        assert hw(n - 1) > h


# =============================================================================
# 2. Graded scores
# =============================================================================


def test_mean_interval_t_matches_scipy():
    x = np.array([1, 0.5, 0, 1, 1, 0.75, 0.25, 1, 0.5, 0.9])
    r = es.mean_interval_t(x, 0.95)
    lo, hi = stats.t.interval(0.95, x.size - 1, loc=x.mean(), scale=stats.sem(x))
    assert r.estimate == pytest.approx(x.mean())
    assert r.lo == pytest.approx(lo, rel=1e-12)
    assert r.hi == pytest.approx(hi, rel=1e-12)
    assert r.n == 10


def test_mean_interval_t_constant_scores_zero_width():
    r = es.mean_interval_t([0.7, 0.7, 0.7])
    assert r.lo == pytest.approx(0.7) and r.hi == pytest.approx(0.7)
    assert r.se == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_deterministic_and_ordered():
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1, 60)
    for method in ("percentile", "bca"):
        r1 = es.bootstrap_interval(x, np.mean, n_boot=2000, seed=3, method=method)
        r2 = es.bootstrap_interval(x, np.mean, n_boot=2000, seed=3, method=method)
        r3 = es.bootstrap_interval(x, np.mean, n_boot=2000, seed=4, method=method)
        assert r1 == r2
        assert (r1.lo, r1.hi) != (r3.lo, r3.hi)
        assert r1.lo <= r1.estimate <= r1.hi
        assert r1.method == method and r1.n_boot == 2000
    med = es.bootstrap_interval(x, np.median, n_boot=500, seed=1, method="percentile")
    assert med.estimate == pytest.approx(float(np.median(x)))


def test_bca_close_to_t_for_large_normal_sample():
    x = np.random.default_rng(5).normal(0, 1, 400)
    b = es.bootstrap_interval(x, np.mean, n_boot=4000, seed=1, method="bca")
    t = es.mean_interval_t(x)
    assert b.lo == pytest.approx(t.lo, abs=0.02)
    assert b.hi == pytest.approx(t.hi, abs=0.02)


def test_paired_bootstrap_equals_bootstrap_of_differences():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 50)
    b = a + rng.normal(0.05, 0.1, 50)
    r1 = es.paired_bootstrap_diff(a, b, n_boot=1000, seed=5)
    r2 = es.bootstrap_interval(b - a, np.mean, 0.95, 1000, 5, "bca")
    assert r1 == r2
    assert r1.estimate == pytest.approx(float(np.mean(b) - np.mean(a)))


# =============================================================================
# 3. Paired comparison
# =============================================================================


def test_paired_counts_mixed_encodings():
    a = ["pass", "pass", "fail", "fail", "yes", "0"]
    b = ["pass", "fail", "pass", "fail", "TRUE", "1"]
    c = es.paired_counts(a, b)
    assert c == es.PairedCounts(n11=2, n10=1, n01=2, n00=1)
    assert sum(c) == 6


def test_exact_mcnemar_textbook_values():
    assert es.exact_mcnemar(0, 6) == pytest.approx(2 * 0.5 ** 6)            # 0.03125
    assert es.exact_mcnemar(1, 9) == pytest.approx(2 * 11 / 1024)          # 0.021484375
    assert es.exact_mcnemar(5, 5) == 1.0
    assert es.exact_mcnemar(0, 0) == 1.0
    assert es.exact_mcnemar(0, 0, "greater") == 1.0


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_exact_mcnemar_matches_scipy_binomtest(alternative):
    for n10 in range(0, 16):
        for n01 in range(0, 16):
            m = n10 + n01
            if m == 0:
                continue
            ref = stats.binomtest(n01, m, 0.5, alternative=alternative).pvalue
            assert es.exact_mcnemar(n10, n01, alternative) == pytest.approx(ref, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("table", [(30, 5, 12, 3), (80, 2, 9, 9), (10, 0, 4, 6), (1, 3, 0, 0),
                                   (0, 0, 7, 0), (50, 10, 10, 30)])
def test_paired_diff_interval_matches_hand_newcombe(table):
    r = es.paired_diff_interval(*table)
    d, lo, hi = newcombe10_ref(*table)
    n11, n10, n01, n00 = table
    assert r.diff == pytest.approx((n01 - n10) / sum(table))
    assert r.diff == pytest.approx(d)
    assert r.lo == pytest.approx(lo, abs=1e-12)
    assert r.hi == pytest.approx(hi, abs=1e-12)
    assert -1.0 <= r.lo <= r.diff <= r.hi <= 1.0


def test_paired_diff_interval_mirror_symmetry():
    r = es.paired_diff_interval(40, 3, 9, 8)
    s = es.paired_diff_interval(40, 9, 3, 8)
    assert s.diff == pytest.approx(-r.diff)
    assert s.lo == pytest.approx(-r.hi, abs=1e-12)
    assert s.hi == pytest.approx(-r.lo, abs=1e-12)


def test_paired_diff_no_discordant_pairs():
    r = es.paired_diff_interval(10, 0, 0, 10)
    assert r.diff == 0.0
    # documented behaviour: phi = 1 and pA = pB = 1/2 -> zero width
    assert r.lo == pytest.approx(0.0, abs=1e-9)
    assert r.hi == pytest.approx(0.0, abs=1e-9)
    r2 = es.paired_diff_interval(25, 0, 0, 0)  # all pass: margins zero -> phi = 0
    assert r2.lo <= 0.0 <= r2.hi


# =============================================================================
# 4. Sample size, power, MDE
# =============================================================================


@pytest.mark.parametrize("p1,p2", [(0.80, 0.84), (0.80, 0.85), (0.5, 0.6), (0.9, 0.8)])
def test_n_unpaired_matches_hand_formulas(p1, p2):
    za, zb = z_quantiles()
    q1, q2 = 1 - p1, 1 - p2
    pbar = (p1 + p2) / 2
    d2 = (p2 - p1) ** 2
    pooled = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * q1 + p2 * q2)) ** 2 / d2
    simple = (za + zb) ** 2 * (p1 * q1 + p2 * q2) / d2
    base = (za + zb) ** 2 * 2 * p1 * q1 / d2
    r = es.n_unpaired(p1, p2)
    assert r.n_pooled == ceil_ref(pooled)
    assert r.n_simple == ceil_ref(simple)
    assert r.n_baseline_variance == ceil_ref(base)
    assert r.two_sided is True and r.alpha == 0.05 and r.power == 0.80
    one = es.n_unpaired(p1, p2, two_sided=False)
    assert one.n_pooled < r.n_pooled


@pytest.mark.parametrize("psi,delta", [(0.10, 0.04), (0.20, 0.05), (0.30, -0.10), (0.05, 0.05)])
def test_n_paired_matches_connor(psi, delta):
    za, zb = z_quantiles()
    ref = (za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2
    assert es.n_paired(psi, delta) == max(ceil_ref(ref), 1)


def test_exact_mcnemar_power_at_connor_n_is_plausible():
    n = es.n_paired(0.10, 0.04)
    pw = es.power_exact_mcnemar_enum(n, 0.03, 0.07)
    assert pw.method == "exact_enumeration"
    # documented: exact conditional test is conservative -> usually below target
    assert 0.65 < pw.power < 0.85


def test_power_enum_matches_brute_force_small_n():
    n, p10, p01, alpha = 12, 0.15, 0.30, 0.05
    total = 0.0
    for n10 in range(n + 1):
        for n01 in range(n + 1 - n10):
            w = stats.multinomial.pmf([n10, n01, n - n10 - n01], n, [p10, p01, 1 - p10 - p01])
            if es.exact_mcnemar(n10, n01) <= alpha * (1 + 1e-9):
                total += w
    assert es.power_exact_mcnemar_enum(n, p10, p01, alpha).power == pytest.approx(total, abs=1e-10)


def test_power_sim_agrees_with_enum():
    sim = es.power_exact_mcnemar(100, 0.05, 0.15, sims=20_000, seed=1)
    enum = es.power_exact_mcnemar_enum(100, 0.05, 0.15)
    assert abs(sim.power - enum.power) <= 4 * sim.mc_se + 1e-9
    assert sim.mc_se == pytest.approx(math.sqrt(sim.power * (1 - sim.power) / 20_000))
    assert es.power_exact_mcnemar(100, 0.05, 0.15, sims=2000, seed=9) == \
        es.power_exact_mcnemar(100, 0.05, 0.15, sims=2000, seed=9)


def test_exact_mcnemar_type_one_error_bounded():
    for n in (20, 60, 150):
        assert es.power_exact_mcnemar_enum(n, 0.1, 0.1).power <= 0.05 + 1e-12
    assert es.power_exact_mcnemar_enum(50, 0.0, 0.0).power == 0.0


def test_mde_round_trips():
    d = es.mde_unpaired(200, 0.80)
    assert 0 < d < 0.2
    assert es.n_unpaired(0.80, 0.80 + d).n_pooled in (200, 201)
    dd = es.mde_unpaired(200, 0.80, direction="down")
    assert es.n_unpaired(0.80, 0.80 - dd).n_pooled in (200, 201)
    dp = es.mde_paired(200, 0.10)
    assert 0 < dp <= 0.10
    assert es.n_paired(0.10, dp) in (200, 201)
    assert math.isnan(es.mde_paired(5, 0.01))
    assert math.isnan(es.mde_unpaired(10, 1.0))


# =============================================================================
# 5-6. Repeats and clusters
# =============================================================================


def test_design_effect_and_effective_n_textbook():
    assert es.design_effect(5, 0.2) == pytest.approx(1.8)
    assert es.design_effect(1, 0.9) == 1.0
    assert es.effective_n_repeats(40, 5, 0.2) == pytest.approx(200 / 1.8)
    assert es.effective_n_repeats(40, 10**6, 0.2) == pytest.approx(200, rel=1e-4)
    assert es.effective_n_repeats(40, 5, 0.0) == pytest.approx(200)


def test_icc_matches_anova_f_identity():
    rng = np.random.default_rng(3)
    item = rng.uniform(0.1, 0.9, 30)
    y = (rng.uniform(size=(30, 4)) < item[:, None]).astype(float)
    f = stats.f_oneway(*y).statistic  # groups = items
    k = y.shape[1]
    assert es.icc_binary(y) == pytest.approx((f - 1) / (f + k - 1), rel=1e-10)


def test_icc_edge_cases():
    assert es.icc_binary([[1, 1, 1], [0, 0, 0], [1, 1, 1]]) == pytest.approx(1.0)
    assert math.isnan(es.icc_binary(np.ones((4, 3))))


def test_cluster_mean_hand_example():
    r = es.cluster_mean_interval([1, 0, 1, 1, 0, 1], ["a", "a", "b", "b", "c", "c"])
    assert r.estimate == pytest.approx(4 / 6)
    assert r.se == pytest.approx(1 / 6)
    assert r.n_clusters == 3 and r.n_obs == 6
    t = float(stats.t.ppf(0.975, 2))
    assert r.lo == pytest.approx(4 / 6 - t / 6)


def test_cluster_singletons_reduce_to_t_interval():
    y = np.random.default_rng(4).integers(0, 2, 50).astype(float)
    r = es.cluster_mean_interval(y, np.arange(50))
    t = es.mean_interval_t(y)
    assert r.design_effect == pytest.approx(1.0)
    assert r.se == pytest.approx(r.se_naive)
    assert (r.lo, r.hi) == pytest.approx((t.lo, t.hi))


def test_cluster_bootstrap_deterministic():
    rng = np.random.default_rng(8)
    ids = np.repeat(np.arange(20), 5)
    y = (rng.uniform(size=100) < 0.7).astype(float)
    r1 = es.cluster_bootstrap_interval(y, ids, n_boot=1000, seed=2)
    r2 = es.cluster_bootstrap_interval(y, ids, n_boot=1000, seed=2)
    assert r1 == r2
    assert r1.lo <= r1.estimate <= r1.hi
    rb = es.cluster_bootstrap_interval(y, ids, n_boot=1000, seed=2, method="bca")
    assert rb.lo <= rb.hi


# =============================================================================
# 7. Judge
# =============================================================================


def test_kappa_textbook_table():
    # 2x2: (A=1,B=1)=20, (1,0)=5, (0,1)=10, (0,0)=15 -> po=0.7, pe=0.5, kappa=0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    r = es.cohen_kappa(a, b, n_boot=1000, seed=0)
    assert r.kappa == pytest.approx(0.4)
    assert r.lo < 0.4 < r.hi
    assert r.n == 50 and r.n_boot_used > 900


def test_kappa_matches_hand_three_categories():
    rng = np.random.default_rng(12)
    a = list(rng.choice(["good", "ok", "bad"], 200))
    b = [x if rng.uniform() < 0.7 else str(rng.choice(["good", "ok", "bad"])) for x in a]
    r = es.cohen_kappa(a, b, n_boot=500, seed=1)
    assert r.kappa == pytest.approx(kappa_ref(a, b), rel=1e-12)
    assert r == es.cohen_kappa(a, b, n_boot=500, seed=1)


def test_kappa_edge_cases():
    assert es.cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1], n_boot=0).kappa == pytest.approx(1.0)
    r = es.cohen_kappa([1, 1, 1], [1, 1, 1], n_boot=200)
    assert math.isnan(r.kappa) and math.isnan(r.lo) and r.n_boot_used == 0
    r0 = es.cohen_kappa([1, 0, 1], [1, 0, 0], n_boot=0)
    assert math.isnan(r0.lo) and r0.n_boot_used == 0


def test_judge_confusion_counts():
    judge = [1, 1, 0, 0, 1, 0]
    human = [1, 0, 1, 0, 1, 0]
    jc = es.judge_confusion(judge, human)
    assert (jc.tp, jc.fn, jc.tn, jc.fp) == (2, 1, 2, 1)
    assert jc.sensitivity == pytest.approx(2 / 3)
    assert jc.specificity == pytest.approx(2 / 3)
    assert math.isnan(es.judge_confusion([1, 0], [0, 0]).sensitivity)


def test_rogan_gladen_textbook_and_clip():
    assert es.rogan_gladen(0.70, 0.90, 0.85) == pytest.approx(0.55 / 0.75)
    assert es.rogan_gladen(0.10, 0.90, 0.85) == 0.0
    assert es.rogan_gladen(0.10, 0.90, 0.85, clip=False) == pytest.approx(-0.05 / 0.75)
    assert es.rogan_gladen(0.99, 0.90, 0.99) == 1.0


def test_rogan_gladen_interval_delta_hand():
    r = es.rogan_gladen_interval(140, 200, 90, 10, 85, 15, method="delta")
    q, se, sp = 0.7, 0.9, 0.85
    j = se + sp - 1
    t = (q + sp - 1) / j
    var = (q * (1 - q) / 200 + t ** 2 * se * (1 - se) / 100 + (1 - t) ** 2 * sp * (1 - sp) / 100) / j ** 2
    h = float(stats.norm.ppf(0.975)) * math.sqrt(var)
    assert r.estimate == pytest.approx(t)
    assert r.lo == pytest.approx(t - h)
    assert r.hi == pytest.approx(t + h)
    assert r.n_draws_used == 0


def test_rogan_gladen_interval_mc_methods():
    for method in ("jeffreys", "bootstrap"):
        r = es.rogan_gladen_interval(140, 200, 90, 10, 85, 15, method=method, n_draws=4000, seed=1)
        assert r.lo < r.estimate < r.hi
        assert r.n_draws_used >= 2000
        assert r == es.rogan_gladen_interval(140, 200, 90, 10, 85, 15, method=method,
                                             n_draws=4000, seed=1)
    perfect = es.rogan_gladen_interval(150, 200, 50, 0, 50, 0, method="jeffreys", n_draws=2000)
    assert perfect.estimate == pytest.approx(0.75)
    assert math.isfinite(perfect.lo) and math.isfinite(perfect.hi)


# =============================================================================
# 8. Multiplicity, non-inferiority, peeking
# =============================================================================


def test_holm_bh_textbook():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    h = es.holm(p)
    assert h.adjusted == pytest.approx((0.05, 0.08, 0.09, 0.09, 0.09))
    assert h.reject == (True, False, False, False, False)
    bh = es.benjamini_hochberg(p)
    assert bh.adjusted == pytest.approx((0.05,) * 5)
    assert bh.reject == (True,) * 5
    # input order is preserved
    hs = es.holm([0.05, 0.01, 0.03, 0.02, 0.04])
    assert hs.adjusted == pytest.approx((0.09, 0.05, 0.09, 0.08, 0.09))


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_holm_bh_match_statsmodels(alpha):
    mt = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(7)
    p = np.concatenate([rng.uniform(0, 0.02, 4), rng.uniform(0, 1, 11)])
    p = np.append(p, p[2])  # a tie
    for method, func in (("holm", es.holm), ("fdr_bh", es.benjamini_hochberg)):
        rej, padj, _, _ = mt.multipletests(p, alpha=alpha, method=method)
        r = func(p, alpha)
        np.testing.assert_allclose(r.adjusted, padj, rtol=0, atol=1e-12)
        assert list(r.reject) == [bool(v) for v in rej]


def test_noninferiority_gate():
    good = es.noninferiority_paired(160, 2, 20, 18, margin=0.02)
    assert good.noninferior and good.superior_by_interval
    assert good.conf == pytest.approx(0.90)
    ref = es.paired_diff_interval(160, 2, 20, 18, 0.90)
    assert good.lo == pytest.approx(ref.lo) and good.hi == pytest.approx(ref.hi)
    bad = es.noninferiority_paired(150, 25, 5, 20, margin=0.02)
    assert not bad.noninferior and not bad.superior_by_interval


def test_peeking_gaussian_single_look_is_alpha():
    r = es.simulate_peeking_gaussian([200], reps=40_000, seed=0)
    assert abs(r.rate - 0.05) <= 4 * r.mc_se
    one = es.simulate_peeking_gaussian([200], reps=40_000, seed=1, two_sided=False)
    assert abs(one.rate - 0.05) <= 4 * one.mc_se


def test_peeking_gaussian_inflation_and_determinism():
    looks10 = list(range(20, 201, 20))
    r1 = es.simulate_peeking_gaussian([200], reps=20_000, seed=3)
    r10 = es.simulate_peeking_gaussian(looks10, reps=20_000, seed=3)
    assert r10.rate > r1.rate + 0.08
    assert r10 == es.simulate_peeking_gaussian(looks10, reps=20_000, seed=3)
    assert es.simulate_peeking_gaussian([100], effect=1.0, reps=1000).rate == 1.0


def test_peeking_mcnemar_single_look_matches_enum():
    r = es.simulate_peeking_mcnemar([100], 0.1, 0.1, reps=20_000, seed=0)
    enum = es.power_exact_mcnemar_enum(100, 0.1, 0.1).power
    assert abs(r.rate - enum) <= 4 * r.mc_se + 1e-9
    assert r == es.simulate_peeking_mcnemar([100], 0.1, 0.1, reps=20_000, seed=0)


# =============================================================================
# Utilities
# =============================================================================


def test_parse_pass_value():
    for v in (True, 1, 1.0, "1", "TRUE", " pass ", "Passed", "y", "yes", "t"):
        assert es.parse_pass_value(v) == 1
    for v in (False, 0, 0.0, "0", "false", "FAIL", "failed", "n", "no", "f"):
        assert es.parse_pass_value(v) == 0


def test_to_jsonable():
    out = es.to_jsonable({"r": es.Interval(np.float64(0.1), float("nan")),
                          "a": np.array([1, 2]), "b": np.bool_(True), "i": np.int64(3),
                          "inf": float("inf"), "t": (1, 2)})
    assert out == {"r": {"lo": 0.1, "hi": None}, "a": [1, 2], "b": True, "i": 3,
                   "inf": None, "t": [1, 2]}
    json.dumps(out, allow_nan=False)


INVALID_CALLS = [
    lambda: es.wilson_interval(5, 4),
    lambda: es.wilson_interval(1, 0),
    lambda: es.wilson_interval(1, 10, conf=1.0),
    lambda: es.wilson_interval(1, 10, conf=0.0),
    lambda: es.wilson_interval(True, 10),
    lambda: es.wilson_interval(1.5, 10),
    lambda: es.clopper_pearson_interval(-1, 10),
    lambda: es.rule_of_three(0),
    lambda: es.n_for_wilson_halfwidth(0.5, 0.6),
    lambda: es.mean_interval_t([1.0]),
    lambda: es.mean_interval_t([1.0, float("nan")]),
    lambda: es.bootstrap_interval([1, 2, 3], method="bogus"),
    lambda: es.bootstrap_interval([1, 2, 3], n_boot=5),
    lambda: es.paired_counts([1, 0], [1]),
    lambda: es.paired_counts([1, 2], [1, 0]),
    lambda: es.exact_mcnemar(-1, 3),
    lambda: es.exact_mcnemar(1, 3, alternative="bigger"),
    lambda: es.paired_diff_interval(0, 0, 0, 0),
    lambda: es.n_unpaired(0.8, 0.8),
    lambda: es.n_unpaired(0.8, 1.2),
    lambda: es.n_paired(0.10, 0.20),
    lambda: es.n_paired(0.0, 0.0),
    lambda: es.n_paired(0.1, 0.0),
    lambda: es.power_exact_mcnemar(100, 0.6, 0.6),
    lambda: es.power_exact_mcnemar_enum(100, 0.6, 0.6),
    lambda: es.mde_unpaired(200, 0.8, direction="sideways"),
    lambda: es.mde_paired(200, 0.0),
    lambda: es.design_effect(0.5, 0.1),
    lambda: es.effective_n_repeats(10, 3, 1.5),
    lambda: es.icc_binary([[1, 0]]),
    lambda: es.icc_binary([1, 0, 1]),
    lambda: es.cluster_mean_interval([1, 0, 1], [1, 1, 1]),
    lambda: es.cohen_kappa([1, 0], [1]),
    lambda: es.cohen_kappa([], []),
    lambda: es.judge_confusion([1, 0], [1]),
    lambda: es.rogan_gladen(0.7, 0.5, 0.5),
    lambda: es.rogan_gladen(1.2, 0.9, 0.9),
    lambda: es.rogan_gladen_interval(10, 20, 0, 0, 5, 5),
    lambda: es.rogan_gladen_interval(10, 20, 9, 1, 9, 1, method="wald"),
    lambda: es.holm([0.1, 1.2]),
    lambda: es.benjamini_hochberg([]),
    lambda: es.noninferiority_paired(10, 1, 1, 10, margin=0.0),
    lambda: es.noninferiority_paired(10, 1, 1, 10, margin=0.05, alpha=0.5),
    lambda: es.simulate_peeking_gaussian([100, 50]),
    lambda: es.simulate_peeking_gaussian([]),
    lambda: es.simulate_peeking_gaussian("100"),
    lambda: es.simulate_peeking_mcnemar([10, 20], 0.7, 0.7),
    lambda: es.parse_pass_value(""),
    lambda: es.parse_pass_value("maybe"),
    lambda: es.parse_pass_value(2),
]


@pytest.mark.parametrize("call", INVALID_CALLS, ids=[f"invalid_{i:02d}" for i in range(len(INVALID_CALLS))])
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


# =============================================================================
# CLI smoke tests via main([...])
# =============================================================================


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = es.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def run_json(capsys, *argv: str) -> dict:
    code, out, err = run_cli(capsys, *argv, "--json")
    assert code == 0, err
    return json.loads(out)


CLI_CASES = [
    (["wilson", "32", "40"], "wilson"),
    (["cp", "32", "40"], "cp"),
    (["mean", "--values", "1,0.5,0,1"], "mean"),
    (["bootstrap", "--values", "1,0.5,0,1,0.7", "--n-boot", "500", "--seed", "3"], "bootstrap"),
    (["bootstrap", "--a", "0.1,0.5,0.4", "--b", "0.3,0.6,0.4", "--n-boot", "200"], "bootstrap"),
    (["mcnemar", "--n10", "1", "--n01", "9"], "mcnemar"),
    (["paired-diff", "--n11", "30", "--n10", "5", "--n01", "12", "--n00", "3"], "paired-diff"),
    (["ni-gate", "--n11", "160", "--n10", "5", "--n01", "7", "--n00", "28", "--margin", "0.05"], "ni-gate"),
    (["n-unpaired", "0.8", "0.85"], "n-unpaired"),
    (["n-paired", "0.10", "0.04", "--check-exact"], "n-paired"),
    (["power-mcnemar", "100", "0.05", "0.15"], "power-mcnemar"),
    (["power-mcnemar", "100", "0.05", "0.15", "--method", "sim", "--sims", "2000"], "power-mcnemar"),
    (["mde", "200", "--design", "unpaired", "--p", "0.8"], "mde"),
    (["mde", "200", "--design", "paired", "--p-disc", "0.1"], "mde"),
    (["design-effect", "--m", "5", "--icc", "0.2", "--n-items", "40"], "design-effect"),
    (["kappa", "--a", "1,1,0,0,1,0", "--b", "1,0,0,0,1,1", "--confusion", "--n-boot", "200"], "kappa"),
    (["rogan-gladen", "--k-obs", "140", "--n-obs", "200", "--tp", "90", "--fn", "10",
      "--tn", "85", "--fp", "15", "--n-draws", "1000"], "rogan-gladen"),
    (["holm", "0.01,0.02", "0.03"], "holm"),
    (["bh", "0.01", "0.02", "0.03"], "bh"),
    (["peeking", "--looks", "50,100", "--reps", "2000"], "peeking"),
    (["peeking", "--looks", "50,100", "--model", "mcnemar", "--p10", "0.1", "--p01", "0.1",
      "--reps", "1000"], "peeking"),
]


@pytest.mark.parametrize("argv,command", CLI_CASES, ids=[" ".join(c[0][:2]) for c in CLI_CASES])
def test_cli_json_smoke(capsys, argv, command):
    out = run_json(capsys, *argv)
    assert out["command"] == command


def test_cli_values_match_library(capsys):
    w = run_json(capsys, "wilson", "32", "40")
    r = es.wilson_interval(32, 40)
    assert (w["lo"], w["hi"]) == pytest.approx((r.lo, r.hi))
    m = run_json(capsys, "mcnemar", "--n10", "1", "--n01", "9")
    assert m["p_value"] == pytest.approx(0.021484375) and m["reject"] is True
    u = run_json(capsys, "n-unpaired", "0.8", "0.85")
    assert u["n_pooled"] == es.n_unpaired(0.8, 0.85).n_pooled
    p = run_json(capsys, "n-paired", "0.10", "0.04")
    assert p["n"] == es.n_paired(0.10, 0.04)
    d = run_json(capsys, "design-effect", "--m", "5", "--icc", "0.2", "--n-items", "40")
    assert d["design_effect"] == pytest.approx(1.8) and d["n_eff"] == pytest.approx(200 / 1.8)
    z = run_json(capsys, "wilson", "0", "20")
    assert "one_sided_exact_upper_event_rate" in z
    nd = run_json(capsys, "paired-diff", "--n11", "10", "--n10", "0", "--n01", "0", "--n00", "10")
    assert "discordance_upper_one_sided" in nd


def test_cli_seeded_output_is_reproducible(capsys):
    args = ("bootstrap", "--values", "1,0.5,0,1,0.7,0.2", "--n-boot", "500", "--seed", "3")
    assert run_json(capsys, *args) == run_json(capsys, *args)


def test_cli_text_output(capsys):
    code, out, _ = run_cli(capsys, "wilson", "32", "40")
    assert code == 0 and out.startswith("Wilson 95%")
    code, out, _ = run_cli(capsys, "mcnemar", "--n10", "1", "--n01", "9")
    assert code == 0 and "reject H0" in out


def test_cli_csv_inputs(capsys, tmp_path):
    pair = tmp_path / "pair.csv"
    pair.write_text("case,A,B\n1,pass,pass\n2,pass,fail\n3,fail,pass\n4,fail,pass\n5,fail,fail\n",
                    encoding="utf-8")
    m = run_json(capsys, "mcnemar", "--csv", str(pair), "--a-col", "A", "--b-col", "B")
    assert (m["n11"], m["n10"], m["n01"], m["n00"]) == (1, 1, 2, 1)

    pv = tmp_path / "p.csv"
    pv.write_text("name,p\nx,0.01\ny,0.04\nz,0.30\n", encoding="utf-8")
    bh = run_json(capsys, "bh", "--csv", str(pv), "--col", "p")
    assert bh["adjusted"] == pytest.approx(list(es.benjamini_hochberg([0.01, 0.04, 0.30]).adjusted))

    mat = tmp_path / "m.csv"
    rows = [[1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
    mat.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n", encoding="utf-8")
    d = run_json(capsys, "design-effect", "--matrix-csv", str(mat))
    assert d["icc"] == pytest.approx(es.icc_binary(rows))

    cl = tmp_path / "c.csv"
    cl.write_text("value,doc\n1,a\n0,a\n1,b\n1,b\n0,c\n1,c\n", encoding="utf-8")
    c = run_json(capsys, "design-effect", "--csv", str(cl), "--value-col", "value",
                 "--cluster-col", "doc", "--n-boot", "200")
    assert c["se"] == pytest.approx(1 / 6) and "cluster_bootstrap" in c


@pytest.mark.parametrize("argv", [
    ["wilson", "50", "40"],
    ["mcnemar"],
    ["n-paired", "0.1", "0.2"],
    ["peeking", "--looks", "50,100", "--model", "mcnemar"],
    ["peeking", "--looks", "100,50"],
    ["holm", "0.1", "1.5"],
    ["mcnemar", "--csv", "/nonexistent/x.csv", "--a-col", "A", "--b-col", "B"],
])
def test_cli_invalid_input_returns_1(capsys, argv):
    code, out, err = run_cli(capsys, *argv)
    assert code == 1
    assert out == ""
    assert err.startswith("error:")


@pytest.mark.parametrize("argv,code", [([], 2), (["wilson", "x", "40"], 2),
                                       (["nosuchcommand"], 2), (["--version"], 0)])
def test_cli_argparse_exits(capsys, argv, code):
    with pytest.raises(SystemExit) as exc:
        es.main(argv)
    assert exc.value.code == code
    capsys.readouterr()


# Newcombe (1998) Stat Med 17:2635-2650, Table III: (e, f, g, h, method 8, method 10).
# theta = (f - g)/n with "first" = B, "second" = A, so n11=e, n01=f, n10=g, n00=h.
NEWCOMBE_TABLE_III = [
    (36, 12, 2, 0, (.0569, .3404), (.0569, .3404)), (20, 12, 2, 16, (.0618, .3242), (.0562, .3292)),
    (18, 12, 2, 18, (.0618, .3239), (.0562, .3290)), (36, 14, 0, 0, (.1528, .4167), (.1528, .4167)),
    (35, 14, 0, 1, (.1573, .4149), (.1461, .4175)), (18, 14, 0, 18, (.1504, .3910), (.1441, .3963)),
    (2, 97, 1, 0, (.8721, .9854), (.8721, .9854)), (1, 97, 1, 1, (.8737, .9850), (.8736, .9850)),
    (0, 29, 1, 0, (.6666, .9882), (.6666, .9882)), (2, 98, 0, 0, (.9178, .9945), (.9178, .9945)),
    (1, 98, 0, 1, (.9174, .9916), (.9171, .9916)), (0, 30, 0, 0, (.8395, 1.0), (.8395, 1.0)),
    (54, 0, 0, 0, (-.0664, .0664), (-.0664, .0664)),
]


@pytest.mark.parametrize("e,f,g,h,m8,m10", NEWCOMBE_TABLE_III)
def test_paired_diff_matches_newcombe_table_iii(e, f, g, h, m8, m10):
    r8 = es.paired_diff_interval(n11=e, n10=g, n01=f, n00=h)
    r10 = es.paired_diff_interval(n11=e, n10=g, n01=f, n00=h, cc_phi=True)
    # Table III is printed to 4 decimals; one entry (e,f,g,h = 1,97,1,1, method 10:
    # 0.8736) differs from the identical method-8 computation (0.8737) by one unit in
    # the last place, so the tolerance is one unit (1e-4) rather than half a unit.
    assert (r8.lo, r8.hi) == pytest.approx(m8, abs=1.01e-4)
    assert (r10.lo, r10.hi) == pytest.approx(m10, abs=1.01e-4)
