#!/usr/bin/env python3
"""shipornot: small-sample statistics for LLM evals (single file, NumPy + SciPy).

Conventions
-----------
* "A" is the baseline and "B" is the candidate. Differences are always B - A.
* Paired binary counts follow this 2x2 table:

      n11 = A pass, B pass        n10 = A pass, B fail   (regressions)
      n01 = A fail, B pass        n00 = A fail, B fail
                                  (n01 = improvements)

  The paired difference in pass rates is (n01 - n10) / n.
* Every random procedure takes an explicit ``seed`` and uses
  ``numpy.random.default_rng(seed)``. The same seed and the same NumPy version
  give the same output.
* Functions return NamedTuples. They unpack like tuples and convert to dicts
  with ``._asdict()``.

Command line
------------
``python shipornot.py --help``. Every subcommand prints JSON to stdout. It
exits 0 on success and 1 on invalid input. argparse usage errors exit 2.

References
----------
Each docstring cites the primary reference for its formula. The citations were
written from memory. Check them against the originals before publication.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import sys
from typing import Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import optimize, special, stats

__version__ = "0.1.0"

__all__ = [
    "wilson_interval", "clopper_pearson_interval", "rule_of_three",
    "mean_interval_t", "bootstrap_interval",
    "paired_counts", "exact_mcnemar", "paired_diff_interval", "paired_bootstrap_diff",
    "n_unpaired", "n_paired", "power_exact_mcnemar", "power_exact_mcnemar_enum",
    "mde_unpaired", "mde_paired",
    "design_effect", "effective_n_repeats", "icc_binary",
    "cluster_mean_interval", "cluster_bootstrap_interval",
    "cohen_kappa", "judge_confusion", "rogan_gladen", "rogan_gladen_interval",
    "n_for_wilson_halfwidth",
    "holm", "benjamini_hochberg", "noninferiority_paired",
    "simulate_peeking_gaussian", "simulate_peeking_mcnemar",
    "parse_pass_value", "to_jsonable", "main",
]

# A p-value counts as "<= alpha" if it is within this relative tolerance of
# alpha. This guards against floating-point rounding at the boundary.
_P_TOL = 1e-9
# Maximum number of array elements materialised per bootstrap chunk.
_CHUNK_ELEMENTS = 2_000_000


# =============================================================================
# Result types
# =============================================================================


class Interval(NamedTuple):
    lo: float
    hi: float


class ZeroOfN(NamedTuple):
    rule_of_three_upper: float
    exact_upper: float
    conf: float


class MeanInterval(NamedTuple):
    estimate: float
    lo: float
    hi: float
    se: float
    n: int
    conf: float


class BootstrapResult(NamedTuple):
    estimate: float
    lo: float
    hi: float
    method: str
    n_boot: int
    conf: float


class PairedCounts(NamedTuple):
    n11: int
    n10: int
    n01: int
    n00: int


class DiffInterval(NamedTuple):
    diff: float
    lo: float
    hi: float
    conf: float


class SampleSizeUnpaired(NamedTuple):
    n_pooled: int
    n_simple: int
    n_baseline_variance: int
    alpha: float
    power: float
    two_sided: bool


class PowerResult(NamedTuple):
    power: float
    mc_se: float
    sims: int
    method: str


class RateResult(NamedTuple):
    rate: float
    mc_se: float
    reps: int


class ClusterMeanResult(NamedTuple):
    estimate: float
    lo: float
    hi: float
    se: float
    se_naive: float
    design_effect: float
    n_clusters: int
    n_obs: int
    conf: float


class KappaResult(NamedTuple):
    kappa: float
    lo: float
    hi: float
    n: int
    conf: float
    n_boot_used: int


class JudgeConfusion(NamedTuple):
    tp: int
    fn: int
    tn: int
    fp: int
    sensitivity: float
    specificity: float


class RoganGladenResult(NamedTuple):
    estimate: float
    lo: float
    hi: float
    observed_rate: float
    sensitivity: float
    specificity: float
    method: str
    conf: float
    n_draws_used: int


class MultipleTestResult(NamedTuple):
    adjusted: tuple
    reject: tuple
    alpha: float
    method: str


class NIResult(NamedTuple):
    noninferior: bool
    superior_by_interval: bool
    diff: float
    lo: float
    hi: float
    conf: float
    margin: float
    alpha: float


# =============================================================================
# Validation helpers
# =============================================================================


def _check_open_unit(x: Any, name: str) -> float:
    """Return x as float if 0 < x < 1, else raise ValueError."""
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in (0, 1), got {x!r}") from exc
    if not 0.0 < v < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {x!r}")
    return v


def _check_prob(x: Any, name: str) -> float:
    """Return x as float if 0 <= x <= 1, else raise ValueError."""
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in [0, 1], got {x!r}") from exc
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {x!r}")
    return v


def _check_int(x: Any, name: str, minimum: int = 0) -> int:
    """Return x as int if it is an integer >= minimum, else raise ValueError."""
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got a bool")
    if isinstance(x, (int, np.integer)):
        xi = int(x)
    elif isinstance(x, (float, np.floating)) and math.isfinite(float(x)) and float(x).is_integer():
        xi = int(x)
    else:
        raise ValueError(f"{name} must be an integer, got {x!r}")
    if xi < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {xi}")
    return xi


def _check_count(k: Any, n: Any) -> tuple[int, int]:
    n_i = _check_int(n, "n", 1)
    k_i = _check_int(k, "k", 0)
    if k_i > n_i:
        raise ValueError(f"k must be <= n, got k={k_i}, n={n_i}")
    return k_i, n_i


def _as_1d_float(x: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


_TRUE = {"1", "1.0", "true", "t", "yes", "y", "pass", "passed"}
_FALSE = {"0", "0.0", "false", "f", "no", "n", "fail", "failed"}


def parse_pass_value(value: Any) -> int:
    """Convert a pass/fail cell to 1/0.

    Accepted values: booleans, 0/1, and (case-insensitive) "1", "0", "true",
    "false", "t", "f", "yes", "no", "y", "n", "pass", "fail", "passed", "failed".
    Anything else raises ValueError. Blank cells are rejected and never
    silently counted as fail.
    """
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return int(value)
    s = str(value).strip().lower()
    if s in _TRUE:
        return 1
    if s in _FALSE:
        return 0
    raise ValueError(f"cannot interpret {value!r} as pass/fail")


def _as_binary(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if arr.dtype.kind in "biuf":
        f = arr.astype(float)
        if not np.all((f == 0.0) | (f == 1.0)):
            raise ValueError(f"{name} must contain only 0/1 values")
        return f.astype(np.int64)
    return np.array([parse_pass_value(v) for v in arr], dtype=np.int64)


def _z_two_sided(conf: float) -> float:
    return float(stats.norm.ppf(0.5 + conf / 2.0))


def _z_alpha(alpha: float, two_sided: bool) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0 if two_sided else 1.0 - alpha))


def _reject(p: Any, alpha: float) -> Any:
    """Decision rule p <= alpha, with a tiny tolerance for rounding."""
    return p <= alpha * (1.0 + _P_TOL)


def _ceil(x: float) -> int:
    """Ceiling that ignores floating-point noise just above an integer."""
    return int(math.ceil(x - 1e-9))


# =============================================================================
# 1. One proportion
# =============================================================================


def _wilson_arrays(k: Any, n: Any, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Wilson score bounds (no input validation; n >= 1 assumed)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(np.maximum(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    lo = np.where(k == 0, 0.0, np.clip(center - half, 0.0, 1.0))
    hi = np.where(k == n, 1.0, np.clip(center + half, 0.0, 1.0))
    return lo, hi


def wilson_interval(k: int, n: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion k/n.

    Formula, with p = k/n and z the two-sided normal quantile:
        center = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z / (1 + z^2/n) * sqrt(p(1-p)/n + z^2/(4 n^2))
        interval = [center - half, center + half]

    Edge cases: for k = 0 the lower bound is exactly 0, and for k = n the
    upper bound is exactly 1. The formula gives these values up to rounding.
    For a one-sided bound after 0 failures or 0 passes, see ``rule_of_three``.

    Reference: Wilson EB (1927). Probable inference, the law of succession,
    and statistical inference. JASA 22(158):209-212.
    """
    k, n = _check_count(k, n)
    conf = _check_open_unit(conf, "conf")
    lo, hi = _wilson_arrays(k, n, _z_two_sided(conf))
    return Interval(float(lo), float(hi))


def clopper_pearson_interval(k: int, n: int, conf: float = 0.95) -> Interval:
    """Clopper-Pearson ("exact") interval for k/n.

    With a = 1 - conf:
        lo = Beta.ppf(a/2; k, n-k+1)    (0 if k = 0)
        hi = Beta.ppf(1-a/2; k+1, n-k)  (1 if k = n)

    Coverage is guaranteed to be at least conf for every true p, so the
    interval is conservative.

    Reference: Clopper CJ, Pearson ES (1934). The use of confidence or
    fiducial limits illustrated in the case of the binomial. Biometrika
    26(4):404-413.
    """
    k, n = _check_count(k, n)
    conf = _check_open_unit(conf, "conf")
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return Interval(lo, hi)


def rule_of_three(n: int, conf: float = 0.95) -> ZeroOfN:
    """One-sided upper bound on an event rate after observing 0 events in n.

    * Approximation: -ln(1 - conf) / n. At 95% this is 2.996/n, about 3/n
      (the "rule of three").
    * Exact one-sided Clopper-Pearson: 1 - (1 - conf)^(1/n).

    For the n-of-n case, apply the bound to failures. The pass rate is then at
    least 1 - upper. These are ONE-sided bounds. The two-sided Wilson or
    Clopper-Pearson intervals at the same conf are wider at the open end.

    Reference: Hanley JA, Lippman-Hand A (1983). If nothing goes wrong, is
    everything all right? JAMA 249(13):1743-1745.
    """
    n = _check_int(n, "n", 1)
    conf = _check_open_unit(conf, "conf")
    a = 1.0 - conf
    return ZeroOfN(-math.log(a) / n, 1.0 - a ** (1.0 / n), conf)


def n_for_wilson_halfwidth(p: float, halfwidth: float, conf: float = 0.95) -> int:
    """Smallest n such that the Wilson interval at an observed rate p has a
    half-width (hi - lo)/2 <= halfwidth.

    The function treats p as continuous (k = p*n). Use it to plan how many
    human labels pin down a judge's sensitivity or specificity, or a pass
    rate, to +/- halfwidth.
    """
    p = _check_prob(p, "p")
    if not 0.0 < float(halfwidth) < 0.5:
        raise ValueError("halfwidth must be in (0, 0.5)")
    conf = _check_open_unit(conf, "conf")
    z = _z_two_sided(conf)

    def hw(n: int) -> float:
        lo, hi = _wilson_arrays(p * n, n, z)
        return float(hi - lo) / 2.0

    if hw(1) <= halfwidth:
        return 1
    hi_n = 1
    while hw(hi_n) > halfwidth:
        hi_n *= 2
        if hi_n > 10**9:
            raise ValueError("halfwidth too small")
    lo_n = hi_n // 2
    while hi_n - lo_n > 1:
        mid = (lo_n + hi_n) // 2
        if hw(mid) <= halfwidth:
            hi_n = mid
        else:
            lo_n = mid
    return hi_n


# =============================================================================
# 2. Graded scores: t interval and bootstrap
# =============================================================================


def mean_interval_t(scores: npt.ArrayLike, conf: float = 0.95) -> MeanInterval:
    """Student-t interval for a mean: mean +/- t_{n-1} * s / sqrt(n).

    The interval is reasonable for graded scores (e.g. 1-5 or 0-1) when n is
    moderate. It is unreliable for very small n with skewed scores. Use
    ``bootstrap_interval`` with BCa there and check coverage in the
    verification report. If all scores are equal, the interval has zero width.

    Reference: Student (1908). The probable error of a mean. Biometrika 6(1):1-25.
    """
    x = _as_1d_float(scores, "scores")
    conf = _check_open_unit(conf, "conf")
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 scores")
    m = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    t = float(stats.t.ppf(0.5 + conf / 2.0, n - 1))
    return MeanInterval(m, m - t * se, m + t * se, se, n, conf)


def _apply_stat(stat: Callable, samples: np.ndarray) -> np.ndarray:
    """Apply stat row-wise. Use the axis= keyword when stat supports it."""
    try:
        vals = np.asarray(stat(samples, axis=1), dtype=float)
        if vals.shape == (samples.shape[0],):
            return vals
    except TypeError:
        pass
    return np.array([float(stat(row)) for row in samples])


def _bootstrap_reps(x: np.ndarray, stat: Callable, n_boot: int,
                    rng: np.random.Generator) -> np.ndarray:
    n = x.size
    chunk = max(1, min(n_boot, _CHUNK_ELEMENTS // n))
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = _apply_stat(stat, x[idx])
    return out


def _jackknife(x: np.ndarray, stat: Callable) -> np.ndarray:
    n = x.size
    if stat is np.mean:
        return (x.sum() - x) / (n - 1)
    return np.array([float(stat(np.delete(x, i))) for i in range(n)])


def _percentile_bounds(reps: np.ndarray, conf: float) -> tuple[float, float]:
    a = 1.0 - conf
    lo, hi = np.quantile(reps, [a / 2.0, 1.0 - a / 2.0])
    return float(lo), float(hi)


def _bca_bounds(theta_hat: float, reps: np.ndarray, jack: np.ndarray,
                conf: float) -> tuple[float, float]:
    """BCa bounds (Efron 1987).

    z0 = Phi^-1(share of reps below theta_hat; ties count one half).
    acc = sum(d^3) / (6 * (sum(d^2))^1.5), where d = mean(jack) - jack.
    Adjusted level for each z_q: Phi(z0 + (z0 + z_q) / (1 - acc (z0 + z_q))).
    """
    B = reps.size
    if np.ptp(reps) == 0.0:
        return float(reps[0]), float(reps[0])
    frac = (np.count_nonzero(reps < theta_hat) + np.count_nonzero(reps <= theta_hat)) / (2.0 * B)
    frac = min(max(frac, 0.5 / B), 1.0 - 0.5 / B)
    z0 = float(special.ndtri(frac))
    d = jack.mean() - jack
    ss = float(np.sum(d * d))
    acc = float(np.sum(d ** 3) / (6.0 * ss ** 1.5)) if ss > 0.0 else 0.0
    a = 1.0 - conf
    qs = []
    for zq in (float(special.ndtri(a / 2.0)), float(special.ndtri(1.0 - a / 2.0))):
        t = z0 + zq
        den = 1.0 - acc * t
        if den > 0.0:
            qs.append(float(special.ndtr(z0 + t / den)))
        else:  # degenerate acceleration; fall back to the extreme quantile
            qs.append(0.0 if zq < 0 else 1.0)
    lo, hi = np.quantile(reps, qs)
    return float(lo), float(hi)


def bootstrap_interval(scores: npt.ArrayLike, stat: Callable = np.mean,
                       conf: float = 0.95, n_boot: int = 10_000, seed: Any = 0,
                       method: str = "bca") -> BootstrapResult:
    """Nonparametric bootstrap interval for stat(scores).

    method="percentile": the (a/2, 1-a/2) quantiles of the bootstrap
    replicates.
    method="bca": bias-corrected and accelerated. The acceleration comes from
    the jackknife. See ``_bca_bounds``.

    ``stat`` should accept ``axis=1`` (np.mean and np.median do). If it does
    not, the function falls back to a slower row-by-row loop.

    References: Efron B (1987). Better bootstrap confidence intervals. JASA
    82(397):171-185. Efron B, Tibshirani RJ (1993). An Introduction to the
    Bootstrap. Chapman & Hall, ch. 13-14.
    """
    x = _as_1d_float(scores, "scores")
    if x.size < 2:
        raise ValueError("need at least 2 scores")
    conf = _check_open_unit(conf, "conf")
    n_boot = _check_int(n_boot, "n_boot", 10)
    if method not in ("percentile", "bca"):
        raise ValueError("method must be 'percentile' or 'bca'")
    rng = np.random.default_rng(seed)
    theta = float(stat(x))
    reps = _bootstrap_reps(x, stat, n_boot, rng)
    if method == "percentile":
        lo, hi = _percentile_bounds(reps, conf)
    else:
        lo, hi = _bca_bounds(theta, reps, _jackknife(x, stat), conf)
    return BootstrapResult(theta, lo, hi, method, n_boot, conf)


# =============================================================================
# 3. Paired comparison on the same cases
# =============================================================================


def paired_counts(a: npt.ArrayLike, b: npt.ArrayLike) -> PairedCounts:
    """Cross-tabulate paired pass/fail results.

    ``a`` is the baseline and ``b`` the candidate, aligned by case.
    Returns (n11, n10, n01, n00). n10 counts cases where A passed and B failed
    (regressions), and n01 counts cases where A failed and B passed
    (improvements).
    """
    x = _as_binary(a, "a")
    y = _as_binary(b, "b")
    if x.size != y.size:
        raise ValueError(f"a and b have different lengths ({x.size} vs {y.size})")
    n11 = int(np.sum((x == 1) & (y == 1)))
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n00 = int(np.sum((x == 0) & (y == 0)))
    return PairedCounts(n11, n10, n01, n00)


def _mcnemar_p_arrays(n10: Any, n01: Any) -> np.ndarray:
    """Vectorised two-sided exact McNemar p-values."""
    n10 = np.asarray(n10)
    n01 = np.asarray(n01)
    m = n10 + n01
    x = np.minimum(n10, n01)
    p = 2.0 * stats.binom.cdf(x, np.maximum(m, 1), 0.5)
    return np.where(m == 0, 1.0, np.minimum(p, 1.0))


def exact_mcnemar(n10: int, n01: int, alternative: str = "two-sided") -> float:
    """Exact (conditional binomial) McNemar test.

    Given m = n10 + n01 discordant pairs, under H0 n01 ~ Binomial(m, 1/2).
      two-sided: p = min(1, 2 * P(X <= min(n10, n01)))
      greater:   p = P(X >= n01)   (H1: B is better)
      less:      p = P(X <= n01)   (H1: B is worse)
    With no discordant pairs the function returns p = 1.

    Reference: McNemar Q (1947). Note on the sampling error of the difference
    between correlated proportions or percentages. Psychometrika 12(2):153-157.
    The exact conditional form is the binomial sign test on discordant pairs.
    """
    n10 = _check_int(n10, "n10")
    n01 = _check_int(n01, "n01")
    m = n10 + n01
    if m == 0:
        return 1.0
    if alternative == "two-sided":
        return float(_mcnemar_p_arrays(n10, n01))
    if alternative == "greater":
        return float(stats.binom.sf(n01 - 1, m, 0.5))
    if alternative == "less":
        return float(stats.binom.cdf(n01, m, 0.5))
    raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")


def _newcombe10_arrays(n11: Any, n10: Any, n01: Any, n00: Any,
                       z: float, cc_phi: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Newcombe (1998) paired score interval for diff = pB - pA (no validation).

    cc_phi=False is Newcombe's method 8 (plain phi; the kit's default).
    cc_phi=True is method 10: the numerator of phi, ad - bc, is replaced by
    max(ad - bc - n/2, 0) when ad > bc. Both checked against Newcombe's Table III.
    (The function name is kept for backward compatibility.)"""
    # Newcombe's notation: a=both, b=first only, c=second only, d=neither,
    # theta = p1 - p2 with p1 = (a+b)/n, p2 = (a+c)/n.
    # Mapping for B - A: "first" = B, "second" = A, so b = n01 and c = n10.
    a = np.asarray(n11, dtype=float)
    b = np.asarray(n01, dtype=float)
    c = np.asarray(n10, dtype=float)
    d = np.asarray(n00, dtype=float)
    n = a + b + c + d
    p1 = (a + b) / n
    p2 = (a + c) / n
    l1, u1 = _wilson_arrays(a + b, n, z)
    l2, u2 = _wilson_arrays(a + c, n, z)
    prod = (a + b) * (c + d) * (a + c) * (b + d)
    safe = np.where(prod > 0, prod, 1.0)
    num = a * d - b * c
    if cc_phi:
        num = np.where(num > 0, np.maximum(num - n / 2.0, 0.0), num)
    phi = np.where(prod > 0, num / np.sqrt(safe), 0.0)
    theta = p1 - p2
    dl = np.sqrt(np.maximum((p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    du = np.sqrt(np.maximum((u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return theta, np.maximum(theta - dl, -1.0), np.minimum(theta + du, 1.0)


def paired_diff_interval(n11: int, n10: int, n01: int, n00: int,
                         conf: float = 0.95, cc_phi: bool = False) -> DiffInterval:
    """Interval for pB - pA on the same cases (Newcombe 1998, method 8 by default).

    Default (cc_phi=False): Newcombe's method 8, the score-based interval with
    the plain phi coefficient. With cc_phi=True: method 10, which applies
    Newcombe's continuity correction to the numerator of phi (when positive).
    Both reproduce Table III of Newcombe (1998) to 4 decimals (see tests).

    This is the hybrid score interval. Let l_i, u_i be the Wilson limits for
    the marginal rates pB = (n11+n01)/n and pA = (n11+n10)/n. Let phi be the
    phi coefficient of the 2x2 table (set to 0 if any margin is 0). Then:
        L = d - sqrt((pB-lB)^2 - 2 phi (pB-lB)(uA-pA) + (uA-pA)^2)
        U = d + sqrt((uB-pB)^2 - 2 phi (uB-pB)(pA-lA) + (pA-lA)^2)
    where d = (n01 - n10)/n.

    Known behaviour: if there are no discordant pairs and both n11, n00 > 0,
    phi = 1 and the interval can be very narrow. It has zero width when
    pA = pB = 1/2. When n10 + n01 = 0, also report ``rule_of_three(n)`` for the
    discordance rate.

    Reference: Newcombe RG (1998). Improved confidence intervals for the
    difference between binomial proportions based on paired data. Statistics
    in Medicine 17(22):2635-2650.
    """
    n11 = _check_int(n11, "n11")
    n10 = _check_int(n10, "n10")
    n01 = _check_int(n01, "n01")
    n00 = _check_int(n00, "n00")
    if n11 + n10 + n01 + n00 < 1:
        raise ValueError("total count must be >= 1")
    conf = _check_open_unit(conf, "conf")
    d, lo, hi = _newcombe10_arrays(n11, n10, n01, n00, _z_two_sided(conf), cc_phi=cc_phi)
    return DiffInterval(float(d), float(lo), float(hi), conf)


def paired_bootstrap_diff(a_scores: npt.ArrayLike, b_scores: npt.ArrayLike,
                          conf: float = 0.95, n_boot: int = 10_000, seed: Any = 0,
                          method: str = "bca") -> BootstrapResult:
    """Paired bootstrap interval for mean(B) - mean(A) on graded scores.

    The function resamples cases with both scores kept together. This equals a
    bootstrap of the per-case differences b - a, because mean(b) - mean(a) =
    mean(b - a).

    Reference: Efron & Tibshirani (1993), An Introduction to the Bootstrap.
    """
    a = _as_1d_float(a_scores, "a_scores")
    b = _as_1d_float(b_scores, "b_scores")
    if a.size != b.size:
        raise ValueError("a_scores and b_scores must have the same length")
    return bootstrap_interval(b - a, np.mean, conf, n_boot, seed, method)


# =============================================================================
# 4. Sample size, power, minimum detectable effect
# =============================================================================


def _n_pooled_cont(p1: float, p2: float, za: float, zb: float) -> float:
    pbar = (p1 + p2) / 2.0
    d = abs(p2 - p1)
    num = za * math.sqrt(2.0 * pbar * (1.0 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return num * num / (d * d)


def n_unpaired(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80,
               two_sided: bool = True) -> SampleSizeUnpaired:
    """Cases per group to detect p1 vs p2 with two independent test sets.

    Returns three versions, each rounded up:
      n_pooled (recommended; Fleiss, no continuity correction):
          [z_a sqrt(2 pbar qbar) + z_b sqrt(p1 q1 + p2 q2)]^2 / (p2 - p1)^2
      n_simple (unpooled):
          (z_a + z_b)^2 (p1 q1 + p2 q2) / (p2 - p1)^2
      n_baseline_variance (shortcut that uses the baseline variance in both
      arms, common in blog posts):
          (z_a + z_b)^2 * 2 p1 q1 / (p2 - p1)^2
    Here z_a is the 1-alpha/2 normal quantile (1-alpha if one-sided) and z_b
    is the `power` quantile.

    Reference: Fleiss JL, Levin B, Paik MC (2003). Statistical Methods for
    Rates and Proportions, 3rd ed., Wiley, ch. 4.
    """
    p1 = _check_prob(p1, "p1")
    p2 = _check_prob(p2, "p2")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    alpha = _check_open_unit(alpha, "alpha")
    power = _check_open_unit(power, "power")
    za = _z_alpha(alpha, two_sided)
    zb = float(stats.norm.ppf(power))
    d2 = (p2 - p1) ** 2
    n_pooled = _ceil(_n_pooled_cont(p1, p2, za, zb))
    n_simple = _ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d2)
    n_base = _ceil((za + zb) ** 2 * 2.0 * p1 * (1 - p1) / d2)
    return SampleSizeUnpaired(max(n_pooled, 1), max(n_simple, 1), max(n_base, 1),
                              alpha, power, bool(two_sided))


def _n_paired_cont(psi: float, delta: float, za: float, zb: float) -> float:
    d = abs(delta)
    num = za * math.sqrt(psi) + zb * math.sqrt(max(psi - d * d, 0.0))
    return num * num / (d * d)


def n_paired(p_disc: float, delta: float, alpha: float = 0.05, power: float = 0.80,
             two_sided: bool = True) -> int:
    """Number of cases for a paired design (same cases run through A and B).

    Connor's formula:
        n = [z_a sqrt(psi) + z_b sqrt(psi - delta^2)]^2 / delta^2
    psi = p10 + p01 is the discordance rate, and delta = p01 - p10 is the
    difference in pass rates. Requires 0 < |delta| <= psi <= 1.

    The formula targets the asymptotic (unconditional) McNemar test. The
    exact conditional test is conservative, so its power at this n is
    usually somewhat lower. Check with ``power_exact_mcnemar_enum``.

    Reference: Connor RJ (1987). Sample size for testing differences in
    proportions for the paired-sample design. Biometrics 43(1):207-211.
    """
    psi = _check_prob(p_disc, "p_disc")
    delta = float(delta)
    if psi == 0.0:
        raise ValueError("p_disc must be > 0")
    if delta == 0.0 or abs(delta) > psi:
        raise ValueError("need 0 < |delta| <= p_disc")
    alpha = _check_open_unit(alpha, "alpha")
    power = _check_open_unit(power, "power")
    za = _z_alpha(alpha, two_sided)
    zb = float(stats.norm.ppf(power))
    return max(_ceil(_n_paired_cont(psi, delta, za, zb)), 1)


def _check_cells(p10: float, p01: float) -> tuple[float, float]:
    p10 = _check_prob(p10, "p10")
    p01 = _check_prob(p01, "p01")
    if p10 + p01 > 1.0 + 1e-12:
        raise ValueError("p10 + p01 must be <= 1")
    return p10, p01


def power_exact_mcnemar(n: int, p10: float, p01: float, alpha: float = 0.05,
                        sims: int = 20_000, seed: Any = 0) -> PowerResult:
    """Power of the two-sided exact McNemar test, by Monte Carlo simulation.

    The simulation draws (n10, n01, rest) ~ Multinomial(n; p10, p01, 1-p10-p01)
    `sims` times and rejects when p <= alpha. The result includes the Monte
    Carlo standard error sqrt(power(1-power)/sims).
    """
    n = _check_int(n, "n", 1)
    sims = _check_int(sims, "sims", 1)
    p10, p01 = _check_cells(p10, p01)
    alpha = _check_open_unit(alpha, "alpha")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [p10, p01, max(0.0, 1.0 - p10 - p01)], size=sims)
    rej = _reject(_mcnemar_p_arrays(draws[:, 0], draws[:, 1]), alpha)
    pw = float(np.mean(rej))
    return PowerResult(pw, math.sqrt(pw * (1 - pw) / sims), sims, "simulation")


@functools.lru_cache(maxsize=None)
def _mcnemar_critical(m: int, alpha: float) -> int:
    """Largest x with two-sided exact McNemar p(x; m) <= alpha, or -1."""
    x = np.arange(m // 2 + 1)
    p = np.minimum(1.0, 2.0 * stats.binom.cdf(x, m, 0.5))
    return int(np.count_nonzero(_reject(p, alpha))) - 1


def power_exact_mcnemar_enum(n: int, p10: float, p01: float,
                             alpha: float = 0.05) -> PowerResult:
    """Power of the two-sided exact McNemar test by exact enumeration.

    power = sum_m P(M = m) * P(reject | m), with M ~ Binomial(n, psi) and
    n01 | m ~ Binomial(m, p01/psi). For each m the rejection region is
    {x <= c_m} U {x >= m - c_m}. Terms with P(M = m) < 1e-16 are skipped. If
    p10 == p01, the result is the exact type-I error rate.
    """
    n = _check_int(n, "n", 1)
    p10, p01 = _check_cells(p10, p01)
    alpha = _check_open_unit(alpha, "alpha")
    psi = p10 + p01
    if psi == 0.0:
        return PowerResult(0.0, 0.0, 0, "exact_enumeration")
    pi = p01 / psi
    m_all = np.arange(n + 1)
    pm = stats.binom.pmf(m_all, n, psi)
    total = 0.0
    for m, w in zip(m_all, pm):
        if m == 0 or w < 1e-16:
            continue
        c = _mcnemar_critical(int(m), float(alpha))
        if c < 0:
            continue
        total += w * (stats.binom.cdf(c, m, pi) + stats.binom.sf(m - c - 1, m, pi))
    return PowerResult(float(min(total, 1.0)), 0.0, 0, "exact_enumeration")


def mde_unpaired(n: int, p: float, alpha: float = 0.05, power: float = 0.80,
                 two_sided: bool = True, direction: str = "up") -> float:
    """Minimum detectable effect with two independent test sets of n each.

    Solves n_pooled(p, p +/- delta) = n for delta (continuous version of
    ``n_unpaired``). Returns NaN if no delta in range reaches the power.
    """
    n = _check_int(n, "n", 1)
    p = _check_prob(p, "p")
    alpha = _check_open_unit(alpha, "alpha")
    power = _check_open_unit(power, "power")
    za = _z_alpha(alpha, two_sided)
    zb = float(stats.norm.ppf(power))
    if direction == "up":
        limit, sign = 1.0 - p, 1.0
    elif direction == "down":
        limit, sign = p, -1.0
    else:
        raise ValueError("direction must be 'up' or 'down'")
    if limit <= 0.0:
        return float("nan")

    def f(d: float) -> float:
        return _n_pooled_cont(p, p + sign * d, za, zb) - n

    if f(limit) > 0.0:
        return float("nan")
    return float(optimize.brentq(f, 1e-12, limit, xtol=1e-12))


def mde_paired(n: int, p_disc: float, alpha: float = 0.05, power: float = 0.80,
               two_sided: bool = True) -> float:
    """Minimum detectable paired difference for n cases with discordance p_disc.

    Solves Connor's formula n_paired(p_disc, delta) = n for delta in
    (0, p_disc]. Returns NaN if even delta = p_disc needs more than n cases.
    """
    n = _check_int(n, "n", 1)
    psi = _check_prob(p_disc, "p_disc")
    if psi == 0.0:
        raise ValueError("p_disc must be > 0")
    alpha = _check_open_unit(alpha, "alpha")
    power = _check_open_unit(power, "power")
    za = _z_alpha(alpha, two_sided)
    zb = float(stats.norm.ppf(power))

    def f(d: float) -> float:
        return _n_paired_cont(psi, d, za, zb) - n

    if f(psi) > 0.0:
        return float("nan")
    return float(optimize.brentq(f, 1e-12, psi, xtol=1e-12))


# =============================================================================
# 5-6. Repeats and clusters
# =============================================================================


def design_effect(m: float, icc: float) -> float:
    """Design effect for m correlated observations per unit: 1 + (m - 1) * icc.

    Reference: Kish L (1965). Survey Sampling. Wiley.
    """
    m = float(m)
    icc = float(icc)
    if not (math.isfinite(m) and m >= 1.0):
        raise ValueError("m must be >= 1")
    if not math.isfinite(icc):
        raise ValueError("icc must be finite")
    return 1.0 + (m - 1.0) * icc


def effective_n_repeats(n_items: int, k_repeats: int, icc: float) -> float:
    """Effective sample size of n_items cases each run k_repeats times.

    n_eff = n_items * k / (1 + (k - 1) * icc), with icc in [0, 1]. As k grows,
    n_eff approaches n_items / icc. Repeats cannot replace new cases when icc
    is high.
    """
    n_items = _check_int(n_items, "n_items", 1)
    k = _check_int(k_repeats, "k_repeats", 1)
    icc = _check_prob(icc, "icc")
    return n_items * k / design_effect(k, icc)


def icc_binary(matrix: npt.ArrayLike) -> float:
    """One-way ANOVA intraclass correlation, ICC(1,1), for an items x repeats
    matrix (rows = items, columns = repeated runs; 0/1 or graded values).

        MSB = k * sum_i (ybar_i - ybar)^2 / (n - 1)
        MSW = sum_ij (y_ij - ybar_i)^2 / (n (k - 1))
        ICC = (MSB - MSW) / (MSB + (k - 1) MSW)

    Returns NaN when MSB + (k-1) MSW = 0 (all values identical). The estimate
    can be negative in small samples.

    Reference: Shrout PE, Fleiss JL (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological Bulletin 86(2):420-428.
    """
    y = np.asarray(matrix, dtype=float)
    if y.ndim != 2:
        raise ValueError("matrix must be 2-D (items x repeats)")
    n, k = y.shape
    if n < 2 or k < 2:
        raise ValueError("need at least 2 items and 2 repeats")
    if not np.all(np.isfinite(y)):
        raise ValueError("matrix contains NaN or infinite values")
    row = y.mean(axis=1)
    grand = y.mean()
    msb = k * float(np.sum((row - grand) ** 2)) / (n - 1)
    msw = float(np.sum((y - row[:, None]) ** 2)) / (n * (k - 1))
    den = msb + (k - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def _group(values: Any, cluster_ids: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    y = _as_1d_float(values, "values")
    ids = np.asarray(cluster_ids)
    if ids.ndim != 1 or ids.size != y.size:
        raise ValueError("cluster_ids must be 1-D and the same length as values")
    _, inv = np.unique(ids, return_inverse=True)
    inv = np.asarray(inv).ravel()
    G = int(inv.max()) + 1
    if G < 2:
        raise ValueError("need at least 2 clusters")
    S = np.bincount(inv, weights=y, minlength=G)
    M = np.bincount(inv, minlength=G).astype(float)
    return y, S, M, G


def cluster_mean_interval(values: npt.ArrayLike, cluster_ids: npt.ArrayLike,
                          conf: float = 0.95) -> ClusterMeanResult:
    """Cluster-robust interval for the overall mean (e.g. pass rate) when
    several cases share a document or conversation.

    ybar = sum(y) / N. With cluster residual totals e_g = S_g - M_g * ybar:
        Var = G/(G-1) * sum_g e_g^2 / N^2         (CR1-style)
        interval = ybar +/- t_{G-1} * sqrt(Var)
    design_effect = Var / (s^2 / N), where s^2 is the plain sample variance.
    With fewer than about 30 clusters, coverage can fall below nominal. See
    the verification report.

    References: Liang KY, Zeger SL (1986). Biometrika 73(1):13-22. Cameron AC,
    Miller DL (2015). A practitioner's guide to cluster-robust inference.
    Journal of Human Resources 50(2):317-372.
    """
    conf = _check_open_unit(conf, "conf")
    y, S, M, G = _group(values, cluster_ids)
    N = y.size
    ybar = float(S.sum() / N)
    e = S - M * ybar
    var = G / (G - 1) * float(np.sum(e * e)) / N ** 2
    se = math.sqrt(var)
    s2 = float(y.var(ddof=1))
    se_naive = math.sqrt(s2 / N)
    deff = var / (s2 / N) if s2 > 0 else float("nan")
    t = float(stats.t.ppf(0.5 + conf / 2.0, G - 1))
    return ClusterMeanResult(ybar, ybar - t * se, ybar + t * se, se, se_naive, deff, G, N, conf)


def cluster_bootstrap_interval(values: npt.ArrayLike, cluster_ids: npt.ArrayLike,
                               conf: float = 0.95, n_boot: int = 10_000, seed: Any = 0,
                               method: str = "percentile") -> BootstrapResult:
    """Cluster bootstrap for the overall mean: resample whole clusters.

    Each replicate is sum(S_g*) / sum(M_g*) over G clusters drawn with
    replacement. method="bca" uses a leave-one-cluster-out jackknife for the
    acceleration.

    Reference: Davison AC, Hinkley DV (1997). Bootstrap Methods and their
    Application. Cambridge University Press, section 3.8.
    """
    conf = _check_open_unit(conf, "conf")
    n_boot = _check_int(n_boot, "n_boot", 10)
    if method not in ("percentile", "bca"):
        raise ValueError("method must be 'percentile' or 'bca'")
    _, S, M, G = _group(values, cluster_ids)
    theta = float(S.sum() / M.sum())
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    chunk = max(1, min(n_boot, _CHUNK_ELEMENTS // G))
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        idx = rng.integers(0, G, size=(stop - start, G))
        reps[start:stop] = S[idx].sum(axis=1) / M[idx].sum(axis=1)
    if method == "percentile":
        lo, hi = _percentile_bounds(reps, conf)
    else:
        jack = (S.sum() - S) / (M.sum() - M)
        lo, hi = _bca_bounds(theta, reps, jack, conf)
    return BootstrapResult(theta, lo, hi, method, n_boot, conf)


# =============================================================================
# 7. The judge as a noisy instrument
# =============================================================================


def _kappa_from_tables(t: Any) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    n = np.asarray(t.sum(axis=(-2, -1)))
    po = np.trace(t, axis1=-2, axis2=-1) / n
    r = t.sum(axis=-1) / n[..., None]
    c = t.sum(axis=-2) / n[..., None]
    pe = np.sum(r * c, axis=-1)
    denom = 1.0 - pe
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (po - pe) / denom
    return np.where(denom > 1e-12, k, np.nan)


def cohen_kappa(a: npt.ArrayLike, b: npt.ArrayLike, conf: float = 0.95,
                n_boot: int = 2000, seed: Any = 0) -> KappaResult:
    """Cohen's kappa between two raters (e.g. LLM judge vs human), with a
    percentile bootstrap interval that resamples items.

        kappa = (p_o - p_e) / (1 - p_e)
        p_e   = sum_c rowshare_c * colshare_c

    Labels may be any hashable values. If p_e = 1 (both raters always use the
    same single label), kappa is undefined and the function returns NaN.
    Bootstrap replicates with undefined kappa are dropped, and n_boot_used
    reports how many remain. Pass n_boot=0 to skip the interval.

    Reference: Cohen J (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement 20(1):37-46.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("a and b must be one-dimensional")
    if a_arr.size == 0:
        raise ValueError("a is empty")
    if a_arr.size != b_arr.size:
        raise ValueError("a and b must have the same length")
    conf = _check_open_unit(conf, "conf")
    n_boot = _check_int(n_boot, "n_boot", 0)
    _, codes = np.unique(np.concatenate([a_arr, b_arr]), return_inverse=True)
    codes = np.asarray(codes).ravel()
    n = a_arr.size
    ca, cb = codes[:n], codes[n:]
    K = int(codes.max()) + 1
    table = np.zeros((K, K))
    np.add.at(table, (ca, cb), 1.0)
    kappa = float(_kappa_from_tables(table))
    if n_boot == 0:
        return KappaResult(kappa, float("nan"), float("nan"), n, conf, 0)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    chunk = max(1, min(n_boot, _CHUNK_ELEMENTS // n))
    for start in range(0, n_boot, chunk):
        bsz = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(bsz, n))
        flat = (np.arange(bsz)[:, None] * (K * K) + ca[idx] * K + cb[idx]).ravel()
        tabs = np.bincount(flat, minlength=bsz * K * K).reshape(bsz, K, K)
        reps[start:start + bsz] = _kappa_from_tables(tabs)
    valid = reps[np.isfinite(reps)]
    if valid.size == 0:
        return KappaResult(kappa, float("nan"), float("nan"), n, conf, 0)
    lo, hi = _percentile_bounds(valid, conf)
    return KappaResult(kappa, lo, hi, n, conf, int(valid.size))


def judge_confusion(judge: npt.ArrayLike, human: npt.ArrayLike) -> JudgeConfusion:
    """Confusion counts of a pass/fail judge against human labels (truth).

    Positive = pass. sensitivity = tp/(tp+fn) is the share of true passes the
    judge passes. specificity = tn/(tn+fp) is the share of true fails the
    judge fails. Either rate is NaN if its denominator is 0.
    """
    j = _as_binary(judge, "judge")
    h = _as_binary(human, "human")
    if j.size != h.size:
        raise ValueError("judge and human must have the same length")
    tp = int(np.sum((j == 1) & (h == 1)))
    fn = int(np.sum((j == 0) & (h == 1)))
    tn = int(np.sum((j == 0) & (h == 0)))
    fp = int(np.sum((j == 1) & (h == 0)))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return JudgeConfusion(tp, fn, tn, fp, sens, spec)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float,
                 clip: bool = True) -> float:
    """Correct a judge-reported pass rate for judge error.

        true_rate = (observed + specificity - 1) / (sensitivity + specificity - 1)

    Requires sensitivity + specificity > 1 (the judge must beat chance). With
    clip=True the result is clipped to [0, 1].

    Reference: Rogan WJ, Gladen B (1978). Estimating prevalence from the
    results of a screening test. American Journal of Epidemiology 107(1):71-76.
    """
    q = _check_prob(observed_rate, "observed_rate")
    se = _check_prob(sensitivity, "sensitivity")
    sp = _check_prob(specificity, "specificity")
    j = se + sp - 1.0
    if j <= 0.0:
        raise ValueError("sensitivity + specificity must exceed 1 (judge no better than chance)")
    est = (q + sp - 1.0) / j
    return min(1.0, max(0.0, est)) if clip else est


def rogan_gladen_interval(k_obs: int, n_obs: int, tp: int, fn: int, tn: int, fp: int,
                          conf: float = 0.95, method: str = "jeffreys",
                          n_draws: int = 10_000, seed: Any = 0) -> RoganGladenResult:
    """Corrected pass rate with an interval that also carries the uncertainty
    from validating the judge.

    Inputs:
    * k_obs of n_obs eval cases were passed by the judge.
    * tp, fn, tn, fp come from a human-labelled validation set. Use
      ``judge_confusion``.

    Methods:
    * "delta": Wald interval with the delta-method variance
          Var = [q(1-q)/n_obs + t^2 se(1-se)/n_pos + (1-t)^2 sp(1-sp)/n_neg] / J^2
      where t is the unclipped estimate and J = se + sp - 1. The interval is
      clipped to [0, 1].
    * "bootstrap": parametric bootstrap. It redraws the three binomial counts
      at their estimated rates and takes the percentile interval. If the
      judge was perfect on validation (se = 1 or sp = 1), the validation
      uncertainty collapses to zero.
    * "jeffreys" (default): Monte Carlo propagation. It draws q, se and sp
      from their Jeffreys posteriors Beta(x + 1/2, n - x + 1/2) and takes the
      percentile interval. This still works when se = 1 or sp = 1 on
      validation.
    Draws with J <= 0 are discarded. n_draws_used reports how many remain.

    Assumptions: the validation cases are a random sample of the same
    population, the human labels are correct, and the judge errors do not
    depend on the model being evaluated.

    Reference for the point estimate: Rogan & Gladen (1978), see
    ``rogan_gladen``.
    """
    k_obs, n_obs = _check_count(k_obs, n_obs)
    tp = _check_int(tp, "tp")
    fn = _check_int(fn, "fn")
    tn = _check_int(tn, "tn")
    fp = _check_int(fp, "fp")
    n_pos, n_neg = tp + fn, tn + fp
    if n_pos < 1 or n_neg < 1:
        raise ValueError("validation set needs at least one human pass and one human fail")
    conf = _check_open_unit(conf, "conf")
    q, se, sp = k_obs / n_obs, tp / n_pos, tn / n_neg
    est = rogan_gladen(q, se, sp)  # raises if se + sp <= 1
    a = 1.0 - conf
    if method == "delta":
        j = se + sp - 1.0
        t_raw = (q + sp - 1.0) / j
        var = (q * (1 - q) / n_obs + t_raw ** 2 * se * (1 - se) / n_pos
               + (1 - t_raw) ** 2 * sp * (1 - sp) / n_neg) / j ** 2
        z = _z_two_sided(conf)
        h = z * math.sqrt(var)
        return RoganGladenResult(est, max(0.0, t_raw - h), min(1.0, t_raw + h),
                                 q, se, sp, method, conf, 0)
    if method not in ("jeffreys", "bootstrap"):
        raise ValueError("method must be 'jeffreys' or 'bootstrap'")
    n_draws = _check_int(n_draws, "n_draws", 100)
    rng = np.random.default_rng(seed)
    if method == "jeffreys":
        qd = rng.beta(k_obs + 0.5, n_obs - k_obs + 0.5, n_draws)
        sed = rng.beta(tp + 0.5, fn + 0.5, n_draws)
        spd = rng.beta(tn + 0.5, fp + 0.5, n_draws)
    else:
        qd = rng.binomial(n_obs, q, n_draws) / n_obs
        sed = rng.binomial(n_pos, se, n_draws) / n_pos
        spd = rng.binomial(n_neg, sp, n_draws) / n_neg
    jd = sed + spd - 1.0
    ok = jd > 0.0
    used = int(np.count_nonzero(ok))
    if used < max(100, n_draws // 2):
        return RoganGladenResult(est, float("nan"), float("nan"), q, se, sp, method, conf, used)
    th = np.clip((qd[ok] + spd[ok] - 1.0) / jd[ok], 0.0, 1.0)
    lo, hi = np.quantile(th, [a / 2.0, 1.0 - a / 2.0])
    return RoganGladenResult(est, float(lo), float(hi), q, se, sp, method, conf, used)


# =============================================================================
# 8. Many comparisons, repeated looks, non-inferiority
# =============================================================================


def _check_pvalues(pvalues: Any) -> np.ndarray:
    p = _as_1d_float(pvalues, "pvalues")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be in [0, 1]")
    return p


def holm(pvalues: npt.ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    """Holm step-down adjustment (controls the family-wise error rate).

    Sort p ascending. Then adjusted_(i) = max_{j<=i} min(1, (m - j + 1) p_(j)).
    A hypothesis is rejected if its adjusted p <= alpha. Results are returned
    in the input order.

    Reference: Holm S (1979). A simple sequentially rejective multiple test
    procedure. Scandinavian Journal of Statistics 6(2):65-70.
    """
    p = _check_pvalues(pvalues)
    alpha = _check_open_unit(alpha, "alpha")
    m = p.size
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return MultipleTestResult(tuple(float(v) for v in adj),
                              tuple(bool(v) for v in _reject(adj, alpha)), alpha, "holm")


def benjamini_hochberg(pvalues: npt.ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    """Benjamini-Hochberg step-up adjustment (controls the false discovery rate
    under independence or positive dependence).

    Sort p ascending. Then adjusted_(i) = min_{j>=i} min(1, m p_(j) / j).
    Results are returned in the input order.

    Reference: Benjamini Y, Hochberg Y (1995). Controlling the false discovery
    rate. JRSS B 57(1):289-300.
    """
    p = _check_pvalues(pvalues)
    alpha = _check_open_unit(alpha, "alpha")
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, m + 1)
    raw = m * p[order] / ranks
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    adj = np.empty(m)
    adj[order] = adj_sorted
    return MultipleTestResult(tuple(float(v) for v in adj),
                              tuple(bool(v) for v in _reject(adj, alpha)), alpha, "bh")


def noninferiority_paired(n11: int, n10: int, n01: int, n00: int, margin: float,
                          alpha: float = 0.05) -> NIResult:
    """Paired non-inferiority gate: is B no worse than A by more than `margin`?

    H0: pB - pA <= -margin   vs   H1: pB - pA > -margin, at one-sided level
    alpha. The function computes the two-sided (1 - 2 alpha) Newcombe interval
    (``paired_diff_interval``). B is declared non-inferior if lo > -margin.
    ``superior_by_interval`` is lo > 0 at the same level.

    Newcombe's interval is asymptotic, so the type-I error rate is not
    guaranteed to be <= alpha. The verification report gives it by
    simulation.
    """
    margin = float(margin)
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be in (0, 1)")
    alpha = _check_open_unit(alpha, "alpha")
    if alpha >= 0.5:
        raise ValueError("alpha must be < 0.5")
    conf = 1.0 - 2.0 * alpha
    d = paired_diff_interval(n11, n10, n01, n00, conf)
    return NIResult(bool(d.lo > -margin), bool(d.lo > 0.0), d.diff, d.lo, d.hi,
                    conf, margin, alpha)


def _check_looks(looks: Any) -> np.ndarray:
    """Validate look points: cumulative sample sizes, strictly increasing, >= 1."""
    if isinstance(looks, (str, bytes)):
        raise ValueError("looks must be a sequence of integers, not a string")
    try:
        vals = [_check_int(v, "looks", 1) for v in looks]
    except TypeError as exc:
        raise ValueError("looks must be a sequence of integers") from exc
    if not vals:
        raise ValueError("looks is empty")
    arr = np.asarray(vals, dtype=np.int64)
    if np.any(np.diff(arr) <= 0):
        raise ValueError("looks must be strictly increasing")
    return arr


def simulate_peeking_gaussian(looks: Any, effect: float = 0.0, alpha: float = 0.05,
                              reps: int = 10_000, seed: Any = 0,
                              two_sided: bool = True) -> RateResult:
    """Probability of at least one rejection when a fixed-sample z-test is
    repeated at several interim looks ("peeking").

    Model: observations are i.i.d. N(effect, 1), so ``effect`` is in standard
    deviation units (for example a standardised per-case difference B - A).
    ``looks`` lists the cumulative sample sizes at which the analyst checks
    the result, e.g. [50, 100, 150, 200]. At every look the test statistic is
    z = S_n / sqrt(n) and H0 is rejected when |z| >= z_{1-alpha/2}
    (z >= z_{1-alpha} if one-sided). The run stops at the first rejection.

    With effect = 0 the returned rate is the actual type-I error rate under
    peeking. With a single look it equals alpha up to Monte Carlo error. With
    effect != 0 it is the power of the peeking procedure.

    The partial sums are simulated exactly through independent normal
    increments between looks, so the cost depends on the number of looks,
    not on the sample size.

    Reference: Armitage P, McPherson CK, Rowe BC (1969). Repeated significance
    tests on accumulating data. JRSS A 132(2):235-244.
    """
    L = _check_looks(looks)
    effect = float(effect)
    if not math.isfinite(effect):
        raise ValueError("effect must be finite")
    alpha = _check_open_unit(alpha, "alpha")
    reps = _check_int(reps, "reps", 1)
    crit = _z_alpha(alpha, two_sided)
    inc = np.diff(np.concatenate(([0], L))).astype(float)
    sd_inc = np.sqrt(inc)
    mean_inc = effect * inc
    root_n = np.sqrt(L.astype(float))
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = max(1, min(reps, _CHUNK_ELEMENTS // L.size))
    for start in range(0, reps, chunk):
        b = min(chunk, reps - start)
        steps = rng.standard_normal((b, L.size)) * sd_inc + mean_inc
        z = np.cumsum(steps, axis=1) / root_n
        stat = np.abs(z) if two_sided else z
        hits += int(np.count_nonzero(np.any(stat >= crit, axis=1)))
    rate = hits / reps
    return RateResult(rate, math.sqrt(rate * (1.0 - rate) / reps), reps)


def simulate_peeking_mcnemar(looks: Any, p10: float, p01: float, alpha: float = 0.05,
                             reps: int = 10_000, seed: Any = 0) -> RateResult:
    """Probability of at least one rejection when the two-sided exact McNemar
    test is repeated at several interim looks on a growing paired eval set.

    New cases arrive in batches. Each case falls into (A pass/B fail,
    A fail/B pass, concordant) with probabilities (p10, p01, 1 - p10 - p01).
    At each cumulative look size the exact McNemar p-value is computed and
    H0 is rejected when p <= alpha. With p10 == p01 the rate is the actual
    type-I error rate under peeking. Otherwise it is the power.

    Reference: Armitage, McPherson & Rowe (1969), see
    ``simulate_peeking_gaussian``. The test is ``exact_mcnemar``.
    """
    L = _check_looks(looks)
    p10, p01 = _check_cells(p10, p01)
    alpha = _check_open_unit(alpha, "alpha")
    reps = _check_int(reps, "reps", 1)
    inc = np.diff(np.concatenate(([0], L)))
    probs = [p10, p01, max(0.0, 1.0 - p10 - p01)]
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = max(1, min(reps, _CHUNK_ELEMENTS // 3))
    for start in range(0, reps, chunk):
        b = min(chunk, reps - start)
        n10 = np.zeros(b, dtype=np.int64)
        n01 = np.zeros(b, dtype=np.int64)
        rejected = np.zeros(b, dtype=bool)
        for step in inc:
            d = rng.multinomial(int(step), probs, size=b)
            n10 += d[:, 0]
            n01 += d[:, 1]
            rejected |= _reject(_mcnemar_p_arrays(n10, n01), alpha)
        hits += int(np.count_nonzero(rejected))
    rate = hits / reps
    return RateResult(rate, math.sqrt(rate * (1.0 - rate) / reps), reps)


# =============================================================================
# 9. JSON conversion
# =============================================================================


def to_jsonable(obj: Any) -> Any:
    """Convert results (NamedTuples, NumPy scalars/arrays, nested containers)
    to plain JSON-compatible Python objects.

    Non-finite floats (NaN, +/-inf) become None, so the output is strict JSON.
    Unknown objects are converted with str().
    """
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return {k: to_jsonable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# =============================================================================
# 10. Command line
# =============================================================================


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.{nd}f}" if math.isfinite(v) else "nan"


def _pct(conf: float) -> str:
    return f"{100.0 * float(conf):g}%"


def _payload(command: str, result: Any = None, **fields: Any) -> dict:
    out: dict = {"command": command}
    out.update(fields)
    if result is not None:
        out.update(result._asdict())
    return out


def _split_list(text: str, name: str) -> list[str]:
    items = [s.strip() for s in str(text).split(",")]
    if any(s == "" for s in items):
        raise ValueError(f"{name}: empty item in comma-separated list")
    return items


def _to_floats(raw: Any, name: str) -> np.ndarray:
    out = []
    for i, v in enumerate(raw):
        try:
            x = float(str(v).strip())
        except ValueError:
            raise ValueError(f"{name}: item {i + 1} ({v!r}) is not a number") from None
        if not math.isfinite(x):
            raise ValueError(f"{name}: item {i + 1} is NaN or infinite")
        out.append(x)
    if not out:
        raise ValueError(f"{name} is empty")
    return np.asarray(out, dtype=float)


def _to_numbers(raw: Any, name: str) -> np.ndarray:
    """Numbers, or pass/fail words mapped to 1/0. Blank cells are rejected."""
    out = []
    for i, v in enumerate(raw):
        s = str(v).strip()
        try:
            x = float(s)
        except ValueError:
            try:
                x = float(parse_pass_value(s))
            except ValueError:
                raise ValueError(f"{name}: item {i + 1} ({v!r}) is not a number "
                                 "or pass/fail value") from None
        if not math.isfinite(x):
            raise ValueError(f"{name}: item {i + 1} is NaN or infinite")
        out.append(x)
    if not out:
        raise ValueError(f"{name} is empty")
    return np.asarray(out, dtype=float)


def _parse_csv_columns(fh: Any, columns: list[str], label: str) -> dict:
    reader = csv.DictReader(fh)
    if not reader.fieldnames:
        raise ValueError(f"{label}: no header row")
    reader.fieldnames = [(f or "").strip() for f in reader.fieldnames]
    missing = [c for c in columns if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"{label}: missing column(s) {missing}; "
                         f"available: {reader.fieldnames}")
    out: dict = {c: [] for c in columns}
    for row in reader:
        for c in columns:
            v = row.get(c)
            if v is None:
                raise ValueError(f"{label} line {reader.line_num}: no value for column {c!r}")
            out[c].append(v.strip())
    if not out[columns[0]]:
        raise ValueError(f"{label}: no data rows")
    return out


def _read_csv(path: str, columns: list[str]) -> dict:
    """Read named columns from a CSV file with a header row ('-' = stdin)."""
    if path == "-":
        return _parse_csv_columns(sys.stdin, columns, "<stdin>")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return _parse_csv_columns(fh, columns, path)


def _read_matrix(path: str, header: bool) -> np.ndarray:
    def parse(fh: Any) -> np.ndarray:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
        if header and rows:
            rows = rows[1:]
        if not rows:
            raise ValueError(f"{path}: no data rows")
        width = len(rows[0])
        out = []
        for i, r in enumerate(rows):
            if len(r) != width:
                raise ValueError(f"{path}: row {i + 1} has {len(r)} cells, expected {width}")
            out.append(_to_numbers(r, f"{path} row {i + 1}"))
        return np.vstack(out)

    if path == "-":
        return parse(sys.stdin)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return parse(fh)


def _values_from(args: argparse.Namespace) -> np.ndarray:
    if getattr(args, "csv", None) and getattr(args, "col", None):
        return _to_numbers(_read_csv(args.csv, [args.col])[args.col], args.col)
    if getattr(args, "values", None):
        return _to_numbers(_split_list(args.values, "--values"), "--values")
    raise ValueError("give --values LIST, or --csv PATH with --col NAME")


def _pair_raw(args: argparse.Namespace) -> tuple[list, list] | None:
    """Paired raw columns from --a/--b lists or --csv with --a-col/--b-col."""
    a_col, b_col = getattr(args, "a_col", None), getattr(args, "b_col", None)
    a_lst, b_lst = getattr(args, "a", None), getattr(args, "b", None)
    has_cols = bool(a_col or b_col)
    has_lists = bool(a_lst or b_lst)
    if has_cols and has_lists:
        raise ValueError("give either --a/--b or --csv with --a-col/--b-col, not both")
    if has_cols:
        if not (a_col and b_col and getattr(args, "csv", None)):
            raise ValueError("--a-col and --b-col must both be given, together with --csv")
        data = _read_csv(args.csv, [a_col, b_col])
        return data[a_col], data[b_col]
    if has_lists:
        if not (a_lst and b_lst):
            raise ValueError("give both --a and --b")
        a = _split_list(a_lst, "--a")
        b = _split_list(b_lst, "--b")
        if len(a) != len(b):
            raise ValueError(f"--a and --b have different lengths ({len(a)} vs {len(b)})")
        return a, b
    return None


def _counts_from(args: argparse.Namespace, need_all: bool) -> PairedCounts:
    pair = _pair_raw(args)
    given = {k: getattr(args, k) for k in ("n11", "n10", "n01", "n00")}
    if pair is not None:
        if any(v is not None for v in given.values()):
            raise ValueError("give either counts (--n11 ...) or raw results, not both")
        return paired_counts(pair[0], pair[1])
    if not need_all:
        if given["n10"] is None or given["n01"] is None:
            raise ValueError("give --n10 and --n01, or raw paired results")
        return PairedCounts(given["n11"] or 0, given["n10"], given["n01"], given["n00"] or 0)
    missing = [f"--{k}" for k, v in given.items() if v is None]
    if missing:
        raise ValueError(f"missing {', '.join(missing)} (or give raw paired results)")
    return PairedCounts(**given)


def _zero_note(k: int, n: int, conf: float, out: dict) -> list[str]:
    if k not in (0, n):
        return []
    z = rule_of_three(n, conf)
    out["one_sided_exact_upper_event_rate"] = z.exact_upper
    out["rule_of_three_upper_event_rate"] = z.rule_of_three_upper
    if k == 0:
        return [f"0 of {n} passed: one-sided {_pct(conf)} upper bound on the pass rate "
                f"{_fmt(z.exact_upper)} (rule of three: {_fmt(z.rule_of_three_upper)})"]
    return [f"{n} of {n} passed: one-sided {_pct(conf)} lower bound on the pass rate "
            f"{_fmt(1.0 - z.exact_upper)} (rule of three: {_fmt(1.0 - z.rule_of_three_upper)})"]


# ----- subcommand handlers: each returns (payload, human-readable text) -----


def _cmd_wilson(args: argparse.Namespace) -> tuple[dict, str]:
    r = wilson_interval(args.k, args.n, args.conf)
    rate = args.k / args.n
    out = _payload("wilson", r, k=args.k, n=args.n, rate=rate, conf=args.conf)
    lines = [f"Wilson {_pct(args.conf)}: {args.k}/{args.n} = {_fmt(rate)}  "
             f"CI [{_fmt(r.lo)}, {_fmt(r.hi)}]"]
    lines += _zero_note(args.k, args.n, args.conf, out)
    return out, "\n".join(lines)


def _cmd_cp(args: argparse.Namespace) -> tuple[dict, str]:
    r = clopper_pearson_interval(args.k, args.n, args.conf)
    rate = args.k / args.n
    out = _payload("cp", r, k=args.k, n=args.n, rate=rate, conf=args.conf)
    lines = [f"Clopper-Pearson {_pct(args.conf)}: {args.k}/{args.n} = {_fmt(rate)}  "
             f"CI [{_fmt(r.lo)}, {_fmt(r.hi)}]"]
    lines += _zero_note(args.k, args.n, args.conf, out)
    return out, "\n".join(lines)


def _cmd_mean(args: argparse.Namespace) -> tuple[dict, str]:
    r = mean_interval_t(_values_from(args), args.conf)
    out = _payload("mean", r)
    return out, (f"mean = {_fmt(r.estimate)}  t {_pct(r.conf)} CI [{_fmt(r.lo)}, {_fmt(r.hi)}]  "
                 f"se = {_fmt(r.se)}  n = {r.n}")


_STATS: dict[str, Callable] = {"mean": np.mean, "median": np.median}


def _cmd_bootstrap(args: argparse.Namespace) -> tuple[dict, str]:
    pair = _pair_raw(args)
    if pair is not None:
        if args.stat != "mean":
            raise ValueError("paired mode supports --stat mean only")
        a = _to_numbers(pair[0], "A")
        b = _to_numbers(pair[1], "B")
        r = paired_bootstrap_diff(a, b, args.conf, args.n_boot, args.seed, args.method)
        label, mode = "mean(B) - mean(A)", "paired"
    else:
        r = bootstrap_interval(_values_from(args), _STATS[args.stat], args.conf,
                               args.n_boot, args.seed, args.method)
        label, mode = args.stat, "one-sample"
    out = _payload("bootstrap", r, mode=mode, stat=args.stat, seed=args.seed)
    return out, (f"{label} = {_fmt(r.estimate)}  {r.method} {_pct(r.conf)} CI "
                 f"[{_fmt(r.lo)}, {_fmt(r.hi)}]  (n_boot = {r.n_boot}, seed = {args.seed})")


def _cmd_mcnemar(args: argparse.Namespace) -> tuple[dict, str]:
    c = _counts_from(args, need_all=False)
    p = exact_mcnemar(c.n10, c.n01, args.alternative)
    alpha = _check_open_unit(args.alpha, "alpha")
    rej = bool(_reject(p, alpha))
    out = _payload("mcnemar", c, alternative=args.alternative, p_value=p,
                   alpha=alpha, reject=rej)
    return out, (f"exact McNemar ({args.alternative}): regressions n10 = {c.n10}, "
                 f"improvements n01 = {c.n01}, p = {p:.4g}  -> "
                 f"{'reject H0' if rej else 'do not reject H0'} at alpha = {alpha:g}")


def _cmd_paired_diff(args: argparse.Namespace) -> tuple[dict, str]:
    c = _counts_from(args, need_all=True)
    r = paired_diff_interval(c.n11, c.n10, c.n01, c.n00, args.conf, cc_phi=args.method10)
    n = sum(c)
    out = _payload("paired-diff", r, counts=c._asdict(), n=n,
                   method="newcombe_method10" if args.method10 else "newcombe_method8")
    lines = [f"pB - pA = {_fmt(r.diff)}  Newcombe ({'method 10' if args.method10 else 'method 8'}) {_pct(r.conf)} CI [{_fmt(r.lo)}, {_fmt(r.hi)}]  "
             f"(n = {n}, n10 = {c.n10}, n01 = {c.n01})"]
    if c.n10 + c.n01 == 0:
        z = rule_of_three(n, args.conf)
        out["discordance_upper_one_sided"] = z.exact_upper
        lines.append(f"no discordant pairs: one-sided {_pct(args.conf)} upper bound on the "
                     f"discordance rate {_fmt(z.exact_upper)}")
    return out, "\n".join(lines)


def _cmd_ni_gate(args: argparse.Namespace) -> tuple[dict, str]:
    c = _counts_from(args, need_all=True)
    r = noninferiority_paired(c.n11, c.n10, c.n01, c.n00, args.margin, args.alpha)
    out = _payload("ni-gate", r, counts=c._asdict())
    verdict = "PASS: B is non-inferior" if r.noninferior else "FAIL: non-inferiority not shown"
    extra = " (also superior by interval)" if r.superior_by_interval else ""
    return out, (f"{verdict}{extra}\n"
                 f"pB - pA = {_fmt(r.diff)}  {_pct(r.conf)} CI [{_fmt(r.lo)}, {_fmt(r.hi)}]  "
                 f"margin = -{r.margin:g}, one-sided alpha = {r.alpha:g}")


def _cmd_n_unpaired(args: argparse.Namespace) -> tuple[dict, str]:
    r = n_unpaired(args.p1, args.p2, args.alpha, args.power, not args.one_sided)
    out = _payload("n-unpaired", r, p1=args.p1, p2=args.p2)
    side = "two-sided" if r.two_sided else "one-sided"
    return out, (f"cases per group for {args.p1:g} vs {args.p2:g} ({side} alpha = {r.alpha:g}, "
                 f"power = {r.power:g}):\n"
                 f"  pooled (recommended): {r.n_pooled}\n"
                 f"  unpooled:             {r.n_simple}\n"
                 f"  baseline-variance:    {r.n_baseline_variance}")


def _cmd_n_paired(args: argparse.Namespace) -> tuple[dict, str]:
    two_sided = not args.one_sided
    n = n_paired(args.p_disc, args.delta, args.alpha, args.power, two_sided)
    out = _payload("n-paired", None, n=n, p_disc=args.p_disc, delta=args.delta,
                   alpha=args.alpha, power=args.power, two_sided=two_sided)
    lines = [f"cases needed (Connor): {n}  (p_disc = {args.p_disc:g}, delta = {args.delta:g}, "
             f"{'two' if two_sided else 'one'}-sided alpha = {args.alpha:g}, power = {args.power:g})"]
    if args.check_exact:
        if not two_sided:
            lines.append("exact check skipped: the exact McNemar power is two-sided only")
            out["exact_power"] = None
        else:
            p01 = (args.p_disc + args.delta) / 2.0
            p10 = (args.p_disc - args.delta) / 2.0
            pr = power_exact_mcnemar_enum(n, p10, p01, args.alpha)
            out["exact_power"] = pr.power
            lines.append(f"power of the exact McNemar test at n = {n}: {_fmt(pr.power)}")
    return out, "\n".join(lines)


def _cmd_power_mcnemar(args: argparse.Namespace) -> tuple[dict, str]:
    if args.method == "enum":
        r = power_exact_mcnemar_enum(args.n, args.p10, args.p01, args.alpha)
        tail = "exact enumeration"
    else:
        r = power_exact_mcnemar(args.n, args.p10, args.p01, args.alpha, args.sims, args.seed)
        tail = f"simulation, MC se = {_fmt(r.mc_se)}, sims = {r.sims}, seed = {args.seed}"
    out = _payload("power-mcnemar", r, n=args.n, p10=args.p10, p01=args.p01, alpha=args.alpha)
    what = "type-I error rate" if args.p10 == args.p01 else "power"
    return out, f"exact McNemar {what} at n = {args.n}: {_fmt(r.power)}  ({tail})"


def _cmd_mde(args: argparse.Namespace) -> tuple[dict, str]:
    two_sided = not args.one_sided
    if args.design == "unpaired":
        if args.p is None:
            raise ValueError("--p (baseline pass rate) is required for --design unpaired")
        d = mde_unpaired(args.n, args.p, args.alpha, args.power, two_sided, args.direction)
        ctx = f"two independent sets of n = {args.n}, baseline p = {args.p:g}, {args.direction}"
    else:
        if args.p_disc is None:
            raise ValueError("--p-disc (discordance rate) is required for --design paired")
        d = mde_paired(args.n, args.p_disc, args.alpha, args.power, two_sided)
        ctx = f"paired, n = {args.n}, p_disc = {args.p_disc:g}"
    out = _payload("mde", None, design=args.design, n=args.n, p=args.p, p_disc=args.p_disc,
                   direction=args.direction, alpha=args.alpha, power=args.power,
                   two_sided=two_sided, mde=d)
    if not math.isfinite(d):
        return out, f"MDE: not reachable at this n and power ({ctx})"
    return out, f"MDE = {_fmt(d)} ({ctx}; alpha = {args.alpha:g}, power = {args.power:g})"


def _cmd_design_effect(args: argparse.Namespace) -> tuple[dict, str]:
    if args.csv and (args.value_col or args.cluster_col):
        if not (args.value_col and args.cluster_col):
            raise ValueError("cluster mode needs both --value-col and --cluster-col")
        data = _read_csv(args.csv, [args.value_col, args.cluster_col])
        y = _to_numbers(data[args.value_col], args.value_col)
        ids = data[args.cluster_col]
        if any(s == "" for s in ids):
            raise ValueError(f"column {args.cluster_col!r} has blank cluster ids")
        r = cluster_mean_interval(y, ids, args.conf)
        out = _payload("design-effect", r, mode="cluster")
        lines = [f"mean = {_fmt(r.estimate)}  cluster-robust {_pct(r.conf)} CI "
                 f"[{_fmt(r.lo)}, {_fmt(r.hi)}]",
                 f"se = {_fmt(r.se)} (naive {_fmt(r.se_naive)}), design effect = "
                 f"{_fmt(r.design_effect, 3)}, clusters = {r.n_clusters}, obs = {r.n_obs}"]
        if r.n_clusters < 30:
            lines.append("note: fewer than 30 clusters; coverage may be below nominal")
        if args.n_boot > 0:
            b = cluster_bootstrap_interval(y, ids, args.conf, args.n_boot, args.seed, args.method)
            out["cluster_bootstrap"] = b._asdict()
            lines.append(f"cluster bootstrap ({b.method}) {_pct(b.conf)} CI "
                         f"[{_fmt(b.lo)}, {_fmt(b.hi)}]  (n_boot = {b.n_boot}, seed = {args.seed})")
        return out, "\n".join(lines)
    if args.matrix_csv:
        mat = _read_matrix(args.matrix_csv, args.header)
        icc = icc_binary(mat)
        n_items, k = mat.shape
        out = _payload("design-effect", None, mode="repeats-matrix", n_items=n_items,
                       k_repeats=k, icc=icc)
        if not math.isfinite(icc):
            out.update(design_effect=None, n_eff=None)
            return out, f"ICC undefined (all values identical); items = {n_items}, repeats = {k}"
        icc_c = min(1.0, max(0.0, icc))
        deff = design_effect(k, icc_c)
        n_eff = effective_n_repeats(n_items, k, icc_c)
        out.update(icc_clipped=icc_c, design_effect=deff, n_eff=n_eff)
        note = "" if icc_c == icc else f" (ICC clipped to {icc_c:g} for n_eff)"
        return out, (f"ICC(1,1) = {_fmt(icc)}, design effect = {_fmt(deff, 3)}, "
                     f"effective n = {_fmt(n_eff, 1)} from {n_items} items x {k} repeats{note}")
    if args.m is not None and args.icc is not None:
        deff = design_effect(args.m, args.icc)
        out = _payload("design-effect", None, mode="formula", m=args.m, icc=args.icc,
                       design_effect=deff)
        lines = [f"design effect = 1 + ({args.m:g} - 1) * {args.icc:g} = {_fmt(deff, 4)}"]
        if args.n_items is not None:
            n_eff = effective_n_repeats(args.n_items, args.m, args.icc)
            out["n_items"] = args.n_items
            out["n_eff"] = n_eff
            out["n_eff_limit"] = args.n_items / args.icc if args.icc > 0 else None
            lines.append(f"effective n = {_fmt(n_eff, 1)} from {args.n_items} items x "
                         f"{args.m:g} repeats")
            if args.icc > 0:
                lines.append(f"limit with unlimited repeats: {_fmt(args.n_items / args.icc, 1)}")
        return out, "\n".join(lines)
    raise ValueError("give --m and --icc, or --matrix-csv PATH, or --csv PATH with "
                     "--value-col and --cluster-col")


def _cmd_kappa(args: argparse.Namespace) -> tuple[dict, str]:
    pair = _pair_raw(args)
    if pair is None:
        raise ValueError("give --a/--b, or --csv with --a-col/--b-col")
    a_raw, b_raw = list(pair[0]), list(pair[1])
    if any(str(v).strip() == "" for v in a_raw + b_raw):
        raise ValueError("blank labels are not allowed")
    try:
        a: Any = [parse_pass_value(v) for v in a_raw]
        b: Any = [parse_pass_value(v) for v in b_raw]
        binary = True
    except ValueError:
        if args.confusion:
            raise
        a, b, binary = [str(v).strip() for v in a_raw], [str(v).strip() for v in b_raw], False
    r = cohen_kappa(a, b, args.conf, args.n_boot, args.seed)
    out = _payload("kappa", r, labels="pass/fail" if binary else "raw", seed=args.seed)
    lines = [f"Cohen's kappa = {_fmt(r.kappa)}  bootstrap {_pct(r.conf)} CI "
             f"[{_fmt(r.lo)}, {_fmt(r.hi)}]  (n = {r.n}, replicates used = {r.n_boot_used})"]
    if args.confusion:
        jc = judge_confusion(a, b)
        out["confusion"] = jc._asdict()
        for name, k, m in (("sensitivity", jc.tp, jc.tp + jc.fn),
                           ("specificity", jc.tn, jc.tn + jc.fp)):
            if m > 0:
                ci = wilson_interval(k, m, args.conf)
                out[f"{name}_ci"] = ci._asdict()
                lines.append(f"{name} = {k}/{m} = {_fmt(k / m)}  Wilson {_pct(args.conf)} CI "
                             f"[{_fmt(ci.lo)}, {_fmt(ci.hi)}]")
            else:
                out[f"{name}_ci"] = None
                lines.append(f"{name}: undefined (no cases in denominator)")
        lines.append(f"tp = {jc.tp}, fn = {jc.fn}, tn = {jc.tn}, fp = {jc.fp} "
                     "(A = judge, B = human)")
    return out, "\n".join(lines)


def _cmd_rogan_gladen(args: argparse.Namespace) -> tuple[dict, str]:
    pair = _pair_raw(args)
    given = {k: getattr(args, k) for k in ("tp", "fn", "tn", "fp")}
    if pair is not None:
        if any(v is not None for v in given.values()):
            raise ValueError("give either --tp/--fn/--tn/--fp or raw judge/human labels, not both")
        jc = judge_confusion(pair[0], pair[1])
        tp, fn, tn, fp = jc.tp, jc.fn, jc.tn, jc.fp
    else:
        missing = [f"--{k}" for k, v in given.items() if v is None]
        if missing:
            raise ValueError(f"missing {', '.join(missing)} (or give raw judge/human labels)")
        tp, fn, tn, fp = given["tp"], given["fn"], given["tn"], given["fp"]
    r = rogan_gladen_interval(args.k_obs, args.n_obs, tp, fn, tn, fp, args.conf,
                              args.method, args.n_draws, args.seed)
    out = _payload("rogan-gladen", r, k_obs=args.k_obs, n_obs=args.n_obs,
                   tp=tp, fn=fn, tn=tn, fp=fp, seed=args.seed)
    lines = [f"judge-reported rate = {_fmt(r.observed_rate)}  (sensitivity = {_fmt(r.sensitivity)}, "
             f"specificity = {_fmt(r.specificity)})",
             f"corrected pass rate = {_fmt(r.estimate)}  {r.method} {_pct(r.conf)} CI "
             f"[{_fmt(r.lo)}, {_fmt(r.hi)}]"]
    if not (math.isfinite(r.lo) and math.isfinite(r.hi)):
        lines.append(f"interval unavailable: only {r.n_draws_used} draws had "
                     "sensitivity + specificity > 1; validate the judge on more cases")
    return out, "\n".join(lines)


def _pvalues_from(args: argparse.Namespace) -> np.ndarray:
    if args.csv:
        if not args.col:
            raise ValueError("--col is required with --csv")
        if args.pvalues:
            raise ValueError("give p-values either as arguments or via --csv, not both")
        return _to_floats(_read_csv(args.csv, [args.col])[args.col], args.col)
    items: list[str] = []
    for tok in args.pvalues:
        items.extend(_split_list(tok, "pvalues"))
    if not items:
        raise ValueError("no p-values given")
    return _to_floats(items, "pvalues")


def _cmd_multi(args: argparse.Namespace, method: str) -> tuple[dict, str]:
    p = _pvalues_from(args)
    r = holm(p, args.alpha) if method == "holm" else benjamini_hochberg(p, args.alpha)
    out = _payload(method, r, pvalues=p.tolist())
    name = "Holm (FWER)" if method == "holm" else "Benjamini-Hochberg (FDR)"
    lines = [f"{name}, alpha = {r.alpha:g}: {sum(r.reject)} of {len(r.reject)} rejected",
             "  #   p          adjusted   reject"]
    for i, (pv, adj, rej) in enumerate(zip(p, r.adjusted, r.reject), 1):
        lines.append(f"  {i:<3} {pv:<10.4g} {adj:<10.4g} {'yes' if rej else 'no'}")
    return out, "\n".join(lines)


def _cmd_peeking(args: argparse.Namespace) -> tuple[dict, str]:
    try:
        looks = [int(s) for s in _split_list(args.looks, "--looks")]
    except ValueError as exc:
        raise ValueError(f"--looks must be comma-separated integers: {exc}") from None
    if args.model == "gaussian":
        r = simulate_peeking_gaussian(looks, args.effect, args.alpha, args.reps, args.seed,
                                      not args.one_sided)
        null = args.effect == 0.0
        ctx = f"z-test, effect = {args.effect:g} SD, {'one' if args.one_sided else 'two'}-sided"
    else:
        if args.p10 is None or args.p01 is None:
            raise ValueError("--p10 and --p01 are required for --model mcnemar")
        r = simulate_peeking_mcnemar(looks, args.p10, args.p01, args.alpha, args.reps, args.seed)
        null = args.p10 == args.p01
        ctx = f"exact McNemar, p10 = {args.p10:g}, p01 = {args.p01:g}"
    what = "false-positive rate under H0" if null else "power"
    out = _payload("peeking", r, model=args.model, looks=looks, alpha=args.alpha,
                   null_true=null, seed=args.seed)
    return out, (f"{len(looks)} looks at {looks} ({ctx}), nominal alpha = {args.alpha:g} per look\n"
                 f"{what} with peeking = {_fmt(r.rate)}  (MC se = {_fmt(r.mc_se)}, "
                 f"reps = {r.reps}, seed = {args.seed})")


# ----- parser -----


def _add_csv(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", metavar="PATH",
                   help="CSV file with a header row ('-' reads stdin)")


def _add_values(p: argparse.ArgumentParser) -> None:
    p.add_argument("--values", metavar="LIST", help="comma-separated scores, e.g. 1,0.5,0")
    p.add_argument("--col", metavar="NAME", help="column name in --csv")


def _add_pair(p: argparse.ArgumentParser, a_help: str = "baseline A",
              b_help: str = "candidate B") -> None:
    p.add_argument("--a", metavar="LIST", help=f"comma-separated values for {a_help}")
    p.add_argument("--b", metavar="LIST", help=f"comma-separated values for {b_help}")
    p.add_argument("--a-col", metavar="NAME", help=f"--csv column for {a_help}")
    p.add_argument("--b-col", metavar="NAME", help=f"--csv column for {b_help}")


def _add_counts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n11", type=int, help="A pass, B pass")
    p.add_argument("--n10", type=int, help="A pass, B fail (regressions)")
    p.add_argument("--n01", type=int, help="A fail, B pass (improvements)")
    p.add_argument("--n00", type=int, help="A fail, B fail")
    _add_csv(p)
    _add_pair(p, "baseline A pass/fail", "candidate B pass/fail")


def _add_test_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--one-sided", action="store_true", help="one-sided test (default two-sided)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shipornot",
        description="Small-sample statistics for LLM evals. A = baseline, B = candidate; "
                    "differences are B - A. Add --json to any subcommand for JSON output.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, help_text: str, func: Callable) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    for name, desc, fn in (("wilson", "Wilson score interval for k/n", _cmd_wilson),
                           ("cp", "Clopper-Pearson exact interval for k/n", _cmd_cp)):
        p = add(name, desc, fn)
        p.add_argument("k", type=int, help="passes")
        p.add_argument("n", type=int, help="cases")
        p.add_argument("--conf", type=float, default=0.95)

    p = add("mean", "Student-t interval for a mean score", _cmd_mean)
    _add_csv(p)
    _add_values(p)
    p.add_argument("--conf", type=float, default=0.95)

    p = add("bootstrap", "bootstrap interval (one sample, or paired B - A with --a/--b)",
            _cmd_bootstrap)
    _add_csv(p)
    _add_values(p)
    _add_pair(p)
    p.add_argument("--stat", choices=sorted(_STATS), default="mean")
    p.add_argument("--method", choices=["bca", "percentile"], default="bca")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.95)

    p = add("mcnemar", "exact McNemar test on paired pass/fail results", _cmd_mcnemar)
    _add_counts(p)
    p.add_argument("--alternative", choices=["two-sided", "greater", "less"],
                   default="two-sided", help="greater: H1 B is better")
    p.add_argument("--alpha", type=float, default=0.05)

    p = add("paired-diff", "Newcombe interval for pB - pA on the same cases", _cmd_paired_diff)
    _add_counts(p)
    p.add_argument("--method10", action="store_true",
                   help="Newcombe's method 10 (continuity-corrected phi); default is method 8")
    p.add_argument("--conf", type=float, default=0.95)

    p = add("n-unpaired", "cases per group for two independent test sets", _cmd_n_unpaired)
    p.add_argument("p1", type=float, help="baseline pass rate")
    p.add_argument("p2", type=float, help="candidate pass rate")
    _add_test_opts(p)

    p = add("n-paired", "cases for a paired design (Connor)", _cmd_n_paired)
    p.add_argument("p_disc", type=float, help="discordance rate p10 + p01")
    p.add_argument("delta", type=float, help="difference p01 - p10 (B - A)")
    _add_test_opts(p)
    p.add_argument("--check-exact", action="store_true",
                   help="also report exact McNemar power at the computed n (slow for large n)")

    p = add("power-mcnemar", "power of the two-sided exact McNemar test", _cmd_power_mcnemar)
    p.add_argument("n", type=int)
    p.add_argument("p10", type=float, help="P(A pass, B fail)")
    p.add_argument("p01", type=float, help="P(A fail, B pass)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--method", choices=["enum", "sim"], default="enum")
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)

    p = add("mde", "minimum detectable effect for a given n", _cmd_mde)
    p.add_argument("n", type=int, help="cases (per group if unpaired)")
    p.add_argument("--design", choices=["unpaired", "paired"], required=True)
    p.add_argument("--p", type=float, help="baseline pass rate (unpaired)")
    p.add_argument("--p-disc", type=float, help="discordance rate (paired)")
    p.add_argument("--direction", choices=["up", "down"], default="up", help="unpaired only")
    _add_test_opts(p)

    p = add("design-effect", "design effect / ICC for repeats, or cluster-robust mean",
            _cmd_design_effect)
    p.add_argument("--m", type=float, help="observations per unit (repeats or cluster size)")
    p.add_argument("--icc", type=float, help="intraclass correlation")
    p.add_argument("--n-items", type=int, help="items, for effective n with --m repeats")
    p.add_argument("--matrix-csv", metavar="PATH",
                   help="items x repeats CSV (no header unless --header) to estimate ICC")
    p.add_argument("--header", action="store_true", help="--matrix-csv has a header row")
    _add_csv(p)
    p.add_argument("--value-col", metavar="NAME", help="cluster mode: value column")
    p.add_argument("--cluster-col", metavar="NAME", help="cluster mode: cluster id column")
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--n-boot", type=int, default=0, help="cluster bootstrap replicates (0 = off)")
    p.add_argument("--method", choices=["percentile", "bca"], default="percentile")
    p.add_argument("--seed", type=int, default=0)

    p = add("kappa", "Cohen's kappa between two raters (labels compared as pass/fail if all "
                     "parse as such, else as exact strings)", _cmd_kappa)
    _add_csv(p)
    _add_pair(p, "rater A (judge)", "rater B (human)")
    p.add_argument("--confusion", action="store_true",
                   help="also report judge sensitivity/specificity (A = judge, B = human)")
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    p = add("rogan-gladen", "judge-error-corrected pass rate with interval", _cmd_rogan_gladen)
    p.add_argument("--k-obs", type=int, required=True, help="cases the judge passed")
    p.add_argument("--n-obs", type=int, required=True, help="cases judged")
    p.add_argument("--tp", type=int)
    p.add_argument("--fn", type=int)
    p.add_argument("--tn", type=int)
    p.add_argument("--fp", type=int)
    _add_csv(p)
    _add_pair(p, "judge pass/fail on validation set", "human pass/fail on validation set")
    p.add_argument("--method", choices=["jeffreys", "bootstrap"], default="jeffreys")
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--n-draws", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)

    for name, desc, method in (("holm", "Holm adjustment (family-wise error)", "holm"),
                               ("bh", "Benjamini-Hochberg adjustment (FDR)", "bh")):
        p = add(name, desc, functools.partial(_cmd_multi, method=method))
        p.add_argument("pvalues", nargs="*", help="p-values (space- or comma-separated)")
        _add_csv(p)
        p.add_argument("--col", metavar="NAME", help="p-value column in --csv")
        p.add_argument("--alpha", type=float, default=0.05)

    p = add("ni-gate", "paired non-inferiority gate (B no worse than A by > margin)",
            _cmd_ni_gate)
    _add_counts(p)
    p.add_argument("--margin", type=float, required=True, help="e.g. 0.02 for 2 points")
    p.add_argument("--alpha", type=float, default=0.05, help="one-sided level")

    p = add("peeking", "simulate repeated looks at a fixed-sample test", _cmd_peeking)
    p.add_argument("--looks", required=True, metavar="LIST",
                   help="cumulative sample sizes, e.g. 50,100,150,200")
    p.add_argument("--model", choices=["gaussian", "mcnemar"], default="gaussian")
    p.add_argument("--effect", type=float, default=0.0, help="gaussian: mean in SD units")
    p.add_argument("--p10", type=float, help="mcnemar: P(A pass, B fail)")
    p.add_argument("--p01", type=float, help="mcnemar: P(A fail, B pass)")
    p.add_argument("--one-sided", action="store_true", help="gaussian only")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns 0 on success, 1 on invalid input (argparse usage
    errors exit 2 via SystemExit)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload, text = args.func(args)
    except (ValueError, OSError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
