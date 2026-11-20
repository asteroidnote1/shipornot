# shipornot

Small-sample statistics for LLM evals: honest intervals, paired comparisons of two prompts or models on the same cases, sample-size planning, and a non-inferiority gate for CI. One file, NumPy + SciPy only, MIT license.

```bash
pip install "git+https://github.com/asteroidnote1/shipornot"
shipornot --help
```

Without installing, `python shipornot.py <subcommand>` works the same (after `pip install numpy scipy`). Python 3.11 or newer.

## Five questions, five commands

**1. What is the honest range for 32 passes out of 40?**

```console
$ shipornot wilson 32 40
Wilson 95%: 32/40 = 0.8000  CI [0.6524, 0.8950]
```

**2. B vs A on the same 200 cases: 134 both pass, 7 regressions, 9 improvements, 50 both fail. How big is the difference?**

```console
$ shipornot paired-diff --n11 134 --n10 7 --n01 9 --n00 50
pB - pA = 0.0100  Newcombe (method 8) 95% CI [-0.0295, 0.0497]  (n = 200, n10 = 7, n01 = 9)
```

**3. Is that difference detectable at all?**

```console
$ shipornot mcnemar --n10 7 --n01 9
exact McNemar (two-sided): regressions n10 = 7, improvements n01 = 9, p = 0.8036  -> do not reject H0 at alpha = 0.05
$ shipornot mde 200 --design paired --p-disc 0.10
MDE = 0.0623 (paired, n = 200, p_disc = 0.1; alpha = 0.05, power = 0.8)
```

With 200 paired cases where A and B disagree on about 10% of them, only differences of about 0.06 or more are reliably detectable.

**4. How many cases do I need to detect a 0.04 gain?**

```console
$ shipornot n-paired 0.10 0.04
cases needed (Connor): 489  (p_disc = 0.1, delta = 0.04, two-sided alpha = 0.05, power = 0.8)
$ shipornot power-mcnemar 489 0.03 0.07 --method enum
exact McNemar power at n = 489: 0.7754  (exact enumeration)
```

The planning formula says 489; the exact test needs 517 for the full 0.80 power. Two independent test sets would need 1,447 cases each for the same gain.

**5. Is the candidate at least no worse than the baseline (CI gate)?**

```console
$ shipornot ni-gate --n11 134 --n10 7 --n01 9 --n00 50 --margin 0.03
PASS: B is non-inferior
pB - pA = 0.0100  90% CI [-0.0231, 0.0432]  margin = -0.03, one-sided alpha = 0.05
```

Every subcommand accepts `--json` after the subcommand name (for scripts and agents), and most accept `--csv` with per-case pass/fail columns. Run `shipornot <subcommand> --help` for the options.

## What is inside

| Subcommand | Method |
|---|---|
| `wilson`, `cp` | Wilson score and Clopper–Pearson intervals for a pass rate |
| `mean`, `bootstrap` | t interval; percentile and BCa bootstrap (one sample or paired) |
| `mcnemar`, `paired-diff` | exact McNemar test; Newcombe (1998) paired-difference interval (method 8 by default, method 10 with `--method10`) |
| `n-unpaired`, `n-paired`, `power-mcnemar`, `mde` | sample size, exact power, minimum detectable effect |
| `design-effect` | design effect / ICC for reruns and clustered cases, cluster-robust interval, cluster bootstrap |
| `kappa`, `rogan-gladen` | judge–human agreement; pass rate corrected for judge sensitivity/specificity |
| `holm`, `bh` | multiple-comparison adjustment |
| `ni-gate` | paired non-inferiority gate |
| `peeking` | false-positive rate when you check results repeatedly |

## How it is checked

```bash
pip install pytest statsmodels
pytest
```

- 170 unit tests, including comparisons against SciPy/statsmodels and all 13 worked examples in Table III of Newcombe (1998) for both method 8 and method 10.
- The interval and test methods were also checked by Monte Carlo simulation (coverage, type-I error, power). The simulation scripts and their saved results are part of the guide described below.

`articles/` holds the files behind the free article *How many judge labels do you need?*: `judge_validation.csv` (synthetic, 100 labels) and `verify_article02.py`, which prints every number in that article as JSON (`python articles/verify_article02.py`).

Results are educational and not a guarantee for any decision.

## Background

`shipornot` is the free companion to *Is B Really Better?*, a decision guide for LLM evals (which question to ask, which number answers it, how many cases to build, and how to write the result up for a non-statistician). **The guide is not on sale.** A free Quick Card and two free articles are at https://asteroidnote1.github.io/. If you want to hear when it is released, watch this repository (Watch → Custom → Releases). No email is needed.

Written with AI assistance.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
