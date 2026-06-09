"""Property tests for the decomposition math: the contributions must RECONCILE
to the total change. This is the invariant the whole engine leans on."""
import math

from inferah_engine.decompose import (
    log_mean, lmdi_factors, segment_additive, mix_shift, reconcile,
)

CASES = [
    # (m0, m1, factors)  with m == product(factors)
    (1000.0, 1000.0, {"orders": (100.0, 100.0), "aov": (10.0, 10.0)}),   # flat
    (1000.0, 800.0, {"orders": (100.0, 100.0), "aov": (10.0, 8.0)}),     # rate down
    (1000.0, 1260.0, {"orders": (100.0, 120.0), "aov": (10.0, 10.5)}),   # both up
    (50.0, 72.0, {"a": (5.0, 6.0), "b": (10.0, 12.0)}),                  # 2 generic factors
]


def test_log_mean_diagonal():
    assert log_mean(7.0, 7.0) == 7.0
    # bounded by the two arguments
    lm = log_mean(2.0, 8.0)
    assert 2.0 < lm < 8.0


def test_lmdi_factors_reconcile_to_total():
    for m0, m1, factors in CASES:
        # the factors must actually multiply to the metric
        p0 = math.prod(f0 for f0, _ in factors.values())
        p1 = math.prod(f1 for _, f1 in factors.values())
        assert math.isclose(p0, m0, rel_tol=1e-9)
        assert math.isclose(p1, m1, rel_tol=1e-9)
        parts = lmdi_factors(m0, m1, factors)
        assert math.isclose(sum(parts.values()), m1 - m0, abs_tol=1e-6)


def test_segment_additive_reconciles():
    seg0 = {"A": 100.0, "B": 200.0, "C": 50.0}
    seg1 = {"A": 90.0, "B": 260.0, "D": 10.0}
    parts = segment_additive(seg0, seg1)
    total = sum(seg1.values()) - sum(seg0.values())
    assert math.isclose(sum(parts.values()), total, abs_tol=1e-9)


def test_mix_shift_identity():
    # rate effect + mix effect == Δ(weighted average), exactly
    t0 = {"normal": (100.0, 3000.0), "promo": (10.0, 100.0)}   # (count, value_sum)
    t1 = {"normal": (80.0, 2400.0), "promo": (60.0, 540.0)}
    rate, mix, _ = mix_shift(t0, t1)
    avg0 = sum(s for _, s in t0.values()) / sum(c for c, _ in t0.values())
    avg1 = sum(s for _, s in t1.values()) / sum(c for c, _ in t1.values())
    assert math.isclose(rate + mix, avg1 - avg0, abs_tol=1e-9)


def test_reconcile_flags_residual():
    ok = reconcile(100.0, {"a": 60.0, "b": 40.0})
    assert ok.ok and ok.residual_pct < 1e-9
    bad = reconcile(100.0, {"a": 10.0})       # 90 unexplained
    assert not bad.ok and math.isclose(bad.residual_pct, 90.0, abs_tol=1e-6)
