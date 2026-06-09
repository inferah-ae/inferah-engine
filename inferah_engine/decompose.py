"""
Deterministic decomposition math. No LLM, no randomness.
Every function returns contributions that reconcile to the total change
(within floating-point error), which is what the engine's gate checks.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


def log_mean(a: float, b: float) -> float:
    """Logarithmic mean L(a,b) = (a-b)/(ln a - ln b), with L(a,a)=a."""
    if a <= 0 or b <= 0:
        # fall back to arithmetic mean if a factor hits zero/negative
        return (a + b) / 2.0
    if abs(a - b) < 1e-12:
        return a
    return (a - b) / (math.log(a) - math.log(b))


def lmdi_factors(m0, m1, factors: dict):
    """
    Decompose a multiplicative metric  M = Π_k F_k  into each factor's
    contribution to ΔM = M1 - M0, using the LMDI index. Exact for any number
    of factors: the contributions sum to ΔM with zero residual (when every
    factor is strictly positive in both periods).

        factors: {factor_name: (f0, f1)}   value of each factor per period

    Returns dict {factor_name: contribution_to_delta}.
    """
    L = log_mean(m1, m0)
    out = {}
    for name, (f0, f1) in factors.items():
        out[name] = L * math.log(f1 / f0) if f0 > 0 and f1 > 0 else 0.0
    return out


def segment_additive(seg0: dict, seg1: dict):
    """
    Additive metric split across a dimension (e.g. GMV by country):
    ΔM = Σ_i (seg1_i - seg0_i). Each segment's contribution is just its delta.
    Reconciles exactly. Returns dict {segment: contribution}.
    """
    keys = set(seg0) | set(seg1)
    return {k: seg1.get(k, 0.0) - seg0.get(k, 0.0) for k in keys}


def mix_shift(table0: dict, table1: dict):
    r"""
    Decompose the change in a RATE metric r = Σ_i w_i * r_i  (e.g. AOV across
    order types / segments) into a 'rate' effect and a 'mix' effect.

    table{0,1} : {segment: (count, value_sum)}  per period.
        r_i = value_sum / count   (segment rate, e.g. segment AOV)
        w_i = count / total_count  (segment share)

    Identity (exact per segment):
        Δ(Σ w_i r_i) = Σ w̄_i Δr_i  +  Σ r̄_i Δw_i
                        \___rate___/    \___mix____/
    Returns (rate_effect, mix_effect, per_segment_detail).
    """
    n0 = sum(c for c, _ in table0.values()) or 1.0
    n1 = sum(c for c, _ in table1.values()) or 1.0
    keys = set(table0) | set(table1)
    rate_effect = 0.0
    mix_effect = 0.0
    detail = {}
    for k in keys:
        c0, s0 = table0.get(k, (0.0, 0.0))
        c1, s1 = table1.get(k, (0.0, 0.0))
        w0, w1 = c0 / n0, c1 / n1
        r0 = (s0 / c0) if c0 else 0.0
        r1 = (s1 / c1) if c1 else 0.0
        wbar = (w0 + w1) / 2.0
        rbar = (r0 + r1) / 2.0
        rate_i = wbar * (r1 - r0)
        mix_i = rbar * (w1 - w0)
        rate_effect += rate_i
        mix_effect += mix_i
        detail[k] = {"rate": rate_i, "mix": mix_i, "w0": w0, "w1": w1, "r0": r0, "r1": r1}
    return rate_effect, mix_effect, detail


@dataclass
class Recon:
    ok: bool
    total: float
    explained: float
    residual: float
    residual_pct: float


def reconcile(total_delta: float, parts: dict, tol_pct: float = 0.5) -> Recon:
    """Gate: do the parts sum to the parent change within tolerance?"""
    explained = sum(parts.values())
    residual = total_delta - explained
    rp = abs(residual / total_delta) * 100 if total_delta else (0.0 if abs(residual) < 1e-9 else float("inf"))
    return Recon(ok=rp <= tol_pct, total=total_delta, explained=explained, residual=residual, residual_pct=rp)
