"""
The investigation engine.

Walks a frozen hypothesis tree over a DataSource. At each node it computes,
deterministically:
  * AXIS A (factor): LMDI split of the parent change into multiplicative children
  * AXIS B (segment): additive split of the parent change across a dimension
It drills into whichever axis is most CONCENTRATED, gating every step on
reconciliation (children must sum to the parent). The LLM is NOT here — the
narration at the end is a fixed template over verified numbers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .decompose import lmdi_two_factor, segment_additive, mix_shift, reconcile

SIGNIFICANCE_PCT = 2.0      # ignore parent moves smaller than this
SEGMENT_DOMINANCE = 0.55    # one segment must own >= this share of the delta to drill into it
RECON_TOL_PCT = 0.5
MAX_STEPS = 8


@dataclass
class Step:
    kind: str                 # 'factor' | 'segment' | 'mix'
    scope: str                # human description of current filter scope
    parts: dict               # contribution per child/segment
    recon: object             # Recon
    winner: str
    winner_share: float


@dataclass
class Result:
    parent_delta: float
    parent_pct: float
    significant: bool
    steps: list = field(default_factory=list)
    leaf: str = ""
    confidence: str = "low"
    unreconciled: str = ""


def _pct(delta, base):
    return (delta / base * 100) if base else 0.0


def investigate(src, tree, p0="0", p1="1", filters=None) -> Result:
    filters = dict(filters or {})
    base = src.aggregate(p0, tree.measure, filters)
    cur = src.aggregate(p1, tree.measure, filters)
    delta = cur - base
    res = Result(parent_delta=delta, parent_pct=_pct(delta, base),
                 significant=abs(_pct(delta, base)) >= SIGNIFICANCE_PCT)
    if not res.significant:
        res.leaf = "Change is below the significance threshold — no investigation."
        return res

    node = tree
    dims_used = set()
    scope = "all"
    worst_recon = 0.0

    for _ in range(MAX_STEPS):
        # ---- recompute the local parent change within current scope ----
        b = src.aggregate(p0, node.measure, filters)
        c = src.aggregate(p1, node.measure, filters)
        local_delta = c - b

        # ---- AXIS B: segment split of GMV by each untested dim ----
        drilled = False
        if node.relation in ("root", "multiplicative") and node.measure in ("gmv", "orders"):
            for dim in node.segment_dims:
                if dim in dims_used:
                    continue
                seg0 = {k: v[1] if node.measure == "gmv" else v[0]
                        for k, v in src.segment_table(p0, dim, filters).items()}
                seg1 = {k: v[1] if node.measure == "gmv" else v[0]
                        for k, v in src.segment_table(p1, dim, filters).items()}
                parts = segment_additive(seg0, seg1)
                rec = reconcile(local_delta, parts, RECON_TOL_PCT)
                worst_recon = max(worst_recon, rec.residual_pct)
                winner = max(parts, key=lambda k: abs(parts[k]))
                share = abs(parts[winner] / local_delta) if local_delta else 0
                res.steps.append(Step("segment", f"{scope} · by {dim}", parts, rec, str(winner), share))
                if share >= SEGMENT_DOMINANCE:
                    # drill into the dominant segment
                    filters[dim] = winner
                    dims_used.add(dim)
                    scope = f"{scope} ∩ {dim}={winner}"
                    drilled = True
                    break
        if drilled:
            continue

        # ---- AXIS A: factor split (multiplicative children) ----
        mult = [ch for ch in node.children if ch.relation == "multiplicative"]
        if mult and node.measure == "gmv":
            a = next(ch for ch in mult if ch.measure == "orders")
            v = next(ch for ch in mult if ch.measure == "aov")
            a0 = src.aggregate(p0, "orders", filters); a1 = src.aggregate(p1, "orders", filters)
            v0 = src.aggregate(p0, "aov", filters); v1 = src.aggregate(p1, "aov", filters)
            lm = lmdi_two_factor(b, c, a0, a1, v0, v1)
            parts = {a.id: lm["A"], v.id: lm["B"]}
            rec = reconcile(local_delta, parts, RECON_TOL_PCT)
            worst_recon = max(worst_recon, rec.residual_pct)
            winner = max(parts, key=lambda k: abs(parts[k]))
            share = abs(parts[winner] / local_delta) if local_delta else 0
            res.steps.append(Step("factor", f"{scope} · orders x AOV", parts, rec, winner, share))
            node = a if winner == a.id else v
            continue

        # ---- AOV leaf: rate vs mix ----
        if node.measure == "aov" and any(ch.relation in ("rate", "mix") for ch in node.children):
            dim = node.children[0].segment_dims[0]
            t0 = src.segment_table(p0, dim, filters)
            t1 = src.segment_table(p1, dim, filters)
            rate_eff, mix_eff, _ = mix_shift(t0, t1)
            # express both effects on the same scale as ΔAOV
            parts = {"aov_rate": rate_eff, "aov_mix": mix_eff}
            daov = src.aggregate(p1, "aov", filters) - src.aggregate(p0, "aov", filters)
            rec = reconcile(daov, parts, RECON_TOL_PCT)
            worst_recon = max(worst_recon, rec.residual_pct)
            winner = max(parts, key=lambda k: abs(parts[k]))
            share = abs(parts[winner] / daov) if daov else 0
            res.steps.append(Step("mix", f"{scope} · rate vs mix by {dim}", parts, rec, winner, share))
            res.leaf = ("mix shift toward cheaper segments" if winner == "aov_mix"
                        else "per-segment basket (rate) change")
            break

        # ---- terminal: orders-driven or no further structure ----
        res.leaf = f"{node.label} is the dominant driver ({scope})."
        break

    # confidence from reconciliation + concentration of the last step
    last_share = res.steps[-1].winner_share if res.steps else 0
    if worst_recon <= RECON_TOL_PCT and last_share >= 0.7:
        res.confidence = "high"
    elif worst_recon <= 1.5:
        res.confidence = "medium"
    else:
        res.confidence = "low"
        res.unreconciled = f"max reconciliation residual {worst_recon:.2f}% — possible missing dimension"
    return res


# ----------------------------- rendering -----------------------------
def render(res: Result, metric_label="GMV") -> str:
    out = []
    out.append(f"{metric_label} change: {res.parent_delta:,.0f} ({res.parent_pct:+.1f}%)")
    if not res.significant:
        out.append("  " + res.leaf); return "\n".join(out)
    out.append("-" * 64)
    for i, s in enumerate(res.steps, 1):
        tag = {"factor": "FACTOR", "segment": "WHERE ", "mix": "RATE/MIX"}[s.kind]
        flag = "OK " if s.recon.ok else f"FAIL {s.recon.residual_pct:.1f}%"
        out.append(f"[{i}] {tag} | {s.scope}   reconcile:{flag}")
        for k, v in sorted(s.parts.items(), key=lambda kv: -abs(kv[1])):
            mark = " <-- winner" if k == s.winner else ""
            out.append(f"        {k:<12} {v:>14,.0f}{mark}")
    out.append("-" * 64)
    out.append(f"LEAF: {res.leaf}")
    out.append(f"CONFIDENCE: {res.confidence.upper()}")
    if res.unreconciled:
        out.append(f"NOTE: {res.unreconciled}")
    return "\n".join(out)


def narrate(res: Result, metric_label="GMV") -> str:
    """Fixed template over verified numbers — this is the ONLY place an LLM
    would later sit, and it may only phrase, never compute."""
    if not res.significant:
        return f"{metric_label} did not move significantly ({res.parent_pct:+.1f}%)."
    where = next((s for s in res.steps if s.kind == "segment" and s.winner_share >= SEGMENT_DOMINANCE), None)
    factor = next((s for s in res.steps if s.kind == "factor"), None)
    mix = next((s for s in res.steps if s.kind == "mix"), None)
    parts = [f"{metric_label} {res.parent_pct:+.1f}%"]
    if where:
        parts.append(f"concentrated in {where.winner} ({where.winner_share*100:.0f}% of the change)")
    if factor:
        parts.append(f"driven by {factor.winner}")
    if mix:
        parts.append(res.leaf)
    return " — ".join(parts) + f". Confidence: {res.confidence}."
