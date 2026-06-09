"""
The investigation engine.

Walks a frozen hypothesis tree over a DataSource. It is GENERIC: it knows
nothing about GMV or AOV. It decides what to do at each node purely from the
node's `relation` and the *kind* of its `measure` (looked up in the source's
measure registry):

  * AXIS A (factor):  node has multiplicative children -> LMDI split of the
                      parent change across those child measures.
  * AXIS B (segment): node's measure is EXTENSIVE (sum/count) -> additive split
                      of the parent change across a dimension.
  * RATE / MIX:       node's measure is a RATIO and it has rate+mix children ->
                      split Δ(ratio) into a within-segment rate effect and a
                      between-segment mix effect (the Simpson split).

Every step is gated on RECONCILIATION (children must sum to the parent). If a
segment split leaves a large residual — rows the dimension can't account for —
the engine does NOT guess: it ABSTAINS and reports the unexplained gap. The LLM
is not in this loop; narration is a fixed template over verified numbers.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .decompose import (lmdi_factors, segment_additive, composition_additive,
                        mix_shift, reconcile)


@dataclass(frozen=True)
class Params:
    significance_pct: float = 2.0     # ignore parent moves smaller than this
    segment_dominance: float = 0.55   # a segment must own >= this share to drill into it
    recon_tol_pct: float = 0.5        # a step "reconciles" if residual <= this
    abstain_residual_pct: float = 1.5  # above this, refuse to localize -> honest abstain
    max_steps: int = 8


# back-compat module constant some callers import for their own reporting
SEGMENT_DOMINANCE = Params().segment_dominance


@dataclass
class Step:
    kind: str                 # 'factor' | 'segment' | 'mix'
    scope: str                # human description of current filter scope
    measure: str              # which measure this step decomposed
    parts: dict               # contribution per child/segment (in measure units)
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
    abstained: bool = False


def _pct(delta, base):
    return (delta / base * 100) if base else 0.0


def investigate(src, tree, p0="0", p1="1", filters=None, *, params: Params = Params()) -> Result:
    measures = src.measures
    filters = dict(filters or {})
    base = src.aggregate(p0, tree.measure, filters)
    cur = src.aggregate(p1, tree.measure, filters)
    delta = cur - base
    res = Result(parent_delta=delta, parent_pct=_pct(delta, base),
                 significant=abs(_pct(delta, base)) >= params.significance_pct)
    if not res.significant:
        res.leaf = "Change is below the significance threshold — no investigation."
        return res

    node = tree
    dims_used = set()
    scope = "all"
    worst_recon = 0.0

    for _ in range(params.max_steps):
        local_delta = (src.aggregate(p1, node.measure, filters)
                       - src.aggregate(p0, node.measure, filters))
        m = measures[node.measure]

        # ---- AXIS B: segment split (extensive measures only) ----
        drilled = False
        if m.extensive:
            for dim in node.segment_dims:
                if dim in dims_used:
                    continue
                seg0 = src.segment(p0, node.measure, dim, filters)
                seg1 = src.segment(p1, node.measure, dim, filters)
                parts = segment_additive(seg0, seg1)
                rec = reconcile(local_delta, parts, params.recon_tol_pct)
                worst_recon = max(worst_recon, rec.residual_pct)
                winner = max(parts, key=lambda k: abs(parts[k])) if parts else None
                share = abs(parts[winner] / local_delta) if (winner is not None and local_delta) else 0.0
                res.steps.append(Step("segment", f"{scope} · by {dim}", node.measure,
                                      parts, rec, str(winner), share))

                # honest abstain: the split can't account for the parent change
                if not rec.ok and rec.residual_pct > params.abstain_residual_pct:
                    res.abstained = True
                    res.unreconciled = (
                        f"splitting by {dim} leaves {rec.residual_pct:.0f}% of the "
                        f"{local_delta:+,.0f} move unexplained — a driver this tree "
                        f"does not model (e.g. a dimension not in the tree).")
                    res.leaf = f"Unreconciled — refusing to localize within {scope}."
                    res.confidence = "low"
                    return res

                if share >= params.segment_dominance:
                    filters[dim] = winner
                    dims_used.add(dim)
                    scope = f"{scope} ∩ {dim}={winner}"
                    drilled = True
                    break
        if drilled:
            continue

        # ---- AXIS A: factor split (multiplicative children) ----
        mult = [ch for ch in node.children if ch.relation == "multiplicative"]
        if mult:
            factors = {ch.id: (src.aggregate(p0, ch.measure, filters),
                               src.aggregate(p1, ch.measure, filters)) for ch in mult}
            parts = lmdi_factors(src.aggregate(p0, node.measure, filters),
                                 src.aggregate(p1, node.measure, filters), factors)
            rec = reconcile(local_delta, parts, params.recon_tol_pct)
            worst_recon = max(worst_recon, rec.residual_pct)
            winner = max(parts, key=lambda k: abs(parts[k]))
            share = abs(parts[winner] / local_delta) if local_delta else 0.0
            labels = " x ".join(ch.label for ch in mult)
            res.steps.append(Step("factor", f"{scope} · {labels}", node.measure,
                                  parts, rec, winner, share))
            node = next(ch for ch in mult if ch.id == winner)
            continue

        # ---- AXIS C: additive composition (signed summand children) ----
        summ = [ch for ch in node.children if ch.relation == "summand"]
        if summ:
            terms = {ch.id: (ch.sign,
                             src.aggregate(p0, ch.measure, filters),
                             src.aggregate(p1, ch.measure, filters)) for ch in summ}
            parts = composition_additive(terms)
            rec = reconcile(local_delta, parts, params.recon_tol_pct)
            worst_recon = max(worst_recon, rec.residual_pct)
            winner = max(parts, key=lambda k: abs(parts[k]))
            share = abs(parts[winner] / local_delta) if local_delta else 0.0
            signed = " ".join(("−" if ch.sign < 0 else "+") + ch.label for ch in summ)
            res.steps.append(Step("compose", f"{scope} · {signed}", node.measure,
                                  parts, rec, winner, share))
            node = next(ch for ch in summ if ch.id == winner)
            continue

        # ---- RATE vs MIX (ratio measure with rate+mix children) ----
        rate_mix = [ch for ch in node.children if ch.relation in ("rate", "mix")]
        if rate_mix and m.kind == "ratio":
            rate_ch = next(ch for ch in rate_mix if ch.relation == "rate")
            mix_ch = next(ch for ch in rate_mix if ch.relation == "mix")
            dim = rate_ch.segment_dims[0]
            t0 = src.ratio_segment(p0, node.measure, dim, filters)
            t1 = src.ratio_segment(p1, node.measure, dim, filters)
            rate_eff, mix_eff, _ = mix_shift(t0, t1)
            parts = {rate_ch.id: rate_eff, mix_ch.id: mix_eff}
            dratio = local_delta
            rec = reconcile(dratio, parts, params.recon_tol_pct)
            worst_recon = max(worst_recon, rec.residual_pct)
            winner = max(parts, key=lambda k: abs(parts[k]))
            share = abs(parts[winner] / dratio) if dratio else 0.0
            res.steps.append(Step("mix", f"{scope} · rate vs mix by {dim}", node.measure,
                                  parts, rec, winner, share))
            res.leaf = (mix_ch.note if winner == mix_ch.id else rate_ch.note) \
                or f"{winner} is the dominant driver."
            break

        # ---- terminal: no further structure ----
        res.leaf = f"{node.label} is the dominant driver ({scope})."
        break

    # confidence from reconciliation + concentration of the last step
    last_share = res.steps[-1].winner_share if res.steps else 0.0
    if worst_recon <= params.recon_tol_pct and last_share >= 0.7:
        res.confidence = "high"
    elif worst_recon <= params.abstain_residual_pct:
        res.confidence = "medium"
    else:
        res.confidence = "low"
        res.unreconciled = res.unreconciled or (
            f"max reconciliation residual {worst_recon:.2f}% — possible missing dimension")
    return res


# ----------------------------- rendering -----------------------------
def render(res: Result, metric_label="GMV") -> str:
    out = []
    out.append(f"{metric_label} change: {res.parent_delta:,.0f} ({res.parent_pct:+.1f}%)")
    if not res.significant:
        out.append("  " + res.leaf); return "\n".join(out)
    out.append("-" * 64)
    for i, s in enumerate(res.steps, 1):
        tag = {"factor": "FACTOR", "segment": "WHERE ", "compose": "COMPOSE",
               "mix": "RATE/MIX"}[s.kind]
        flag = "OK " if s.recon.ok else f"FAIL {s.recon.residual_pct:.1f}%"
        out.append(f"[{i}] {tag} | {s.scope}   [Δ{s.measure}]   reconcile:{flag}")
        for k, v in sorted(s.parts.items(), key=lambda kv: -abs(kv[1])):
            mark = " <-- winner" if k == s.winner else ""
            out.append(f"        {k:<12} {v:>14,.2f}{mark}")
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
    if res.abstained:
        return (f"{metric_label} moved {res.parent_pct:+.1f}%, but the tree can't "
                f"reconcile it — {res.unreconciled} Refusing to guess.")
    where = next((s for s in res.steps
                  if s.kind == "segment" and s.winner_share >= SEGMENT_DOMINANCE), None)
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
