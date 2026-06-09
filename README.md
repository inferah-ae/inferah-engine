# Inferah Engine

**A deterministic engine for *why did this metric change?*** — it decomposes the change
over a hypothesis tree you define, checks every step reconciles, and refuses to guess
when it doesn't.

---

## The problem

Point an LLM at a warehouse and ask "why did GMV drop 12%?" and you get a plausible,
confident story — and a *different* story next run, none of it reproducible or auditable.
On real analytics questions a raw LLM lands around **51%** accuracy: good enough to sound
right, not good enough to trust. The win here isn't a smarter model — it's **constraint**:
pin the math, pin the tree, and only let a model speak about numbers it did not compute.

## How it works

The **engine owns the math.** At every node it runs two deterministic splits:

- **Factor (axis A)** — a multiplicative metric such as `GMV = orders × AOV` is split with
  **LMDI** (Log-Mean Divisia Index): *exact*, zero residual. Tells you which **lever** moved.
- **Segment (axis B)** — an additive metric is split across a dimension (e.g. GMV by
  country) to tell you **where** it moved.
- For a **rate** metric (e.g. AOV) it further splits **rate vs mix**, so a drop caused by the
  *mix* shifting toward cheaper segments is never misread as the per-segment rate falling
  (Simpson-proof).

It walks a **frozen hypothesis tree** that you author, drills into whichever axis is most
**concentrated**, and **gates every step on reconciliation**: the children must sum to the
parent within tolerance. If a step doesn't reconcile, the engine reports
*"unreconciled — likely a missing dimension"* instead of narrating a guess.

An **LLM is optional and not in this repo.** Its only job would be to translate the user's
question into a tree/measure and to narrate the *already-verified* result. **It never
computes a number** — `narrate()` is a fixed template over figures the engine produced.

## Why it's different (honestly)

Metric-tree / root-cause tooling is an **established category** — see PyRCA, riskloc, and
friends. This isn't "nobody does this." The difference is *what kind of answer* you get:

- **Factor decomposition of the mechanism** — orders-vs-AOV, rate-vs-mix — not just "which
  segment is biggest." That's what makes it robust to Simpson's paradox.
- **A reconciliation gate.** Most tools hand you a ranked list of suspects. This one refuses
  to present a decomposition that doesn't add up, and tells you a dimension is probably
  missing rather than guessing.

Think of it as a **reconciled variance investigator**, not a magic root-cause oracle.

## Quickstart

```bash
pip install -r requirements.txt   # pandas + numpy are enough for the demo
```

```python
from data.synthetic import make_orders
from inferah_engine import SyntheticSource, GMV_TREE, investigate, render, narrate

src = SyntheticSource(make_orders())          # base-view filters applied inside
res = investigate(src, GMV_TREE, p0="0", p1="1")
print(render(res))
print(narrate(res))
```

Expected output:

```
GMV change: -42,400 (-11.2%)
----------------------------------------------------------------
[1] WHERE  | all · by country   reconcile:OK
        Westland            -42,400 <-- winner
        Eastland                  0
        Northland                 0
[2] FACTOR | all ∩ country=Westland · orders x AOV   reconcile:OK
        aov                 -47,576 <-- winner
        orders                5,176
[3] RATE/MIX | all ∩ country=Westland · rate vs mix by order_type   reconcile:OK
        aov_mix                  -9 <-- winner
        aov_rate                  1
----------------------------------------------------------------
LEAF: mix shift toward cheaper segments
CONFIDENCE: HIGH
```

Read it as: **GMV −11.2% → concentrated in Westland → driven by AOV → a mix shift toward
cheaper promo orders**, every step reconciled, confidence high. Or open
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) for the annotated walkthrough.

## Generic core, swappable packs

The engine is **business-agnostic** — it knows LMDI, segment splits, rate/mix, and
reconciliation, and nothing else. All domain knowledge lives in the **hypothesis-tree
packs** you feed it (`trees/*.yaml`):

- [`trees/gmv_drop.yaml`](trees/gmv_drop.yaml) — the food-delivery GMV example
  (`GMV = orders × AOV`, AOV split rate vs mix).
- [`trees/acquisition.yaml`](trees/acquisition.yaml) — a second domain
  (`subscriptions = paywall_visits × conversion`, conversion split rate vs mix).

The food-delivery example is exactly that — **an included example, not the product.**

## Built on

This reuses **established** decomposition / RCA ideas rather than inventing them: the
**LMDI** index from energy-economics index-decomposition analysis, and a design informed by
[riskloc](https://github.com/shaido987/riskloc), [PyRCA](https://github.com/salesforce/PyRCA),
and [ruptures](https://github.com/deepcharles/ruptures). Maturity, not reinvention.

## Status

**Real, today:**

- the deterministic engine — LMDI factor split, additive segment split, rate/mix
- the reconciliation gate + confidence scoring
- the synthetic, zero-setup demo
- a read-only Postgres source (same interface, swap-in — see notebook §3)

**Roadmap (not here yet):**

- LLM-assisted tree auto-construction from a plain-English question
- beam search over top-k branches (compound causes)
- feedback-driven branch priors (changes search *order*, never the math)
- more tree packs (fail-rate, CSAT, supply)

## License

[Apache-2.0](LICENSE).
