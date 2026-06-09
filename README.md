# Inferah Engine

**A deterministic engine for *why did this metric change?*** — it decomposes the change
over a hypothesis tree you define, checks every step reconciles, and refuses to guess
when it doesn't.

---

## The problem

Point an LLM at a warehouse and ask "why did GMV drop 12%?" and you get a plausible,
confident story — and a *different* story next run, none of it reproducible or auditable.
Text-to-SQL / analytics benchmarks (e.g. BIRD, Spider 2.0) still put even strong models
well short of the accuracy you'd trust for a decision: good enough to sound right, not good
enough to act on. The win here isn't a smarter model — it's **constraint**: pin the math,
pin the tree, and only let a model speak about numbers it did not compute.

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
pip install -e .                  # pandas + numpy + PyYAML; add ".[postgres]" for the DB source
python -m inferah_engine.demo     # runs all three scenarios below
```

```python
from inferah_engine import SyntheticSource, GMV_TREE, GMV_MEASURES, investigate, render, narrate
from inferah_engine.synthetic import make_orders

src = SyntheticSource(make_orders(), GMV_MEASURES)   # base-view filters applied inside
res = investigate(src, GMV_TREE, p0="0", p1="1")
print(render(res))
print(narrate(res))
```

Expected output:

```
GMV change: -42,400 (-11.2%)
----------------------------------------------------------------
[1] WHERE  | all · by country   [Δgmv]   reconcile:OK
        Westland         -42,400.00 <-- winner
        Northland              0.00
        Eastland               0.00
[2] FACTOR | all ∩ country=Westland · Orders x AOV   [Δgmv]   reconcile:OK
        aov              -47,575.94 <-- winner
        orders             5,175.94
[3] RATE/MIX | all ∩ country=Westland · rate vs mix by order_type   [Δaov]   reconcile:OK
        aov_mix               -9.23 <-- winner
        aov_rate               1.42
----------------------------------------------------------------
LEAF: Order mix shifted toward cheaper segments (promo, market shift).
CONFIDENCE: HIGH
```

The `[Δ…]` tag names the measure each step is decomposing, so the units are explicit:
steps 1–2 are in **GMV dollars**, step 3 is in **AOV dollars** (the rate/mix split is on the
average, not the total). Read it as: **GMV −11.2% → concentrated in Westland → driven by AOV
→ a mix shift toward cheaper promo orders**, every step reconciled, confidence high. Or open
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) for the annotated walkthrough.

### It refuses to guess

When the real driver is a dimension the tree doesn't model, the segment split can't add up to
the parent change — and the engine **abstains** rather than blaming the biggest visible slice:

```python
from inferah_engine.synthetic import make_orders_unmapped
res = investigate(SyntheticSource(make_orders_unmapped(), GMV_MEASURES), GMV_TREE)
print(narrate(res))
# GMV moved -15.6%, but the tree can't reconcile it — splitting by country leaves 100% of
# the -84,000 move unexplained — a driver this tree does not model … Refusing to guess.
```

## Generic core, swappable packs

The engine is **business-agnostic** — it knows LMDI, segment splits, rate/mix, and
reconciliation, and nothing else. There are **no measure names baked into the engine**: it
decides what to do at each node purely from the node's `relation` and the *kind* of its
measure (`sum` / `count` / `ratio`) in the pack's measure registry. All domain knowledge
lives in the **packs** you feed it — a pack is a `measures:` registry plus a `tree:`, loaded
from YAML:

```python
from inferah_engine import load_pack, SyntheticSource, investigate
pack = load_pack("trees/acquisition.yaml")           # a different metric, the same walk
res = investigate(SyntheticSource(df, pack.measures), pack.tree)
```

- [`trees/gmv_drop.yaml`](trees/gmv_drop.yaml) — the food-delivery GMV example
  (`GMV = orders × AOV`, AOV split rate vs mix).
- [`trees/acquisition.yaml`](trees/acquisition.yaml) — a second domain
  (`subscriptions = paywall_visits × conversion`, conversion split rate vs mix).
- [`trees/gmv_foodtech_deep.yaml`](trees/gmv_foodtech_deep.yaml) — a **deep** food-delivery
  pack exercising every primitive (see below).

Both packs run through the identical engine; a test asserts the YAML parses to the same
objects as the Python constants in `tree.py` so they can't drift. The food-delivery example
is exactly that — **an included example, not the product.**

### The deep pack: one identity, all four primitives

[`trees/gmv_foodtech_deep.yaml`](trees/gmv_foodtech_deep.yaml) is the full master identity,
all algebra, no drivers:

```
GMV = MAU × OrderConv × Frequency × AOV
  MAU = New + Resurrected + Retained                       (signed additive)
    New      = Installs × FirstOrderConv
    Retained = PrevActives × RetentionRate
  AOV = Food + DeliveryFee + ServiceFee + SmallOrderFee − Discount   (signed additive)
    Food     = ItemsPerOrder × PricePerItem
    Discount = PromoPenetration × AvgDiscount
```

The bundled order-grain scenario (`make_foodtech_deep`) is a textbook **Simpson trap**: GMV
falls ~3.9% with orders flat, and the engine walks the identity end to end —

```
WHERE  → city=Metro (100% of the move; Harbor flat)
FACTOR → AOV (MAU, OrderConv, Frequency all flat)
COMPOSE→ −Discount (Food/Delivery/Service/Small unchanged)
FACTOR → PromoPenetration (AvgDiscount flat: promos never got deeper)
RATE/MIX by cuisine → MIX wins: within-cuisine promo rates never moved; penetration
         rose only because the order mix shifted toward a high-promo cuisine.
```

Every step reconciles exactly, so it lands on the promo-penetration **MIX** leaf with HIGH
confidence — not "promos got more aggressive." A second scenario (`make_foodtech_offtree`)
hides the whole drop in NULL-city rows the WHERE axis can't see, and the engine **abstains**.

Supporting this pack meant generalizing the engine twice (no GMV hardcoding): a **signed
additive composition** node (`M = Σ ±child`, e.g. AOV's `− Discount`) and **count-distinct**
measures (`MAU = COUNT(DISTINCT user_id)`, intensive — never dim-split). Both are driven off
the registry and covered by tests.

## Built on

This reuses **established** decomposition / RCA ideas rather than inventing them: the
**LMDI** index from energy-economics index-decomposition analysis, and a design informed by
[riskloc](https://github.com/shaido987/riskloc), [PyRCA](https://github.com/salesforce/PyRCA),
and [ruptures](https://github.com/deepcharles/ruptures). Maturity, not reinvention.

## Status

**Real, today:**

- the generic deterministic engine — n-factor LMDI split, additive segment split, rate/mix —
  driven entirely off the tree's `relation` + measure-kind, with **zero metric names hardcoded**
- the reconciliation gate + confidence scoring, and an **honest abstain** when a split can't
  account for the parent change
- a real YAML pack loader (`load_pack`) — two packs (`gmv_drop`, `acquisition`) run through the
  same engine
- property tests + CI (`pytest`, GitHub Actions) and a zero-setup demo
  (`python -m inferah_engine.demo`)
- a read-only Postgres source (same interface, swap-in — see notebook §3)

**Roadmap (not here yet):**

- LLM-assisted tree auto-construction from a plain-English question
- beam search over top-k branches (compound causes; today the walk is greedy)
- feedback-driven branch priors (changes search *order*, never the math)
- more tree packs (fail-rate, CSAT, supply)

## License

[Apache-2.0](LICENSE).
