"""
Synthetic but DETERMINISTIC data, so the engine runs with zero setup and the
same inputs always produce the same investigation. Three scenarios, one per
thing worth showing:

  make_orders()          GMV pack   — a MIX shift (Simpson): AOV falls because a
                                       promo flooded one region with cheap orders,
                                       per-order price unchanged.
  make_visits()          ACQ pack   — a RATE drop: paywall conversion fell within
                                       every traffic source; not localizable to one.
  make_orders_unmapped() GMV pack   — the drop hides in rows whose dimension is
                                       NULL, so the country split can't reconcile
                                       and the engine ABSTAINS instead of guessing.

Region / source names are fictional placeholders.
"""
from __future__ import annotations
import pandas as pd


def make_orders() -> pd.DataFrame:
    """GMV down ~12%, orders flat, AOV down — concentrated in Westland, and the
    cause is MIX (a promo flooded Westland with cheap orders), not a price cut."""
    rows = []

    def add(period, country, order_type, n, price, status="delivered", is_test=False):
        for _ in range(n):
            rows.append(dict(period=period, country=country, order_type=order_type,
                             gmv_usd=price, status=status, is_test=is_test))

    # period 0 (baseline)
    add("0", "Northland", "normal", 5000, 30)
    add("0", "Westland",  "normal", 6000, 30)
    add("0", "Eastland",  "normal", 4000, 12)
    add("0", "Westland", "normal", 50, 30, status="cancelled")   # base view must drop
    add("0", "Westland", "normal", 30, 999, is_test=True)        # base view must drop
    # period 1 (current): Westland flooded with cheap promo orders
    add("1", "Northland", "normal", 5000, 30)
    add("1", "Westland",  "normal", 4000, 30)
    add("1", "Westland",  "promo",  2200, 8)
    add("1", "Eastland",  "normal", 4000, 12)
    add("1", "Westland", "normal", 40, 30, status="cancelled")
    return pd.DataFrame(rows)


def make_visits() -> pd.DataFrame:
    """Acquisition pack: subscriptions = paywall_visits x paywall_conversion.
    Visits are flat; conversion falls 10%%->7%% inside BOTH sources equally. The
    cause is a real RATE drop, and it is NOT concentrated in one source — so the
    segment split finds no winner and the rate/mix split carries the story."""
    rows = []

    def add(period, mkt_source, n, converted):
        for _ in range(n):
            rows.append(dict(period=period, mkt_source=mkt_source, converted=converted))

    # period 0: each source 10000 visits @ 10% conversion -> 1000 subs each
    add("0", "search", 1000, 1); add("0", "search", 9000, 0)
    add("0", "social", 1000, 1); add("0", "social", 9000, 0)
    # period 1: visits flat, conversion 7% in BOTH -> 700 subs each
    add("1", "search", 700, 1);  add("1", "search", 9300, 0)
    add("1", "social", 700, 1);  add("1", "social", 9300, 0)
    return pd.DataFrame(rows)


def make_orders_unmapped() -> pd.DataFrame:
    """The entire GMV drop sits in rows whose `country` is NULL (an unmapped /
    newly-added segment the tree doesn't model). The country split silently
    excludes those rows and therefore CANNOT reconcile to the parent change —
    the engine abstains rather than blaming a country at random."""
    rows = []

    def add(period, country, n, price):
        for _ in range(n):
            rows.append(dict(period=period, country=country, order_type="normal",
                             gmv_usd=price, status="delivered", is_test=False))

    for period in ("0", "1"):
        add(period, "Northland", 5000, 30)
        add(period, "Westland",  5000, 30)
        add(period, "Eastland",  5000, 30)
    # the move lives entirely in NULL-country rows
    add("0", None, 3000, 30)     # 90,000
    add("1", None, 200, 30)      #  6,000  -> -84,000, invisible to a country split
    return pd.DataFrame(rows)
