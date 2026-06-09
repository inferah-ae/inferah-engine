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

  make_foodtech_deep()   DEEP pack  — order-grain rows covering the full
                                       GMV = MAU × OrderConv × Frequency × AOV tree.
                                       A Simpson MIX shift: promo penetration rises
                                       in one city because a high-promo cuisine grew
                                       its share — not because promos got deeper.
  make_foodtech_offtree() DEEP pack — the drop sits in NULL-city rows -> abstain.

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


def make_foodtech_deep() -> pd.DataFrame:
    """DEEP pack, order-grain (1 row = 1 order). GMV falls ~3.9% with orders flat.

    The cause is a classic Simpson MIX shift, reachable only by walking the full
    GMV = MAU × OrderConv × Frequency × AOV identity down to a leaf:

      WHERE   the move localizes to city=Metro (Harbor is flat);
      FACTOR  within Metro, MAU/OrderConv/Frequency are flat -> AOV carries it;
      COMPOSE AOV = Food + Delivery + Service + Small − Discount; only Discount moved;
      FACTOR  Discount = PromoPenetration × AvgDiscount; AvgDiscount flat (8 always)
              -> PromoPenetration carries it (0.35 -> 0.50);
      RATE/MIX by cuisine: the within-cuisine promo rate NEVER changed (burgers 0.60,
              pizza 0.10 in both periods). Penetration rose purely because the order
              MIX shifted toward burgers (share 0.5 -> 0.8) — the high-promo cuisine.
              Promos did NOT get deeper or more frequent within any cuisine.

    Every step reconciles exactly, so the engine lands on the promo-penetration MIX
    leaf with HIGH confidence. Per-order economics are identical across the whole
    frame (food 25, delivery 3, service 1, small 0; promo orders carry an 8 discount),
    so nothing but the cuisine mix can move AOV — that is what makes it Simpson-clean.
    """
    rows = []
    counter = {"0": 0, "1": 0}   # per-period order counter (2 orders/user, disjoint by city)

    def add(period, city, cuisine, order_type, n):
        promo = order_type == "promo"
        gmv = 21 if promo else 29        # 25 + 3 + 1 + 0 − (8 if promo else 0)
        for i in range(n):
            idx = counter[period]
            counter[period] += 1
            user_idx = idx // 2          # two orders per user
            uid = f"{period}-u{user_idx}"
            c = user_idx % 10            # deterministic cohort split: 20% new, 10% resurr, 70% retained
            cohort_status = "new" if c < 2 else ("resurrected" if c == 2 else "retained")
            rows.append(dict(
                period=period, city=city, channel=("app" if i % 2 == 0 else "web"),
                cuisine=cuisine, order_type=order_type,
                signup_cohort=f"2025Q{(user_idx % 4) + 1}",
                gmv_usd=gmv, food_usd=25, delivery_fee_usd=3, service_fee_usd=1,
                small_order_fee_usd=0, discount_usd=(8 if promo else 0),
                items=3, is_paying_delivery=1, is_promo=(1 if promo else 0),
                user_id=uid,
                new_user_id=(uid if cohort_status == "new" else None),
                resurrected_user_id=(uid if cohort_status == "resurrected" else None),
                retained_user_id=(uid if cohort_status == "retained" else None),
                install_user_id=(uid if cohort_status == "new" else None),
                prev_active_user_id=(uid if cohort_status == "retained" else None),
            ))

    # period 0 — Metro: burgers 1000 (60% promo), pizza 1000 (10% promo); penetration 0.35
    add("0", "Metro", "burgers", "promo", 600); add("0", "Metro", "burgers", "normal", 400)
    add("0", "Metro", "pizza",   "promo", 100); add("0", "Metro", "pizza",   "normal", 900)
    add("0", "Harbor", "sushi",  "normal", 300)
    # period 1 — Metro: burger share 0.5 -> 0.8 (the MIX shift), within-cuisine promo unchanged;
    #            penetration 0.50, AOV 26.2 -> 25.0, orders held at 2000.
    add("1", "Metro", "burgers", "promo", 960); add("1", "Metro", "burgers", "normal", 640)
    add("1", "Metro", "pizza",   "promo",  40); add("1", "Metro", "pizza",   "normal", 360)
    add("1", "Harbor", "sushi",  "normal", 300)
    return pd.DataFrame(rows)


def make_foodtech_offtree() -> pd.DataFrame:
    """DEEP pack abstain scenario. The whole GMV drop sits in rows whose `city` is
    NULL (an unmapped segment the tree's WHERE axis can't see). The city split
    silently drops those rows via groupby(dropna) and therefore can't reconcile to
    the parent change — so the engine ABSTAINS instead of blaming Metro at random."""
    rows = []
    counter = {"0": 0, "1": 0}

    def add(period, city, n):
        for i in range(n):
            idx = counter[period]
            counter[period] += 1
            uid = f"{period}-u{idx // 2}"
            rows.append(dict(
                period=period, city=city, channel=("app" if i % 2 == 0 else "web"),
                cuisine="burgers", order_type="normal", signup_cohort="2025Q1",
                gmv_usd=29, food_usd=25, delivery_fee_usd=3, service_fee_usd=1,
                small_order_fee_usd=0, discount_usd=0, items=3, is_paying_delivery=1,
                is_promo=0, user_id=uid,
                new_user_id=None, resurrected_user_id=None, retained_user_id=uid,
                install_user_id=None, prev_active_user_id=uid,
            ))

    add("0", "Metro", 1000); add("1", "Metro", 1000)   # mapped city: flat
    add("0", None, 1000);    add("1", None, 200)        # NULL-city: −23,200, invisible to the split
    return pd.DataFrame(rows)
