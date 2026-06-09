"""Synthetic but DETERMINISTIC orders. Encodes one scenario:
GMV down ~12%, orders flat, AOV down — concentrated in Westland, mix-driven
(a promo flooded Westland with cheap orders). Lets the engine run with zero setup.
Region names are fictional placeholders — this is a made-up scenario, not real data."""
import pandas as pd

def make_orders() -> pd.DataFrame:
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
