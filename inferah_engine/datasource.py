"""
DataSource: the only thing that touches data. The engine asks it for small
AGGREGATES (sums/counts), never raw rows. Two implementations share one
interface so the engine code is identical for synthetic data and a real DB.

    aggregate(period, measure, filters)            -> float
    segment_table(period, dim, filters)            -> {seg: (orders, gmv)}
"""
from __future__ import annotations
import pandas as pd

# period = which window. We keep it abstract: "0" = baseline, "1" = current.
# In a real deployment these map to date ranges.


class SyntheticSource:
    """Backed by an in-memory orders DataFrame. Already filtered to the base
    view (delivered, non-test) when constructed."""

    def __init__(self, df: pd.DataFrame):
        # base-view discipline: drop test + non-delivered up front
        self.df = df[(df["status"] == "delivered") & (~df["is_test"])].copy()

    def _slice(self, period, filters):
        d = self.df[self.df["period"] == period]
        for col, val in (filters or {}).items():
            d = d[d[col] == val]
        return d

    def aggregate(self, period, measure, filters=None) -> float:
        d = self._slice(period, filters)
        if measure == "gmv":
            return float(d["gmv_usd"].sum())
        if measure == "orders":
            return float(len(d))
        if measure == "aov":
            n = len(d)
            return float(d["gmv_usd"].sum() / n) if n else 0.0
        raise ValueError(f"unknown measure {measure}")

    def segment_table(self, period, dim, filters=None):
        """For a dimension, return {segment: (order_count, gmv_sum)} — enough
        for both additive (GMV) and mix-shift (AOV) decomposition."""
        d = self._slice(period, filters)
        g = d.groupby(dim)["gmv_usd"].agg(["count", "sum"])
        return {k: (float(r["count"]), float(r["sum"])) for k, r in g.iterrows()}


class PostgresSource:
    """Same interface, backed by a read-only Postgres connection over a pinned
    base view. Pseudocode-complete; needs `pip install sqlalchemy psycopg2-binary`
    and a real base view. The engine code does not change.

        src = PostgresSource(
            url="postgresql+psycopg2://readonly:***@localhost:5432/inferah",
            base_view="finance.net_revenue_orders",
            period_col="day",
            periods={"0": ("2026-05-11","2026-05-18"), "1": ("2026-05-18","2026-05-25")},
        )
    """

    def __init__(self, url, base_view, period_col, periods,
                 gmv_col="net_revenue_usd"):
        from sqlalchemy import create_engine
        self.engine = create_engine(url)            # use a READ-ONLY role
        self.base_view = base_view
        self.period_col = period_col
        self.periods = periods                      # {"0": (start,end), "1": (start,end)}
        self.gmv = gmv_col

    def _where(self, period, filters):
        s, e = self.periods[period]
        clauses = [f"{self.period_col} >= '{s}'", f"{self.period_col} < '{e}'"]
        for col, val in (filters or {}).items():
            clauses.append(f"{col} = '{val}'")
        return " AND ".join(clauses)

    def aggregate(self, period, measure, filters=None) -> float:
        import pandas as pd
        w = self._where(period, filters)
        if measure == "gmv":
            sql = f"SELECT COALESCE(SUM({self.gmv}),0) v FROM {self.base_view} WHERE {w}"
        elif measure == "orders":
            sql = f"SELECT COUNT(*) v FROM {self.base_view} WHERE {w}"
        elif measure == "aov":
            sql = f"SELECT COALESCE(SUM({self.gmv}),0)/NULLIF(COUNT(*),0) v FROM {self.base_view} WHERE {w}"
        else:
            raise ValueError(measure)
        return float(pd.read_sql(sql, self.engine)["v"].iloc[0] or 0.0)

    def segment_table(self, period, dim, filters=None):
        import pandas as pd
        w = self._where(period, filters)
        sql = (f"SELECT {dim} seg, COUNT(*) c, COALESCE(SUM({self.gmv}),0) s "
               f"FROM {self.base_view} WHERE {w} GROUP BY {dim}")
        df = pd.read_sql(sql, self.engine)
        return {r["seg"]: (float(r["c"]), float(r["s"])) for _, r in df.iterrows()}
