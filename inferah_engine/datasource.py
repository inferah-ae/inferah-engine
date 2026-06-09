"""
DataSource: the only thing that touches data. The engine asks it for small
AGGREGATES (sums / counts / ratios), never raw rows. Two implementations share
one interface so the engine code is identical for synthetic data and a real DB.

    aggregate(period, measure, filters)        -> float
    segment(period, measure, dim, filters)     -> {seg: float}          (extensive only)
    ratio_segment(period, measure, dim, filters) -> {seg: (den, num)}   (ratio only)

Both are constructed with a MEASURE REGISTRY ({name: Measure}); that registry —
not hardcoded column names — is what every aggregate resolves through.
"""
from __future__ import annotations
import pandas as pd

from .measures import Measure

# period = which window. We keep it abstract: "0" = baseline, "1" = current.
# In a real deployment these map to date ranges.


class SyntheticSource:
    """Backed by an in-memory DataFrame. If the frame carries the base-view
    discipline columns (status / is_test) we apply them up front; otherwise the
    frame is taken as already-clean."""

    def __init__(self, df: pd.DataFrame, measures: dict[str, Measure]):
        if "status" in df.columns and "is_test" in df.columns:
            df = df[(df["status"] == "delivered") & (~df["is_test"])]
        self.df = df.copy()
        self.measures = measures

    def _slice(self, period, filters):
        d = self.df[self.df["period"] == period]
        for col, val in (filters or {}).items():
            d = d[d[col] == val]
        return d

    def aggregate(self, period, measure, filters=None) -> float:
        m = self.measures[measure]
        if m.kind == "ratio":
            num = self.aggregate(period, m.numerator, filters)
            den = self.aggregate(period, m.denominator, filters)
            return float(num / den) if den else 0.0
        d = self._slice(period, filters)
        if m.kind == "count":
            return float(len(d))
        if m.kind == "sum":
            return float(d[m.column].sum())
        raise ValueError(f"unknown measure kind {m.kind!r}")

    def segment(self, period, measure, dim, filters=None) -> dict:
        """{segment: value} for an EXTENSIVE measure (sum/count). NaN keys in
        `dim` are dropped by groupby — so rows with an unmapped dimension value
        silently fall out of the split, which is exactly what the reconciliation
        gate is there to catch."""
        m = self.measures[measure]
        d = self._slice(period, filters)
        if m.kind == "count":
            g = d.groupby(dim).size()
            return {k: float(v) for k, v in g.items()}
        if m.kind == "sum":
            g = d.groupby(dim)[m.column].sum()
            return {k: float(v) for k, v in g.items()}
        raise ValueError(f"segment() needs an extensive measure, got {m.kind!r}")

    def ratio_segment(self, period, measure, dim, filters=None) -> dict:
        """{segment: (denominator, numerator)} for a RATIO measure — enough for
        the rate/mix split. denominator is the share weight; numerator/denominator
        is the per-segment rate."""
        m = self.measures[measure]
        if m.kind != "ratio":
            raise ValueError(f"ratio_segment() needs a ratio measure, got {m.kind!r}")
        num = self.segment(period, m.numerator, dim, filters)
        den = self.segment(period, m.denominator, dim, filters)
        keys = set(num) | set(den)
        return {k: (den.get(k, 0.0), num.get(k, 0.0)) for k in keys}


class PostgresSource:
    """Same interface, backed by a read-only Postgres connection over a pinned
    base view. Needs `pip install sqlalchemy psycopg2-binary` and a real base
    view + read-only role. The engine code does not change.

        src = PostgresSource(
            url="postgresql+psycopg2://readonly:***@localhost:5432/inferah",
            base_view="finance.orders_clean",
            period_col="day",
            periods={"0": ("2026-05-11","2026-05-18"), "1": ("2026-05-18","2026-05-25")},
            measures=GMV_MEASURES,
        )
    """

    def __init__(self, url, base_view, period_col, periods, measures: dict[str, Measure]):
        from sqlalchemy import create_engine
        self.engine = create_engine(url)            # use a READ-ONLY role
        self.base_view = base_view
        self.period_col = period_col
        self.periods = periods                      # {"0": (start,end), "1": (start,end)}
        self.measures = measures

    def _where(self, period, filters):
        s, e = self.periods[period]
        clauses = [f"{self.period_col} >= '{s}'", f"{self.period_col} < '{e}'"]
        for col, val in (filters or {}).items():
            clauses.append(f"{col} = '{val}'")
        return " AND ".join(clauses)

    def _scalar(self, sql) -> float:
        return float(pd.read_sql(sql, self.engine)["v"].iloc[0] or 0.0)

    def aggregate(self, period, measure, filters=None) -> float:
        m = self.measures[measure]
        if m.kind == "ratio":
            num = self.aggregate(period, m.numerator, filters)
            den = self.aggregate(period, m.denominator, filters)
            return float(num / den) if den else 0.0
        w = self._where(period, filters)
        if m.kind == "count":
            return self._scalar(f"SELECT COUNT(*) v FROM {self.base_view} WHERE {w}")
        if m.kind == "sum":
            return self._scalar(
                f"SELECT COALESCE(SUM({m.column}),0) v FROM {self.base_view} WHERE {w}")
        raise ValueError(f"unknown measure kind {m.kind!r}")

    def segment(self, period, measure, dim, filters=None) -> dict:
        m = self.measures[measure]
        w = self._where(period, filters)
        if m.kind == "count":
            expr = "COUNT(*)"
        elif m.kind == "sum":
            expr = f"COALESCE(SUM({m.column}),0)"
        else:
            raise ValueError(f"segment() needs an extensive measure, got {m.kind!r}")
        # NULL dimension values are excluded by GROUP BY here just as NaN keys are
        # dropped in pandas — the reconciliation gate is what flags the resulting gap.
        sql = (f"SELECT {dim} seg, {expr} v FROM {self.base_view} "
               f"WHERE {w} AND {dim} IS NOT NULL GROUP BY {dim}")
        df = pd.read_sql(sql, self.engine)
        return {r["seg"]: float(r["v"]) for _, r in df.iterrows()}

    def ratio_segment(self, period, measure, dim, filters=None) -> dict:
        m = self.measures[measure]
        if m.kind != "ratio":
            raise ValueError(f"ratio_segment() needs a ratio measure, got {m.kind!r}")
        num = self.segment(period, m.numerator, dim, filters)
        den = self.segment(period, m.denominator, dim, filters)
        keys = set(num) | set(den)
        return {k: (den.get(k, 0.0), num.get(k, 0.0)) for k in keys}
