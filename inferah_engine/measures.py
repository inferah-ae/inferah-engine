"""
Measure registry: the vocabulary a tree is allowed to ask the data for.

A tree references measures by NAME (e.g. "gmv", "orders", "aov"). The registry
maps each name to a small spec the DataSource knows how to evaluate:

    sum            extensive, additive   -> SUM(column)                  e.g. gmv
    count          extensive, additive   -> COUNT(*)                     e.g. orders
    count+distinct intensive             -> COUNT(DISTINCT column)       e.g. mau
    ratio          intensive, NOT additive -> numerator / denominator    e.g. aov = gmv / orders

"extensive" measures (plain sum/count) add up across a segment dimension, so they
support the additive *segment* split. "intensive" measures do not — averaging an
average, or counting distinct users per city and summing, is the Simpson trap —
so they are decomposed via rate/mix or factor/composition instead.

A COUNT DISTINCT (e.g. monthly active users) is a count, but it is NOT additive
across an arbitrary dimension (a user active in two cities is one MAU, not two),
so it is treated as intensive: never dim-split, only used as a factor or composed
from a partition via disjoint helper columns.

The registry is what makes the engine generic: there are no measure names baked
into engine.py. Swap the registry + tree and the same walk investigates a
different metric.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Measure:
    name: str
    kind: str                 # "sum" | "count" | "ratio"
    column: str | None = None        # for sum: column to SUM; for count: column to COUNT DISTINCT
    distinct: bool = False           # for count: COUNT(DISTINCT column) instead of COUNT(*)
    numerator: str | None = None     # for ratio: name of another measure
    denominator: str | None = None   # for ratio: name of another measure

    @property
    def extensive(self) -> bool:
        """Additive across a segment dimension. Plain sum/count are; a COUNT
        DISTINCT is not (users overlap across dimensions); ratios are not."""
        return self.kind == "sum" or (self.kind == "count" and not self.distinct)

    def __post_init__(self):
        if self.kind == "sum" and not self.column:
            raise ValueError(f"sum measure {self.name!r} needs a column")
        if self.kind == "count" and self.distinct and not self.column:
            raise ValueError(f"distinct-count measure {self.name!r} needs a column")
        if self.kind == "ratio" and not (self.numerator and self.denominator):
            raise ValueError(f"ratio measure {self.name!r} needs numerator and denominator")
        if self.kind not in ("sum", "count", "ratio"):
            raise ValueError(f"measure {self.name!r}: unknown kind {self.kind!r}")


def registry_from_dict(d: dict) -> dict[str, Measure]:
    """Build {name: Measure} from a plain dict (e.g. parsed YAML).

        measures:
          gmv:    {kind: sum, column: gmv_usd}
          orders: {kind: count}
          aov:    {kind: ratio, numerator: gmv, denominator: orders}
    """
    reg: dict[str, Measure] = {}
    for name, spec in d.items():
        reg[name] = Measure(
            name=name,
            kind=spec["kind"],
            column=spec.get("column"),
            distinct=bool(spec.get("distinct", False)),
            numerator=spec.get("numerator"),
            denominator=spec.get("denominator"),
        )
    return reg
