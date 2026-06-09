"""
Measure registry: the vocabulary a tree is allowed to ask the data for.

A tree references measures by NAME (e.g. "gmv", "orders", "aov"). The registry
maps each name to a small spec the DataSource knows how to evaluate:

    sum     extensive, additive   -> SUM(column)            e.g. gmv
    count   extensive, additive   -> COUNT(*)               e.g. orders
    ratio   intensive, NOT additive -> numerator / denominator   e.g. aov = gmv / orders

"extensive" measures (sum, count) add up across segments, so they support the
additive segment split. "intensive" measures (ratio) do not — averaging an
average is the Simpson trap — so they are decomposed via rate/mix instead.

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
    column: str | None = None        # for sum: the column to SUM
    numerator: str | None = None     # for ratio: name of another measure
    denominator: str | None = None   # for ratio: name of another measure

    @property
    def extensive(self) -> bool:
        """Additive across segments (sum/count). Ratios are intensive."""
        return self.kind in ("sum", "count")

    def __post_init__(self):
        if self.kind == "sum" and not self.column:
            raise ValueError(f"sum measure {self.name!r} needs a column")
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
            numerator=spec.get("numerator"),
            denominator=spec.get("denominator"),
        )
    return reg
