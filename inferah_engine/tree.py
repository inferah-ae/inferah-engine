"""
Tree = the frozen 'algorithm' the engine follows. In production this is
authored in an external UI and pinned/versioned. Here it's plain data so you
can read and edit it. Nodes are DATA, not code.

A pack is a (tree, measures) pair: the tree names measures, the registry says
how to compute them. The Python constants below are the same content as
trees/gmv_drop.yaml — a test asserts they don't drift apart.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .measures import Measure


@dataclass
class Node:
    id: str
    label: str
    measure: str                      # a name in the pack's measure registry
    relation: str = "root"            # root | multiplicative | summand | rate | mix
    children: list = field(default_factory=list)
    segment_dims: list = field(default_factory=list)   # axis-B dims to test, e.g. ["country"]
    sign: int = 1                     # for summand children: +1 (adds) or -1 (subtracts)
    note: str = ""                    # human hint, read by the narrator only


# ---- example pack: GMV drop investigation (food-delivery flavour) ----
GMV_MEASURES = {
    "gmv": Measure("gmv", "sum", column="gmv_usd"),
    "orders": Measure("orders", "count"),
    "aov": Measure("aov", "ratio", numerator="gmv", denominator="orders"),
}

GMV_TREE = Node(
    id="gmv",
    label="GMV",
    measure="gmv",
    relation="root",
    segment_dims=["country"],         # where did it move (axis B)
    note="Top-down GMV decomposition. GMV = orders x AOV.",
    children=[
        Node(id="orders", label="Orders", measure="orders", relation="multiplicative",
             segment_dims=["country"],
             note="Demand / acquisition / retention side."),
        Node(id="aov", label="AOV", measure="aov", relation="multiplicative",
             segment_dims=["country"],
             note="Basket size. Split rate vs mix below.",
             children=[
                 Node(id="aov_rate", label="AOV · rate", measure="aov", relation="rate",
                      segment_dims=["order_type"],
                      note="Per-segment basket actually changed (pricing/upsell)."),
                 Node(id="aov_mix", label="AOV · mix", measure="aov", relation="mix",
                      segment_dims=["order_type"],
                      note="Order mix shifted toward cheaper segments (promo, market shift)."),
             ]),
    ],
)
