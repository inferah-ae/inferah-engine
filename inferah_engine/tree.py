"""
Tree = the frozen 'algorithm' the engine follows. In production this is
authored in an external UI and pinned/versioned. Here it's a plain dict so you
can read and edit it. Nodes are DATA, not code.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    label: str
    measure: str                      # which frozen measure: gmv / orders / aov
    relation: str = "root"            # root | multiplicative | rate | mix
    children: list = field(default_factory=list)
    segment_dims: list = field(default_factory=list)   # axis-B dims to test, e.g. ["country","city"]
    threshold_pct: float = 1.0        # don't recurse below this share of the parent change
    note: str = ""                    # human hint, read by the narrator only


# ---- example: GMV drop investigation (food-delivery pack) ----
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
