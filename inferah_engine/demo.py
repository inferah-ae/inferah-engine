"""
Zero-setup demo: run the engine on the bundled synthetic scenarios.

    python -m inferah_engine.demo

Shows the three things worth showing — a Simpson MIX shift the engine
decomposes correctly, a different metric (acquisition) on the SAME walk, and a
case where the driver is off-tree and the engine honestly ABSTAINS.
"""
from . import investigate, render, narrate, SyntheticSource, GMV_TREE, GMV_MEASURES
from .loader import load_pack
from .synthetic import (make_orders, make_visits, make_orders_unmapped,
                        make_foodtech_deep, make_foodtech_offtree)


def _section(title, res, label):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(render(res, metric_label=label))
    print("\n> " + narrate(res, metric_label=label) + "\n")


def main():
    _section("GMV pack — a MIX shift (Simpson trap)",
             investigate(SyntheticSource(make_orders(), GMV_MEASURES), GMV_TREE),
             "GMV")

    acq = load_pack("trees/acquisition.yaml")
    _section("Acquisition pack — same walk, different metric (a RATE drop)",
             investigate(SyntheticSource(make_visits(), acq.measures), acq.tree),
             "Subscriptions")

    _section("GMV pack — driver is off-tree: the engine ABSTAINS",
             investigate(SyntheticSource(make_orders_unmapped(), GMV_MEASURES), GMV_TREE),
             "GMV")

    deep = load_pack("trees/gmv_foodtech_deep.yaml")
    _section("DEEP pack — full GMV = MAU × OrderConv × Frequency × AOV tree, "
             "Simpson MIX to the promo-penetration leaf",
             investigate(SyntheticSource(make_foodtech_deep(), deep.measures), deep.tree),
             "GMV")

    _section("DEEP pack — drop hides in NULL-city rows: the engine ABSTAINS",
             investigate(SyntheticSource(make_foodtech_offtree(), deep.measures), deep.tree),
             "GMV")


if __name__ == "__main__":
    main()
