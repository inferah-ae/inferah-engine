"""The YAML packs are REAL, not decorative: they parse into the same objects the
engine walks, both packs run, and the Python constants don't drift from YAML."""
from inferah_engine import investigate, SyntheticSource, GMV_TREE, GMV_MEASURES
from inferah_engine.loader import load_pack
from inferah_engine.synthetic import (make_orders, make_visits,
                                      make_foodtech_deep, make_foodtech_offtree)


def _flatten(node):
    yield (node.id, node.measure, node.relation, tuple(node.segment_dims))
    for c in node.children:
        yield from _flatten(c)


def test_gmv_yaml_matches_python_constants():
    pack = load_pack("trees/gmv_drop.yaml")
    assert list(_flatten(pack.tree)) == list(_flatten(GMV_TREE))
    assert {k: (m.kind, m.column, m.numerator, m.denominator)
            for k, m in pack.measures.items()} == \
           {k: (m.kind, m.column, m.numerator, m.denominator)
            for k, m in GMV_MEASURES.items()}


def test_gmv_pack_runs_from_yaml():
    pack = load_pack("trees/gmv_drop.yaml")
    res = investigate(SyntheticSource(make_orders(), pack.measures), pack.tree)
    assert res.significant and not res.abstained


def test_acquisition_pack_runs_and_finds_rate():
    pack = load_pack("trees/acquisition.yaml")
    res = investigate(SyntheticSource(make_visits(), pack.measures), pack.tree)
    assert res.significant and not res.abstained
    # subscriptions fell via a conversion RATE drop, not localizable to one source
    assert not any(s.winner_share >= 0.55 for s in res.steps if s.kind == "segment")
    mix = next(s for s in res.steps if s.kind == "mix")
    assert mix.winner == "conv_rate"


def test_deep_pack_loads_and_exercises_every_node_type():
    """The deep pack uses all four primitives — factor (LMDI), segment (WHERE),
    signed compose (AOV = Σ ±term), and rate/mix. Loading it proves the loader
    parses summand `sign: -1` and count-distinct measures."""
    pack = load_pack("trees/gmv_foodtech_deep.yaml")
    assert pack.measures["mau"].kind == "count" and pack.measures["mau"].distinct
    assert pack.measures["mau"].extensive is False        # distinct user count is intensive
    assert pack.measures["orders"].extensive is True
    # the discount summand carries a negative sign in the AOV composition
    aov = next(c for c in pack.tree.children if c.id == "aov")
    discount = next(c for c in aov.children if c.id == "discount")
    assert discount.relation == "summand" and discount.sign == -1


def test_deep_pack_reaches_promo_mix_leaf_and_reconciles():
    """GMV falls via a Simpson MIX shift; the engine walks WHERE→FACTOR→COMPOSE→
    FACTOR→RATE/MIX and lands on promo-penetration MIX, reconciling at every step."""
    pack = load_pack("trees/gmv_foodtech_deep.yaml")
    res = investigate(SyntheticSource(make_foodtech_deep(), pack.measures), pack.tree)
    assert res.significant and not res.abstained
    assert res.confidence == "high"
    # localized to Metro on the WHERE axis
    where = next(s for s in res.steps if s.kind == "segment" and s.winner_share >= 0.55)
    assert where.winner == "Metro"
    # AOV carried it (a factor step), the signed compose blamed discount,
    compose = next(s for s in res.steps if s.kind == "compose")
    assert compose.winner == "discount"
    # and the final leaf is the MIX side of promo penetration, not the rate side
    mix = res.steps[-1]
    assert mix.kind == "mix" and mix.winner == "promo_mix"
    assert all(s.recon.ok for s in res.steps)


def test_deep_pack_abstains_on_null_city_rows():
    """The whole drop sits in NULL-city rows the WHERE axis can't see → the city
    split can't reconcile and the engine abstains instead of blaming Metro."""
    pack = load_pack("trees/gmv_foodtech_deep.yaml")
    res = investigate(SyntheticSource(make_foodtech_offtree(), pack.measures), pack.tree)
    assert res.significant and res.abstained
    assert "city" in res.unreconciled
