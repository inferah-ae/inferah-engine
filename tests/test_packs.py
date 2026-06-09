"""The YAML packs are REAL, not decorative: they parse into the same objects the
engine walks, both packs run, and the Python constants don't drift from YAML."""
from inferah_engine import investigate, SyntheticSource, GMV_TREE, GMV_MEASURES
from inferah_engine.loader import load_pack
from inferah_engine.synthetic import make_orders, make_visits


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
