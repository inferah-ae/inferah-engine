"""End-to-end engine behaviour on the synthetic scenarios: it finds the cause
when the tree can explain it, ABSTAINS when it can't, and is deterministic."""
from inferah_engine import investigate, narrate, SyntheticSource, GMV_TREE, GMV_MEASURES
from inferah_engine.synthetic import make_orders, make_orders_unmapped


def _gmv_src(df):
    return SyntheticSource(df, GMV_MEASURES)


def test_mix_shift_is_localized_and_explained():
    res = investigate(_gmv_src(make_orders()), GMV_TREE)
    assert res.significant and not res.abstained
    # localized to the flooded region
    where = next(s for s in res.steps if s.kind == "segment")
    assert where.winner == "Westland" and where.winner_share > 0.9
    # and attributed to MIX, not a price cut
    mix = next(s for s in res.steps if s.kind == "mix")
    assert mix.winner == "aov_mix"
    assert "mix" in res.leaf.lower()


def test_abstains_when_driver_is_off_tree():
    res = investigate(_gmv_src(make_orders_unmapped()), GMV_TREE)
    assert res.significant
    assert res.abstained
    assert res.unreconciled                      # carries the unexplained gap
    assert "Refusing to guess" in narrate(res)
    # it must NOT have invented a country winner
    assert not any(s.winner_share >= 0.55 for s in res.steps if s.kind == "segment")


def test_every_step_reconciles_on_clean_data():
    res = investigate(_gmv_src(make_orders()), GMV_TREE)
    for s in res.steps:
        assert s.recon.ok, f"step {s.kind} failed to reconcile: {s.recon.residual_pct}%"


def test_deterministic():
    a = investigate(_gmv_src(make_orders()), GMV_TREE)
    b = investigate(_gmv_src(make_orders()), GMV_TREE)
    assert narrate(a) == narrate(b)
    assert [(s.kind, s.winner) for s in a.steps] == [(s.kind, s.winner) for s in b.steps]
