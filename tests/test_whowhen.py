import pytest

from sentinel_eval import whowhen
from sentinel_eval.attribution import Attribution

pytestmark = pytest.mark.skipif(not (whowhen.DEFAULT_DIR / "Hand-Crafted.parquet").exists(),
                                reason="Who&When data not downloaded (run `sentinel-eval download`)")


def test_loader_shapes():
    hand, algo = whowhen.load("hand"), whowhen.load("algo")
    assert (len(hand), len(algo)) == (58, 126)
    assert all(0 <= t.meta["mistake_step"] < len(t.steps) for t in algo)
    # Hand-Crafted labels are not cleanly indexable: 3/58 are out of range and only ~34/55 in-range steps carry the
    # labelled agent's name. We keep the raw labels (like published baselines) and flag it in the README.
    assert sum(not 0 <= t.meta["mistake_step"] < len(t.steps) for t in hand) <= 3
    assert sum(t.steps[t.meta["mistake_step"]].agent == t.meta["mistake_agent"] for t in algo) >= 120


def test_oracle_scores_100_and_random_is_far_below():
    traces = whowhen.load("algo")
    oracle = whowhen.evaluate(traces, lambda t: Attribution(t.meta["mistake_agent"], t.meta["mistake_step"]))
    assert oracle["agent_acc"] == oracle["step_acc"] == 1.0
    assert whowhen.expected_random(traces)["step_acc"] < 0.3
