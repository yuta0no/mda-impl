from typing import Any

import numpy as np
import pytest

from mda.bench import (
    BOUNDS,
    LOG_AXES,
    World,
    WorldSpec,
    canonical_worlds,
    from_unit,
    list_worlds,
    to_unit,
)
from mda.metrics import rmsle

LO, HI = BOUNDS[:, 0], BOUNDS[:, 1]


def test_list_worlds():
    # Note: original code of AutoSciLab contains a stale comment on its size
    assert len(list_worlds()) == 57


def test_canonical_worlds():
    assert len(canonical_worlds()) == 9


def test_canonical_is_subset():
    assert set(canonical_worlds()).issubset(set(list_worlds()))


def test_bounds_map_to_unit_corners():
    with np.errstate(all="raise"):  # C_I and C_P have lower bound 0: no log(0)
        assert np.allclose(to_unit(LO), 0.0)
        assert np.allclose(to_unit(HI), 1.0)
    assert np.allclose(from_unit(np.zeros(7)), LO)
    assert np.allclose(from_unit(np.ones(7)), HI)


def test_round_trip():
    X = World(WorldSpec("c0_michaelis_menten")).heldout(seed=0, n=200)
    assert np.allclose(from_unit(to_unit(X)), X)
    U = np.random.default_rng(0).uniform(size=(200, 7))
    assert np.allclose(to_unit(from_unit(U)), U)


def test_midpoint_is_geometric_on_log_axes():
    mid = from_unit(np.full(7, 0.5))
    expected = np.where(LOG_AXES, np.sqrt(LO * HI), (LO + HI) / 2)
    assert np.allclose(mid, expected)


def test_from_unit_clips_to_bounds():
    U = np.random.default_rng(0).uniform(-0.5, 1.5, size=(200, 7))
    X = from_unit(U)
    assert np.all(X >= LO) and np.all(X <= HI)  # exact, not allclose


def test_heldout_is_from_unit_of_uniform_draws():
    # World.heldout draws each axis in turn; from_unit of the same uniform draws must give it.
    # mda.design.random_design is from_unit of uniform draws too, so it samples this distribution.
    n = 100
    rng = np.random.default_rng(1)
    U = np.stack([rng.uniform(size=n) for _ in range(7)], axis=1)
    X = World(WorldSpec("c0_michaelis_menten")).heldout(seed=1, n=n)
    np.testing.assert_allclose(from_unit(U), X, rtol=1e-12)


@pytest.mark.parametrize("shape", [(7,), (5, 7), (3, 5, 7)])
def test_shapes(shape):
    U = np.full(shape, 0.3)
    assert from_unit(U).shape == shape
    assert to_unit(from_unit(U)).shape == shape


@pytest.mark.parametrize("domain", list_worlds())
def test_observe_close_to_truth(domain):
    world = World(WorldSpec(domain))
    X = world.heldout(seed=0, n=20)
    y = np.array([world.observe(x) for x in X])
    assert np.allclose(y, world.truth(X), rtol=0.10)


MISMATCHED = {"c49_sinh_uncompetitive", "c63_sinh_uncompetitive_arrhenius"}


@pytest.mark.parametrize("domain", [d for d in list_worlds() if d not in MISMATCHED])
def test_ground_truth_law_scores_zero(domain):
    world = World(WorldSpec(domain))
    law_str = world.oracle.get_ground_truth_law_str()

    # upstream scorer
    upstream = world.oracle.evaluate_law(law_str=law_str)
    assert upstream["error"] is None
    assert upstream["rmsle"] < 1e-3

    # our truth + held-out set + metric give the same score
    namespace: dict[str, Any] = {"np": np}
    exec(law_str, namespace)
    X = world.heldout(seed=42, n=1000)  # same seed and n as `world.oracle.evaluate_law`
    y_pred = np.array([namespace["discovered_law"](*row) for row in X.tolist()], dtype=float)
    assert rmsle(y_pred=y_pred, y_true=world.truth(X)) == pytest.approx(
        upstream["rmsle"], abs=1e-12
    )
