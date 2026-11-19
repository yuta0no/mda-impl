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


@pytest.mark.parametrize("shape", [(7,), (5, 7), (3, 5, 7)])
def test_shapes(shape):
    U = np.full(shape, 0.3)
    assert from_unit(U).shape == shape
    assert to_unit(from_unit(U)).shape == shape
