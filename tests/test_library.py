import numpy as np
import pytest

from mda.bench import N_TEST, TEST_SEED, World, WorldSpec, canonical_worlds
from mda.fit import fit
from mda.hypothesis import Hypothesis, compile_hypothesis
from mda.library import SEED_BY_WORLD, SEED_LIBRARY
from mda.metrics import rmsle


def true_constants(world: str) -> dict[str, float]:
    """The constants of the oracle `World` builds, which the seed's parameters are named after."""
    p = World(WorldSpec(world)).oracle._law_params
    return {k: float(v) for k, v in p.items()}


def true_theta(world: str, h: Hypothesis) -> np.ndarray:
    c = true_constants(world)
    return np.array([c[p.name] for p in h.params])


def test_one_seed_per_canonical_world():
    assert list(SEED_BY_WORLD) == canonical_worlds()
    assert tuple(SEED_BY_WORLD.values()) == SEED_LIBRARY
    assert len(SEED_LIBRARY) == 9


def test_seeds_compile_and_are_unique():
    for h in SEED_LIBRARY:
        compile_hypothesis(h)
    assert len({h.name for h in SEED_LIBRARY}) == 9
    assert len({h.expr for h in SEED_LIBRARY}) == 9


def test_library_is_read_only():
    with pytest.raises(TypeError):
        SEED_BY_WORLD["extra"] = SEED_LIBRARY[0]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("world", canonical_worlds())
def test_world_constants_within_bounds(world):
    """Checks the D7 bounds against the worlds; the bounds are not tuned to pass it."""
    h = SEED_BY_WORLD[world]
    c = true_constants(world)
    assert set(c) == {p.name for p in h.params}
    for p in h.params:
        assert p.low <= c[p.name] <= p.high, p.name


@pytest.mark.parametrize("world", canonical_worlds())
def test_seed_at_true_constants_matches_world(world):
    h = SEED_BY_WORLD[world]
    w = World(WorldSpec(world))
    X = w.heldout(1, 200)
    np.testing.assert_allclose(
        compile_hypothesis(h)(true_theta(world, h), X), w.truth(X), rtol=1e-10
    )


@pytest.mark.parametrize("world", canonical_worlds())
def test_seed_fits_its_world(world):
    h = SEED_BY_WORLD[world]
    w = World(WorldSpec(world))
    X = w.heldout(0, 25)
    res = fit(h, X, w.truth(X), rng=np.random.default_rng(0), n_starts=8)
    assert res.ok
    X_test = w.heldout(TEST_SEED, N_TEST)
    assert rmsle(compile_hypothesis(h)(res.theta, X_test), w.truth(X_test)) < 0.05
