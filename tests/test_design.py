import inspect
import time

import numpy as np
import pytest

import mda.design
from mda.bench import INPUT_NAMES, World, WorldSpec, from_unit, to_unit
from mda.design import (
    N_EVALS,
    SIGMA0,
    DesignError,
    between_model_variance,
    cma_design,
    random_design,
)
from mda.hypothesis import compile_hypothesis
from mda.library import SEED_BY_WORLD, SEED_LIBRARY

CA = INPUT_NAMES.index("C_A")
N_INPUTS = len(INPUT_NAMES)
CENTRE = np.full((1, N_INPUTS), 0.5)


def const(c: float):
    return lambda X: np.full(X.shape[0], c)


def c_a(X):
    return X[:, CA].copy()


def c_a_curved(X):
    # Agrees with c_a at low C_A and departs from it as C_A grows: the gap is C_A^2 / 50
    return X[:, CA] * (1 + X[:, CA] / 50)


class Recorder:
    """A predictor that records every condition it is asked about."""

    def __init__(self, f):
        self.f = f
        self.seen: list[np.ndarray] = []

    def __call__(self, X):
        self.seen.append(X.copy())
        return self.f(X)

    @property
    def X(self) -> np.ndarray:
        return np.concatenate(self.seen)


def unit_grid(n: int = 50, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(n, N_INPUTS))


def mm_and_competitive():
    """Two laws that both explain D0 of `c1_competitive_inhibition` and differ once C_I > 0.

    Michaelis-Menten and competitive inhibition with the world's own constants: D0 holds C_I at
    0, where the two are the same function.
    """
    world = World(WorldSpec("c1_competitive_inhibition"))
    k = world.oracle._law_params
    mm, ci = SEED_BY_WORLD["c0_michaelis_menten"], SEED_BY_WORLD["c1_competitive_inhibition"]
    f_mm, f_ci = compile_hypothesis(mm), compile_hypothesis(ci)
    theta_mm = np.array([k[p.name] for p in mm.params])
    theta_ci = np.array([k[p.name] for p in ci.params])
    d0 = np.array([[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)])
    np.testing.assert_allclose(f_mm(theta_mm, d0), f_ci(theta_ci, d0), rtol=1e-12)
    return [lambda X: f_mm(theta_mm, X), lambda X: f_ci(theta_ci, X)]


def test_defaults_follow_spec():
    # Spec §6 and D10; change the spec in the same PR as these
    assert (N_EVALS, SIGMA0) == (48, 0.3)


# between_model_variance


def test_variance_matches_hand_computation():
    # mean 0.25 * 0 + 0.75 * 4 = 3, so V = 0.25 * 9 + 0.75 * 1 = 3
    v = between_model_variance(CENTRE, np.array([0.25, 0.75]), [const(0.0), const(4.0)])
    np.testing.assert_allclose(v, [3.0], rtol=1e-14)


def test_variance_in_log1p_space():
    # log1p(0) = 0 and log1p(e^2 - 1) = 2: mean 1, V = 0.5 * 1 + 0.5 * 1 = 1
    w = np.array([0.5, 0.5])
    v = between_model_variance(CENTRE, w, [const(0.0), const(np.e**2 - 1)], space="log1p")
    np.testing.assert_allclose(v, [1.0], rtol=1e-14)


def test_variance_evaluates_predictors_in_natural_units():
    rec = Recorder(c_a)
    U = unit_grid(5)
    between_model_variance(U, np.array([0.5, 0.5]), [rec, const(1.0)])
    np.testing.assert_allclose(rec.X, from_unit(U))


def test_variance_is_zero_when_models_agree():
    w = np.array([0.2, 0.3, 0.5])
    v = between_model_variance(unit_grid(), w, [c_a, c_a, c_a])
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_variance_is_zero_with_one_model_of_weight_one():
    w = np.array([0.0, 1.0, 0.0])
    v = between_model_variance(unit_grid(), w, [const(5.0), c_a, c_a_curved])
    np.testing.assert_array_equal(v, 0.0)


def test_variance_is_non_negative():
    rng = np.random.default_rng(1)
    hs = [
        SEED_BY_WORLD[d]
        for d in ("c0_michaelis_menten", "c8_hill_cooperativity", "c3_arrhenius_temperature")
    ]
    predictors = []
    for h in hs:
        f = compile_hypothesis(h)
        theta = np.array([np.sqrt(p.low * p.high) for p in h.params])
        predictors.append(lambda X, f=f, theta=theta: f(theta, X))
    for _ in range(5):
        v = between_model_variance(
            unit_grid(200, int(rng.integers(1000))), rng.dirichlet([1, 1, 1]), predictors
        )
        assert np.all(np.isfinite(v)) and np.all(v >= 0) and np.any(v > 0)


def test_variance_with_no_weight_is_zero():
    v = between_model_variance(unit_grid(), np.zeros(2), [const(np.nan), const(1.0)])
    np.testing.assert_array_equal(v, 0.0)


@pytest.mark.filterwarnings("error")
def test_zero_weight_models_are_skipped():
    # 0 * nan would otherwise make every V NaN
    broken = Recorder(const(np.nan))
    v = between_model_variance(CENTRE, np.array([0.5, 0.5, 0.0]), [const(0.0), const(2.0), broken])
    np.testing.assert_allclose(v, [1.0])
    assert broken.seen == []


@pytest.mark.filterwarnings("error")
def test_non_finite_predictions_give_nan_where_they_occur():
    def half_broken(X):
        return np.where(X[:, CA] > 1.0, np.inf, X[:, CA])

    U = unit_grid(100)
    v = between_model_variance(U, np.array([0.5, 0.5]), [half_broken, const(0.0)])
    broken = from_unit(U)[:, CA] > 1.0
    assert broken.any() and (~broken).any()
    assert np.all(np.isnan(v[broken]))
    assert np.all(np.isfinite(v[~broken]))


@pytest.mark.filterwarnings("error")
def test_log1p_of_rates_at_or_below_minus_one_gives_nan():
    v = between_model_variance(
        CENTRE, np.array([0.5, 0.5]), [const(-1.0), const(0.0)], space="log1p"
    )
    assert np.isnan(v[0])


@pytest.mark.filterwarnings("error")
def test_overflowing_variance_is_nan():
    # Both predictions are finite, but their squared gap is not: V is undefined, not infinite
    v = between_model_variance(CENTRE, np.array([0.5, 0.5]), [const(0.0), const(1e200)])
    assert np.isnan(v[0])


def test_variance_rejects_unknown_space():
    # Under pytest the runtime type check rejects the argument first. A real run has no such
    # check, so call the undecorated function.
    f = inspect.unwrap(between_model_variance)
    rec = Recorder(c_a)
    with pytest.raises(ValueError, match="unknown space 'log'"):
        f(CENTRE, np.array([0.5, 0.5]), [rec, c_a], space="log")
    assert rec.seen == []  # rejected before any model is evaluated


def test_variance_rejects_bad_weights():
    with pytest.raises(ValueError, match="2 weights for 3 predictors"):
        between_model_variance(CENTRE, np.array([0.5, 0.5]), [c_a, c_a, c_a])
    with pytest.raises(ValueError, match="non-negative"):
        between_model_variance(CENTRE, np.array([1.5, -0.5]), [c_a, c_a])
    with pytest.raises(ValueError, match="finite"):
        between_model_variance(CENTRE, np.array([np.nan, 1.0]), [c_a, c_a])


# cma_design


@pytest.mark.parametrize("n_evals", [N_EVALS, 9, 10, 1])
def test_cma_design_uses_exactly_n_evals(n_evals):
    rec = Recorder(c_a_curved)
    x = cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(0), n_evals=n_evals)
    assert rec.X.shape == (n_evals, N_INPUTS)
    assert x.shape == (N_INPUTS,)


def test_cma_design_uses_48_evals_by_default():
    rec = Recorder(c_a_curved)
    cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(0))
    assert rec.X.shape[0] == 48


@pytest.mark.filterwarnings("error")
def test_cma_design_with_flat_variance_still_uses_n_evals():
    # pycma would stop on flat fitness after one generation
    rec = Recorder(c_a)
    x = cma_design(np.array([1.0]), [rec], rng=np.random.default_rng(0))
    assert rec.X.shape[0] == N_EVALS
    np.testing.assert_array_equal(x, rec.X[0])  # ties go to the earliest point evaluated


def test_cma_proposals_stay_in_the_unit_box(monkeypatch):
    # cma_design clips pycma's proposals and from_unit clips again, so the evaluated conditions
    # cannot show whether pycma kept to the bounds. Without them, the clip would pile the
    # proposals that left the box onto its faces.
    asked: list[np.ndarray] = []
    ask = mda.design.cma.CMAEvolutionStrategy.ask

    def spy(self, *args, **kwargs):
        out = ask(self, *args, **kwargs)
        asked.extend(out)
        return out

    monkeypatch.setattr(mda.design.cma.CMAEvolutionStrategy, "ask", spy)
    rec = Recorder(c_a_curved)
    cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(0))
    U = np.array(asked)
    assert U.shape == (54, N_INPUTS)  # 6 generations of 9; the last 6 points are not evaluated
    assert np.all((U >= 0) & (U <= 1))
    np.testing.assert_array_equal(rec.X, from_unit(U[:N_EVALS]))


def test_cma_design_returns_best_point_evaluated():
    rec = Recorder(c_a_curved)
    w = np.array([0.3, 0.7])
    x = cma_design(w, [c_a, rec], rng=np.random.default_rng(4))
    v = between_model_variance(to_unit(rec.X), w, [c_a, c_a_curved])
    np.testing.assert_allclose(x, rec.X[np.argmax(v)])


def test_cma_design_moves_to_where_models_disagree():
    # The two models agree at low C_A and part as C_A grows, so the design goes to high C_A.
    # This only checks the direction: 48 random points would nearly always pass it too.
    # test_cma_design_beats_random_search checks the search itself.
    w = np.array([0.5, 0.5])
    for seed in range(3):
        x = cma_design(w, [c_a, c_a_curved], rng=np.random.default_rng(seed))
        assert to_unit(x)[CA] > 0.9


def test_cma_design_beats_random_search():
    # V of the design against the best V of 48 uniform points, the random-design control with
    # the same budget. The disagreement peaks in a corner of the box (Enz at its maximum), far
    # from where CMA-ES starts, so the search has to adapt to get there. Measured over these 10
    # seeds: 10 wins and a median ratio of 4.4. A search that never adapts wins 5 with a median
    # of 1.8, and one that minimises V wins 1 with a median of 0.15.
    w = np.array([0.5, 0.5])
    predictors = mm_and_competitive()
    ratios = []
    for seed in range(10):
        x = cma_design(w, predictors, rng=np.random.default_rng(seed))
        v_cma = between_model_variance(to_unit(x)[None], w, predictors)[0]
        U = np.random.default_rng([seed, 1]).uniform(size=(N_EVALS, N_INPUTS))
        ratios.append(v_cma / np.max(between_model_variance(U, w, predictors)))
    assert np.sum(np.array(ratios) > 1) >= 8
    assert np.median(ratios) > 2.5


@pytest.mark.filterwarnings("error")
def test_cma_design_never_chooses_a_nan_point():
    # Disagreement grows with C_A, but one model is non-finite above C_A = 1 (unit 0.5)
    def half_broken(X):
        return np.where(X[:, CA] > 1.0, np.nan, c_a_curved(X))

    rec = Recorder(half_broken)
    x = cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(0))
    assert (rec.X[:, CA] > 1.0).any()  # it did try the broken region
    assert x[CA] <= 1.0


@pytest.mark.filterwarnings("error")
def test_cma_search_retreats_from_nan_points():
    # The same setup. pycma is told that a NaN point is the worst, so the search stays mostly
    # on the finite side although V grows towards the broken region. After the first two
    # generations, 13% to 43% of the points are NaN over these seeds. If NaN ranked as the
    # best, 77% to 97% would be.
    def half_broken(X):
        return np.where(X[:, CA] > 1.0, np.nan, c_a_curved(X))

    for seed in range(10):
        rec = Recorder(half_broken)
        cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(seed))
        assert np.mean(rec.X[18:, CA] > 1.0) < 0.6


@pytest.mark.filterwarnings("error")
def test_cma_design_raises_when_no_point_is_defined():
    # Returning the first point evaluated would break "never chooses a NaN point"
    rec = Recorder(const(np.nan))
    with pytest.raises(DesignError, match="undefined at all 48 points"):
        cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(0))
    assert rec.X.shape[0] == N_EVALS  # it used the whole budget before giving up


@pytest.mark.filterwarnings("error")
def test_cma_design_uses_a_single_defined_point():
    # V is defined only in the first generation: later, all-NaN generations must not replace it
    calls = 0

    def first_call_only(X):
        nonlocal calls
        calls += 1
        return c_a_curved(X) if calls == 1 else np.full(X.shape[0], np.nan)

    rec = Recorder(c_a)
    w = np.array([0.5, 0.5])
    x = cma_design(w, [rec, first_call_only], rng=np.random.default_rng(0))
    first = rec.seen[0]
    v = between_model_variance(to_unit(first), w, [c_a, c_a_curved])
    np.testing.assert_array_equal(x, first[np.argmax(v)])


@pytest.mark.filterwarnings("error")
def test_cma_design_never_chooses_an_overflowing_point():
    # Above C_A = 1 the gap between the models is finite but its square overflows
    def huge_above_one(X):
        return np.where(X[:, CA] > 1.0, 1e200, c_a_curved(X))

    rec = Recorder(huge_above_one)
    x = cma_design(np.array([0.5, 0.5]), [c_a, rec], rng=np.random.default_rng(0))
    assert (rec.X[:, CA] > 1.0).any()
    assert x[CA] <= 1.0


def test_cma_design_is_deterministic_for_a_seed():
    w = np.array([0.5, 0.5])
    a = cma_design(w, [c_a, c_a_curved], rng=np.random.default_rng(7))
    b = cma_design(w, [c_a, c_a_curved], rng=np.random.default_rng(7))
    c = cma_design(w, [c_a, c_a_curved], rng=np.random.default_rng(8))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cma_design_ignores_and_keeps_the_global_random_state():
    # pycma samples from and seeds the legacy global state unless told otherwise
    w = np.array([0.5, 0.5])
    legacy = np.random  # the legacy API is the point of this test  # noqa: NPY002
    legacy.seed(1)  # noqa: NPY002
    before = legacy.get_state()  # noqa: NPY002
    a = cma_design(w, [c_a, c_a_curved], rng=np.random.default_rng(7))
    after = legacy.get_state()  # noqa: NPY002
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2] == before[2]
    legacy.seed(2)  # noqa: NPY002
    b = cma_design(w, [c_a, c_a_curved], rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_cma_design_rejects_non_positive_n_evals():
    with pytest.raises(ValueError, match="n_evals"):
        cma_design(np.array([1.0]), [c_a], rng=np.random.default_rng(0), n_evals=0)


def test_cma_design_is_fast_for_a_full_pool():
    # Done-when of feat/design: under a second for a 24-model pool (N_MAX in mda.posterior)
    rng = np.random.default_rng(0)
    predictors = []
    for i in range(24):
        h = SEED_LIBRARY[i % len(SEED_LIBRARY)]
        f = compile_hypothesis(h)
        theta = np.array([rng.uniform(p.low, p.high) for p in h.params])
        predictors.append(lambda X, f=f, theta=theta: f(theta, X))
    w = rng.dirichlet(np.ones(24))
    t0 = time.perf_counter()
    for _ in range(5):
        cma_design(w, predictors, rng=rng)
    assert (time.perf_counter() - t0) / 5 < 1.0


def test_cma_design_separates_laws_that_agree_on_the_passive_data():
    # Both rates are proportional to Enz, so V grows as Enz^2 and the design takes Enz to its
    # maximum. The laws only differ when C_I > 0, so it also leaves C_I = 0, where D0 sits.
    w = np.array([0.5, 0.5])
    predictors = mm_and_competitive()
    for seed in range(5):
        x = cma_design(w, predictors, rng=np.random.default_rng(seed))
        u = to_unit(x)
        assert u[INPUT_NAMES.index("Enz")] > 0.9
        assert between_model_variance(u[None], w, predictors)[0] > 0


# random_design


def test_random_design_is_uniform_in_the_unit_box():
    # from_unit of uniform draws, as World.heldout is (test_bench.py), so the two share a distribution
    for seed in range(5):
        x = random_design(np.random.default_rng(seed))
        assert x.shape == (N_INPUTS,)
        np.testing.assert_allclose(to_unit(x), np.random.default_rng(seed).uniform(size=N_INPUTS))
