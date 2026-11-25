import subprocess
import sys
import time

import numpy as np
import pytest

from mda.bench import INPUT_NAMES, from_unit, to_unit
from mda.fit import (
    REL_FLOOR,
    RHO,
    SIGMA_FLOOR,
    FitResult,
    fit,
    relative_residuals,
    residual_input_spearman,
    residuals,
    sigma,
)
from mda.hypothesis import Hypothesis, Param, compile_hypothesis

MM = Hypothesis(
    "michaelis_menten",
    "Vmax * C_A / (Km + C_A)",
    (Param("Vmax", 1e-3, 1e3), Param("Km", 1e-3, 1e3)),
)
HILL = Hypothesis(
    "hill",
    "Vmax * C_A**n / (K**n + C_A**n)",
    (Param("Vmax", 1e-3, 1e3), Param("K", 1e-3, 1e3), Param("n", 0.1, 8.0)),
)
MM_TRUE = np.array([2.5, 0.7])
HILL_TRUE = np.array([4.0, 1.5, 2.3])
CA, CI = INPUT_NAMES.index("C_A"), INPUT_NAMES.index("C_I")

# Passive data D0 (spec §1): C_A in {0.03, 0.3, 3}, other inputs at (0, 1, 0, 1, 310, 7)
D0 = np.array([[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)])


def conditions(n: int, seed: int = 0) -> np.ndarray:
    return from_unit(np.random.default_rng(seed).uniform(size=(n, len(INPUT_NAMES))))


def data(h: Hypothesis, theta: np.ndarray, n: int = 25) -> tuple[np.ndarray, np.ndarray]:
    X = conditions(n)
    return X, compile_hypothesis(h)(theta, X)


def run(h: Hypothesis, X, y, seed: int = 0, n_starts: int = 8, **kwargs) -> FitResult:
    return fit(h, X, y, rng=np.random.default_rng(seed), n_starts=n_starts, **kwargs)


def in_bounds(h: Hypothesis, theta: np.ndarray) -> bool:
    return all(p.low <= t <= p.high for p, t in zip(h.params, theta, strict=True))


def test_sigma_formula():
    # Non-default rho and floor, so a constant hard-coded in the formula would fail
    y = np.array([-10.0, -1.0, -0.2, 0.0, 0.2, 1.0, 10.0])
    expected = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(sigma(y, rho=0.1, sigma_floor=0.5), expected)


def test_sigma_defaults_follow_spec():
    # Spec §3: rho = 0.03, sigma_floor = 0.01; change the spec in the same PR as these
    assert (RHO, SIGMA_FLOOR, REL_FLOOR) == (0.03, 0.01, 0.01)
    y = np.array([-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0])
    expected = np.array([0.03, 0.015, 0.01, 0.01, 0.01, 0.015, 0.03])
    np.testing.assert_allclose(sigma(y), expected)


def test_residuals():
    y, y_hat = np.array([0.0, 1.0, 10.0]), np.array([0.01, 0.97, 10.3])
    np.testing.assert_allclose(residuals(y, y_hat), [-1.0, 1.0, -1.0])
    np.testing.assert_allclose(residuals(y, y_hat, "log1p"), np.log1p(y) - np.log1p(y_hat))
    assert np.isnan(residuals(np.array([1.0]), np.array([-2.0]), "log1p")[0])


@pytest.mark.parametrize(("h", "truth"), [(MM, MM_TRUE), (HILL, HILL_TRUE)])
@pytest.mark.parametrize("residual", ["whitened", "log1p"])
def test_recovers_noiseless_constants(h, truth, residual):
    X, y = data(h, truth)
    res = run(h, X, y, residual=residual)
    assert res.ok
    np.testing.assert_allclose(res.theta, truth, rtol=1e-4)
    assert res.rss < 1e-8
    np.testing.assert_allclose(res.y_hat, y, rtol=1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_stays_in_bounds(seed):
    # The data want Vmax = 50, beyond the upper bound, so the optimum sits on the bound
    h = Hypothesis("mm", MM.expr, (Param("Vmax", 1e-3, 10.0), Param("Km", 1e-3, 1e3)))
    X, y = data(MM, np.array([50.0, 0.7]))
    res = run(h, X, y, seed=seed)
    assert res.ok
    assert in_bounds(h, res.theta)
    assert res.theta[0] == pytest.approx(10.0)


def test_never_leaves_bounds_on_noisy_data():
    X, y = data(HILL, HILL_TRUE)
    y = y * np.exp(np.random.default_rng(1).normal(0.0, 0.5, size=y.shape))
    for seed in range(5):
        res = run(HILL, X, y, seed=seed)
        assert res.ok
        assert in_bounds(HILL, res.theta)


@pytest.mark.parametrize(
    "expr",
    [
        "k * log(C_I - 100)",  # nan everywhere: C_I is within [0, 50], so the log's argument < 0
        "k / C_I",  # inf at C_I = 0
        "k * exp(1000 * C_A)",  # overflows
    ],
)
def test_non_finite_predictions_give_not_ok(expr):
    h = Hypothesis("bad", expr, (Param("k", 0.1, 10.0),))
    X = conditions(10)
    X[:, CI] = 0.0
    res = run(h, X, np.ones(10))
    assert not res.ok
    assert res.rss == np.inf
    assert in_bounds(h, res.theta)


def test_negative_prediction_with_log1p_is_not_ok():
    h = Hypothesis("neg", "-k", (Param("k", 2.0, 10.0),))
    res = run(h, D0, np.ones(3), residual="log1p")
    assert not res.ok
    assert res.rss == np.inf


def test_partly_non_finite_region_is_avoided():
    # log(k - C_A) is nan wherever k < C_A, so starts there are non-finite but the optimum is not
    h = Hypothesis("partial", "log(k - C_A)", (Param("k", 0.0, 200.0),))
    X = conditions(15)
    X[:, CA] = np.linspace(0.1, 5.0, 15)
    y = np.log(20.0 - X[:, CA])
    res = run(h, X, y)
    assert res.ok
    assert res.theta[0] == pytest.approx(20.0, rel=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_huge_finite_residuals_keep_their_gradient(seed):
    # Residuals up to 1e12: a low penalty cap would flatten them and stop the fit at its start
    h = Hypothesis("huge", "1e10 * k", (Param("k", 0.0, 1.0),))
    res = run(h, D0, np.zeros(3), seed=seed, n_starts=1)
    assert res.theta[0] == pytest.approx(0.0, abs=1e-12)


def test_arrhenius_single_starts_mostly_converge():
    h = Hypothesis(
        "arrhenius",
        "A * exp(-Ea / (8.314 * T)) * C_A / (Km + C_A)",
        (Param("A", 1e-3, 1e30), Param("Ea", 1e3, 2e5), Param("Km", 1e-3, 1e3)),
    )
    X, y = data(h, np.array([1e9, 5e4, 0.7]))
    rss = np.array([run(h, X, y, seed=s, n_starts=1).rss for s in range(20)])
    assert np.mean(rss < 1e-6) >= 0.75  # about 0.5 with a penalty cap of 1e8


def test_random_starts_rescue_a_stuck_warm_start():
    # Every residual is nan at the warm start, where k < C_A everywhere, so that start cannot move
    h = Hypothesis("partial", "log(k - C_A)", (Param("k", 0.0, 200.0),))
    X = conditions(15)
    X[:, CA] = np.linspace(0.1, 5.0, 15)
    y = np.log(20.0 - X[:, CA])
    warm = np.array([0.05])
    assert not run(h, X, y, n_starts=0, warm_start=warm).ok
    res = run(h, X, y, warm_start=warm)
    assert res.ok
    assert res.theta[0] == pytest.approx(20.0, rel=1e-5)


def test_ties_go_to_the_warm_start():
    # No start gives finite residuals, so all tie at rss = inf and the first start is kept
    h = Hypothesis("bad", "k * log(C_I - 100)", (Param("k", 0.1, 10.0),))
    res = run(h, conditions(10), np.ones(10), warm_start=np.array([5.0]))
    assert not res.ok
    assert res.theta[0] == pytest.approx(5.0)


def test_deterministic_for_a_given_rng():
    X, y = data(HILL, HILL_TRUE)
    y = y * np.exp(np.random.default_rng(1).normal(0.0, 0.3, size=y.shape))
    a, b = run(HILL, X, y, seed=7), run(HILL, X, y, seed=7)
    np.testing.assert_array_equal(a.theta, b.theta)
    assert a.rss == b.rss


def test_warm_start_is_the_initial_point():
    X, y = data(HILL, HILL_TRUE)
    warm = np.array([3.0, 2.0, 1.7])
    f = compile_hypothesis(HILL)
    evaluated = []

    def recording(theta, X):
        evaluated.append(np.array(theta))
        return f(theta, X)

    run(HILL, X, y, n_starts=0, warm_start=warm, f=recording)
    np.testing.assert_allclose(evaluated[0], warm, rtol=1e-12)


# j does not affect the predictions, so the optimiser never moves it: the fitted j shows which
# start produced the result
FLAT = Hypothesis("flat", "k * C_A + 0 * j", (Param("k", 0.1, 10.0), Param("j", 0.0, 100.0)))


def test_warm_start_determines_the_result():
    X = conditions(10)
    y = 2.0 * X[:, CA]
    warm = run(FLAT, X, y, n_starts=0, warm_start=np.array([1.0, 37.0]))
    assert warm.theta[1] == pytest.approx(37.0)
    assert warm.theta[0] == pytest.approx(2.0)
    cold = [run(FLAT, X, y, seed=s, n_starts=1).theta[1] for s in range(3)]
    assert all(c != pytest.approx(37.0) for c in cold)  # cold starts land elsewhere


def test_warm_start_is_clipped_into_bounds():
    X, y = data(MM, MM_TRUE)
    res = run(MM, X, y, n_starts=0, warm_start=np.array([1e6, -5.0]))
    assert res.ok
    assert in_bounds(MM, res.theta)


def test_warm_start_shape_is_checked():
    X, y = data(MM, MM_TRUE)
    with pytest.raises(ValueError, match="warm_start"):
        run(MM, X, y, warm_start=np.array([1.0, 2.0, 3.0]))


def test_no_starts_raises():
    X, y = data(MM, MM_TRUE)
    with pytest.raises(ValueError, match="no starts"):
        run(MM, X, y, n_starts=0)


def test_caller_errors_raise_without_runtime_type_checks():
    # pytest type-checks every call into mda, which hides these errors; a production run does
    # not. They must raise, not come back as a failed fit.
    code = """
import numpy as np
from mda.fit import fit
from mda.hypothesis import Hypothesis, Param

h = Hypothesis("mm", "V * C_A / (K + C_A)", (Param("V", 1e-3, 1e3), Param("K", 1e-3, 1e3)))
X = np.ones((5, 7))
for kwargs in ({"y": np.ones(5), "residual": "bogus"}, {"y": np.ones(4)}):
    try:
        fit(h, X, rng=np.random.default_rng(0), n_starts=2, **kwargs)
        print("returned")
    except ValueError:
        print("raised")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["raised", "raised"]


def test_fixed_parameter_is_held():
    h = Hypothesis("mm", MM.expr, (Param("Vmax", 1e-3, 1e3), Param("Km", 0.7, 0.7)))
    X, y = data(MM, MM_TRUE)
    res = run(h, X, y)
    assert res.theta[1] == 0.7
    assert res.theta[0] == pytest.approx(2.5, rel=1e-6)


def test_all_parameters_fixed_just_evaluates():
    h = Hypothesis("mm", MM.expr, (Param("Vmax", 2.5, 2.5), Param("Km", 0.7, 0.7)))
    X, y = data(MM, MM_TRUE)
    res = run(h, X, y, n_starts=0)
    np.testing.assert_array_equal(res.theta, MM_TRUE)
    assert res.ok
    assert res.rss < 1e-20


def test_wide_bounds_reach_small_values():
    # Km = 1e-4 sits seven decades below the upper bound and is still recovered
    h = Hypothesis("mm", MM.expr, (Param("Vmax", 1e-3, 1e3), Param("Km", 1e-6, 1e3)))
    X = D0.copy()
    X[:, CA] = [1e-2, 3e-4, 1e-4]  # below the benchmark's C_A bound, to make Km identifiable
    y = compile_hypothesis(h)(np.array([2.5, 1e-4]), X)
    res = run(h, X, y)
    np.testing.assert_allclose(res.theta, [2.5, 1e-4], rtol=1e-4)


def test_wide_bounds_are_drawn_in_log_space():
    # j does not affect the predictions, so the fitted j is the start that was drawn for it
    h = Hypothesis("flat", FLAT.expr, (Param("k", 0.1, 10.0), Param("j", 1e-6, 1e6)))
    X = conditions(10)
    y = 2.0 * X[:, CA]
    j = np.array([run(h, X, y, seed=s, n_starts=1).theta[1] for s in range(20)])
    # Log-uniform draws fall on both sides of 1; uniform draws would all be far above it
    assert np.sum(j < 1.0) >= 5
    assert np.sum(j > 1.0) >= 5


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_linear_space_parameters_do_not_warn():
    # Neither Ea nor b spans more than two decades, so both stay in linear space. b has a
    # negative bound, and Ea is far beyond the point where exp overflows.
    h = Hypothesis(
        "linear",
        "k * exp(-Ea / 8.314 * (1 / T - 1 / 310)) * C_A + b",
        (Param("k", 1e-3, 1e3), Param("Ea", 1e4, 1e5), Param("b", -5.0, 5.0)),
    )
    X = conditions(25)
    truth = np.array([2.0, 5e4, -1.0])
    y = compile_hypothesis(h)(truth, X)
    res = run(h, X, y, warm_start=np.array([1.0, 2e4, -2.0]))
    assert res.ok
    np.testing.assert_allclose(res.theta, truth, rtol=1e-4)


@pytest.mark.parametrize("residual", ["whitened", "log1p"])
def test_all_zero_d0_does_not_crash(residual):
    # The activator worlds (c67, c78, c92) return 0 at every D0 condition
    y = np.zeros(3)
    res = run(HILL, D0, y, residual=residual)
    assert res.ok
    assert in_bounds(HILL, res.theta)
    assert np.all(np.isfinite(res.y_hat))
    # Only C_A varies in D0, so every other input's correlation is 0
    rho = residual_input_spearman(D0, y, res.y_hat)
    assert np.all(np.isfinite(rho))
    np.testing.assert_array_equal(np.delete(rho, CA), np.zeros(6))


def test_underdetermined_fit():
    X, y = data(HILL, HILL_TRUE, n=2)
    res = run(HILL, X, y)
    assert res.ok
    assert res.rss < 1e-10


def test_precompiled_function_is_used():
    X, y = data(MM, MM_TRUE)
    calls = []
    f = compile_hypothesis(MM)

    def counted(theta, X):
        calls.append(1)
        return f(theta, X)

    res = run(MM, X, y, f=counted)
    assert calls
    np.testing.assert_allclose(res.theta, MM_TRUE, rtol=1e-4)


def test_fast_enough():
    X, y = data(HILL, HILL_TRUE)
    y = y * np.exp(np.random.default_rng(1).normal(0.0, 0.05, size=y.shape))
    f = compile_hypothesis(HILL)
    run(HILL, X, y, f=f)  # warm up
    t0 = time.perf_counter()
    for seed in range(5):
        run(HILL, X, y, seed=seed, f=f, warm_start=HILL_TRUE)
    # "Milliseconds" per fit (blueprint §1); the bound is loose to keep CI stable
    assert (time.perf_counter() - t0) / 5 < 0.5


def test_relative_residuals():
    y = np.array([0.0, 0.005, 1.0, -2.0])
    y_hat = np.array([0.002, 0.0, 1.1, -1.0])
    np.testing.assert_allclose(relative_residuals(y, y_hat), [0.2, 0.5, 0.1, 0.5])
    np.testing.assert_allclose(relative_residuals(y, y_hat, c=0.1), [0.02, 0.05, 0.1, 0.5])


def test_spearman_picks_the_correlated_input():
    X = conditions(30, seed=3)
    y = np.ones(30)
    y_hat = y - X[:, CA] ** 2  # residual increases monotonically with C_A
    rho = residual_input_spearman(X, y, y_hat)
    assert rho.shape == (7,)
    assert rho[CA] == pytest.approx(1.0)
    assert np.all(np.abs(np.delete(rho, CA)) < 0.5)
    rho_neg = residual_input_spearman(X, y, y + X[:, CA])
    assert rho_neg[CA] == pytest.approx(-1.0)


def test_spearman_unit_coordinates_do_not_matter():
    # D6 specifies to_unit coordinates; ranks are invariant to that monotone map
    X = conditions(20, seed=4)
    rng = np.random.default_rng(5)
    y, y_hat = rng.normal(size=20), rng.normal(size=20)
    np.testing.assert_allclose(
        residual_input_spearman(X, y, y_hat),
        residual_input_spearman(to_unit(X), y, y_hat),
        atol=1e-12,
    )


@pytest.mark.filterwarnings("error")
def test_spearman_constant_gives_zero_without_warnings():
    y = np.array([1.0, 2.0, 3.0])
    # Constant residual: every correlation is 0
    np.testing.assert_array_equal(residual_input_spearman(D0, y, y), np.zeros(7))
    # Varying residual on D0: only C_A varies, the 6 constant inputs give 0
    rho = residual_input_spearman(D0, y, np.zeros(3))
    assert rho[CA] == pytest.approx(1.0)
    np.testing.assert_array_equal(np.delete(rho, CA), np.zeros(6))
    # A single point
    np.testing.assert_array_equal(residual_input_spearman(D0[:1], y[:1], y[:1] - 1), np.zeros(7))


def test_spearman_non_finite_residual_gives_nan():
    y = np.array([1.0, 2.0, 3.0])
    rho = residual_input_spearman(D0, y, np.array([1.0, np.inf, 3.0]))
    assert np.all(np.isnan(rho))
