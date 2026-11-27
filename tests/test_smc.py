import math

import numpy as np
import pytest
from jaxtyping import TypeCheckError
from scipy.integrate import quad
from scipy.stats import norm

from mda.bench import INPUT_NAMES, from_unit
from mda.config import DEFAULTS
from mda.design import between_model_variance, cma_design
from mda.fit import sigma
from mda.hypothesis import Hypothesis, Param, compile_hypothesis
from mda.posterior import log_prior, log_weights, prune_top
from mda.smc import (
    SMCResult,
    log_likelihood,
    log_model_weights,
    particle_predictor,
    tempered_smc,
)

CA = INPUT_NAMES.index("C_A")
N_INPUTS = len(INPUT_NAMES)
# The settings a call takes when it names none: those of the config (spec §8, D18)
N_PARTICLES, ESS_FRACTION, N_MOVES = DEFAULTS.n_particles, DEFAULTS.ess_fraction, DEFAULTS.n_moves
# The structure prior of the SMC control, p(m) ∝ exp(-2 C_m) (spec §8). An arm config sets it:
# `DEFAULTS.lam` is the BIC arm's 0
LAMBDA = 2.0


def conditions(*c_a: float) -> np.ndarray:
    """Conditions that differ only in C_A; the other inputs are those of D0 (spec §1)."""
    X = np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0], (len(c_a), 1))
    X[:, CA] = c_a
    return X


# The toy problem: five rates from r0 = 2.5 C_A with errors of up to 2%. Under `linear`, the
# likelihood is Gaussian in k, centred on K_HAT with standard deviation K_SD
X = conditions(0.1, 0.3, 1.0, 3.0, 10.0)
Y = 2.5 * X[:, CA] * np.array([1.01, 0.99, 1.02, 0.98, 1.0])
S = sigma(Y)
K_HAT = float(np.sum(X[:, CA] * Y / S**2) / np.sum(X[:, CA] ** 2 / S**2))
K_SD = float(1 / np.sqrt(np.sum(X[:, CA] ** 2 / S**2)))


def linear(low: float = 0.0, high: float = 10.0) -> Hypothesis:
    return Hypothesis("linear", "k * C_A", (Param("k", low, high),))


LINEAR = linear()
# Michaelis-Menten in a box a few thousand times the size of its posterior on the data below
MM = Hypothesis(
    "michaelis_menten",
    "Vmax * C_A / (Km + C_A)",
    (Param("Vmax", 1.0, 10.0), Param("Km", 0.01, 1.0)),
)
X_MM = conditions(0.03, 0.1, 0.3, 1.0, 3.0)
Y_MM = 5.0 * X_MM[:, CA] / (0.1 + X_MM[:, CA]) * np.array([1.01, 0.99, 1.02, 0.98, 1.0])
# Non-finite everywhere: C_I is within [0, 50], so the argument of the log is negative
BROKEN = Hypothesis("broken", "k * log(C_I - 100)", (Param("k", 0.1, 10.0),))


def partly_broken(j_low: float, j_high: float) -> Hypothesis:
    """`linear` where j > 0 and NaN where j < 0: log(j) is NaN there, and so is 0 * NaN."""
    return Hypothesis(
        "partly_broken", "k * C_A + 0 * log(j)", (Param("k", 0.0, 10.0), Param("j", j_low, j_high))
    )


def smc(h: Hypothesis = LINEAR, seed: int = 0, **kwargs) -> SMCResult:
    return tempered_smc(h, X, Y, rng=np.random.default_rng(seed), **kwargs)


def one_rung(**kwargs) -> SMCResult:
    """A run of 100 likelihood evaluations, for tests that only compare two runs."""
    return smc(max_rungs=1, n_moves=0, **kwargs)


def log_z_linear(low: float, high: float, *, log_uniform: bool = False) -> float:
    """log of the integral of p(D | k) p(k) over [low, high], by numerical quadrature."""
    f = compile_hypothesis(LINEAR)

    def ll(k: float) -> float:
        return log_likelihood(Y, f(np.array([k]), X), S)

    def prior(k: float) -> float:
        return 1 / (k * math.log(high / low)) if log_uniform else 1 / (high - low)

    peak = ll(min(max(K_HAT, low), high))  # factored out so the integrand does not underflow
    points = [k for k in (K_HAT - 8 * K_SD, K_HAT, K_HAT + 8 * K_SD) if low < k < high]
    integral, _ = quad(
        lambda k: math.exp(ll(k) - peak) * prior(k), low, high, points=points, epsabs=0, limit=500
    )
    return peak + math.log(integral)


class Recorder:
    """A compiled hypothesis that records every parameter vector it is evaluated at."""

    def __init__(self, h: Hypothesis):
        self.f = compile_hypothesis(h)
        self.seen: list[np.ndarray] = []

    def __call__(self, theta, X):
        self.seen.append(np.array(theta))
        return self.f(theta, X)

    @property
    def thetas(self) -> np.ndarray:
        return np.array(self.seen)


# log_likelihood


def test_log_likelihood_matches_hand_computation():
    # Residuals of 1 and 0 standard deviations: -1/2 - log(0.5) - log(1) - log(2 pi)
    y, y_hat, s = np.array([1.0, 2.0]), np.array([1.5, 2.0]), np.array([0.5, 1.0])
    expected = -0.5 + math.log(2) - math.log(2 * math.pi)
    assert log_likelihood(y, y_hat, s) == pytest.approx(expected, rel=1e-14)
    assert log_likelihood(y, y_hat, s) == pytest.approx(norm.logpdf(y, y_hat, s).sum(), rel=1e-14)


def test_log_likelihood_is_minus_half_rss_up_to_a_constant():
    # The RSS of mda.fit: the likelihood is the one behind the fixed-scale BIC (spec §3)
    f = compile_hypothesis(LINEAR)
    rss = [float(np.sum(((Y - f(np.array([k]), X)) / S) ** 2)) for k in (2.0, 2.5, 3.0)]
    ll = [log_likelihood(Y, f(np.array([k]), X), S) for k in (2.0, 2.5, 3.0)]
    np.testing.assert_allclose(np.diff(ll), -0.5 * np.diff(rss), rtol=1e-12)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e200])
def test_log_likelihood_of_an_unusable_prediction_is_minus_inf(bad):
    # 1e200 is finite, but its squared residual overflows
    y_hat = Y.copy()
    y_hat[2] = bad
    assert log_likelihood(Y, y_hat, S) == -math.inf


def test_log_likelihood_of_no_data_is_zero():
    none = np.empty(0)
    assert log_likelihood(none, none, none) == 0.0


# tempered_smc: the evidence


def test_log_z_matches_quadrature():
    # Done-when of feat/smc. With 400 particles, the estimates of 30 seeds were within 0.24
    assert smc(n_particles=400).log_z == pytest.approx(log_z_linear(0.0, 10.0), abs=0.5)


def test_log_z_matches_quadrature_with_the_default_settings():
    # 100 particles are noisier: a standard deviation of about 0.25 over seeds. The evidence
    # itself, not its log, is what SMC estimates without bias, so its mean is checked too
    err = np.array([smc(seed=seed).log_z for seed in range(8)]) - log_z_linear(0.0, 10.0)
    assert np.all(np.abs(err) < 1.5)
    assert abs(err.mean()) < 0.4
    assert np.exp(err).mean() == pytest.approx(1.0, abs=0.35)


def test_log_z_matches_quadrature_in_two_dimensions():
    # The posterior is within 0.5 of Vmax = 5 and 0.02 of Km = 0.1: a grid over that patch
    s = sigma(Y_MM)
    vmax, km = np.linspace(3.0, 7.0, 801), np.linspace(0.03, 0.25, 801)
    c_a = X_MM[:, CA]
    y_hat = vmax[:, None, None] * c_a / (km[None, :, None] + c_a)
    ll = -0.5 * np.sum(((Y_MM - y_hat) / s) ** 2, axis=2)
    ll += -np.sum(np.log(s)) - 0.5 * len(Y_MM) * math.log(2 * math.pi)
    integral = np.trapezoid(np.trapezoid(np.exp(ll - ll.max()), km, axis=1), vmax)
    expected = ll.max() + math.log(integral) - math.log(9.0 * 0.99)  # the prior density
    f = compile_hypothesis(MM)
    assert ll[400, 0] == pytest.approx(log_likelihood(Y_MM, f(np.array([5.0, 0.03]), X_MM), s))

    r = tempered_smc(MM, X_MM, Y_MM, rng=np.random.default_rng(0), n_particles=300)
    assert r.log_z == pytest.approx(expected, abs=1.0)  # 30 seeds were within 0.56
    np.testing.assert_allclose(r.particles.mean(axis=0), [5.0, 0.1], rtol=0.03)


def test_single_rung_is_importance_sampling_from_the_prior():
    # At the rung cap the whole step is taken at once: log Z = log mean_i p(D | theta_i) over
    # the prior draws, whatever is left of the effective sample size
    rec = Recorder(LINEAR)
    r = one_rung(f=rec)
    assert rec.thetas.shape == (N_PARTICLES, 1) and r.n_loglik == N_PARTICLES
    ll = np.array([log_likelihood(Y, rec.f(theta, X), S) for theta in rec.thetas])
    expected = ll.max() + math.log(np.mean(np.exp(ll - ll.max())))
    assert r.log_z == pytest.approx(expected, rel=1e-12)
    np.testing.assert_array_equal(r.betas, [1.0])
    assert r.ess[0] < ESS_FRACTION * N_PARTICLES
    assert math.isnan(r.acceptance[0])  # no proposals were made
    assert np.all(np.isin(r.particles, rec.thetas))  # resampled, not moved


def test_evidence_of_no_data_is_one():
    r = tempered_smc(LINEAR, np.empty((0, N_INPUTS)), np.empty(0), rng=np.random.default_rng(0))
    assert r.log_z == 0.0 and r.ok
    np.testing.assert_array_equal(r.betas, [1.0])
    np.testing.assert_array_equal(r.ess, [N_PARTICLES])


def test_sigma_n_defaults_to_the_whitening_scale():
    default = one_rung()
    assert one_rung(sigma_n=S).log_z == default.log_z
    # Twice the noise scale halves every whitened residual: a different likelihood
    doubled = one_rung(sigma_n=2 * S)
    assert doubled.log_z != default.log_z
    f = compile_hypothesis(LINEAR)
    expected = [log_likelihood(Y, f(theta, X), 2 * S) for theta in doubled.particles]
    np.testing.assert_allclose(doubled.loglik, expected, rtol=1e-12)


# tempered_smc: the rungs


@pytest.mark.parametrize("eta", [0.3, ESS_FRACTION, 0.9])
def test_ess_stays_at_or_above_the_target(eta):
    # Done-when of feat/smc
    # The moves do not enter into it, so one per rung keeps the test cheap
    for h in (LINEAR, MM):
        r = smc(h, ess_fraction=eta, n_moves=1)
        assert len(r.betas) > 2
        assert np.all(r.ess >= eta * N_PARTICLES)
        assert np.all(r.ess <= N_PARTICLES)
        # Every rung but the last was bisected onto the target
        np.testing.assert_allclose(r.ess[:-1], eta * N_PARTICLES, rtol=1e-6)


def test_temperatures_rise_to_one():
    r = smc()
    assert len(r.betas) == len(r.ess) == len(r.acceptance) > 2
    assert np.all(np.diff(r.betas) > 0) and r.betas[0] > 0
    assert r.betas[-1] == 1.0  # exactly: the last step is the whole of what remained
    assert np.all((r.acceptance > 0) & (r.acceptance < 1))


def test_a_higher_target_takes_more_rungs():
    low, high = smc(ess_fraction=0.5, n_moves=1), smc(ess_fraction=0.9, n_moves=1)
    assert len(high.betas) > len(low.betas)


def test_rung_cap_ends_the_run():
    r = smc(max_rungs=3)
    assert len(r.betas) == 3 and r.betas[-1] == 1.0
    assert np.all(r.ess[:2] >= ESS_FRACTION * N_PARTICLES)
    assert r.ess[2] < ESS_FRACTION * N_PARTICLES  # the forced step ignores the target


def test_result_fields():
    r = smc(MM)
    assert r.particles.shape == (N_PARTICLES, 2) and r.loglik.shape == (N_PARTICLES,)
    np.testing.assert_array_equal(r.weights, 1 / N_PARTICLES)  # every rung resamples
    assert r.ok and math.isfinite(r.log_z)
    f = compile_hypothesis(MM)
    expected = [log_likelihood(Y, f(theta, X), S) for theta in r.particles]
    np.testing.assert_array_equal(r.loglik, expected)
    # The prior draws, then up to n_moves proposals per particle and rung
    assert N_PARTICLES < r.n_loglik <= N_PARTICLES * (1 + N_MOVES * len(r.betas))


def test_best_is_the_particle_with_the_largest_likelihood():
    r = smc()
    assert r.loglik[r.best] == r.loglik.max()
    # The closest of 100 posterior draws to the least-squares estimate, the mode
    assert abs(r.particles[r.best, 0] - K_HAT) < 0.2 * K_SD
    assert np.all(np.abs(r.particles[:, 0] - K_HAT) >= abs(r.particles[r.best, 0] - K_HAT))
    # Resampling without moves leaves copies of the best particle: the first one counts
    copies = one_rung()
    ties = np.flatnonzero(copies.loglik == copies.loglik.max())
    assert len(ties) > 1 and copies.best == ties[0]
    # No particle with a finite likelihood: the first one
    assert smc(BROKEN).best == 0


def test_result_arrays_are_read_only():
    # A result is shared by reference: by a candidate, and by every snapshot of the pool
    for r in (smc(MM), smc(BROKEN)):
        for name in ("particles", "weights", "loglik", "betas", "ess", "acceptance"):
            with pytest.raises(ValueError, match="read-only"):
                getattr(r, name)[...] = 0.0


# tempered_smc: the bounds


def test_particles_never_leave_the_bounds():
    # Done-when of feat/smc. Every parameter vector the likelihood sees is inside the box
    for h in (LINEAR, MM, linear(1e-3, 1e3)):
        rec = Recorder(h)
        r = smc(h, n_moves=3, f=rec)
        lo, hi = [p.low for p in h.params], [p.high for p in h.params]
        assert np.all((rec.thetas >= lo) & (rec.thetas <= hi))
        assert np.all((r.particles >= lo) & (r.particles <= hi))


def test_proposals_outside_the_bounds_are_rejected_not_clipped():
    # The upper bound cuts the likelihood at its peak, so half the proposals of a particle
    # near it land outside. Clipping would park them on the bound itself (spec §11)
    h = linear(0.0, K_HAT)
    rec = Recorder(h)
    r = smc(h, n_particles=400, f=rec)
    assert np.all(rec.thetas < K_HAT) and np.all(rec.thetas > 0.0)
    assert np.all(r.particles < K_HAT)
    assert np.mean(r.particles > K_HAT - K_SD) > 0.5  # they do crowd against it
    # The posterior is the lower half of a Gaussian: its mean is sqrt(2 / pi) sd below the peak
    assert r.particles.mean() == pytest.approx(K_HAT - math.sqrt(2 / math.pi) * K_SD, abs=0.006)
    assert r.log_z == pytest.approx(log_z_linear(0.0, K_HAT), abs=0.5)


def test_step_floor_keeps_a_collapsed_cloud_moving():
    # One particle has a standard deviation of 0, so the proposal scale is the floor times
    # `step_scale`: 0.05 here. Without the floor every proposal would be the particle itself
    rec = Recorder(LINEAR)
    smc(n_particles=1, n_moves=50, step_floor=0.1, f=rec)
    steps = np.abs(np.diff(rec.thetas[:, 0]))
    assert len(steps) > 40  # the prior draw, then one proposal per move unless it left the box
    assert np.all(steps > 0) and np.all(steps < 0.5)
    assert 0.01 < np.median(steps) < 0.1


def test_fixed_parameter_is_held():
    # D20: b has low == high. It is not sampled, so the run is that of `linear`, draw for draw
    h = Hypothesis("offset", "k * C_A + b", (Param("b", 0.0, 0.0), Param("k", 0.0, 10.0)))
    r, plain = smc(h), smc(LINEAR)
    np.testing.assert_array_equal(r.particles[:, 0], 0.0)
    np.testing.assert_array_equal(r.particles[:, 1], plain.particles[:, 0])
    assert r.log_z == plain.log_z


def test_hypothesis_without_free_parameters():
    # One point: the evidence is its likelihood, evaluated once
    f = compile_hypothesis(LINEAR)
    for h in (linear(2.5, 2.5), Hypothesis("constant", "2.5 * C_A", ())):
        r = smc(h)
        assert r.log_z == pytest.approx(log_likelihood(Y, f(np.array([2.5]), X), S), rel=1e-14)
        assert r.n_loglik == 1 and r.particles.shape == (N_PARTICLES, len(h.params))
        np.testing.assert_array_equal(r.betas, [1.0])
        assert math.isnan(r.acceptance[0])


# tempered_smc: non-finite models


@pytest.mark.filterwarnings("error")
def test_model_that_is_non_finite_in_part_of_its_box():
    # j < 0, a quarter of the box, has likelihood 0, and j > 0 does not matter: three quarters
    # of the evidence of `linear`
    r = smc(partly_broken(-1.0, 3.0), n_particles=200)
    assert r.ok
    assert r.log_z == pytest.approx(log_z_linear(0.0, 10.0) + math.log(0.75), abs=1.0)
    assert np.all(r.particles[:, 1] > 0)
    assert np.all(np.isfinite(r.loglik))
    assert np.all(r.ess >= ESS_FRACTION * 200)


@pytest.mark.filterwarnings("error")
def test_first_rung_only_removes_non_finite_particles_when_they_are_the_majority():
    # j > 0 on a tenth of the box: no step leaves an effective sample size of half the
    # particles, so the first rung takes the smallest step there is and the rest go on as usual
    r = smc(partly_broken(-9.0, 1.0), n_particles=200)
    assert r.ok
    assert 0 < r.betas[0] < 1e-12
    assert 0 < r.ess[0] < 50  # the particles that drew j > 0, about 20
    assert r.ess[0] == pytest.approx(round(r.ess[0]), abs=1e-9)  # all of them, equally weighted
    assert np.all(r.ess[1:] >= ESS_FRACTION * 200)
    assert r.log_z == pytest.approx(log_z_linear(0.0, 10.0) + math.log(0.1), abs=1.2)
    assert np.all(r.particles[:, 1] > 0)


@pytest.mark.filterwarnings("error")
def test_model_that_is_non_finite_everywhere():
    r = smc(BROKEN)
    assert r.log_z == -math.inf and not r.ok
    assert r.betas.size == r.ess.size == r.acceptance.size == 0
    assert r.particles.shape == (N_PARTICLES, 1) and r.n_loglik == N_PARTICLES
    assert np.all((r.particles >= 0.1) & (r.particles <= 10.0))  # the prior draws
    np.testing.assert_array_equal(r.loglik, -math.inf)
    np.testing.assert_array_equal(r.weights, 1 / N_PARTICLES)


# tempered_smc: randomness and arguments


def test_same_rng_state_gives_the_same_result():
    a, b, c = (smc(MM, seed=seed, n_moves=2) for seed in (1, 1, 2))
    assert a.log_z == b.log_z and a.n_loglik == b.n_loglik
    np.testing.assert_array_equal(a.particles, b.particles)
    np.testing.assert_array_equal(a.betas, b.betas)
    assert a.log_z != c.log_z


def test_global_random_state_is_not_used():
    np.random.seed(0)  # noqa: NPY002
    before = np.random.get_state()[1].copy()  # noqa: NPY002
    a = smc(n_moves=1)
    np.testing.assert_array_equal(np.random.get_state()[1], before)  # noqa: NPY002
    np.random.seed(1)  # noqa: NPY002
    assert smc(n_moves=1).log_z == a.log_z


def test_compiles_the_hypothesis_when_f_is_not_given():
    assert one_rung(f=compile_hypothesis(LINEAR)).log_z == one_rung().log_z


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_particles": 0},
        {"ess_fraction": 1.0},
        {"ess_fraction": -0.1},
        {"n_moves": -1},
        {"max_rungs": 0},
        {"step_scale": 0.0},
        {"step_floor": -1.0},
        {"log_span": 0.5},
        {"sigma_n": np.zeros(5)},
        {"sigma_n": np.full(5, np.nan)},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        smc(**kwargs)


def test_sigma_n_of_the_wrong_length_raises():
    # ValueError from the function itself; under pytest, jaxtyping's shape check comes first
    with pytest.raises((ValueError, TypeCheckError)):
        smc(sigma_n=np.ones(3))


# tempered_smc: the log-uniform prior of wide bounds, a deviation from the paper (D18)


def test_wide_parameters_have_a_log_uniform_prior_by_default():
    # k spans six decades, more than the two of the default `log_span`: its prior is uniform
    # in log k
    assert DEFAULTS.log_span == 100.0
    h = linear(1e-3, 1e3)
    expected = log_z_linear(1e-3, 1e3, log_uniform=True)
    assert smc(h, n_particles=200).log_z == pytest.approx(expected, abs=1.0)
    np.testing.assert_array_equal(smc(h).particles, smc(h, log_span=100.0).particles)
    # Without a span, the prior is the paper's uniform one, which gives a different evidence
    uniform = log_z_linear(1e-3, 1e3)
    assert expected - uniform > 3
    assert smc(h, log_span=None).log_z == pytest.approx(uniform, abs=1.5)


def test_log_span_leaves_narrow_and_non_positive_parameters_alone():
    # [0, 10] has no logarithm at its lower bound, and [1, 10] spans one decade, not two
    for h in (LINEAR, linear(1.0, 10.0)):
        a, b = smc(h, n_moves=2, log_span=None), smc(h, n_moves=2, log_span=100.0)
        assert a.log_z == b.log_z
        np.testing.assert_array_equal(a.particles, b.particles)


def test_log_span_keeps_particles_within_the_bounds():
    h = linear(1e-3, 2.0)  # the upper bound cuts into the posterior
    rec = Recorder(h)
    r = smc(h, log_span=100.0, f=rec)
    assert np.all((rec.thetas >= 1e-3) & (rec.thetas <= 2.0))
    assert np.all((r.particles >= 1e-3) & (r.particles <= 2.0))


# particle_predictor


def test_particle_predictor_matches_hand_computation():
    # 0.25 * 1 C_A + 0.75 * 3 C_A = 2.5 C_A
    f = compile_hypothesis(LINEAR)
    predict = particle_predictor(f, np.array([[1.0], [3.0]]), np.array([0.25, 0.75]))
    np.testing.assert_allclose(predict(X), 2.5 * X[:, CA], rtol=1e-14)
    # Equal weights by default: 2 C_A
    np.testing.assert_allclose(
        particle_predictor(f, np.array([[1.0], [3.0]]))(X), 2.0 * X[:, CA], rtol=1e-14
    )


def test_particle_predictor_of_an_smc_result_is_the_posterior_mean_prediction():
    r = smc()
    predict = particle_predictor(compile_hypothesis(LINEAR), r.particles, r.weights)
    # The model is linear in k, so the mean prediction is the prediction at the mean k
    np.testing.assert_allclose(predict(X), r.particles.mean() * X[:, CA], rtol=1e-12)
    np.testing.assert_allclose(predict(X), K_HAT * X[:, CA], rtol=0.02)


def test_particle_predictor_copies_its_particles():
    particles = np.array([[1.0], [3.0]])
    predict = particle_predictor(compile_hypothesis(LINEAR), particles)
    particles[:] = 100.0
    np.testing.assert_allclose(predict(X), 2.0 * X[:, CA], rtol=1e-14)


@pytest.mark.filterwarnings("error")
def test_particle_predictor_skips_particles_of_weight_zero():
    # 0 * nan would otherwise make the average NaN
    f = compile_hypothesis(partly_broken(-1.0, 1.0))
    particles = np.array([[2.0, -0.5], [3.0, 0.5]])
    np.testing.assert_allclose(
        particle_predictor(f, particles, np.array([0.0, 1.0]))(X), 3.0 * X[:, CA], rtol=1e-14
    )
    # With weight, the non-finite particle makes the average non-finite
    assert not np.any(np.isfinite(particle_predictor(f, particles)(X)))


@pytest.mark.parametrize(
    "weights",
    [np.array([0.5, -0.5]), np.array([0.5, np.nan]), np.zeros(2)],
    ids=["negative", "nan", "all-zero"],
)
def test_particle_predictor_rejects_invalid_weights(weights):
    f = compile_hypothesis(LINEAR)
    with pytest.raises(ValueError):
        particle_predictor(f, np.array([[1.0], [3.0]]), weights)
    # A wrong length raises ValueError too; under pytest, jaxtyping's shape check comes first
    with pytest.raises((ValueError, TypeCheckError)):
        particle_predictor(f, np.array([[1.0], [3.0]]), np.array([1.0]))


def test_particle_predictor_in_the_design_objective():
    # Models predicting 2 C_A (two particles) and 4 C_A (a point estimate), with weights 1/4
    # and 3/4: the mean is 3.5 C_A, so V = (0.25 * 1.5^2 + 0.75 * 0.5^2) C_A^2 = 0.75 C_A^2
    f = compile_hypothesis(LINEAR)
    predictors = [
        particle_predictor(f, np.array([[1.0], [3.0]])),
        lambda X: f(np.array([4.0]), X),
    ]
    U = np.random.default_rng(0).uniform(size=(20, N_INPUTS))
    v = between_model_variance(U, np.array([0.25, 0.75]), predictors)
    np.testing.assert_allclose(v, 0.75 * from_unit(U)[:, CA] ** 2, rtol=1e-12)
    # One particle is a point estimate: at 2 C_A, it gives the same V
    one = [particle_predictor(f, np.array([[2.0]])), predictors[1]]
    np.testing.assert_allclose(
        between_model_variance(U, np.array([0.25, 0.75]), one), v, rtol=1e-12
    )


# log_model_weights


def test_model_weights_match_hand_computation():
    # Equal evidence, 1 and 2 parameters: softmax(-2, -4) under p(m) ∝ exp(-2 C_m)
    log_w = log_model_weights(np.array([0.0, 0.0]), np.array([1, 2]), LAMBDA)
    np.testing.assert_allclose(np.exp(log_w), [1 / (1 + math.exp(-2)), 1 / (1 + math.exp(2))])
    # Evidence makes up for parameters: 2 more in log Z cancel one more parameter
    log_w = log_model_weights(np.array([-7.0, -5.0]), np.array([1, 2]), LAMBDA)
    np.testing.assert_allclose(np.exp(log_w), [0.5, 0.5], rtol=1e-14)


def test_model_weights_without_a_structure_prior():
    log_w = log_model_weights(np.array([0.0, math.log(3)]), np.array([1, 5]), lam=0.0)
    np.testing.assert_allclose(np.exp(log_w), [0.25, 0.75], rtol=1e-14)


def test_model_weights_are_the_posterior_weights_of_minus_two_log_z():
    log_z, c = np.array([3.5, -120.0, 7.25, -2933.0]), np.array([2, 3, 3, 1])
    expected = log_weights(-2 * log_z, log_prior(c, LAMBDA))
    np.testing.assert_array_equal(log_model_weights(log_z, c, LAMBDA), expected)
    np.testing.assert_allclose(expected[[0, 2]], np.log([0.14804720, 0.85195280]), rtol=1e-6)
    assert np.exp(log_model_weights(log_z, c, LAMBDA)).sum() == pytest.approx(1.0)


@pytest.mark.filterwarnings("error")
def test_model_without_evidence_has_weight_zero():
    log_w = log_model_weights(np.array([-math.inf, 1.0, 1.0]), np.array([1, 2, 2]), LAMBDA)
    np.testing.assert_allclose(np.exp(log_w), [0.0, 0.5, 0.5])
    keep, log_w = prune_top(log_w, 2)
    np.testing.assert_array_equal(keep, [1, 2])
    # No model with evidence: every weight is 0, as in mda.posterior
    none = log_model_weights(np.full(2, -math.inf), np.array([1, 2]), LAMBDA)
    np.testing.assert_array_equal(none, -math.inf)


# The control's model-posterior step, as in the module docstring


def test_evidence_and_design_for_a_small_pool():
    # The data are linear in C_A: `linear` explains them, Michaelis-Menten in this box cannot
    hypotheses = [LINEAR, MM, BROKEN]
    fs = [compile_hypothesis(h) for h in hypotheses]
    rng = np.random.default_rng(0)
    results = [tempered_smc(h, X, Y, rng=rng, f=f) for h, f in zip(hypotheses, fs, strict=True)]
    n_free = np.array([len(h.params) for h in hypotheses])
    log_w = log_model_weights(np.array([r.log_z for r in results]), n_free, LAMBDA)
    w = np.exp(log_w)
    assert w[0] > 0.999 and w[2] == 0.0
    # Their particles disagree most where C_A is largest
    predictors = [
        particle_predictor(f, r.particles, r.weights) for f, r in zip(fs, results, strict=True)
    ]
    x = cma_design(np.array([0.5, 0.5, 0.0]), predictors, rng=rng)
    assert x.shape == (N_INPUTS,) and x[CA] > 50.0
