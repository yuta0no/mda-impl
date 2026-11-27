"""Tempered-SMC evidence for one hypothesis: the SMC control arm (spec §8, Alg. 6, D18).

The primary arm scores a model by the BIC of its point estimate. The SMC control replaces that
with a log evidence, log Z_m = log of the integral of p(D | theta) p(theta) over the declared
bounds, estimated by adaptive-tempering SMC with `n_particles` particles per structure. Its
particles also replace the point estimate in the design objective: mu-bar_m(a) is their average.
Where the agent needs a point estimate all the same (to predict, and for the fit error of the
predictive check and the adaptive cap), it is the best-fitting particle (D21).

The functions take one hypothesis, or arrays with one entry per model, so the module does not
depend on `mda.pool`. One model-posterior step (Alg. 9) and design of the control, on the
observations `(X, y)` and with one result per candidate `c` of a pool:

    results = [tempered_smc(c.hypothesis, X, y, rng=rng, f=c.f) for c in pool]
    log_w = log_model_weights(np.array([r.log_z for r in results]), pool.n_free, lam)
    predictors = [particle_predictor(c.f, r.particles, r.weights) for c, r in zip(pool, results)]
    x = cma_design(np.exp(log_w), predictors, rng=rng)

`Pool.refit_smc` runs the first line and keeps each result with its candidate. A model that is
non-finite over its whole box has log Z = -inf and weight 0.

The settings are fields of `MDAConfig`, which the SMC arm reads when `evidence` is "smc".
"""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from mda.arrays import Conditions, ParamCounts, PerModel, Rates
from mda.config import DEFAULTS
from mda.design import Predictor
from mda.fit import sigma
from mda.hypothesis import Hypothesis, Model, compile_hypothesis
from mda.posterior import log_prior, log_weights

# Halvings of the step when searching for the next temperature. Sixty exhaust the resolution of
# a float, so a step that cannot reach the target comes out as small as a step can be.
_BISECTIONS = 60

Particles = Float[np.ndarray, "n_particles p"]  # one parameter vector per row, in natural units
ParticleValues = Float[np.ndarray, " n_particles"]  # one value per particle: weight, log-lik
PerRung = Float[np.ndarray, " rungs"]  # one value per tempering rung


@dataclass(frozen=True)
class SMCResult:
    """The particle posterior of one hypothesis, and the evidence estimate that came with it.

    Every rung resamples, so the weights are uniform. They are kept because Alg. 6 returns them
    and `particle_predictor` takes them. The arrays are marked read-only: a result is shared by
    reference, as a `FitResult` is.
    """

    particles: Particles  # within the declared bounds; approximately draws from p(theta | D)
    weights: ParticleValues  # normalised
    loglik: ParticleValues  # log p(D | theta_i); the largest marks the best-fitting particle
    log_z: float  # the estimate of log p(D | m); -inf when no particle had a finite likelihood
    betas: PerRung  # the temperature reached by each rung: increasing, and 1 at the last
    ess: PerRung  # effective sample size of each rung's reweighted particles, before resampling
    acceptance: PerRung  # fraction of each rung's proposals accepted; NaN when it made none
    n_loglik: int  # likelihood evaluations made: the cost of the run
    ok: bool  # False when log_z is -inf

    def __post_init__(self) -> None:
        for a in (self.particles, self.weights, self.loglik, self.betas, self.ess, self.acceptance):
            a.setflags(write=False)

    @property
    def best(self) -> int:
        """Index of the best-fitting particle: the largest likelihood, the first one on a tie.

        That particle is the point estimate of the SMC control (D21). The index is 0 when no
        particle has a finite likelihood.
        """
        return int(np.argmax(self.loglik))


def log_likelihood(y: Rates, y_hat: Rates, sigma_n: Rates) -> float:
    """Gaussian log-likelihood of the whitened raw residuals (D1), constants included.

    -1/2 sum_n ((y_n - y_hat_n) / sigma_n)^2 - sum_n log sigma_n - N/2 log(2 pi). Up to the
    constants, this is -RSS / 2 with the RSS of `mda.fit`, the likelihood behind the fixed-scale
    BIC (spec §3). It is -inf when a prediction is non-finite or the sum of squares overflows,
    so such a parameter vector has zero posterior mass.
    """
    return _log_likelihood(y, y_hat, sigma_n, _log_norm(sigma_n))


def _log_norm(sigma_n: np.ndarray) -> float:
    """The constant of `log_likelihood`: -sum_n log sigma_n - N/2 log(2 pi)."""
    return -float(np.sum(np.log(sigma_n))) - 0.5 * sigma_n.shape[0] * math.log(2 * math.pi)


def _log_likelihood(
    y: np.ndarray, y_hat: np.ndarray, sigma_n: np.ndarray, log_norm: float
) -> float:
    # `log_likelihood` with its constant passed in: the sampler calls this once per proposal
    with np.errstate(all="ignore"):
        ll = log_norm - 0.5 * float(np.sum(((y - y_hat) / sigma_n) ** 2))
    return ll if math.isfinite(ll) else -math.inf


def tempered_smc(
    h: Hypothesis,
    X: Conditions,
    y: Rates,
    *,
    rng: np.random.Generator,
    n_particles: int = DEFAULTS.n_particles,
    ess_fraction: float = DEFAULTS.ess_fraction,
    n_moves: int = DEFAULTS.n_moves,
    max_rungs: int = DEFAULTS.max_rungs,
    step_scale: float = DEFAULTS.step_scale,
    step_floor: float = DEFAULTS.step_floor,
    log_span: float | None = DEFAULTS.log_span,
    sigma_n: Rates | None = None,
    f: Model | None = None,
) -> SMCResult:
    """Estimate the log evidence of `h` on (X, y) by adaptive-tempering SMC (Alg. 6, D18).

    The prior of a parameter is uniform over its bounds, in natural units (App. A.4), unless
    the bounds are wide: a parameter with `low > 0` and `high / low > log_span` gets a prior
    uniform in its logarithm, and moves in its logarithm. That is a deviation from the paper
    (D18), for the reason given below; `log_span=None` gives every parameter the paper's prior.

    The likelihood is `log_likelihood` with the whitening scale `sigma_n`, which defaults to
    `mda.fit.sigma(y)`. `f` is `compile_hypothesis(h)`; pass it to avoid recompiling.

    The particles start as draws from the prior at temperature beta = 0, and each rung raises
    beta until it reaches 1, the posterior:

    1. Choose the step. The whole remaining step is taken when it leaves an effective sample
       size of at least `ess_fraction * n_particles`, and on rung `max_rungs`, whatever it
       leaves. Otherwise the step is bisected to the largest one that does.
    2. Add log sum_i W_i exp(step * loglik_i) to log Z, and reweight.
    3. Resample, with systematic resampling. The paper names no scheme; this one is ours.
    4. Make `n_moves` random-walk Metropolis moves of every particle at the new beta. The
       proposal is Gaussian with a diagonal scale of `step_scale` times the standard deviation
       of the resampled particles, floored at `step_floor` before scaling and fixed for the
       rung. A proposal outside the bounds is rejected without evaluating it. Clipping it to
       the bounds instead would break detailed balance (spec §11).

    A parameter with `low == high` is held fixed and never proposed (D20). A parameter vector
    whose predictions are non-finite has likelihood 0: it is dropped at the first rung and no
    move to one is accepted. When more than `1 - ess_fraction` of the prior draws are like
    that, no step reaches the target, and the first rung takes the smallest step there is,
    which only removes them. When all of them are, the result has `log_z = -inf` and
    `ok = False`, with the prior draws as particles and no rungs.

    The estimate is only as good as the moves. Measured on a two-parameter law under the
    uniform prior (docs/blueprint.md, §11), it is within 1 or 2 of the true log evidence
    while the prior box is up to about 10^5 times the volume the posterior occupies. Far
    beyond that, the particles reach the posterior too late, or not at all, and log Z comes
    out too low: by tens to thousands for a seed law, whose bounds span six decades (D7).
    This is why wide bounds get the log-uniform prior by default, with the two-decade
    threshold of the fit (D8). It changes the prior, so the evidence is not the paper's.

    All randomness comes from `rng`, and the result depends only on its state. Raises
    `ValueError` for caller errors: settings out of range, or a `sigma_n` that is not positive
    or does not match `y`.
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be positive, got {n_particles}")
    if not 0.0 <= ess_fraction < 1.0:
        raise ValueError(f"ess_fraction must be in [0, 1), got {ess_fraction}")
    if n_moves < 0:
        raise ValueError(f"n_moves must be non-negative, got {n_moves}")
    if max_rungs < 1:
        raise ValueError(f"max_rungs must be positive, got {max_rungs}")
    if not (step_scale > 0 and step_floor >= 0):
        raise ValueError(
            f"need step_scale > 0 and step_floor >= 0, got {step_scale} and {step_floor}"
        )
    if log_span is not None and not log_span >= 1.0:
        raise ValueError(f"log_span must be at least 1, got {log_span}")
    s = sigma(y) if sigma_n is None else np.asarray(sigma_n, dtype=float)
    if s.shape != y.shape or not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError(f"sigma_n must be positive with the shape of y, {y.shape}; got {s}")
    if f is None:
        f = compile_hypothesis(h)

    lo = np.array([p.low for p in h.params], dtype=float)
    hi = np.array([p.high for p in h.params], dtype=float)
    free = lo < hi
    n_free = int(free.sum())
    in_log = np.zeros_like(free)  # the parameters with a log-uniform prior
    if log_span is not None:
        in_log = free & (lo > 0) & (hi > log_span * lo)

    # The sampler works on z, one coordinate per free parameter: the log of a parameter with a
    # log-uniform prior, the value itself otherwise. The prior is uniform in z either way.
    def to_theta(z: np.ndarray) -> np.ndarray:
        theta = np.tile(lo, (z.shape[0], 1))
        theta[:, free] = z
        if in_log.any():
            with np.errstate(over="ignore"):  # only for a proposal far outside the bounds
                theta[:, in_log] = np.exp(theta[:, in_log])
            # exp(log(high)) can round to just above high. This is not how a proposal outside
            # the bounds is handled: that one is rejected below, before it is ever used.
            np.clip(theta, lo, hi, out=theta)
        return theta

    log_norm = _log_norm(s)

    def loglik(theta: np.ndarray) -> float:
        return _log_likelihood(y, f(theta, X), s, log_norm)

    z_lo, z_hi = lo[free], hi[free]
    z_lo[in_log[free]], z_hi[in_log[free]] = np.log(lo[in_log]), np.log(hi[in_log])
    z = rng.uniform(z_lo, z_hi, size=(n_particles, n_free))
    theta = to_theta(z)
    if n_free:
        ll = np.array([loglik(row) for row in theta])
        n_loglik = n_particles
    else:  # every particle is the same point
        ll = np.full(n_particles, loglik(theta[0]))
        n_loglik = 1
    uniform = np.full(n_particles, 1.0 / n_particles)
    if not np.any(np.isfinite(ll)):
        none = np.empty(0)
        return SMCResult(theta, uniform, ll, -math.inf, none, none, none, n_loglik, False)

    beta, log_z = 0.0, 0.0
    betas: list[float] = []
    ess: list[float] = []
    acceptance: list[float] = []
    while beta < 1.0:
        step, full = _next_step(
            ll, 1.0 - beta, ess_fraction * n_particles, force=len(betas) + 1 >= max_rungs
        )
        w, shift = _increments(ll, step)
        # The weights are uniform at the start of a rung, so sum_i W_i exp(step * ll_i) is a mean
        log_z += shift + math.log(w.sum() / n_particles)
        beta = 1.0 if full else beta + step
        betas.append(beta)
        ess.append(float(w.sum() ** 2 / np.sum(w**2)))

        keep = _systematic(w, rng)
        z, ll = z[keep], ll[keep]

        n_accepted, n_proposed = 0, n_moves * n_particles * bool(n_free)
        if n_proposed:
            scale = step_scale * np.maximum(z.std(axis=0), step_floor)
            for _ in range(n_moves):
                proposals = z + scale * rng.standard_normal((n_particles, n_free))
                log_u = np.log1p(-rng.uniform(size=n_particles))  # log of a draw in (0, 1]
                inside = np.all((proposals >= z_lo) & (proposals <= z_hi), axis=1)
                candidates = to_theta(proposals)
                for i in np.flatnonzero(inside).tolist():
                    ll_new = loglik(candidates[i])
                    n_loglik += 1
                    # Every particle here has a finite likelihood, so a proposal at -inf
                    # gives -inf and is never accepted
                    if log_u[i] < beta * (ll_new - ll[i]):
                        z[i], ll[i] = proposals[i], ll_new
                        n_accepted += 1
        acceptance.append(n_accepted / n_proposed if n_proposed else math.nan)

    return SMCResult(
        to_theta(z),
        uniform,
        ll,
        log_z,
        np.array(betas),
        np.array(ess),
        np.array(acceptance),
        n_loglik,
        True,
    )


def particle_predictor(
    f: Model, particles: Particles, weights: ParticleValues | None = None
) -> Predictor:
    """mu-bar(a) = sum_i W_i f(theta_i, a): the prediction averaged over a model's particles (Eq. 60).

    This is the predictor the SMC control passes to `mda.design.between_model_variance` and
    `cma_design`, where the BIC arm passes the prediction at its point estimate (spec §6, §8).
    `weights` are the particle weights, normally summing to 1 as `SMCResult.weights` do; None
    weights the particles equally.

    Particles with weight 0 are skipped. The average is non-finite wherever a particle with
    positive weight predicts a non-finite rate, which leaves V(a) undefined there, as for a
    point estimate. Each call evaluates `f` once per particle.
    """
    theta = np.array(particles, dtype=float)
    n = theta.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.array(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"{w.shape[0]} weights for {n} particles")
    if not np.all(np.isfinite(w) & (w >= 0)):
        raise ValueError(f"weights must be finite and non-negative, got {w}")
    active = np.flatnonzero(w > 0).tolist()
    if not active:
        raise ValueError("no particle has positive weight")
    w = w[active]

    def predict(X: Conditions) -> Rates:
        with np.errstate(all="ignore"):
            return w @ np.stack([f(theta[i], X) for i in active])

    return predict


def log_model_weights(log_z: PerModel, n_params: ParamCounts, lam: float) -> PerModel:
    """log p(m | D) = log softmax_m(log Z_m + log p(m)), with p(m) ∝ exp(-lam C_m) (Alg. 9).

    `log_z` holds the log evidence of each model, as `SMCResult.log_z`, and `n_params` its number
    of free parameters C_m. `lam` is 2 in the SMC control (spec §8); it has no default, because
    `MDAConfig.lam` defaults to the BIC arm's 0. A model at -inf gets -inf, a weight of 0. The
    result takes the place of `mda.posterior.log_weights` in the SMC control: exponentiate it
    for prediction and design, and pass it as it is to `mda.posterior.prune_top`.
    """
    # log_weights takes a BIC and applies -BIC / 2, which undoes the factor exactly
    return log_weights(-2.0 * log_z, log_prior(n_params, lam))


def _increments(ll: ParticleValues, step: float) -> tuple[ParticleValues, float]:
    """exp(step * ll - shift) for each particle, and the shift: the largest step * ll.

    The largest weight is then 1, so the weights cannot all underflow. `ll` must have a finite
    entry. A particle at -inf gets 0, also for a step so small that step * ll would be 0.
    """
    log_inc = np.where(np.isfinite(ll), step * ll, -np.inf)
    shift = float(log_inc.max())
    return np.exp(log_inc - shift), shift


def _next_step(
    ll: ParticleValues, remaining: float, target: float, *, force: bool
) -> tuple[float, bool]:
    """The next temperature step, and whether it is the whole of `remaining`.

    The step is the largest one whose reweighted particles have an effective sample size of at
    least `target`, which decreases as the step grows. When not even the smallest step reaches
    `target`, that smallest step is returned, so the temperature always rises.
    """

    def ess(step: float) -> float:
        w, _ = _increments(ll, step)
        return float(w.sum() ** 2 / np.sum(w**2))

    if force or ess(remaining) >= target:
        return remaining, True
    reached, missed = 0.0, remaining
    for _ in range(_BISECTIONS):
        mid = 0.5 * (reached + missed)
        if ess(mid) >= target:
            reached = mid
        else:
            missed = mid
    return (reached if reached > 0.0 else missed), False


def _systematic(w: ParticleValues, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: as many indices as weights, drawn in proportion to `w`.

    One uniform draw places an evenly spaced comb over the cumulative weights, so particle i is
    kept floor(n W_i) or ceil(n W_i) times. A particle with weight 0 is never kept.
    """
    n = w.shape[0]
    cumulative = np.cumsum(w)
    cumulative /= cumulative[-1]  # the last entry is exactly 1, above every comb tooth
    teeth = (rng.uniform() + np.arange(n)) / n
    return np.searchsorted(cumulative, teeth, side="right")
