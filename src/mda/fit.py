"""Point estimates for one hypothesis: bounded multistart least squares (spec §3, D1, D8).

`fit` computes the point estimate in the BIC approximation (App. A.6.3, Alg. 8):
theta_hat = argmin RSS(theta) over the declared bounds. `relative_residuals` feeds the predictive
check (Alg. 13, D5) and the adaptive cap (Alg. 10), and `residual_input_spearman` the correlation
hint of the expansion prompt (Alg. 12, D6).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from scipy.optimize import least_squares
from scipy.stats import ConstantInputWarning, spearmanr

from mda.arrays import Conditions, Rates, Theta
from mda.bench import INPUT_NAMES
from mda.config import DEFAULTS, Residual
from mda.hypothesis import Hypothesis, Model, compile_hypothesis

# Inside the optimiser, non-finite residuals become this value and larger ones are clipped to it,
# so a step into a region where the model overflows is rejected instead of aborting the start.
# It must be far above any meaningful residual: clipping flattens the objective, and a start on
# the plateau has zero gradient and stops where it began (a cap of 1e8 left half the random
# starts of an Arrhenius law stuck, 1e20 a fifth). It must also stay well below 1e43: at the edge
# of a non-finite region the finite-difference Jacobian is about cap / 1e-8, and scipy's
# trust-region solver raises its singular values to the sixth power, which then overflows.
# Final RSS values are never penalised.
_PENALTY = 1e30


@dataclass(frozen=True)
class FitResult:
    theta: Theta  # within the declared bounds, in natural (not log) units; (theta hat in the paper)
    rss: float  # inf when no start gave finite residuals
    y_hat: Rates  # predictions at theta on the fitted X; may be non-finite when not ok
    ok: bool

    def __post_init__(self) -> None:
        # A result is shared by reference (a candidate, a warm start, a snapshot of the pool),
        # so its arrays are marked read-only
        self.theta.setflags(write=False)
        self.y_hat.setflags(write=False)


def sigma(y: Rates, rho: float = DEFAULTS.rho, sigma_floor: float = DEFAULTS.sigma_floor) -> Rates:
    """Whitening scale sigma_n = max(rho |y_n|, sigma_floor) (eq. 46, spec §3)."""
    return np.maximum(rho * np.abs(y), sigma_floor)


def residuals(
    y: Rates,
    y_hat: Rates,
    residual: Residual = DEFAULTS.residual,
    *,
    rho: float = DEFAULTS.rho,
    sigma_floor: float = DEFAULTS.sigma_floor,
) -> Rates:
    """Residuals whose sum of squares is the RSS (D1).

    `whitened` (default): (y - y_hat) / sigma(y, rho, sigma_floor). `log1p`:
    log1p(y) - log1p(y_hat), which is non-finite when y_hat <= -1 and does not use the scale.
    """
    with np.errstate(all="ignore"):
        if residual == "whitened":
            return (y - y_hat) / sigma(y, rho, sigma_floor)
        if residual == "log1p":
            return np.log1p(y) - np.log1p(y_hat)
    raise ValueError(f"unknown residual {residual!r}")


def fit(
    h: Hypothesis,
    X: Conditions,
    y: Rates,
    *,
    rng: np.random.Generator,
    n_starts: int,
    warm_start: Theta | None = None,
    residual: Residual = DEFAULTS.residual,
    f: Model | None = None,
    rho: float = DEFAULTS.rho,
    sigma_floor: float = DEFAULTS.sigma_floor,
    log_span: float = DEFAULTS.log_span,
) -> FitResult:
    """Minimise the RSS of `h` on (X, y) within the bounds of `h.params` (spec §3, D8).

    Runs `scipy.optimize.least_squares(method="trf")` from the warm start (clipped into the
    bounds), if given, and from `n_starts` points drawn uniformly from the bounds, and keeps the
    start with the smallest RSS; ties go to the earlier start, so the warm start wins them. All
    random starts are drawn before any optimisation, so the result depends only on `rng`'s state.

    Parameters with `low > 0` and `high / low > log_span` are optimised, and drawn, in log space.
    Parameters with `low == high` are held fixed. `f` is `compile_hypothesis(h)`; pass it to
    avoid recompiling. `rho` and `sigma_floor` set the whitening scale of `residuals`.

    Never raises for numerical problems: if no start reaches finite residuals, the result has
    `ok=False` and `rss=inf`. Caller errors raise `ValueError`: nothing to start from, a
    `warm_start` of the wrong shape, an unknown `residual`, or shapes that do not broadcast.
    """
    if f is None:
        f = compile_hypothesis(h)
    lo = np.array([p.low for p in h.params], dtype=float)
    hi = np.array([p.high for p in h.params], dtype=float)
    free = lo < hi
    log_free = (free & (lo > 0) & (hi > log_span * lo))[free]

    # The optimiser works on z: the log of a log-space parameter, the value itself otherwise.
    # Only the log-space entries go through log and exp, because a linear-space bound can be zero
    # or negative and a linear-space value can exceed the point where exp overflows.
    def to_z(theta_free: np.ndarray) -> np.ndarray:
        return np.where(log_free, np.log(np.where(log_free, theta_free, 1.0)), theta_free)

    def to_theta(z: np.ndarray) -> np.ndarray:
        theta = lo.copy()
        theta[free] = np.where(log_free, np.exp(np.where(log_free, z, 0.0)), z)
        return np.clip(theta, lo, hi)

    z_lo, z_hi = to_z(lo[free]), to_z(hi[free])

    def res(y_hat: np.ndarray) -> np.ndarray:
        return residuals(y, y_hat, residual, rho=rho, sigma_floor=sigma_floor)

    def evaluate(theta: np.ndarray) -> tuple[np.ndarray, float]:
        y_hat = f(theta, X)
        with np.errstate(over="ignore"):
            rss = float(np.sum(res(y_hat) ** 2))
        return y_hat, rss if np.isfinite(rss) else np.inf

    if not free.any():
        theta = lo.copy()
        y_hat, rss = evaluate(theta)
        return FitResult(theta, rss, y_hat, bool(np.isfinite(rss)))

    starts: list[np.ndarray] = []
    if warm_start is not None:
        if warm_start.shape != lo.shape:
            raise ValueError(f"warm_start has shape {warm_start.shape}, expected {lo.shape}")
        w = np.clip(np.where(np.isfinite(warm_start), warm_start, lo), lo, hi)[free]
        starts.append(to_z(w))
    starts.extend(rng.uniform(z_lo, z_hi, size=(n_starts, len(z_lo))))
    if not starts:
        raise ValueError("no starts: pass n_starts > 0 or a warm_start")

    def fun(z: np.ndarray) -> np.ndarray:
        r = res(f(to_theta(z), X))
        return np.clip(np.where(np.isfinite(r), r, _PENALTY), -_PENALTY, _PENALTY)

    # Numerical problems never raise in `fun`, so whatever does is a caller error, such as an
    # unknown residual or shapes that do not broadcast. Evaluate once outside the try below,
    # which would otherwise report it as a failed fit.
    fun(starts[0])

    best: FitResult | None = None
    for z0 in starts:
        try:
            sol = least_squares(fun, z0, bounds=(z_lo, z_hi), method="trf")
        except (ValueError, np.linalg.LinAlgError):
            continue
        theta = to_theta(sol.x)
        y_hat, rss = evaluate(theta)
        if best is None or rss < best.rss:
            best = FitResult(theta, rss, y_hat, bool(np.isfinite(rss)))
    if best is None:  # every start raised
        theta = to_theta(starts[0])
        return FitResult(theta, np.inf, f(theta, X), False)
    return best


def relative_residuals(y: Rates, y_hat: Rates, c: float = DEFAULTS.rel_floor) -> Rates:
    """Per-datum relative error |y - y_hat| / max(|y|, c), with F(y) = y (D5)."""
    with np.errstate(all="ignore"):
        return np.abs(y - y_hat) / np.maximum(np.abs(y), c)


def residual_input_spearman(X: Conditions, y: Rates, y_hat: Rates) -> Float[np.ndarray, "7"]:
    """Spearman correlation between the signed residual y - y_hat and each input (D6).

    Values are in the order of `INPUT_NAMES`. D6 asks for inputs in `to_unit` coordinates; that
    map is monotone in each input, so it leaves a rank correlation unchanged and is not applied.
    The value is 0 for an input or a residual that is constant over the data, which includes
    fewer than two points. All values are NaN when any residual is non-finite.
    """
    r = y - y_hat
    if not np.all(np.isfinite(r)):
        return np.full(len(INPUT_NAMES), np.nan)
    with warnings.catch_warnings():
        # Constant inputs are routine (6 of 7 on D0); spearmanr returns NaN for them, mapped to 0
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = np.array([spearmanr(X[:, j], r).statistic for j in range(X.shape[1])])
    return np.where(np.isfinite(rho), rho, 0.0)
