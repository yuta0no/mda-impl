"""Model weights from point estimates: BIC, weights, pruning and the adaptive cap (spec §3-4).

Every function takes arrays with one entry per model, in pool order, so the module does not
depend on `mda.pool`. One model-posterior step (Alg. 9) on a refitted `pool` with `n` observations:

    bic = bic_profiled(pool.rss, pool.n_free, n)  # or bic_fixed
    log_w = log_weights(bic, log_prior(pool.n_free, lam))
    keep, log_w = prune_top(log_w, cap)
    pool.keep(keep)
    w = np.exp(log_w)

A model whose RSS is not finite has BIC = inf and weight 0, and is the first to be pruned.
"""

import numpy as np
from jaxtyping import Float, Int

from mda.arrays import ParamCounts, PerModel

RSS_FLOOR = 1e-12  # floor of RSS / N in the profiled-scale BIC (D2)
TAU_C = 0.9  # the cap drops when the largest weight exceeds this (spec §4, App. B.2)
TAU_FIT = 0.05  # ... and the MAP model's median relative residual is at most this
N_MIN = 6  # the dropped cap
N_MAX = 24  # the cap otherwise


def bic_profiled(
    rss: PerModel, n_params: ParamCounts, n: int, rss_floor: float = RSS_FLOOR
) -> PerModel:
    """Profiled-scale BIC, N log(RSS / N) + C log N (spec §3, Alg. 8).

    `n` is the number of observations and `n_params` the number of free parameters C. RSS / N is
    floored at `rss_floor`: an exact fit would otherwise give -inf, so exact fits tie on the
    first term and C log N decides between them (D2). A non-finite RSS gives inf.
    """
    _check_n(n)
    bic = n * np.log(np.maximum(rss / n, rss_floor)) + n_params * np.log(n)
    return np.where(np.isfinite(rss), bic, np.inf)


def bic_fixed(rss: PerModel, n_params: ParamCounts, n: int) -> PerModel:
    """Fixed-scale BIC, RSS + C log N: the tau = 1 control (spec §3). A non-finite RSS gives inf."""
    _check_n(n)
    return np.where(np.isfinite(rss), rss + n_params * np.log(n), np.inf)


def log_prior(n_params: ParamCounts, lam: float = 0.0) -> PerModel:
    """Unnormalised log of the structure prior p(m) ∝ exp(-lam C_m) (spec §3).

    `lam` is 0 in the BIC arm (Table 2), a uniform prior, and 2 in the SMC control (spec §8).
    """
    return -lam * n_params.astype(float)


def log_weights(bic: PerModel, log_p: PerModel | None = None) -> PerModel:
    """log p(m | D) = log softmax_m(-BIC_m / 2 + log p(m)) (spec §3, Alg. 9).

    `log_p` is the log structure prior, as `log_prior` returns it; None is a uniform prior.
    A model with a non-finite BIC gets -inf, a weight of 0. When no model has a finite BIC,
    every entry is -inf: the weights are then all 0 and do not sum to 1.
    """
    score = -0.5 * bic if log_p is None else -0.5 * bic + log_p
    return _normalise(np.where(np.isfinite(score), score, -np.inf))


def weights(bic: PerModel, log_p: PerModel | None = None) -> PerModel:
    """p(m | D), the exponential of `log_weights`.

    Use it for prediction and design. Rank and prune with `log_weights`: the weight of a model
    far behind the best underflows to exactly 0 (routinely under the fixed-scale BIC, where the
    RSS is not logged), which loses its order among the other models at 0.
    """
    return np.exp(log_weights(bic, log_p))


def prune_top(log_w: PerModel, n: int) -> tuple[Int[np.ndarray, " k"], Float[np.ndarray, " k"]]:
    """Keep the `n` models with the largest weights and renormalise (spec §3).

    Returns the indices of the kept models, in pool order, and their renormalised log weights.
    Ties go to the earlier model. Models at -inf (a non-finite fit) are dropped first, also
    before a model whose weight is finite in log space but underflows to 0. All models are kept
    when there are at most `n`.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    order = np.argsort(-np.where(np.isnan(log_w), -np.inf, log_w), kind="stable")
    keep = np.sort(order[:n])
    return keep, _normalise(log_w[keep])


def adaptive_cap(
    w: PerModel,
    fit_err: float,
    *,
    tau_c: float = TAU_C,
    tau_fit: float = TAU_FIT,
    n_min: int = N_MIN,
    n_max: int = N_MAX,
) -> int:
    """Pool-size cap after an update (Alg. 10; spec §4).

    `n_min` when the largest of the weights `w` exceeds `tau_c` and `fit_err`, the MAP model's
    median relative residual, is at most `tau_fit`; otherwise `n_max`. A NaN `fit_err` and an
    empty pool give `n_max`.
    """
    if w.size and np.max(w) > tau_c and fit_err <= tau_fit:
        return n_min
    return n_max


def _normalise(log_w: Float[np.ndarray, " k"]) -> Float[np.ndarray, " k"]:
    """Subtract the log of the sum of exp(log_w), leaving it unchanged when that sum is 0."""
    finite = np.isfinite(log_w)
    if not finite.any():
        return np.full(log_w.shape, -np.inf)
    # Shift by the maximum before exponentiating, so that the largest term is exp(0) = 1
    shifted = np.where(finite, log_w, -np.inf) - log_w[finite].max()
    return shifted - np.log(np.sum(np.exp(shifted)))


def _check_n(n: int) -> None:
    if n < 1:
        raise ValueError(f"need at least one observation, got n = {n}")
