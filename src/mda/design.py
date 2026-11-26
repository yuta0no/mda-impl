"""Experiment design: the model-disagreement objective V(a), CMA-ES and random design (spec §6).

The primary arm chooses the next condition by maximising the between-model variance of the
predictions (App. B.2) with CMA-ES on the normalised 7-d box (Alg. 14, D10). The random-design
control draws it uniformly from the same box.

The functions take the model weights and one predictor per model, so the module does not depend
on `mda.pool`. With the BIC weights `w` of a refitted pool:

    predictors = [lambda X, c=c: c.f(c.fit.theta, X) for c in pool]
    x = cma_design(w, predictors, rng=rng)

A predictor maps conditions in natural units to predicted rates. The SMC control passes one
that averages over a model's particles instead.
"""

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from jaxtyping import Float

from mda.arrays import Condition, Conditions, PerModel, Rates
from mda.bench import INPUT_NAMES, from_unit
from mda.config import DEFAULTS, Space

# pycma warns on import when matplotlib, which it only needs for plotting, is missing
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Could not import matplotlib", category=UserWarning)
    import cma

type Predictor = Callable[[Conditions], Rates]

UnitDesigns = Float[np.ndarray, "n 7"]  # conditions in the unit box, as `mda.bench.to_unit` maps
Values = Float[np.ndarray, " n"]  # one objective value per design

_DIM = len(INPUT_NAMES)


class DesignError(RuntimeError):
    """`cma_design` found no condition at which V(a) is defined."""


def between_model_variance(
    U: UnitDesigns,
    weights: PerModel,
    predictors: Sequence[Predictor],
    *,
    space: Space = DEFAULTS.design_space,
) -> Values:
    """V(a) = sum_m w_m (mu_m(a) - sum_m' w_m' mu_m'(a))^2 at each row a of `U` (App. B.2).

    `U` is in the unit box and is mapped to natural units with `from_unit`. `weights` are the
    model weights in the order of `predictors`, normally summing to 1 as `posterior.weights`
    returns them, and `predictors[m]` gives mu_m in natural units. `space` is the space in which
    the predictions are compared: the raw rate (D9) or log(1 + rate).

    Models with weight 0 are skipped, so a pruned or unfittable model cannot affect V. V is NaN
    where it is undefined: where a model with positive weight predicts a non-finite value (or
    one at or below -1 in log1p space), or where predictions are so large that V overflows.
    `cma_design` never chooses such a point.
    """
    if space not in ("raw", "log1p"):
        raise ValueError(f"unknown space {space!r}")
    w = np.asarray(weights, dtype=float)
    if len(predictors) != w.shape[0]:
        raise ValueError(f"{w.shape[0]} weights for {len(predictors)} predictors")
    if not np.all(np.isfinite(w) & (w >= 0)):
        raise ValueError(f"weights must be finite and non-negative, got {w}")
    X = from_unit(U)
    active = np.flatnonzero(w > 0)
    if active.size == 0:
        return np.zeros(X.shape[0])
    with np.errstate(all="ignore"):
        mu = np.stack([predictors[m](X) for m in active.tolist()])
        if space == "log1p":
            mu = np.log1p(mu)
        w = w[active]
        mean = w @ mu
        v = w @ (mu - mean) ** 2
    # A non-finite prediction makes v NaN or inf, and so does an overflow of finite ones
    return np.where(np.isfinite(v), v, np.nan)


def cma_design(
    weights: PerModel,
    predictors: Sequence[Predictor],
    *,
    rng: np.random.Generator,
    n_evals: int = DEFAULTS.n_evals,
    sigma0: float = DEFAULTS.sigma0,
    space: Space = DEFAULTS.design_space,
) -> Condition:
    """The next condition, in natural units: the CMA-ES maximiser of V(a) (Alg. 14, D10).

    CMA-ES starts at the centre of the unit box with step size `sigma0`, uses pycma's default
    population size (9 in 7-d) and keeps every candidate in [0, 1]^7. It stops after exactly
    `n_evals` evaluations of V, cutting the last generation short if needed, and returns the
    best point evaluated, not the final mean. Ties go to the earliest point evaluated.

    A point where V is NaN is never returned, and pycma is told it is the worst of its
    generation. Raises `DesignError` when V is NaN at every point evaluated, which leaves no
    valid design to return; the caller decides what to do instead, such as `random_design`.

    pycma ignores its termination criteria here: a flat V (one model with weight 1, or models
    that agree everywhere) still takes `n_evals` evaluations. All randomness comes from `rng`;
    the global NumPy random state is neither used nor changed.
    """
    if n_evals < 1:
        raise ValueError(f"n_evals must be positive, got {n_evals}")
    es = cma.CMAEvolutionStrategy(
        np.full(_DIM, 0.5),
        sigma0,
        {
            "bounds": [0.0, 1.0],
            "randn": lambda lam, n: rng.standard_normal((lam, n)),
            "seed": np.nan,  # do not seed (or use) the global NumPy random state
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
        },
    )
    best_u, best_v = None, -np.inf
    n_left = n_evals
    while n_left > 0:
        U = np.clip(np.array(es.ask()), 0.0, 1.0)
        full = U.shape[0] <= n_left
        U = U[:n_left]
        v = between_model_variance(U, weights, predictors, space=space)
        n_left -= U.shape[0]
        defined = ~np.isnan(v)
        if defined.any():
            i = int(np.nanargmax(v))  # the earliest of the largest defined values
            if v[i] > best_v:
                best_u, best_v = U[i], v[i]
        if full:
            # pycma minimises; a NaN V ranks below every defined value
            es.tell(list(U), list(np.where(defined, -v, np.inf)))
    if best_u is None:
        raise DesignError(
            f"V(a) is undefined at all {n_evals} points evaluated: "
            "a model with positive weight predicts no finite rate at any of them"
        )
    return from_unit(best_u)


def random_design(rng: np.random.Generator) -> Condition:
    """A condition drawn uniformly from the unit box, in natural units: the random-design control.

    This is the distribution of `World.heldout`: log-uniform on the log axes, uniform elsewhere.
    """
    return from_unit(rng.uniform(size=_DIM))
