"""The MDA agent: Alg. 2 with point estimates and BIC weights (spec §3-6).

One (world, seed) cell drives an agent as Alg. 1 does (`mda.run.run_cell`):

    agent = MDAAgent(X0, y0, seed=seed, config=config, proposer=proposer)
    for _ in range(config.n_active):
        x = agent.design()
        agent.update_state(x, world.observe(x))
        predict = agent.get_predictor()

The agent holds the data, the candidate pool and the model weights. Every hyperparameter comes
from its `MDAConfig`, and every hypothesis beyond the seed library from its `Proposer`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mda.arrays import Condition, Conditions, PerModel, Rates
from mda.config import DEFAULTS, MDAConfig
from mda.design import DesignError, Predictor, cma_design, random_design
from mda.fit import FitResult, relative_residuals
from mda.hypothesis import Hypothesis
from mda.library import SEED_LIBRARY
from mda.pool import Candidate, Origin, Pool
from mda.posterior import (
    adaptive_cap,
    bic_fixed,
    bic_profiled,
    log_prior,
    log_weights,
    prune_top,
)
from mda.propose import CallRecord, NullProposer, Proposer

# How a design was chosen: by CMA-ES, at random because the arm asks for it, or at random
# because CMA-ES found no condition where V(a) is defined (spec §6)
type DesignMethod = Literal["cma", "random", "fallback"]

# Streams of the per-step random generators, seeded by (cell seed, step, stream) (D8). They are
# non-zero because a seed sequence ignores trailing zeros
_FIT, _DESIGN = 1, 2


@dataclass(frozen=True)
class StepInfo:
    """What happened in the initialisation or in one `update_state`, for the run log."""

    # The predictive-check error that gated the first expansion round; None at initialisation,
    # which has no check
    e_star: float | None
    n_added: int  # candidates the proposer added to the pool, before any pruning
    calls: tuple[CallRecord, ...]  # the LLM requests the proposer made


class MDAAgent:
    """Alg. 2: `design`, `update_state`, `get_model` and `get_predictor`; the constructor is `init`.

    Public state. Each name is a property without a setter, and what it returns cannot be used
    to change the agent: only `update_state` and `design` do that.

    - `X`, `y`: the data, the passive observations first (arrays marked read-only);
    - `pool`: the candidates, fitted to all of the data (a tuple, in pool order);
    - `weights`, `log_weights`, `bic`: one value per candidate, in pool order (copies);
    - `map_index`: the pool index of the MAP candidate;
    - `cap`: the pool-size cap chosen by the last update (Alg. 10);
    - `n_e`: the number of expansion rounds so far;
    - `step`: the number of observations absorbed by `update_state`;
    - `info`: the `StepInfo` of the initialisation or the last update;
    - `last_design`: how the last `design` call chose its condition;
    - `seed`, `config`, `proposer`: the constructor's arguments.
    """

    def __init__(
        self,
        X0: Conditions,
        y0: Rates,
        *,
        seed: int,
        config: MDAConfig = DEFAULTS,
        proposer: Proposer | None = None,
        library: Sequence[Hypothesis] = SEED_LIBRARY,
    ) -> None:
        """Initialise from the passive data (X0, y0) (Alg. 2 `init`, Alg. 11).

        The pool starts as `library` plus what `proposer` returns for one "init" call; the
        default proposer returns nothing. The pool is then fitted and weighted, and pruned to
        `config.n_max`. The adaptive cap is first applied by `update_state`.

        `seed` is the cell seed: with the step, it seeds every fit and every design (D8), so two
        agents given the same arguments and observations behave identically. Raises `ValueError`
        when the pool is empty: no library and no usable proposal.
        """
        self._config = config
        self._proposer: Proposer = NullProposer() if proposer is None else proposer
        self._seed = seed
        # Copies of the caller's arrays. The data are replaced at each update, never edited
        self._X: Conditions = _frozen(np.array(X0, dtype=float))
        self._y: Rates = _frozen(np.array(y0, dtype=float))
        self._pool = Pool()
        self._step = 0
        self._n_e = 0
        self._cap = config.n_max
        self._last_design: DesignMethod | None = None
        self._log_w: PerModel = np.empty(0)
        self._bic: PerModel = np.empty(0)

        n = len(self._y)
        self._pool.add(library, Origin.SEED, added_at=n)
        proposal = self._proposer.propose(
            "init", self.pool, self._X, self._y, weights=None, budget_left=config.n_active
        )
        added = self._pool.add(proposal.hypotheses, Origin.INIT, added_at=n)
        if len(self._pool) == 0:
            raise ValueError("the initial pool is empty: no seed library and no usable proposal")
        self._score(self._rng(_FIT), config.n_max)
        self._e_cached = self._in_sample_error()
        self._info = StepInfo(None, len(added), proposal.calls)

    # Alg. 2

    def design(self) -> Condition:
        """The next condition to observe, in natural units (Alg. 2 `design`, spec §6).

        With `config.design == "cma"`, the CMA-ES maximiser of the between-model variance V(a) of
        the fitted candidates' predictions. If V(a) is undefined everywhere CMA-ES looks, the
        condition is drawn at random instead and `last_design` is "fallback". With "random", it
        is always drawn at random.

        Calling it again before the next `update_state` returns the same condition.
        """
        cfg = self._config
        rng = self._rng(_DESIGN)
        if cfg.design == "random":
            self._last_design = "random"
            return random_design(rng)
        if cfg.design != "cma":
            raise ValueError(f"unknown design {cfg.design!r}")
        try:
            x = cma_design(
                self.weights,
                [_predictor(c) for c in self._pool],
                rng=rng,
                n_evals=cfg.n_evals,
                sigma0=cfg.sigma0,
                space=cfg.design_space,
            )
        except DesignError:
            self._last_design = "fallback"
            return random_design(rng)
        self._last_design = "cma"
        return x

    def update_state(self, x: Condition, y: float) -> None:
        """Absorb the observation `y` at `x` (Alg. 2 `update-state`, spec §5).

        1. The predictive check gives e*, from the fit made before (x, y) arrived (D5).
        2. (x, y) joins the data; every candidate is refitted, reweighted and pruned to
           `config.n_max`.
        3. Up to `config.r_m` expansion rounds. A round runs while e* exceeds `config.tau_e` and
           fewer than `config.n_e_max` rounds have run in the cell. It asks the proposer for
           candidates, rescores the pool if any were added, and replaces e* by the in-sample
           error of the new MAP model. A round counts towards `n_e` whatever it adds.
        4. The adaptive cap (Alg. 10): when it is below the pool size, the candidates with the
           largest weights are kept and refitted.
        """
        cfg = self._config
        e = self._predictive_check(x, y)
        self._X = _frozen(np.vstack([self._X, x]))
        self._y = _frozen(np.append(self._y, y))
        self._step += 1
        rng = self._rng(_FIT)
        self._score(rng, cfg.n_max)

        e_star, n_added, calls = e, 0, []
        for _ in range(cfg.r_m):
            if e <= cfg.tau_e or (cfg.n_e_max is not None and self._n_e >= cfg.n_e_max):
                break
            proposal = self._proposer.propose(
                "expand",
                self.pool,
                self._X,
                self._y,
                weights=self.weights,
                budget_left=max(cfg.n_active - self._step, 0),
            )
            added = self._pool.add(proposal.hypotheses, Origin.EXPAND, added_at=len(self._y))
            self._n_e += 1
            n_added += len(added)
            calls.extend(proposal.calls)
            if added:
                self._score(rng, cfg.n_max)
            e = self._in_sample_error()

        self._adapt_and_prune(rng)
        self._e_cached = self._in_sample_error()
        self._info = StepInfo(e_star, n_added, tuple(calls))

    def get_model(self) -> Candidate:
        """The MAP candidate on the current data: the one with the largest weight (spec §3).

        Ties go to the candidate added first. When no candidate has a finite fit, every weight
        is 0 and this is the first candidate, whose predictions may be non-finite.
        """
        m = self._map()
        return self._pool[0 if m is None else m]

    def get_predictor(self) -> Predictor:
        """The plug-in predictor of the MAP candidate, f(x; theta hat). No model averaging."""
        return _predictor(self.get_model())

    # Public state: properties without setters

    @property
    def X(self) -> Conditions:
        """The conditions observed so far, the passive ones first. Marked read-only."""
        return self._X

    @property
    def y(self) -> Rates:
        """The rate observed at each row of `X`. Marked read-only."""
        return self._y

    @property
    def pool(self) -> tuple[Candidate, ...]:
        """The candidates, in pool order. A snapshot: a later update does not change it."""
        return tuple(self._pool)

    @property
    def step(self) -> int:
        """The number of observations absorbed by `update_state`."""
        return self._step

    @property
    def n_e(self) -> int:
        """The number of expansion rounds so far in the cell."""
        return self._n_e

    @property
    def cap(self) -> int:
        """The pool-size cap chosen by the last update (Alg. 10); `config.n_max` before any."""
        return self._cap

    @property
    def info(self) -> StepInfo:
        """What happened in the initialisation or in the last `update_state`."""
        return self._info

    @property
    def last_design(self) -> DesignMethod | None:
        """How the last `design` call chose its condition; None before any."""
        return self._last_design

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def config(self) -> MDAConfig:
        return self._config

    @property
    def proposer(self) -> Proposer:
        return self._proposer

    @property
    def log_weights(self) -> PerModel:
        """log p(m | D) of each candidate; -inf for one without a finite fit."""
        return self._log_w.copy()

    @property
    def weights(self) -> PerModel:
        """p(m | D) of each candidate. All 0 when no candidate has a finite fit."""
        return np.exp(self._log_w)

    @property
    def bic(self) -> PerModel:
        """The BIC of each candidate, of the kind `config.bic` names; inf without a finite fit."""
        return self._bic.copy()

    @property
    def map_index(self) -> int | None:
        """The pool index of the MAP candidate, or None when no candidate has a finite fit."""
        return self._map()

    # Internals

    def _rng(self, stream: int) -> np.random.Generator:
        return np.random.default_rng([self._seed, self._step, stream])

    def _score(self, rng: np.random.Generator, cap: int) -> None:
        """One model-posterior step (Alg. 9): refit, reweight, and prune to `cap`."""
        cfg = self._config
        self._pool.refit(
            self._X,
            self._y,
            rng=rng,
            n_starts=cfg.n_starts,
            residual=cfg.residual,
            rel_floor=cfg.rel_floor,
            rho=cfg.rho,
            sigma_floor=cfg.sigma_floor,
            log_span=cfg.log_span,
        )
        n, n_free, rss = len(self._y), self._pool.n_free, self._pool.rss
        if cfg.bic == "profiled":
            bic = bic_profiled(rss, n_free, n, cfg.rss_floor)
        elif cfg.bic == "fixed":
            bic = bic_fixed(rss, n_free, n)
        else:
            raise ValueError(f"unknown bic {cfg.bic!r}")
        keep, self._log_w = prune_top(log_weights(bic, log_prior(n_free, cfg.lam)), cap)
        self._pool.keep(keep)
        self._bic = bic[keep]

    def _map(self) -> int | None:
        if not np.isfinite(self._log_w).any():
            return None
        return int(np.argmax(self._log_w))

    def _in_sample_error(self) -> float:
        """The MAP candidate's median relative residual on the data; inf when there is no MAP."""
        m = self._map()
        return math.inf if m is None else self._pool[m].fit_err

    def _predictive_check(self, x: Condition, y: float) -> float:
        """e* for the new observation, before it joins the data (Alg. 13, D5).

        "cached": the in-sample error left by the preceding fit, as the executed ChemBench agent
        used (App. A.8). It does not look at (x, y), so a surprising observation triggers an
        expansion one update later. "prequential": the relative error of the MAP candidate's
        prediction at `x`, which its fit has not seen; with `config.backtest`, the median of that
        error and the in-sample ones. A check that cannot be evaluated gives inf.
        """
        cfg = self._config
        if cfg.check == "cached":
            return self._e_cached
        if cfg.check != "prequential":
            raise ValueError(f"unknown check {cfg.check!r}")
        m = self._map()
        if m is None:
            return math.inf
        c = self._pool[m]
        fit = _fit_of(c)
        errors = relative_residuals(np.array([y]), c.f(fit.theta, x[None, :]), cfg.rel_floor)
        if cfg.backtest:
            stored = relative_residuals(self._y, fit.y_hat, cfg.rel_floor)
            errors = np.concatenate([stored, errors])
        e = float(np.median(errors))
        return e if math.isfinite(e) else math.inf

    def _adapt_and_prune(self, rng: np.random.Generator) -> None:
        """Choose the cap for this update and prune to it (Alg. 10, spec §4)."""
        cfg = self._config
        self._cap = adaptive_cap(
            self.weights,
            self._in_sample_error(),
            tau_c=cfg.tau_c,
            tau_fit=cfg.tau_fit,
            n_min=cfg.n_min,
            n_max=cfg.n_max,
        )
        if len(self._pool) > self._cap:
            keep, _ = prune_top(self._log_w, self._cap)
            self._pool.keep(keep)
            self._score(rng, self._cap)


def _frozen(a: np.ndarray) -> np.ndarray:
    """`a` itself, marked read-only."""
    a.setflags(write=False)
    return a


def _fit_of(c: Candidate) -> FitResult:
    if c.fit is None:
        raise RuntimeError(f"candidate {c.hypothesis.name!r} has no fit")
    return c.fit


def _predictor(c: Candidate) -> Predictor:
    """f(x; theta hat) of a fitted candidate."""
    theta = _fit_of(c).theta
    return lambda X: c.f(theta, X)
