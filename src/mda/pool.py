"""The candidate pool: the hypotheses under consideration and their fits (spec §3–4).

`Pool.add` takes new hypotheses, dropping exact-expression duplicates (D13). `Pool.refit` fits
every candidate to all the data, as each update does (Alg. 9). `mda.posterior` turns the pool's
`rss` and `n_free` into weights and chooses which candidates `Pool.keep` retains.

In the SMC control (spec §8), `Pool.refit_smc` takes the place of `Pool.refit`: it runs tempered
SMC for every candidate, and the weights come from the pool's `log_z`.
"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from jaxtyping import Int

from mda.arrays import Conditions, ParamCounts, PerModel, Rates
from mda.config import DEFAULTS, Residual
from mda.fit import FitResult, fit, relative_residuals, residuals, sigma
from mda.hypothesis import Hypothesis, Model, compile_hypothesis, normalise_expr
from mda.smc import SMCResult, tempered_smc


class Origin(StrEnum):
    """Where a candidate came from (spec §4)."""

    SEED = "seed"  # the seed library
    INIT = "init"  # the LLM call at initialisation (Alg. 11)
    EXPAND = "expand"  # an LLM expansion call (Alg. 12)


@dataclass(frozen=True, eq=False)
class Candidate:
    """One hypothesis of the pool. Candidates compare by identity."""

    hypothesis: Hypothesis
    f: Model  # compile_hypothesis(hypothesis)
    key: str  # normalise_expr(hypothesis.expr), which identifies the candidate in the pool (D13)
    origin: Origin
    added_at: int  # the number of observations B when it was added: 3 for a seed or an init
    fit: FitResult | None = None  # on the data of the last `Pool.refit`; None until then
    # Median relative residual of `fit` on that data (D5): inf when the fit is not ok, NaN until
    # the first fit
    fit_err: float = math.nan
    # The particle posterior on that data, when the pool was last refitted by `Pool.refit_smc`;
    # `fit` is then its best-fitting particle (D21). None after `Pool.refit`
    smc: SMCResult | None = None

    @property
    def n_free(self) -> int:
        """C_m in the BIC: the number of parameters with `low < high` (D20).

        `fit` holds a parameter with `low == high` fixed, so it is not counted.
        """
        return sum(p.low < p.high for p in self.hypothesis.params)


class Pool:
    """The candidates, in the order they were added. Indices and per-model arrays follow it."""

    def __init__(self) -> None:
        self._candidates: list[Candidate] = []

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self._candidates)

    def __getitem__(self, i: int) -> Candidate:
        return self._candidates[i]

    def add(
        self, hypotheses: Iterable[Hypothesis], origin: Origin, *, added_at: int
    ) -> list[Candidate]:
        """Add the hypotheses that are not yet in the pool, and return their candidates.

        `added_at` is the number of observations at the time, which each candidate records.

        A hypothesis is a duplicate when its expression matches, up to formatting, a candidate's
        or an earlier one's in `hypotheses` (D13); the earlier one stays, whatever the names and
        bounds. The expression of a candidate removed by `keep` is no longer in the pool, so it
        can be added again. New candidates have no fit until the next `refit`.

        Raises `HypothesisError` for a hypothesis that does not compile, and then leaves the pool
        unchanged.
        """
        keys = {c.key for c in self._candidates}
        added: list[Candidate] = []
        for h in hypotheses:
            key = normalise_expr(h.expr)
            compiled = compile_hypothesis(h)
            if key in keys:
                continue
            keys.add(key)
            added.append(Candidate(h, compiled, key, origin, added_at))
        self._candidates.extend(added)
        return added

    def refit(
        self,
        X: Conditions,
        y: Rates,
        *,
        rng: np.random.Generator,
        n_starts: int = DEFAULTS.n_starts,
        residual: Residual = DEFAULTS.residual,
        rel_floor: float = DEFAULTS.rel_floor,
        rho: float = DEFAULTS.rho,
        sigma_floor: float = DEFAULTS.sigma_floor,
        log_span: float = DEFAULTS.log_span,
    ) -> None:
        """Fit every candidate to all of (X, y), replacing its `fit` and `fit_err` (spec §3).

        Each fit runs `n_starts` random starts and, when the candidate has a fit, a warm start
        from its estimate (D8). The candidates draw from `rng` in pool order, so the result
        depends only on the pool, the data and `rng`'s state. `rel_floor` is the floor c of the
        relative residual (D5); `rho`, `sigma_floor` and `log_span` are passed to `fit`. If a fit
        raises, the pool is left unchanged. A candidate's `smc` is dropped: it is not of this fit.
        """
        refitted: list[Candidate] = []
        for c in self._candidates:
            res = fit(
                c.hypothesis,
                X,
                y,
                rng=rng,
                n_starts=n_starts,
                warm_start=None if c.fit is None else c.fit.theta,
                residual=residual,
                f=c.f,
                rho=rho,
                sigma_floor=sigma_floor,
                log_span=log_span,
            )
            err = np.median(relative_residuals(y, res.y_hat, rel_floor)) if res.ok else math.inf
            refitted.append(replace(c, fit=res, fit_err=float(err), smc=None))
        self._candidates = refitted

    def refit_smc(
        self,
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
        rel_floor: float = DEFAULTS.rel_floor,
        rho: float = DEFAULTS.rho,
        sigma_floor: float = DEFAULTS.sigma_floor,
    ) -> None:
        """Run tempered SMC for every candidate on all of (X, y): the SMC control (spec §8).

        It replaces each candidate's `smc`, the particle posterior with its log evidence, and
        with it `fit` and `fit_err`. The point estimate is the best-fitting particle (D21), and
        `fit.rss` the whitened RSS there, so whatever reads a fit works as it does after `refit`:
        prediction, the predictive check, the adaptive cap and the proposer. A candidate that
        is non-finite over its whole box gets `log_z = -inf` and a fit that is not ok.

        Every run starts from the prior: no particle is carried over from the previous update.
        The candidates draw from `rng` in pool order, so the result depends only on the pool,
        the data and `rng`'s state. `log_span` gives a parameter whose bounds span more than it
        a log-uniform prior (D18); None is the paper's uniform prior. `rho` and `sigma_floor`
        set the noise scale of the likelihood, which is the whitening scale of `mda.fit`;
        `rel_floor` is the floor c of the relative residual (D5); the other settings are passed
        to `tempered_smc`. If a run raises, the pool is left unchanged.
        """
        s = sigma(y, rho, sigma_floor)
        refitted: list[Candidate] = []
        for c in self._candidates:
            run = tempered_smc(
                c.hypothesis,
                X,
                y,
                rng=rng,
                n_particles=n_particles,
                ess_fraction=ess_fraction,
                n_moves=n_moves,
                max_rungs=max_rungs,
                step_scale=step_scale,
                step_floor=step_floor,
                log_span=log_span,
                sigma_n=s,
                f=c.f,
            )
            theta = run.particles[run.best].copy()
            y_hat = c.f(theta, X)
            with np.errstate(over="ignore"):
                rss = float(
                    np.sum(residuals(y, y_hat, "whitened", rho=rho, sigma_floor=sigma_floor) ** 2)
                )
            ok = math.isfinite(rss)  # exactly when the particle's likelihood is positive
            res = FitResult(theta, rss if ok else math.inf, y_hat, ok)
            err = np.median(relative_residuals(y, y_hat, rel_floor)) if ok else math.inf
            refitted.append(replace(c, fit=res, fit_err=float(err), smc=run))
        self._candidates = refitted

    def keep(self, indices: Iterable[int] | Int[np.ndarray, " k"]) -> None:
        """Remove every candidate whose index is not in `indices`. The others keep their order."""
        kept = sorted({int(i) for i in indices})
        if kept and (kept[0] < 0 or kept[-1] >= len(self)):
            raise IndexError(f"indices must be within [0, {len(self)}), got {kept}")
        self._candidates = [self._candidates[i] for i in kept]

    @property
    def n_free(self) -> ParamCounts:
        """The number of free parameters of each candidate (C_m)."""
        return np.array([c.n_free for c in self._candidates], dtype=int)

    @property
    def rss(self) -> PerModel:
        """The RSS of each candidate's fit: inf where the fit is not ok."""
        return np.array([r.rss for r in self._fits()], dtype=float)

    @property
    def fit_err(self) -> PerModel:
        """The in-sample median relative residual of each candidate: inf where the fit is not ok."""
        self._fits()
        return np.array([c.fit_err for c in self._candidates], dtype=float)

    @property
    def log_z(self) -> PerModel:
        """The log evidence of each candidate, from `refit_smc`: -inf where it has none."""
        runs = [c.smc for c in self._candidates if c.smc is not None]
        if len(runs) < len(self._candidates):
            names = [c.hypothesis.name for c in self._candidates if c.smc is None]
            raise RuntimeError(f"candidates without an SMC run on the current data: {names}")
        return np.array([r.log_z for r in runs], dtype=float)

    def _fits(self) -> list[FitResult]:
        fits = [c.fit for c in self._candidates if c.fit is not None]
        if len(fits) < len(self._candidates):
            names = [c.hypothesis.name for c in self._candidates if c.fit is None]
            raise RuntimeError(f"candidates added since the last refit have no fit: {names}")
        return fits
