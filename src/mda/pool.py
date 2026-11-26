"""The candidate pool: the hypotheses under consideration and their fits (spec §3–4).

`Pool.add` takes new hypotheses, dropping exact-expression duplicates (D13). `Pool.refit` fits
every candidate to all the data, as each update does (Alg. 9). `mda.posterior` turns the pool's
`rss` and `n_free` into weights and chooses which candidates `Pool.keep` retains.
"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from jaxtyping import Int

from mda.arrays import Conditions, ParamCounts, PerModel, Rates
from mda.fit import REL_FLOOR, FitResult, Residual, fit, relative_residuals
from mda.hypothesis import Hypothesis, Model, compile_hypothesis, normalise_expr

N_STARTS = 8  # random starts per fit, on top of the warm start (D8)


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
        n_starts: int = N_STARTS,
        residual: Residual = "whitened",
        rel_floor: float = REL_FLOOR,
    ) -> None:
        """Fit every candidate to all of (X, y), replacing its `fit` and `fit_err` (spec §3).

        Each fit runs `n_starts` random starts and, when the candidate has a fit, a warm start
        from its estimate (D8). The candidates draw from `rng` in pool order, so the result
        depends only on the pool, the data and `rng`'s state. `rel_floor` is the floor c of the
        relative residual (D5). If a fit raises, the pool is left unchanged.
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
            )
            err = np.median(relative_residuals(y, res.y_hat, rel_floor)) if res.ok else math.inf
            refitted.append(replace(c, fit=res, fit_err=float(err)))
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

    def _fits(self) -> list[FitResult]:
        fits = [c.fit for c in self._candidates if c.fit is not None]
        if len(fits) < len(self._candidates):
            names = [c.hypothesis.name for c in self._candidates if c.fit is None]
            raise RuntimeError(f"candidates added since the last refit have no fit: {names}")
        return fits
