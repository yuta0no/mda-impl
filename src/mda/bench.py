import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from mda._vendor.autoscilab.oracle import chembench as _chembench
from mda._vendor.autoscilab.oracle.chembench import (
    CHEM_BENCHMARK_DOMAINS,
    CHEM_INPUT_BOUNDS,
    CHEM_INPUT_VARS,
    CHEM_LOG_VARS,
    ChemBenchOracle,
)
from mda.arrays import Condition, Conditions, Rates

_log = logging.getLogger("mda.vendor.autoscilab")


def _print_to_log(*args, sep=" ", **_kwargs):
    _log.debug(sep.join(str(a) for a in args))


# use logger instead of print
_chembench.print = _print_to_log  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[missing-attribute]


# re-export parameters from vendored autoscilab
INPUT_NAMES = tuple(CHEM_INPUT_VARS)
BOUNDS = np.array([CHEM_INPUT_BOUNDS[n] for n in INPUT_NAMES])
LOG_AXES = np.array([n in CHEM_LOG_VARS for n in INPUT_NAMES])

# settings for MDA paper (p.36 of arXiv v5)
NOISE = 0.01  # multiplicative noise (1% of the true value)
DIFFICULTY = "medium"

# the benchmark's fixed test grid, `World.heldout(TEST_SEED, N_TEST)`: the set the upstream
# `evaluate_law` scores on, and the one RMSLE and the pass criterion use (App. B.1)
TEST_SEED = 42
N_TEST = 1000

# settings not explicitly specified in the MDA paper
LAW_VERSION = "v0"
GRAMMAR_MODE = "none"  # not relevant to MDA


def list_worlds() -> list[str]:
    return list(CHEM_BENCHMARK_DOMAINS)


def canonical_worlds() -> list[str]:
    return [d for d in CHEM_BENCHMARK_DOMAINS if int(d.split("_")[0][1:]) < 10]


# CMA-ES requires normalised variables (p.37 of arXiv v5)
def _warp(X):
    """Log on the log axes, identity elsewhere."""
    return np.where(LOG_AXES, np.log(np.where(LOG_AXES, X, 1.0)), X)


_LO, _HI = _warp(BOUNDS[:, 0]), _warp(BOUNDS[:, 1])


def to_unit(X: Float[np.ndarray, "*batch 7"]) -> Float[np.ndarray, "*batch 7"]:
    return (_warp(X) - _LO) / (_HI - _LO)


def from_unit(U: Float[np.ndarray, "*batch 7"]) -> Float[np.ndarray, "*batch 7"]:
    Z = _LO + np.clip(U, 0.0, 1.0) * (_HI - _LO)
    X = np.where(LOG_AXES, np.exp(np.where(LOG_AXES, Z, 0.0)), Z)
    return np.clip(X, BOUNDS[:, 0], BOUNDS[:, 1])


@dataclass(frozen=True)
class WorldSpec:
    domain: str
    law_version: str = LAW_VERSION
    difficulty: str = DIFFICULTY
    noise: float = NOISE


class World:
    def __init__(self, spec: WorldSpec) -> None:
        self.spec = spec
        self.oracle = ChemBenchOracle(
            domain_id=spec.domain,
            difficulty=spec.difficulty,
            noise_level=spec.noise,
            law_version=spec.law_version,
            grammar_mode=GRAMMAR_MODE,
        )

    def observe(self, x: Condition) -> float:
        params = dict(zip(INPUT_NAMES, x.tolist(), strict=True))
        return float(self.oracle.run(params=params).measurement)

    def truth(self, X: Conditions) -> Rates:
        fn, p = self.oracle._rate_fn, self.oracle._law_params
        return np.array([fn(p, *row) for row in X.tolist()], dtype=float)

    def heldout(self, seed: int, n: int) -> Conditions:
        rng = np.random.default_rng(seed=seed)
        ret = []
        for (lo, hi), is_log in zip(BOUNDS, LOG_AXES, strict=True):
            if is_log:
                ret.append(np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
            else:
                ret.append(rng.uniform(lo, hi, n))
        return np.stack(ret, axis=1)
