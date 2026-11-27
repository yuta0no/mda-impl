"""One (world, seed) cell of the experiment: Alg. 1, and the run log it produces (spec §1, §7).

`run_cell` starts an agent on the passive data, then alternates design, observation and update
for `config.n_active` rounds. After the initialisation and after every update it evaluates the
agent's predictor, so a cell yields one learning curve over the budget B (the number of
observations).

A `RunLog` is JSON-compatible: `RunLog.from_json(log.to_json()) == log`. The JSON is strict, so a
non-finite number is written as the string "inf", "-inf" or "nan".
"""

import json
import math
import zlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mda.agent import MDAAgent
from mda.arrays import Condition, Conditions
from mda.bench import N_TEST, TEST_SEED, World, WorldSpec
from mda.config import DEFAULTS, MDAConfig
from mda.metrics import nmsle, passed, rmsle
from mda.propose import CallRecord, NullProposer, Proposer

SCHEMA = 1  # version of the run-log format; raise it when a field changes meaning


@dataclass(frozen=True)
class ModelRecord:
    """One candidate of the pool, as fitted to the data of its step."""

    name: str
    expr: str
    params: list[str]  # parameter names, in the order of `theta`
    theta: list[float]  # theta hat
    rss: float  # inf when the fit is not finite
    bic: float  # of the kind `config.bic` names
    weight: float  # p(m | D)
    fit_err: float  # median relative residual on the data
    origin: str  # "seed", "init" or "expand"
    added_at: int  # the budget B at which the candidate joined the pool


@dataclass(frozen=True)
class Evaluation:
    """The MAP predictor's scores (spec §7). An infinite score is a failure."""

    rmsle: float  # on the benchmark's test grid
    nmsle: float  # on the cell's query set (D11)
    passed: bool  # rmsle below the pass threshold


@dataclass(frozen=True)
class StepRecord:
    """The agent after the initialisation (the first record) or after one update."""

    budget: int  # B, the number of observations so far
    x: list[float] | None  # the condition observed in this step; None at initialisation
    y: float | None  # its observed rate
    design: str | None  # how `x` was chosen: "cma", "random", or "fallback" (spec §6)
    e_star: float | None  # the predictive-check error that gated expansion (spec §5)
    n_added: int  # candidates the proposer added in this step, before pruning
    n_e: int  # expansion rounds so far in the cell
    cap: int  # the pool-size cap after this step (spec §4)
    map: int | None  # index in `pool` of the MAP candidate; None when no fit is finite
    pool: list[ModelRecord]
    llm_calls: list[CallRecord]  # the LLM requests made in this step
    eval: Evaluation

    @property
    def expanded(self) -> bool:
        """Whether the pool gained candidates in this step."""
        return self.n_added > 0


@dataclass(frozen=True)
class RunLog:
    """Everything recorded for one cell: `steps[i]` is the state at B = len(y0) + i."""

    world: str
    seed: int
    proposer: str  # class name of the proposer
    config: dict[str, Any]  # `MDAConfig.to_dict()`
    x0: list[list[float]]  # the passive conditions
    y0: list[float]  # and their observed rates
    steps: list[StepRecord]
    schema: int = SCHEMA

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(_encode(asdict(self)), allow_nan=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "RunLog":
        d = json.loads(text)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"run log has schema {d.get('schema')!r}, expected {SCHEMA}")
        return cls(
            world=d["world"],
            seed=d["seed"],
            proposer=d["proposer"],
            config=d["config"],
            x0=[[_float(v) for v in row] for row in d["x0"]],
            y0=[_float(v) for v in d["y0"]],
            steps=[_step(s) for s in d["steps"]],
            schema=d["schema"],
        )


def passive_design(config: MDAConfig = DEFAULTS) -> Conditions:
    """The conditions of the passive data D0 (spec §1, App. B.2)."""
    return np.array([[c_a, *config.passive_rest] for c_a in config.passive_c_a], dtype=float)


def query_seed(world: str, seed: int) -> int:
    """The seed of a cell's query set (D11): the CRC-32 of "<world>:<seed>".

    Unlike `hash`, it is the same in every process. It shares nothing with the generators that
    seed fitting and design, so the query set is the same for every arm.
    """
    return zlib.crc32(f"{world}:{seed}".encode())


def query_set(world: World, seed: int, n: int = DEFAULTS.n_query) -> Conditions:
    """The held-out conditions on which nMSLE is computed for the cell (world, seed) (D11)."""
    return world.heldout(query_seed(world.spec.domain, seed), n)


def run_cell(
    world: str,
    seed: int,
    config: MDAConfig = DEFAULTS,
    proposer: Proposer | None = None,
) -> RunLog:
    """Run one cell (Alg. 1) and return its log, with an evaluation at every budget.

    `world` is a benchmark world ID, and `seed` the replicate: it seeds the agent's fits and
    designs and the query set, not the world's noise (spec §1). `proposer` defaults to
    `NullProposer`, the fixed-library arm, which makes no LLM call. A proposer that holds a
    config must hold `config`, which the log records: `MDAAgent` raises `ValueError` otherwise.
    """
    w = World(WorldSpec(world))
    proposer = NullProposer() if proposer is None else proposer
    X_test = w.heldout(TEST_SEED, N_TEST)
    X_query = query_set(w, seed, config.n_query)
    y_test, y_query = w.truth(X_test), w.truth(X_query)

    def record(x: Condition | None, y: float | None) -> StepRecord:
        predict = agent.get_predictor()
        y_hat = predict(X_test)
        evaluation = Evaluation(
            rmsle=rmsle(y_hat, y_test),
            nmsle=nmsle(predict(X_query), y_query),
            passed=passed(y_hat, y_test),
        )
        return _record(agent, x, y, evaluation)

    X0 = passive_design(config)
    y0 = np.array([w.observe(x) for x in X0])
    agent = MDAAgent(X0, y0, seed=seed, config=config, proposer=proposer)
    steps = [record(None, None)]
    for _ in range(config.n_active):
        x = agent.design()
        y = w.observe(x)
        agent.update_state(x, y)
        steps.append(record(x, y))
    return RunLog(
        world=world,
        seed=seed,
        proposer=type(proposer).__name__,
        config=config.to_dict(),
        x0=X0.tolist(),
        y0=y0.tolist(),
        steps=steps,
    )


def _record(
    agent: MDAAgent, x: Condition | None, y: float | None, evaluation: Evaluation
) -> StepRecord:
    weights, bic = agent.weights, agent.bic
    pool = []
    for i, c in enumerate(agent.pool):
        assert c.fit is not None  # the agent scores the pool before returning
        pool.append(
            ModelRecord(
                name=c.hypothesis.name,
                expr=c.hypothesis.expr,
                params=[p.name for p in c.hypothesis.params],
                theta=c.fit.theta.tolist(),
                rss=float(c.fit.rss),
                bic=float(bic[i]),
                weight=float(weights[i]),
                fit_err=float(c.fit_err),
                origin=c.origin.value,
                added_at=c.added_at,
            )
        )
    return StepRecord(
        budget=len(agent.y),
        x=None if x is None else x.tolist(),
        y=None if y is None else float(y),
        design=None if x is None else agent.last_design,
        e_star=agent.info.e_star,
        n_added=agent.info.n_added,
        n_e=agent.n_e,
        cap=agent.cap,
        map=agent.map_index,
        pool=pool,
        llm_calls=list(agent.info.calls),
        eval=evaluation,
    )


def _encode(obj: Any) -> Any:
    """`obj` with every non-finite float replaced by its string, so that it is strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)  # "inf", "-inf" or "nan", which `float` reads back
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_encode(v) for v in obj]
    return obj


def _float(v: Any) -> float:
    return float(v)  # a JSON number, or the string of a non-finite float


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _step(d: dict[str, Any]) -> StepRecord:
    return StepRecord(
        budget=d["budget"],
        x=None if d["x"] is None else [_float(v) for v in d["x"]],
        y=_opt_float(d["y"]),
        design=d["design"],
        e_star=_opt_float(d["e_star"]),
        n_added=d["n_added"],
        n_e=d["n_e"],
        cap=d["cap"],
        map=d["map"],
        pool=[_model(m) for m in d["pool"]],
        llm_calls=[CallRecord(**c) for c in d["llm_calls"]],
        eval=Evaluation(
            rmsle=_float(d["eval"]["rmsle"]),
            nmsle=_float(d["eval"]["nmsle"]),
            passed=d["eval"]["passed"],
        ),
    )


def _model(d: dict[str, Any]) -> ModelRecord:
    return ModelRecord(
        name=d["name"],
        expr=d["expr"],
        params=d["params"],
        theta=[_float(v) for v in d["theta"]],
        rss=_float(d["rss"]),
        bic=_float(d["bic"]),
        weight=_float(d["weight"]),
        fit_err=_float(d["fit_err"]),
        origin=d["origin"],
        added_at=d["added_at"],
    )
