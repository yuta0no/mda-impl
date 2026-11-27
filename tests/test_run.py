import json
import math
import zlib
from dataclasses import replace
from functools import cache

import numpy as np
import pytest

from mda.agent import MDAAgent
from mda.bench import BOUNDS, N_TEST, TEST_SEED, World, WorldSpec
from mda.config import DEFAULTS, MDAConfig
from mda.hypothesis import Hypothesis, Param, compile_hypothesis
from mda.library import SEED_LIBRARY
from mda.metrics import PASS_THRESHOLD, nmsle, rmsle
from mda.propose import CallRecord, NullProposer, Proposal, Proposer
from mda.run import (
    SCHEMA,
    Evaluation,
    ModelRecord,
    RunLog,
    StepRecord,
    passive_design,
    query_seed,
    query_set,
    run_cell,
)

C0 = "c0_michaelis_menten"
# C_I activates this world, and D0 holds it at 0, so its passive rates are all 0 (README,
# "Benchmark behaviour"). No seed law comes near it within a few experiments
ZERO_D0 = "c67_allosteric_act"
FRACTAL = "c93_fractal_competitive"  # no seed law fits it within tau_e
# Two experiments, one random start and a short design search: enough to exercise the loop
TINY = replace(DEFAULTS, n_active=2, n_starts=1, n_evals=9)


# The fixed-library arm as it is run, but for the random starts per fit: 2 for the default 8,
# which makes the cell three times faster under the tests' runtime type checks
SMOKE = replace(DEFAULTS, n_starts=2)


@pytest.fixture(scope="module")
def smoke() -> RunLog:
    """One whole cell of the fixed-library arm, on the Michaelis-Menten world."""
    return run_cell(C0, 0, SMOKE)


@cache
def tiny(world: str, seed: int, config: MDAConfig = TINY) -> RunLog:
    """A short run, shared by the tests that only read it."""
    return run_cell(world, seed, config)


def best(step: StepRecord) -> ModelRecord:
    """The MAP candidate of a step."""
    assert step.map is not None
    return step.pool[step.map]


def logged_predictions(step: StepRecord, X: np.ndarray) -> np.ndarray:
    """The MAP candidate's predictions at `X`, from what the log holds of it.

    The snapshot is enough to re-evaluate a step: the expression, its parameter names and
    theta hat.
    """
    top = best(step)
    params = tuple(Param(n, v, v) for n, v in zip(top.params, top.theta, strict=True))
    return compile_hypothesis(Hypothesis(top.name, top.expr, params))(np.array(top.theta), X)


def model_records(agent: MDAAgent) -> list[ModelRecord]:
    """What a log should hold of each candidate of `agent`."""
    records = []
    for c, weight, bic in zip(agent.pool, agent.weights, agent.bic, strict=True):
        assert c.fit is not None
        records.append(
            ModelRecord(
                name=c.hypothesis.name,
                expr=c.hypothesis.expr,
                params=[p.name for p in c.hypothesis.params],
                theta=c.fit.theta.tolist(),
                rss=c.fit.rss,
                bic=float(bic),
                weight=float(weight),
                fit_err=c.fit_err,
                origin=c.origin.value,
                added_at=c.added_at,
            )
        )
    return records


def in_box(x: list[float] | None) -> bool:
    assert x is not None
    return bool(np.all(np.array(x) >= BOUNDS[:, 0]) and np.all(np.array(x) <= BOUNDS[:, 1]))


def strict_loads(text: str):
    def reject(token: str):
        raise AssertionError(f"{token} is not strict JSON")

    return json.loads(text, parse_constant=reject)


class OneLawPerCall:
    """A proposer that returns one new power law per call, and names its calls."""

    def __init__(self) -> None:
        self.n = 0

    def propose(self, kind, pool, X, y, *, weights, budget_left) -> Proposal:
        self.n += 1
        law = Hypothesis(f"power_{self.n}", f"k * C_A ** {self.n + 1}", (Param("k", 1e-3, 1e3),))
        return Proposal((law,), (CallRecord(f"{kind}/call{self.n}", f"req_{self.n}"),))


class OffData:
    """Proposes one law at initialisation: a seed law plus a term that is 0 at T = 310, the
    temperature of D0, and not finite at any other."""

    def propose(self, kind, pool, X, y, *, weights, budget_left) -> Proposal:
        if kind != "init":
            return Proposal()
        seed = SEED_LIBRARY[0]
        law = Hypothesis("off_data", f"{seed.expr} + sqrt(0 - (T - 310) ** 2)", seed.params)
        return Proposal((law,))


@cache
def expanding() -> RunLog:
    """A short run in which every update expands, by one law."""
    return run_cell(FRACTAL, 0, replace(TINY, n_active=3), OneLawPerCall())


# Protocol pieces (spec §1, D11)


def test_passive_design_is_d0():
    expected = [[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)]
    np.testing.assert_array_equal(passive_design(), expected)
    config = replace(DEFAULTS, passive_c_a=(1.0,), passive_rest=(2.0, 3.0, 4.0, 5.0, 300.0, 6.0))
    np.testing.assert_array_equal(passive_design(config), [[1.0, 2.0, 3.0, 4.0, 5.0, 300.0, 6.0]])


def test_query_seed_is_a_crc_of_world_and_seed():
    assert query_seed(C0, 0) == zlib.crc32(b"c0_michaelis_menten:0") == 1887845923
    seeds = {query_seed(w, s) for w in (C0, "c1_competitive_inhibition") for s in range(3)}
    assert len(seeds) == 6


def test_query_set_is_60_heldout_conditions_of_the_cell():
    world = World(WorldSpec(C0))
    Q = query_set(world, 0)
    assert Q.shape == (60, 7)
    assert np.all(BOUNDS[:, 0] <= Q) and np.all(BOUNDS[:, 1] >= Q)
    np.testing.assert_array_equal(Q, world.heldout(query_seed(C0, 0), 60))
    np.testing.assert_array_equal(Q, query_set(World(WorldSpec(C0)), 0))  # the same in any arm
    assert not np.array_equal(Q, query_set(world, 1))
    assert not np.array_equal(Q, query_set(World(WorldSpec("c1_competitive_inhibition")), 0))
    assert not np.array_equal(Q, world.heldout(TEST_SEED, N_TEST)[:60])  # not the test grid
    assert query_set(world, 0, 7).shape == (7, 7)


# One full cell (Alg. 1)


def test_smoke_run_passes_at_b_25(smoke: RunLog):
    # Done-when of feat/agent: one cell runs end to end with no API call, and the fixed library
    # passes on a world whose law it contains
    assert (smoke.world, smoke.seed, smoke.proposer) == (C0, 0, "NullProposer")
    assert smoke.steps[-1].budget == 25
    assert smoke.steps[-1].eval.passed
    assert smoke.steps[-1].eval.rmsle < 0.05


def test_run_log_has_23_evaluation_points(smoke: RunLog):
    assert [s.budget for s in smoke.steps] == list(range(3, 26))
    for s in smoke.steps:
        assert math.isfinite(s.eval.rmsle) and math.isfinite(s.eval.nmsle)
        assert s.eval.passed == (s.eval.rmsle < 0.05)


def test_run_log_survives_a_json_round_trip(smoke: RunLog):
    text = smoke.to_json()
    assert RunLog.from_json(text) == smoke
    assert strict_loads(text)["schema"] == SCHEMA
    assert RunLog.from_json(smoke.to_json(indent=2)) == smoke


def test_run_log_records_the_config_and_the_passive_data(smoke: RunLog):
    assert smoke.config == SMOKE.to_dict()
    assert MDAConfig.from_dict(smoke.config) == SMOKE
    world = World(WorldSpec(C0))
    np.testing.assert_array_equal(smoke.x0, passive_design())
    assert smoke.y0 == [world.observe(x) for x in passive_design()]


def test_first_record_is_the_initialisation(smoke: RunLog):
    first = smoke.steps[0]
    assert (first.x, first.y, first.design, first.e_star) == (None, None, None, None)
    assert (first.n_added, first.n_e, first.cap, first.llm_calls) == (0, 0, DEFAULTS.n_max, [])
    assert [m.expr for m in first.pool] == [h.expr for h in SEED_LIBRARY]
    assert [m.name for m in first.pool] == [h.name for h in SEED_LIBRARY]
    assert [m.params for m in first.pool] == [[p.name for p in h.params] for h in SEED_LIBRARY]


def test_each_later_record_holds_one_observation(smoke: RunLog):
    world = World(WorldSpec(C0))
    for s in smoke.steps[1:]:
        assert s.x is not None and s.y is not None and s.e_star is not None
        assert s.design == "cma" and in_box(s.x)
        assert s.y == world.observe(np.array(s.x))
    # A new design at every step
    assert len({tuple(s.x or ()) for s in smoke.steps[1:]}) == 22


def test_pool_is_cut_to_n_min_once_one_law_dominates(smoke: RunLog):
    sizes = [len(s.pool) for s in smoke.steps]
    for s in smoke.steps:
        assert s.cap in (DEFAULTS.n_min, DEFAULTS.n_max) and len(s.pool) <= s.cap
    # The cap is not applied at initialisation, and a discarded law does not come back
    assert (smoke.steps[0].cap, sizes[0]) == (DEFAULTS.n_max, len(SEED_LIBRARY))
    assert sizes == sorted(sizes, reverse=True)
    last = smoke.steps[-1]
    assert (last.cap, sizes[-1]) == (DEFAULTS.n_min, DEFAULTS.n_min)
    assert best(last).weight > DEFAULTS.tau_c and best(last).fit_err <= DEFAULTS.tau_fit


def test_fixed_library_never_expands(smoke: RunLog):
    assert all(not s.expanded and s.n_added == 0 and s.llm_calls == [] for s in smoke.steps)
    seeds = {h.expr for h in SEED_LIBRARY}
    for s in smoke.steps:
        assert {m.expr for m in s.pool} <= seeds
        assert all(m.origin == "seed" and m.added_at == 3 for m in s.pool)


def test_pool_snapshot_is_consistent(smoke: RunLog):
    for s in smoke.steps:
        weights = [m.weight for m in s.pool]
        assert sum(weights) == pytest.approx(1.0)
        assert s.map == int(np.argmax(weights))
        top = best(s)
        assert top.bic == min(m.bic for m in s.pool)  # lam = 0: the weights follow the BIC
        for m in s.pool:
            assert len(m.theta) == len(m.params)
            assert m.rss >= 0 and m.fit_err >= 0


def test_logged_scores_are_those_of_the_logged_map_model(smoke: RunLog):
    # RMSLE on the benchmark's test grid, and nMSLE on the query set of the cell: that of its
    # own seed, of the configured size (spec §7, D11)
    few_queries = replace(TINY, n_query=7)
    logs = [smoke, tiny(C0, 1), tiny(C0, 2), tiny(ZERO_D0, 0), tiny(C0, 1, few_queries)]
    assert {log.seed for log in logs} == {0, 1, 2} and logs[-1].config["n_query"] == 7
    for log in logs:
        world = World(WorldSpec(log.world))
        X_test = world.heldout(TEST_SEED, N_TEST)
        X_query = query_set(world, log.seed, log.config["n_query"])
        y_test, y_query = world.truth(X_test), world.truth(X_query)
        for s in log.steps:
            on_test, on_query = logged_predictions(s, X_test), logged_predictions(s, X_query)
            assert s.eval.rmsle == pytest.approx(rmsle(on_test, y_test), rel=1e-12)
            assert s.eval.nmsle == pytest.approx(nmsle(on_query, y_query), rel=1e-12)
            assert s.eval.passed == (s.eval.rmsle < PASS_THRESHOLD)


def test_pass_is_judged_on_the_test_grid_and_not_on_the_query_set():
    # At B = 5 this cell is under the threshold on the test grid (an RMSLE of 0.037) and over
    # it on its query set (0.058)
    log = tiny("c51_sinh_product", 1)
    world, last = World(WorldSpec(log.world)), log.steps[-1]
    X_test, X_query = world.heldout(TEST_SEED, N_TEST), query_set(world, log.seed)
    on_test = rmsle(logged_predictions(last, X_test), world.truth(X_test))
    on_query = rmsle(logged_predictions(last, X_query), world.truth(X_query))
    assert on_test < PASS_THRESHOLD < on_query
    assert last.eval.passed


def test_cell_over_the_threshold_is_recorded_as_failed():
    log = tiny(ZERO_D0, 0)
    assert all(s.eval.rmsle > 10 * PASS_THRESHOLD for s in log.steps)
    assert [s.eval.passed for s in log.steps] == [False, False, False]


# Arms and proposers, on short runs


def test_same_cell_gives_the_same_log():
    assert run_cell(C0, 1, TINY) == tiny(C0, 1)


def test_seed_changes_the_designs_but_not_the_passive_data():
    a, b = tiny(C0, 1), tiny(C0, 2)
    assert (a.x0, a.y0) == (b.x0, b.y0)
    assert a.steps[1].x != b.steps[1].x


def test_passive_data_follow_the_config():
    config = replace(
        TINY,
        n_active=1,
        passive_c_a=(0.1, 1.0, 10.0, 30.0),
        passive_rest=(5.0, 2.0, 1.0, 0.5, 300.0, 6.5),
    )
    log = run_cell(C0, 0, config)
    world, X0 = World(WorldSpec(C0)), passive_design(config)
    assert X0.shape == (4, 7)
    np.testing.assert_array_equal(log.x0, X0)
    assert log.y0 == [world.observe(x) for x in X0]
    assert [s.budget for s in log.steps] == [4, 5]
    assert all(m.added_at == 4 for m in log.steps[0].pool)


@pytest.mark.parametrize(
    ("make_log", "make_proposer", "n_e", "caps"),
    [
        (lambda: tiny(C0, 2), NullProposer, 0, [24, 6, 6]),
        (expanding, OneLawPerCall, 3, [24, 24, 24, 24]),
    ],
    ids=["cap drop", "expansions"],
)
def test_log_is_what_a_fresh_agent_does_with_the_logged_observations(
    make_log, make_proposer, n_e, caps
):
    # The log alone is enough to repeat the cell: its config, seed, passive data and observations.
    # Every record then holds the state of the agent after that step
    log: RunLog = make_log()
    assert (log.steps[-1].n_e, [s.cap for s in log.steps]) == (n_e, caps)
    proposer: Proposer = make_proposer()
    agent = MDAAgent(
        np.array(log.x0),
        np.array(log.y0),
        seed=log.seed,
        config=MDAConfig.from_dict(log.config),
        proposer=proposer,
    )
    for i, s in enumerate(log.steps):
        if i > 0:
            assert s.x is not None and s.y is not None
            # One design per step, made in the state the record before describes
            np.testing.assert_array_equal(agent.design(), s.x)
            assert s.design == agent.last_design
            agent.update_state(np.array(s.x), s.y)
        assert s.budget == len(agent.y) == 3 + i
        assert (s.e_star, s.n_added, s.n_e) == (agent.info.e_star, agent.info.n_added, agent.n_e)
        assert (s.cap, s.map) == (agent.cap, agent.map_index)
        assert s.llm_calls == list(agent.info.calls)
        assert s.pool == model_records(agent)


def test_agent_is_given_the_cells_config_and_seed_and_the_logged_observations(monkeypatch):
    seen = []
    update_state = MDAAgent.update_state

    def spy(self, x, y):
        seen.append((self.config, self.seed, x.tolist(), y))
        return update_state(self, x, y)

    monkeypatch.setattr(MDAAgent, "update_state", spy)
    log = run_cell(C0, 3, TINY)
    assert len(seen) == 2
    assert all(config is TINY and seed == 3 for config, seed, _, _ in seen)
    assert [(x, y) for _, _, x, y in seen] == [(s.x, s.y) for s in log.steps[1:]]


def test_n_active_sets_the_number_of_updates():
    assert [s.budget for s in tiny(C0, 1).steps] == [3, 4, 5]
    assert [s.budget for s in run_cell(C0, 0, replace(TINY, n_active=0)).steps] == [3]


def test_random_design_arm_is_recorded():
    log = run_cell(C0, 0, replace(TINY, design="random"))
    assert [s.design for s in log.steps] == [None, "random", "random"]
    assert log.config["design"] == "random"


def test_proposals_and_their_llm_calls_are_logged():
    # No seed law fits this world within tau_e, so every update expands
    log = expanding()
    assert log.proposer == "OneLawPerCall"
    assert all(s.e_star is not None and s.e_star > DEFAULTS.tau_e for s in log.steps[1:])
    assert [s.llm_calls for s in log.steps] == [
        [CallRecord("init/call1", "req_1")],
        [CallRecord("expand/call2", "req_2")],
        [CallRecord("expand/call3", "req_3")],
        [CallRecord("expand/call4", "req_4")],
    ]
    assert [s.n_added for s in log.steps] == [1, 1, 1, 1]
    assert [s.expanded for s in log.steps] == [True, True, True, True]
    assert [s.n_e for s in log.steps] == [0, 1, 2, 3]
    # The pool stays under its cap, so every law is still there at the end, with where it came
    # from and the budget at which it joined
    assert [len(s.pool) for s in log.steps] == [len(SEED_LIBRARY) + k for k in (1, 2, 3, 4)]
    added = {m.name: (m.origin, m.added_at) for m in log.steps[-1].pool}
    assert {name: at for name, at in added.items() if name.startswith("power_")} == {
        "power_1": ("init", 3),
        "power_2": ("expand", 4),
        "power_3": ("expand", 5),
        "power_4": ("expand", 6),
    }
    assert RunLog.from_json(log.to_json()) == log


def test_fallback_design_is_recorded():
    # The proposed law fits D0 as the seed it extends does, so it has a weight, and it predicts
    # nothing finite at any other temperature: V(a) is undefined for the first design, which
    # falls back to a random condition (spec §6). The observation there takes its weight away
    log = run_cell(C0, 0, TINY, OffData())
    off = {s.budget: m for s in log.steps for m in s.pool if m.name == "off_data"}
    assert off[3].weight > 0
    assert [s.design for s in log.steps] == [None, "fallback", "cma"]
    assert log.config["design"] == "cma"
    assert in_box(log.steps[1].x)
    assert off[4].weight == 0.0 and off[4].rss == off[4].bic == off[4].fit_err == math.inf
    assert RunLog.from_json(log.to_json()) == log


def test_world_with_all_zero_passive_rates_runs():
    log = tiny(ZERO_D0, 0)
    assert log.y0 == [0.0, 0.0, 0.0]
    assert [s.budget for s in log.steps] == [3, 4, 5]
    assert all(s.map is not None for s in log.steps)
    assert RunLog.from_json(log.to_json()) == log


# The JSON format


def log_with(rmsle_: float, nmsle_: float, rss: float = 1.0, e_star: float | None = 0.5) -> RunLog:
    model = ModelRecord("m", "k * C_A", ["k"], [2.0], rss, 3.5, 1.0, 0.25, "seed", 3)
    step = StepRecord(
        budget=4,
        x=[1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0],
        y=0.5,
        design="fallback",
        e_star=e_star,
        n_added=0,
        n_e=1,
        cap=24,
        map=None,
        pool=[model],
        llm_calls=[CallRecord("a/b"), CallRecord("a/c", "req_1")],
        eval=Evaluation(rmsle_, nmsle_, False),
    )
    return RunLog("w", 7, "P", {"tau_e": 0.05, "n_e_max": None}, [[1.0] * 7], [0.25], [step])


def test_non_finite_numbers_are_written_as_strings():
    log = log_with(math.inf, -math.inf, rss=math.inf, e_star=math.inf)
    text = log.to_json()
    d = strict_loads(text)
    step = d["steps"][0]
    assert (step["eval"]["rmsle"], step["eval"]["nmsle"]) == ("inf", "-inf")
    assert (step["pool"][0]["rss"], step["e_star"]) == ("inf", "inf")
    assert RunLog.from_json(text) == log
    nan = RunLog.from_json(log_with(math.nan, 1.0).to_json())
    assert math.isnan(nan.steps[0].eval.rmsle) and nan.steps[0].eval.nmsle == 1.0


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_every_number_is_read_back_when_it_is_not_finite(value):
    model = ModelRecord("m", "k * C_A", ["k"], [value], value, value, value, value, "seed", 3)
    step = StepRecord(
        budget=4,
        x=[value] * 7,
        y=value,
        design="cma",
        e_star=value,
        n_added=0,
        n_e=0,
        cap=24,
        map=0,
        pool=[model],
        llm_calls=[],
        eval=Evaluation(value, value, False),
    )
    log = RunLog("w", 7, "P", {}, [[value] * 7], [value], [step])
    text = log.to_json()
    strict_loads(text)
    assert RunLog.from_json(text) == log


def test_json_keeps_none_apart_from_numbers():
    log = log_with(0.1, 0.2, e_star=None)
    d = strict_loads(log.to_json())
    assert d["steps"][0]["e_star"] is None and d["steps"][0]["map"] is None
    assert d["steps"][0]["llm_calls"] == [
        {"call_id": "a/b", "request_id": None},
        {"call_id": "a/c", "request_id": "req_1"},
    ]
    assert d["config"] == {"tau_e": 0.05, "n_e_max": None}
    assert RunLog.from_json(log.to_json()) == log


def test_numbers_read_back_as_floats():
    # JSON has one number type: a rate of 1.0 written by another tool as 1 is still a float
    d = json.loads(log_with(0.1, 0.2).to_json())
    d["steps"][0]["y"] = 1
    d["steps"][0]["pool"][0]["theta"] = [2]
    d["y0"] = [0]
    log = RunLog.from_json(json.dumps(d))
    assert (log.steps[0].y, log.steps[0].pool[0].theta, log.y0) == (1.0, [2.0], [0.0])
    assert all(
        isinstance(v, float) for v in (log.steps[0].y, log.steps[0].pool[0].theta[0], log.y0[0])
    )


def test_unknown_schema_is_rejected():
    d = json.loads(log_with(0.1, 0.2).to_json())
    d["schema"] = SCHEMA + 1
    with pytest.raises(ValueError, match="schema"):
        RunLog.from_json(json.dumps(d))
    del d["schema"]
    with pytest.raises(ValueError, match="schema"):
        RunLog.from_json(json.dumps(d))
