import math
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mda.agent import MDAAgent, StepInfo
from mda.bench import BOUNDS, INPUT_NAMES, from_unit
from mda.config import DEFAULTS, MDAConfig, Space
from mda.design import Predictor, cma_design, random_design
from mda.fit import FitResult, relative_residuals
from mda.hypothesis import Hypothesis, HypothesisError, Param, compile_hypothesis
from mda.library import SEED_LIBRARY
from mda.pool import Candidate, Origin, Pool
from mda.posterior import bic_fixed, bic_profiled
from mda.posterior import weights as bic_weights
from mda.propose import CallRecord, NullProposer, Proposal

MM = Hypothesis(
    "michaelis_menten",
    "Vmax * C_A / (Km + C_A)",
    (Param("Vmax", 1e-3, 1e3), Param("Km", 1e-3, 1e3)),
)
HILL = Hypothesis(
    "hill",
    "Vmax * C_A**n / (K**n + C_A**n)",
    (Param("Vmax", 1e-3, 1e3), Param("K", 1e-3, 1e3), Param("n", 0.1, 8.0)),
)
LINEAR = Hypothesis("linear", "k * C_A", (Param("k", 1e-3, 1e3),))
CONST = Hypothesis("const", "c + 0 * C_A", (Param("c", 1e-3, 1e3),))
# Non-finite everywhere: C_I is within [0, 50], so the argument of the log is negative
BROKEN = Hypothesis("broken", "k * log(C_I - 100)", (Param("k", 0.1, 10.0),))
# Finite at T = 310, the temperature of D0, and at no other
OFF_DATA = Hypothesis("off_data", "k * C_A + sqrt(0 - (T - 310) ** 2)", (Param("k", 1e-3, 1e3),))
MM_TRUE = np.array([2.5, 0.7])
CA, TEMP = INPUT_NAMES.index("C_A"), INPUT_NAMES.index("T")

# Passive data D0 (spec §1): C_A in {0.03, 0.3, 3}, other inputs at (0, 1, 0, 1, 310, 7)
D0 = np.array([[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)])

# Fewer starts and design evaluations than the defaults, to keep the tests fast
FAST = replace(DEFAULTS, n_starts=2, n_evals=9)

# The generators of a step are seeded by (cell seed, step, stream), with one stream for the fits
# and one for the design (D8). The stream numbers are written out here because a change to them
# changes every run
FIT_STREAM, DESIGN_STREAM = 1, 2


# D0 moved far below Km, where Michaelis-Menten is close to its tangent at the origin, and one
# condition at saturation, where the two differ by a factor of about 70
LOW = D0.copy()
LOW[:, CA] = [0.01, 0.02, 0.03]
SATURATED = D0[0].copy()
SATURATED[CA] = 50.0


def tangent(X: np.ndarray) -> np.ndarray:
    """Rates exactly linear in C_A, with the slope Vmax / Km of `mm` at the origin."""
    return MM_TRUE[0] / MM_TRUE[1] * X[:, CA]


def mm(X: np.ndarray) -> np.ndarray:
    """The noise-free rates of the Michaelis-Menten law the test data follow."""
    return compile_hypothesis(MM)(MM_TRUE, X)


# The rates of `mm` on D0 with a relative error of a few percent, so that fits are not exact
NOISY_Y0 = mm(D0) * np.array([1.01, 0.99, 1.01])


def conditions(n: int, seed: int = 0) -> np.ndarray:
    return from_unit(np.random.default_rng(seed).uniform(size=(n, len(INPUT_NAMES))))


def power_laws(*exponents: float) -> list[Hypothesis]:
    """Distinct one-parameter laws, none of which fits Michaelis-Menten data."""
    return [
        Hypothesis(f"power_{p}", f"k * C_A ** {p}", (Param("k", 1e-3, 1e3),)) for p in exponents
    ]


class FakeProposer:
    """Returns scripted proposals and records what it was asked.

    `init` is the proposal of the initialisation call, and `expand[i]` that of the i-th expansion
    call; once `expand` runs out, expansion calls return nothing.
    """

    def __init__(self, init=(), expand=()):
        self.init = tuple(init)
        self.expand = [tuple(batch) for batch in expand]
        self.asked: list[SimpleNamespace] = []

    def propose(self, kind, pool, X, y, *, weights, budget_left) -> Proposal:
        self.asked.append(
            SimpleNamespace(
                kind=kind,
                pool=pool,
                X=X,
                y=y,
                names=[c.hypothesis.name for c in pool],
                fitted=[c.fit is not None for c in pool],
                n=len(y),
                weights=None if weights is None else weights.copy(),
                budget_left=budget_left,
            )
        )
        if kind == "init":
            hypotheses = self.init
        else:
            i = sum(a.kind == "expand" for a in self.asked) - 1
            hypotheses = self.expand[i] if i < len(self.expand) else ()
        return Proposal(
            hypotheses, (CallRecord(f"call{len(self.asked)}", f"req{len(self.asked)}"),)
        )

    @property
    def expansions(self) -> list[SimpleNamespace]:
        return [a for a in self.asked if a.kind == "expand"]


def agent_on_d0(
    library=(MM,), proposer=None, config: MDAConfig = FAST, seed: int = 0, y0=None
) -> MDAAgent:
    return MDAAgent(
        D0,
        mm(D0) if y0 is None else y0,
        seed=seed,
        config=config,
        proposer=proposer,
        library=library,
    )


def feed(agent: MDAAgent, n: int, seed: int = 1) -> None:
    """`n` updates at fixed random conditions, observing `mm` without noise."""
    for x in conditions(n, seed):
        agent.update_state(x, float(mm(x[None, :])[0]))


def e_star(agent: MDAAgent) -> float:
    """The predictive-check error of the last update."""
    assert agent.info.e_star is not None
    return agent.info.e_star


def names(agent: MDAAgent) -> list[str]:
    return [c.hypothesis.name for c in agent.pool]


def fitted(c: Candidate) -> FitResult:
    assert c.fit is not None
    return c.fit


def rss(pool: Sequence[Candidate]) -> np.ndarray:
    return np.array([fitted(c).rss for c in pool])


def n_free(pool: Sequence[Candidate]) -> np.ndarray:
    return np.array([c.n_free for c in pool])


def predictors(agent: MDAAgent) -> list[Predictor]:
    """The plug-in predictor f(x; theta hat) of each candidate, in pool order."""
    return [lambda X, c=c: c.f(fitted(c).theta, X) for c in agent.pool]


def design_rng(agent: MDAAgent) -> np.random.Generator:
    """The generator of the agent's next design (D8)."""
    return np.random.default_rng([agent.seed, agent.step, DESIGN_STREAM])


def in_box(x: np.ndarray) -> bool:
    return bool(np.all(x >= BOUNDS[:, 0]) and np.all(x <= BOUNDS[:, 1]))


@pytest.fixture
def refits(monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    """Every `Pool.refit` call made during the test: its number of observations and settings."""
    seen: list[SimpleNamespace] = []
    refit = Pool.refit

    def spy(self, X, y, **kwargs):
        seen.append(SimpleNamespace(n=len(y), settings=kwargs))
        return refit(self, X, y, **kwargs)

    monkeypatch.setattr(Pool, "refit", spy)
    return seen


# Initialisation (Alg. 11)


def test_default_library_is_the_seed_library_and_default_proposer_proposes_nothing():
    agent = MDAAgent(D0, mm(D0), seed=0, config=FAST)
    assert [c.hypothesis for c in agent.pool] == list(SEED_LIBRARY)
    assert isinstance(agent.proposer, NullProposer)
    assert all(c.origin is Origin.SEED and c.added_at == 3 for c in agent.pool)
    assert agent.info == StepInfo(None, 0, ())
    assert (agent.step, agent.n_e, agent.cap) == (0, 0, FAST.n_max)


def test_init_adds_the_proposal_to_the_library():
    twin = Hypothesis("twin", "Vmax*C_A/(Km+C_A)", MM.params)  # the library's MM, reformatted
    proposer = FakeProposer(init=[HILL, twin, LINEAR])
    agent = agent_on_d0(library=(MM,), proposer=proposer)
    assert names(agent) == ["michaelis_menten", "hill", "linear"]
    assert [c.origin for c in agent.pool] == [Origin.SEED, Origin.INIT, Origin.INIT]
    assert [c.added_at for c in agent.pool] == [3, 3, 3]
    # One init call: it saw the unfitted library, no weights, and the whole budget
    (asked,) = proposer.asked
    assert (asked.kind, asked.names, asked.fitted) == ("init", ["michaelis_menten"], [False])
    assert (asked.weights, asked.n, asked.budget_left) == (None, 3, FAST.n_active)
    # The duplicate is not counted; the call is recorded
    assert agent.info == StepInfo(None, 2, (CallRecord("call1", "req1"),))
    # The pool is fitted and weighted on return
    assert all(c.fit is not None and c.fit.y_hat.shape == (3,) for c in agent.pool)
    assert agent.weights.sum() == pytest.approx(1.0)


def test_init_without_a_library_uses_the_proposal_alone():
    agent = agent_on_d0(library=(), proposer=FakeProposer(init=[MM]))
    assert names(agent) == ["michaelis_menten"]
    assert agent.pool[0].origin is Origin.INIT


def test_empty_initial_pool_raises():
    with pytest.raises(ValueError, match="initial pool is empty"):
        agent_on_d0(library=())
    with pytest.raises(ValueError, match="initial pool is empty"):
        agent_on_d0(library=(), proposer=FakeProposer())


def test_init_prunes_to_n_max_and_leaves_the_adaptive_cap_to_updates():
    # tau_c = 0 and an infinite tau_fit would drop the cap to n_min at the first chance
    config = replace(FAST, n_max=2, n_min=1, tau_c=0.0, tau_fit=math.inf)
    agent = agent_on_d0(library=(MM, LINEAR, CONST), config=config)
    assert len(agent.pool) == 2 and "michaelis_menten" in names(agent)
    assert agent.cap == 2


def test_a_proposal_that_does_not_compile_is_the_proposers_error():
    bad = Hypothesis("bad", "k * unknown", (Param("k", 0.1, 1.0),))
    with pytest.raises(HypothesisError):
        agent_on_d0(proposer=FakeProposer(init=[bad]))


class ConfiguredProposer(FakeProposer):
    """A fake proposer that holds a config, as a proposer reading hyperparameters does."""

    def __init__(self, config, init=()):
        super().__init__(init=init)
        self.config = config


def test_proposer_holding_another_config_is_refused():
    # A cell has one config, the one the run log records
    proposer = ConfiguredProposer(replace(FAST, n_new=2, tau_fit=0.1), init=[HILL])
    with pytest.raises(ValueError, match=r"config differs .*\['n_new', 'tau_fit'\]"):
        agent_on_d0(proposer=proposer)
    assert proposer.asked == []  # refused before the initialisation call
    # An equal config is the same config, whichever object holds it
    same = ConfiguredProposer(replace(FAST), init=[HILL])
    assert names(agent_on_d0(proposer=same)) == ["michaelis_menten", "hill"]
    assert len(same.asked) == 1
    # A `config` that is not an `MDAConfig` is the proposer's own business
    other = ConfiguredProposer({"n_new": 2}, init=[HILL])
    assert names(agent_on_d0(proposer=other)) == ["michaelis_menten", "hill"]


# update_state: data and scoring (Alg. 2, Alg. 9)


def test_update_appends_the_observation_and_refits_on_all_data():
    agent = agent_on_d0(library=(MM, LINEAR))
    X = conditions(4)
    for k, x in enumerate(X, start=1):
        agent.update_state(x, float(mm(x[None, :])[0]))
        assert agent.step == k
        assert agent.X.shape == (3 + k, 7) and agent.y.shape == (3 + k,)
        np.testing.assert_array_equal(agent.X[-1], x)
        assert all(c.fit is not None and c.fit.y_hat.shape == (3 + k,) for c in agent.pool)
    np.testing.assert_array_equal(agent.X, np.vstack([D0, X]))
    np.testing.assert_allclose(agent.y, mm(agent.X))
    # Noise-free Michaelis-Menten data: the law is recovered and takes all the weight
    best = agent.get_model()
    assert best.hypothesis is MM
    assert best.fit is not None
    np.testing.assert_allclose(best.fit.theta, MM_TRUE, rtol=1e-6)
    assert agent.weights[agent.map_index] == pytest.approx(1.0)


def test_weights_and_bic_are_in_pool_order():
    agent = agent_on_d0(library=(LINEAR, MM, CONST))
    feed(agent, 3)
    n = len(agent.y)
    assert agent.weights.shape == agent.bic.shape == agent.log_weights.shape == (len(agent.pool),)
    np.testing.assert_allclose(agent.bic, bic_profiled(rss(agent.pool), n_free(agent.pool), n))
    np.testing.assert_allclose(agent.weights, np.exp(agent.log_weights))
    assert agent.weights.sum() == pytest.approx(1.0)
    assert (
        agent.map_index == int(np.argmax(agent.weights)) == names(agent).index("michaelis_menten")
    )


def test_bic_kind_and_structure_prior_come_from_the_config():
    fixed = agent_on_d0(library=(LINEAR, MM, CONST), config=replace(FAST, bic="fixed"))
    feed(fixed, 3)
    expected = bic_fixed(rss(fixed.pool), n_free(fixed.pool), len(fixed.y))
    np.testing.assert_allclose(fixed.bic, expected)
    # The Hill law fits the three noisy points exactly and Michaelis-Menten nearly so: under the
    # fixed-scale BIC the penalty decides for the smaller law. A prior that favours parameters
    # turns the order around
    config = replace(FAST, bic="fixed")
    plain = agent_on_d0(library=(MM, HILL), config=config, y0=NOISY_Y0)
    prior = agent_on_d0(library=(MM, HILL), config=replace(config, lam=-50.0), y0=NOISY_Y0)
    np.testing.assert_allclose(prior.bic, plain.bic, rtol=1e-6)
    assert plain.get_model().hypothesis is MM
    assert prior.get_model().hypothesis is HILL
    expected = plain.log_weights + 50.0 * n_free(plain.pool)
    np.testing.assert_allclose(
        prior.log_weights, expected - np.logaddexp.reduce(expected), rtol=1e-6, atol=1e-6
    )


def test_rss_floor_comes_from_the_config():
    # Both laws fit noise-free Michaelis-Menten data exactly, so RSS / N is at its floor and only
    # the penalty tells them apart (D2)
    floor = 1e-6
    agent = agent_on_d0(library=(MM, HILL), config=replace(FAST, rss_floor=floor))
    assert np.all(rss(agent.pool) / 3 < DEFAULTS.rss_floor)
    np.testing.assert_allclose(agent.bic, 3 * math.log(floor) + n_free(agent.pool) * math.log(3))
    assert agent.get_model().hypothesis is MM


def test_update_refits_once_and_again_after_each_change_to_the_pool(refits):
    # The proposer's first answer is empty and its second supplies the law that fits
    proposer = FakeProposer(expand=[[], [MM]])
    agent = agent_on_d0(library=(LINEAR, CONST), proposer=proposer, config=replace(FAST, n_min=1))
    assert [r.n for r in refits] == [3]
    # A round that adds nothing leaves the fits as they are (spec §5)
    feed(agent, 1)
    assert (agent.n_e, agent.info.n_added) == (1, 0)
    assert [r.n for r in refits] == [3, 4]
    # One refit on the new observation, one with the added law, and one of the law that is left
    # once the cap has dropped (spec §4)
    feed(agent, 1, seed=2)
    assert (agent.n_e, agent.info.n_added, agent.cap) == (2, 1, 1)
    assert names(agent) == ["michaelis_menten"]
    assert [r.n for r in refits] == [3, 4, 5, 5, 5]
    # Nothing to change: the law fits and the pool is at its cap
    feed(agent, 1, seed=3)
    assert agent.n_e == 2
    assert [r.n for r in refits] == [3, 4, 5, 5, 5, 6]


def test_fit_settings_reach_the_pool(refits):
    config = replace(
        FAST,
        n_starts=3,
        residual="log1p",
        rel_floor=0.02,
        rho=0.05,
        sigma_floor=0.02,
        log_span=10.0,
    )
    agent = agent_on_d0(config=config)
    feed(agent, 1)
    assert len(refits) == 2  # the initialisation and the update
    for r in refits:
        rng = r.settings.pop("rng")
        assert isinstance(rng, np.random.Generator)
        assert r.settings == {
            "n_starts": 3,
            "residual": "log1p",
            "rel_floor": 0.02,
            "rho": 0.05,
            "sigma_floor": 0.02,
            "log_span": 10.0,
        }


# Expansion (Alg. 2, Alg. 12)


def test_null_proposer_never_expands():
    # The linear law misfits throughout, so every update asks for candidates and gets none
    agent = agent_on_d0(library=(LINEAR,))
    for x in conditions(4):
        agent.update_state(x, float(mm(x[None, :])[0]))
        assert e_star(agent) > FAST.tau_e
        assert (agent.info.n_added, agent.info.calls) == (0, ())
        assert names(agent) == ["linear"]
    assert agent.pool[0].origin is Origin.SEED
    assert agent.n_e == 4  # the rounds are counted, although they added nothing


def test_pool_expands_exactly_when_the_error_exceeds_tau_e():
    # The linear law misfits until the proposer supplies Michaelis-Menten, in its second answer
    proposer = FakeProposer(expand=[[CONST], [MM]])
    agent = agent_on_d0(library=(LINEAR,), proposer=proposer)
    asked_at = []
    for k, x in enumerate(conditions(8), start=1):
        before = len(proposer.expansions)
        agent.update_state(x, float(mm(x[None, :])[0]))
        asked = len(proposer.expansions) > before
        assert asked == (e_star(agent) > FAST.tau_e)
        if asked:
            asked_at.append(k)
    assert asked_at == [1, 2]  # no more once a law fits
    assert agent.n_e == 2
    assert agent.get_model().hypothesis is MM
    by_name = {c.hypothesis.name: c for c in agent.pool}
    assert by_name["michaelis_menten"].origin is Origin.EXPAND
    assert by_name["michaelis_menten"].added_at == 5  # the budget B of the second update


def test_tau_e_comes_from_the_config_and_an_error_equal_to_it_does_not_expand():
    def first_update(tau_e: float) -> tuple[float, int]:
        """e* of the first update and the number of expansion calls it made."""
        proposer = FakeProposer()
        config = replace(FAST, tau_e=tau_e)
        agent = agent_on_d0(library=(LINEAR,), proposer=proposer, config=config)
        feed(agent, 1)
        return e_star(agent), len(proposer.expansions)

    # The linear law misses Michaelis-Menten data by far more than the default threshold
    e, asked = first_update(FAST.tau_e)
    assert e > 10 * FAST.tau_e and asked == 1
    # Alg. 2 expands when e* > tau_e
    assert first_update(e) == (e, 0)
    assert first_update(math.nextafter(e, 0.0)) == (e, 1)


def test_expansion_call_sees_the_fitted_pool_and_the_remaining_budget():
    proposer = FakeProposer(expand=[[MM]])
    agent = agent_on_d0(
        library=(LINEAR, CONST), proposer=proposer, config=replace(FAST, n_active=5)
    )
    feed(agent, 1)
    asked = proposer.expansions[0]
    assert (asked.n, asked.budget_left) == (4, 4)  # after the first of 5 experiments
    assert asked.names == ["linear", "const"] and asked.fitted == [True, True]
    # The fits and the weights are those of the data it is given, the new observation included
    assert [fitted(c).y_hat.shape for c in asked.pool] == [(4,), (4,)]
    shown = bic_weights(bic_profiled(rss(asked.pool), n_free(asked.pool), 4))
    np.testing.assert_allclose(asked.weights, shown)
    assert asked.weights.sum() == pytest.approx(1.0)
    assert agent.info.n_added == 1
    assert agent.info.calls == (CallRecord("call2", "req2"),)
    # The new candidate is fitted and weighted before the update returns
    assert names(agent) == ["linear", "const", "michaelis_menten"]
    assert agent.weights.shape == (3,) and agent.get_model().hypothesis is MM


def test_budget_left_does_not_go_negative():
    proposer = FakeProposer()
    agent = agent_on_d0(library=(LINEAR,), proposer=proposer, config=replace(FAST, n_active=1))
    feed(agent, 3)
    assert [a.budget_left for a in proposer.expansions] == [0, 0, 0]


def test_duplicate_proposals_add_nothing():
    proposer = FakeProposer(expand=[[LINEAR, Hypothesis("again", "k*C_A", LINEAR.params)]])
    agent = agent_on_d0(library=(LINEAR,), proposer=proposer)
    feed(agent, 1)
    assert names(agent) == ["linear"]
    assert (agent.info.n_added, agent.n_e) == (0, 1)
    assert len(agent.info.calls) == 1  # the call is recorded even so


@pytest.mark.parametrize("r_m", [0, 1, 2, 3])
def test_r_m_bounds_the_rounds_of_one_update(r_m):
    # Nothing proposed ever fits, so every allowed round runs
    proposer = FakeProposer(expand=[[h] for h in power_laws(2, 3, 4, 5, 6, 7, 8, 9, 10)])
    agent = agent_on_d0(library=(LINEAR,), proposer=proposer, config=replace(FAST, r_m=r_m))
    feed(agent, 3)
    assert [a.n for a in proposer.expansions] == [n for n in (4, 5, 6) for _ in range(r_m)]
    assert agent.n_e == 3 * r_m
    assert len(agent.pool) == 1 + 3 * r_m


def test_rounds_stop_once_the_refitted_map_model_fits():
    # R_m = 3, but the second round supplies Michaelis-Menten and the in-sample check passes
    proposer = FakeProposer(expand=[power_laws(2), [MM], power_laws(3)])
    agent = agent_on_d0(library=(LINEAR,), proposer=proposer, config=replace(FAST, r_m=3))
    feed(agent, 1)
    assert [a.n for a in proposer.expansions] == [4, 4]
    assert agent.info.n_added == 2 and len(agent.info.calls) == 2
    assert e_star(agent) > FAST.tau_e  # the gating value
    feed(agent, 2, seed=2)
    assert len(proposer.expansions) == 2


@pytest.mark.parametrize("n_e_max", [0, 1, 3])
def test_n_e_max_caps_the_rounds_of_a_cell(n_e_max):
    proposer = FakeProposer()
    config = replace(FAST, r_m=2, n_e_max=n_e_max)
    agent = agent_on_d0(library=(LINEAR,), proposer=proposer, config=config)
    feed(agent, 4)
    assert len(proposer.expansions) == agent.n_e == n_e_max
    assert e_star(agent) > FAST.tau_e  # still misfitting


def test_expansion_respects_the_pool_cap():
    # Each expansion offers 6 laws to a pool capped at 4. None fits, so every update expands
    laws = power_laws(*np.arange(2, 20).tolist())
    proposer = FakeProposer(expand=[laws[:6], laws[6:12], laws[12:]])
    agent = agent_on_d0(
        library=(LINEAR,), proposer=proposer, config=replace(FAST, n_max=4, n_min=2)
    )
    (x,) = conditions(1)
    agent.update_state(x, float(mm(x[None, :])[0]))
    assert agent.info.n_added == 6 and len(agent.pool) == 4
    # The 4 kept are those with the largest weights among the old and the new
    everyone = Pool()
    everyone.add([LINEAR, *laws[:6]], Origin.SEED, added_at=3)
    everyone.refit(agent.X, agent.y, rng=np.random.default_rng(0))
    bic = bic_profiled(everyone.rss, everyone.n_free, len(agent.y))
    best = sorted(np.argsort(bic)[:4].tolist())
    assert names(agent) == [everyone[i].hypothesis.name for i in best]
    for seed in (2, 3):
        feed(agent, 1, seed=seed)
        assert agent.info.n_added == 6 and len(agent.pool) == 4
        assert agent.weights.shape == (4,)
    assert agent.cap == 4


# The predictive check (Alg. 13, D5)


@pytest.mark.parametrize("library", [(LINEAR, CONST), (LINEAR, CONST, MM)], ids=["misfit", "fit"])
def test_cached_check_is_the_in_sample_error_of_the_preceding_fit(library):
    # Noisy rates, so that no fit is exact. With Michaelis-Menten in the library the cap drops,
    # and the error carried to the next update is that of the refit which follows the pruning
    agent = agent_on_d0(library=library, config=replace(FAST, n_min=2), y0=NOISY_Y0)
    X = conditions(5, seed=3)
    y = mm(X) * np.random.default_rng(0).normal(1.0, 0.02, size=5)
    for x, y_x in zip(X, y.tolist(), strict=True):
        expected = agent.get_model().fit_err
        agent.update_state(x, y_x)
        assert agent.info.e_star == expected
    assert (agent.cap == 2) == (MM in library)


def test_cached_check_reacts_to_a_surprise_one_update_later():
    # A linear law fits the low-C_A data; the first observation at saturation contradicts it
    proposer = FakeProposer()
    agent = MDAAgent(LOW, tangent(LOW), seed=0, config=FAST, proposer=proposer, library=(LINEAR,))
    agent.update_state(SATURATED, float(mm(SATURATED[None, :])[0]))
    # The check only knows the fit made before the observation arrived
    assert agent.info.e_star == pytest.approx(0.0, abs=1e-9)
    assert proposer.expansions == []
    (x,) = conditions(1)
    agent.update_state(x, float(mm(x[None, :])[0]))
    assert e_star(agent) > FAST.tau_e
    assert [a.n for a in proposer.expansions] == [5]


def test_prequential_check_without_backtest_is_the_error_at_the_new_point():
    # The same surprise as above now triggers the expansion at once
    config = replace(FAST, check="prequential", backtest=False)
    proposer = FakeProposer()
    agent = MDAAgent(LOW, tangent(LOW), seed=0, config=config, proposer=proposer, library=(LINEAR,))
    y = float(mm(SATURATED[None, :])[0])
    predicted = agent.get_predictor()(SATURATED[None, :])
    agent.update_state(SATURATED, y)
    expected = relative_residuals(np.array([y]), predicted, FAST.rel_floor)[0]
    assert agent.info.e_star == pytest.approx(expected)
    assert e_star(agent) > 10.0  # the linear law overshoots the saturated rate many times
    assert [a.n for a in proposer.expansions] == [4]


def test_prequential_check_with_backtest_is_the_median_over_old_and_new():
    # A floor of 1 is above the low-C_A rates, so it enters the stored errors
    config = replace(FAST, check="prequential", rel_floor=1.0)
    agent = agent_on_d0(library=(LINEAR,), config=config)
    for x in conditions(3):
        y = float(mm(x[None, :])[0])
        best = agent.get_model()
        assert best.fit is not None
        stored = np.abs(agent.y - best.fit.y_hat) / np.maximum(np.abs(agent.y), 1.0)
        new = abs(y - best.f(best.fit.theta, x[None, :])[0]) / max(abs(y), 1.0)
        agent.update_state(x, y)
        assert agent.info.e_star == pytest.approx(float(np.median([*stored, new])))


def test_prequential_check_uses_the_configured_floor():
    # The saturated rate is about 2.5: a floor of 10 replaces it in the denominator
    config = replace(FAST, check="prequential", backtest=False, rel_floor=10.0)
    agent = MDAAgent(LOW, tangent(LOW), seed=0, config=config, library=(LINEAR,))
    y = float(mm(SATURATED[None, :])[0])
    predicted = float(agent.get_predictor()(SATURATED[None, :])[0])
    agent.update_state(SATURATED, y)
    assert y < 10.0
    assert agent.info.e_star == pytest.approx(abs(y - predicted) / 10.0)


def test_backtest_makes_one_surprise_a_minority():
    # With three stored points that fit, the median ignores the one new point that does not
    config = replace(FAST, check="prequential", backtest=True)
    proposer = FakeProposer()
    agent = MDAAgent(LOW, tangent(LOW), seed=0, config=config, proposer=proposer, library=(LINEAR,))
    agent.update_state(SATURATED, float(mm(SATURATED[None, :])[0]))
    assert agent.info.e_star == pytest.approx(0.0, abs=1e-9)
    assert proposer.expansions == []


# The adaptive cap (Alg. 10)


def test_cap_drops_once_the_fit_is_adequate_and_the_best_models_are_kept():
    config = replace(FAST, n_min=2)
    agent = agent_on_d0(library=(LINEAR, CONST, MM), config=config)
    assert agent.cap == config.n_max and len(agent.pool) == 3  # not applied at initialisation
    # An agent that keeps all three shows their order: the constant law has the smallest weight
    everyone = agent_on_d0(library=(LINEAR, CONST, MM), config=replace(FAST, n_min=3))
    for a in (agent, everyone):
        feed(a, 1)
    assert names(everyone)[int(np.argmin(everyone.log_weights))] == "const"
    assert agent.cap == 2 and names(agent) == ["linear", "michaelis_menten"]
    assert agent.weights.max() > config.tau_c
    assert agent.get_model().fit_err <= config.tau_fit
    # The survivors are refitted and reweighted
    assert [fitted(c).y_hat.shape for c in agent.pool] == [(4,), (4,)]
    np.testing.assert_allclose(agent.bic, bic_profiled(rss(agent.pool), n_free(agent.pool), 4))
    np.testing.assert_allclose(agent.weights, bic_weights(agent.bic))
    feed(agent, 2, seed=2)
    assert agent.cap == 2 and names(agent) == ["linear", "michaelis_menten"]


def test_cap_is_chosen_on_the_pool_as_the_update_leaves_it():
    # The library misfits until the expansion of this very update supplies Michaelis-Menten. The
    # cap follows the fit made after it, not the error that gated the expansion
    proposer = FakeProposer(expand=[[MM]])
    config = replace(FAST, n_min=1)
    agent = agent_on_d0(library=(LINEAR, CONST), proposer=proposer, config=config)
    feed(agent, 1)
    assert e_star(agent) > FAST.tau_fit and agent.info.n_added == 1
    assert agent.cap == 1 and names(agent) == ["michaelis_menten"]


def test_cap_needs_a_dominant_model():
    # Michaelis-Menten and the Hill law that contains it both fit exactly, and share the weight
    agent = agent_on_d0(library=(MM, HILL, LINEAR), config=replace(FAST, n_min=1))
    feed(agent, 4)
    assert agent.get_model().fit_err <= FAST.tau_fit
    assert agent.weights.max() < FAST.tau_c
    assert agent.cap == FAST.n_max and len(agent.pool) == 3
    # The largest weight never exceeds a tau_c of 1
    agent = agent_on_d0(library=(LINEAR, CONST, MM), config=replace(FAST, n_min=1, tau_c=1.0))
    feed(agent, 4)
    assert agent.get_model().fit_err <= FAST.tau_fit
    assert agent.cap == FAST.n_max and len(agent.pool) == 3


def test_cap_needs_an_adequate_fit_by_the_configured_tau_fit():
    # The linear law outweighs the constant one, here by more than a tau_c of 0.5, and misses
    # Michaelis-Menten data by far more than tau_fit
    strict = replace(FAST, n_min=1, tau_c=0.5)
    agent = agent_on_d0(library=(LINEAR, CONST), config=strict)
    feed(agent, 4)
    worst = agent.get_model().fit_err
    assert agent.weights.max() > strict.tau_c and worst > strict.tau_fit
    assert agent.cap == FAST.n_max and len(agent.pool) == 2
    # The same run under a tau_fit that the linear law meets
    lenient = replace(strict, tau_fit=1.0)
    agent = agent_on_d0(library=(LINEAR, CONST), config=lenient)
    feed(agent, 4)
    assert worst <= lenient.tau_fit
    assert agent.cap == 1 and names(agent) == ["linear"]


def test_raising_the_cap_does_not_restore_discarded_models():
    agent = agent_on_d0(library=(LINEAR, CONST, MM), config=replace(FAST, n_min=1))
    feed(agent, 1)
    assert agent.cap == 1 and names(agent) == ["michaelis_menten"]
    # Rates that follow the enzyme level, which the law does not contain: the fit is no longer
    # adequate, so the cap is n_max again
    enz = INPUT_NAMES.index("Enz")
    for x in conditions(6, seed=5):
        agent.update_state(x, 5.0 * x[enz])
    assert agent.get_model().fit_err > FAST.tau_fit
    assert agent.cap == FAST.n_max
    assert names(agent) == ["michaelis_menten"]


def test_candidates_added_after_a_cap_drop_are_not_pruned_to_the_old_cap():
    # The cap of one update does not bind the next: within an update the pool is pruned to n_max,
    # and the cap is chosen afresh at its end
    proposer = FakeProposer(expand=[power_laws(2, 3, 4)])
    config = replace(FAST, n_min=1, check="prequential", backtest=False)
    agent = agent_on_d0(library=(LINEAR, CONST, MM), proposer=proposer, config=config)
    feed(agent, 1)
    assert agent.cap == 1 and names(agent) == ["michaelis_menten"]
    enz = INPUT_NAMES.index("Enz")
    for x in conditions(6, seed=5):
        agent.update_state(x, 5.0 * x[enz])
    assert len(proposer.expansions) >= 1
    assert agent.cap == FAST.n_max
    assert len(agent.pool) == 4


# get_model and get_predictor


def test_predictor_is_the_plug_in_of_the_map_model():
    agent = agent_on_d0(library=(LINEAR, MM, CONST))
    feed(agent, 3)
    best = agent.get_model()
    assert best is agent.pool[int(np.argmax(agent.weights))]
    assert best.fit is not None
    X = conditions(20, seed=9)
    np.testing.assert_array_equal(agent.get_predictor()(X), best.f(best.fit.theta, X))
    np.testing.assert_allclose(agent.get_predictor()(X), mm(X), rtol=1e-6)


def test_map_model_follows_the_current_data():
    # Exactly linear data at first, then data only Michaelis-Menten explains: the MAP model changes
    agent = MDAAgent(LOW, tangent(LOW), seed=0, config=FAST, library=(LINEAR, MM))
    assert agent.get_model().hypothesis is LINEAR
    feed(agent, 4)
    assert agent.get_model().hypothesis is MM


def test_map_ties_go_to_the_candidate_added_first():
    # Two laws with the same predictions and as many parameters. Both fit exactly linear data
    # exactly, which the floor of RSS / N turns into an exact tie (D2)
    twin = Hypothesis("twin", "k * C_A + 0 * C_I", LINEAR.params)
    for library in ((LINEAR, twin), (twin, LINEAR)):
        agent = MDAAgent(LOW, tangent(LOW), seed=0, config=FAST, library=library)
        np.testing.assert_array_equal(agent.weights, [0.5, 0.5])
        assert agent.map_index == 0
        assert agent.get_model().hypothesis is library[0]


def test_pool_without_a_finite_fit_has_no_map_model():
    agent = agent_on_d0(library=(BROKEN,))
    assert agent.map_index is None
    np.testing.assert_array_equal(agent.weights, [0.0])
    np.testing.assert_array_equal(agent.bic, [np.inf])
    # get_model still answers, with the first candidate
    assert agent.get_model() is agent.pool[0]
    assert not np.isfinite(agent.get_predictor()(D0)).any()


def test_pool_without_a_finite_fit_still_designs_and_asks_for_candidates():
    proposer = FakeProposer()
    agent = agent_on_d0(library=(BROKEN,), proposer=proposer)
    # No law has a weight, so V(a) is 0 everywhere: defined, and CMA-ES returns a condition
    x = agent.design()
    assert agent.last_design == "cma" and in_box(x)
    # The check cannot be evaluated, which counts as a misfit and asks for candidates
    agent.update_state(x, 1.0)
    assert agent.info.e_star == math.inf
    assert len(proposer.expansions) == 1
    assert agent.cap == FAST.n_max


@pytest.mark.parametrize("check", ["cached", "prequential"])
def test_new_candidates_rescue_a_pool_without_a_finite_fit(check):
    proposer = FakeProposer(expand=[[MM]])
    agent = agent_on_d0(library=(BROKEN,), proposer=proposer, config=replace(FAST, check=check))
    feed(agent, 1)
    assert agent.info.e_star == math.inf
    assert agent.get_model().hypothesis is MM
    np.testing.assert_allclose(agent.weights, [0.0, 1.0])


# design (Alg. 14)


def test_design_is_a_condition_in_the_box_and_repeats_until_the_next_update():
    agent = agent_on_d0(library=(MM, LINEAR, CONST))
    assert agent.last_design is None
    x = agent.design()
    assert agent.last_design == "cma"
    assert x.shape == (7,) and in_box(x)
    np.testing.assert_array_equal(agent.design(), x)
    agent.update_state(x, float(mm(x[None, :])[0]))
    assert not np.array_equal(agent.design(), x)


def test_design_goes_where_the_models_disagree():
    # A linear and a saturating law agree on the low-C_A data; they part as C_A grows
    agent = MDAAgent(LOW, mm(LOW), seed=0, config=DEFAULTS, library=(LINEAR, MM))
    assert np.all(agent.weights > 0)
    assert agent.design()[CA] > 10.0


def test_design_is_the_cma_es_maximiser_of_v_for_the_pool():
    # CMA-ES on the weights and the plug-in predictors of the pool, each weight with its own
    # law's predictor, with the configured settings and the generator of the step (D8)
    config = replace(FAST, n_evals=20, sigma0=0.1, design_space="log1p")
    agent = agent_on_d0(library=(MM, HILL, LINEAR), config=config, seed=1, y0=NOISY_Y0)

    def reference(
        weights: np.ndarray, n_evals: int = 20, sigma0: float = 0.1, space: Space = "log1p"
    ) -> np.ndarray:
        """The design for the agent's current state, by `cma_design` itself."""
        rng = design_rng(agent)
        return cma_design(
            weights, predictors(agent), rng=rng, n_evals=n_evals, sigma0=sigma0, space=space
        )

    for _ in range(2):
        x = agent.design()
        np.testing.assert_array_equal(x, reference(agent.weights))
        agent.update_state(x, 1.01 * float(mm(x[None, :])[0]))
    x = agent.design()
    np.testing.assert_array_equal(x, reference(agent.weights))
    # In this state each setting changes the design, and so does the pairing of weights and
    # laws, so the equality above checks all four
    assert not np.array_equal(reference(agent.weights, n_evals=9), x)
    assert not np.array_equal(reference(agent.weights, sigma0=0.3), x)
    assert not np.array_equal(reference(agent.weights, space="raw"), x)
    assert not np.array_equal(reference(agent.weights[::-1].copy()), x)


def test_random_design_is_a_uniform_draw_that_ignores_the_models():
    agent = agent_on_d0(library=(MM, LINEAR), config=replace(FAST, design="random"), seed=4)
    xs = []
    for _ in range(5):
        x = agent.design()
        assert agent.last_design == "random" and in_box(x)
        # The first draw of the step's generator, which no search has used (D8)
        np.testing.assert_array_equal(x, random_design(design_rng(agent)))
        np.testing.assert_array_equal(agent.design(), x)
        xs.append(x)
        agent.update_state(x, float(mm(x[None, :])[0]))
    assert len({tuple(x) for x in xs}) == 5


def test_design_falls_back_to_a_random_condition_when_v_is_undefined():
    # The only law predicts nothing finite away from T = 310, so V(a) is undefined wherever
    # CMA-ES looks (spec §6)
    agent = agent_on_d0(library=(OFF_DATA,))
    np.testing.assert_array_equal(agent.weights, [1.0])
    x = agent.design()
    assert agent.last_design == "fallback"
    assert in_box(x) and x[TEMP] != 310.0
    np.testing.assert_array_equal(agent.design(), x)
    # The observation there leaves the law without a finite fit, and so without a weight: V(a)
    # is 0 everywhere, which CMA-ES can search again
    agent.update_state(x, 1.0)
    assert agent.map_index is None
    assert in_box(agent.design()) and agent.last_design == "cma"


# Public state is for reading

PUBLIC_STATE = {
    *("X", "y", "pool", "weights", "log_weights", "bic", "map_index"),
    *("cap", "n_e", "step", "info", "last_design", "seed", "config", "proposer"),
}


def test_public_state_is_read_only_properties():
    agent = agent_on_d0(library=(MM, LINEAR))
    feed(agent, 1)
    agent.design()
    # Everything public is a method or a documented property. A plain attribute, which a caller
    # could assign to, would show up among the latter
    public = {name for name in dir(agent) if not name.startswith("_")}
    methods = {name for name in public if callable(getattr(MDAAgent, name, None))}
    assert methods == {"design", "update_state", "get_model", "get_predictor"}
    assert public - methods == PUBLIC_STATE
    for name in sorted(PUBLIC_STATE):
        assert isinstance(getattr(MDAAgent, name), property)
        with pytest.raises(AttributeError):
            setattr(agent, name, getattr(agent, name))


def test_per_model_arrays_are_copies():
    agent = agent_on_d0(library=(MM, LINEAR))
    for name in ("weights", "log_weights", "bic"):
        before = getattr(agent, name).copy()
        getattr(agent, name).fill(99.0)
        np.testing.assert_array_equal(getattr(agent, name), before)


def test_data_are_read_only_copies_of_the_callers_arrays():
    X0, y0 = D0.copy(), mm(D0)
    agent = MDAAgent(X0, y0, seed=0, config=FAST, library=(MM,))
    for _ in range(2):  # after the initialisation, then after an update
        with pytest.raises(ValueError, match="read-only"):
            agent.X[0, 0] = 99.0
        with pytest.raises(ValueError, match="read-only"):
            agent.y[0] = 99.0
        feed(agent, 1)
    # The caller's arrays stay its own
    X0[:], y0[:] = 99.0, 99.0
    np.testing.assert_array_equal(agent.X[:3], D0)
    np.testing.assert_array_equal(agent.y[:3], mm(D0))


def test_pool_is_a_snapshot_of_frozen_candidates():
    agent = agent_on_d0(library=(MM, LINEAR))
    before = agent.pool
    assert isinstance(before, tuple)
    feed(agent, 1)
    # The update replaced the agent's candidates and left the snapshot as it was
    assert [fitted(c).y_hat.shape for c in before] == [(3,), (3,)]
    assert [fitted(c).y_hat.shape for c in agent.pool] == [(4,), (4,)]
    c = agent.get_model()
    with pytest.raises(AttributeError):
        setattr(c, "fit_err", 0.0)  # noqa: B010  (a plain assignment does not type-check)
    with pytest.raises(ValueError, match="read-only"):
        fitted(c).theta[0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        fitted(c).y_hat[0] = 99.0


class Meddler:
    """A proposer that proposes nothing and tries to change everything it is given."""

    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.written: list[str] = []  # the writes that were not refused

    def propose(self, kind, pool, X, y, *, weights, budget_left) -> Proposal:
        self.kinds.append(kind)
        attempts = {
            "X": lambda: X.fill(0.0),
            "y": lambda: y.fill(0.0),
            "pool": lambda: pool.keep([]),
        }
        if kind == "expand":
            weights.fill(0.0)  # a copy, so this is allowed and changes nothing
            attempts["theta"] = lambda: pool[0].fit.theta.fill(0.0)
            attempts["y_hat"] = lambda: pool[0].fit.y_hat.fill(0.0)
        for name, write in attempts.items():
            try:
                write()
            except (ValueError, AttributeError):
                continue
            self.written.append(f"{kind}: {name}")
        return Proposal()


def test_proposer_cannot_change_the_agent_through_its_arguments():
    # Neither library law fits, so every update expands and calls the proposer
    meddler = Meddler()
    meddled = agent_on_d0(library=(LINEAR, CONST), proposer=meddler)
    clean = agent_on_d0(library=(LINEAR, CONST), proposer=FakeProposer())
    for agent in (meddled, clean):
        feed(agent, 2)
    assert meddler.kinds == ["init", "expand", "expand"]
    assert meddler.written == []
    np.testing.assert_array_equal(meddled.X, clean.X)
    np.testing.assert_array_equal(meddled.y, clean.y)
    np.testing.assert_array_equal(meddled.weights, clean.weights)
    assert names(meddled) == names(clean)
    for a, b in zip(meddled.pool, clean.pool, strict=True):
        np.testing.assert_array_equal(fitted(a).theta, fitted(b).theta)
        np.testing.assert_array_equal(fitted(a).y_hat, fitted(b).y_hat)


def test_proposer_is_given_a_snapshot_and_read_only_data():
    proposer = FakeProposer(expand=[[MM]])
    agent = agent_on_d0(library=(LINEAR, CONST), proposer=proposer)
    feed(agent, 1)
    assert [a.kind for a in proposer.asked] == ["init", "expand"]
    for asked in proposer.asked:
        assert isinstance(asked.pool, tuple)
        assert not asked.X.flags.writeable and not asked.y.flags.writeable
    # The expansion added a candidate to the agent's pool, not to what the proposer was given
    assert len(proposer.expansions[0].pool) == 2 and len(agent.pool) == 3


# Reproducibility (D8)


def self_driven(seed: int, n: int = 4) -> MDAAgent:
    """An agent after `n` rounds of observing `mm` at its own designs."""
    agent = agent_on_d0(library=(MM, HILL, LINEAR, CONST), seed=seed)
    for _ in range(n):
        x = agent.design()
        agent.update_state(x, float(mm(x[None, :])[0]))
    return agent


def test_same_seed_gives_the_same_run():
    a, b = self_driven(seed=3), self_driven(seed=3)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert names(a) == names(b)
    for ca, cb in zip(a.pool, b.pool, strict=True):
        np.testing.assert_array_equal(fitted(ca).theta, fitted(cb).theta)


def test_seed_changes_the_designs():
    a, b = self_driven(seed=3), self_driven(seed=4)
    assert not np.array_equal(a.X[3:], b.X[3:])


def test_fits_are_seeded_by_cell_seed_step_and_stream():
    # At step k every candidate is fitted in pool order with one generator, seeded by
    # (seed, k, stream), and starts from its previous estimate (D8). No law is added or pruned
    # here, so a pool refitted that way stays in step with the agent's
    library = (MM, HILL, LINEAR)
    agent = agent_on_d0(library=library, seed=5, y0=NOISY_Y0)
    reference = Pool()
    reference.add(library, Origin.SEED, added_at=3)
    for step, x in enumerate([None, *conditions(2)]):
        if x is not None:
            agent.update_state(x, 1.01 * float(mm(x[None, :])[0]))
        rng = np.random.default_rng([5, step, FIT_STREAM])
        reference.refit(agent.X, agent.y, rng=rng, n_starts=FAST.n_starts)
        assert agent.step == step and names(agent) == [h.name for h in library]
        for c, expected in zip(agent.pool, reference, strict=True):
            np.testing.assert_array_equal(fitted(c).theta, fitted(expected).theta)


def test_global_random_state_is_not_used():
    np.random.seed(0)  # noqa: NPY002
    before = np.random.get_state()[1].copy()  # noqa: NPY002
    a = self_driven(seed=3, n=2)
    np.testing.assert_array_equal(np.random.get_state()[1], before)  # noqa: NPY002
    np.random.seed(1)  # noqa: NPY002
    np.testing.assert_array_equal(self_driven(seed=3, n=2).X, a.X)
