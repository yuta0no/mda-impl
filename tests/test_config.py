import inspect
import json
from dataclasses import fields, replace
from typing import TypeAliasType

import jaxtyping
import pytest

from mda.config import DEFAULTS, MDAConfig
from mda.design import between_model_variance, cma_design
from mda.fit import fit, relative_residuals, residuals, sigma
from mda.llm import ClaudeLLM
from mda.pool import Pool
from mda.posterior import adaptive_cap, bic_profiled, log_prior


def test_defaults_follow_spec():
    # Each value is the paper's or a D-default of docs/chembench-spec.md; change the spec in the
    # same PR as these
    c = MDAConfig()
    # Spec §1
    assert c.passive_c_a == (0.03, 0.3, 3.0)
    assert c.passive_rest == (0.0, 1.0, 0.0, 1.0, 310.0, 7.0)
    assert c.n_active == 22
    # Spec §3, D1, D2, D8
    assert (c.rho, c.sigma_floor, c.residual) == (0.03, 0.01, "whitened")
    assert (c.n_starts, c.log_span) == (8, 100.0)
    assert (c.bic, c.rss_floor, c.lam) == ("profiled", 1e-12, 0.0)
    # Spec §4
    assert (c.n_init, c.n_new, c.n_max, c.n_min) == (12, 6, 24, 6)
    assert (c.tau_c, c.tau_fit) == (0.9, 0.05)
    # Spec §5, D3, D4, D5
    assert (c.tau_e, c.r_m, c.n_e_max) == (0.05, 1, None)
    assert (c.check, c.backtest, c.rel_floor) == ("cached", True, 0.01)
    # Spec §6, D9, D10
    assert (c.design, c.n_evals, c.sigma0, c.design_space) == ("cma", 48, 0.3, "raw")
    # D11
    assert c.n_query == 60
    # Spec §9, D6, D14, D15
    assert (c.llm_model, c.llm_effort) == ("claude-haiku-5-5", "medium")
    assert (c.llm_max_tokens, c.llm_max_retries) == (16_000, 5)
    assert (c.max_attempts, c.structured_output, c.prompt_phase) == (3, True, True)
    assert MDAConfig() == DEFAULTS


def test_every_field_is_checked_by_the_spec_test():
    # A new field must be added to `test_defaults_follow_spec`, with its source
    source = inspect.getsource(test_defaults_follow_spec)
    assert [f.name for f in fields(MDAConfig) if f"c.{f.name}" not in source] == []


@pytest.mark.parametrize(
    ("function", "parameter", "field"),
    [
        (sigma, "rho", "rho"),
        (sigma, "sigma_floor", "sigma_floor"),
        (residuals, "residual", "residual"),
        (residuals, "rho", "rho"),
        (residuals, "sigma_floor", "sigma_floor"),
        (fit, "residual", "residual"),
        (fit, "rho", "rho"),
        (fit, "sigma_floor", "sigma_floor"),
        (fit, "log_span", "log_span"),
        (relative_residuals, "c", "rel_floor"),
        (Pool.refit, "n_starts", "n_starts"),
        (Pool.refit, "residual", "residual"),
        (Pool.refit, "rel_floor", "rel_floor"),
        (Pool.refit, "rho", "rho"),
        (Pool.refit, "sigma_floor", "sigma_floor"),
        (Pool.refit, "log_span", "log_span"),
        (bic_profiled, "rss_floor", "rss_floor"),
        (log_prior, "lam", "lam"),
        (adaptive_cap, "tau_c", "tau_c"),
        (adaptive_cap, "tau_fit", "tau_fit"),
        (adaptive_cap, "n_min", "n_min"),
        (adaptive_cap, "n_max", "n_max"),
        (between_model_variance, "space", "design_space"),
        (cma_design, "n_evals", "n_evals"),
        (cma_design, "sigma0", "sigma0"),
        (cma_design, "space", "design_space"),
        (ClaudeLLM.__init__, "model", "llm_model"),
        (ClaudeLLM.__init__, "effort", "llm_effort"),
        (ClaudeLLM.__init__, "max_tokens", "llm_max_tokens"),
        (ClaudeLLM.__init__, "max_retries", "llm_max_retries"),
    ],
    ids=lambda v: getattr(v, "__qualname__", v),
)
def test_keyword_defaults_come_from_the_config(function, parameter, field):
    # No module keeps its own copy of a hyperparameter
    # `inspect.unwrap` reaches the function under the runtime type-checking wrapper
    default = inspect.signature(inspect.unwrap(function)).parameters[parameter].default
    assert default == getattr(DEFAULTS, field)


def test_config_is_frozen_and_replaceable():
    with pytest.raises(AttributeError):
        DEFAULTS.tau_e = 0.1  # ty: ignore[invalid-assignment]
    control = replace(DEFAULTS, bic="fixed", design="random")
    assert (control.bic, control.design) == ("fixed", "random")
    assert (DEFAULTS.bic, DEFAULTS.design) == ("profiled", "cma")


def test_dict_round_trip_through_json():
    config = replace(DEFAULTS, n_e_max=4, check="prequential", passive_c_a=(0.1, 1.0))
    d = config.to_dict()
    assert d["passive_c_a"] == [0.1, 1.0]  # tuples become lists, as JSON has them
    assert set(d) == {f.name for f in fields(MDAConfig)}
    assert MDAConfig.from_dict(json.loads(json.dumps(d))) == config
    assert MDAConfig.from_dict(json.loads(json.dumps(DEFAULTS.to_dict()))) == DEFAULTS


def test_from_dict_fills_missing_fields_and_rejects_unknown_ones():
    assert MDAConfig.from_dict({"tau_e": 0.1}) == replace(DEFAULTS, tau_e=0.1)
    with pytest.raises(TypeError, match="no_such_field"):
        MDAConfig.from_dict({"no_such_field": 1})


@pytest.mark.parametrize(
    "changes",
    [
        {"rho": 0.0},
        {"sigma_floor": -0.01},
        {"rss_floor": 0.0},
        {"rel_floor": 0.0},
        {"n_evals": 0},
        {"sigma0": 0.0},
        {"n_min": 0},
        {"n_query": 0},
        {"max_attempts": 0},
        {"n_active": -1},
        {"n_starts": -1},
        {"r_m": -1},
        {"tau_e": -0.1},
        {"n_e_max": -1},
        {"n_min": 25},
        {"passive_c_a": ()},
        {"passive_rest": (0.0, 1.0)},
    ],
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()),
)
def test_invalid_values_raise(changes):
    with pytest.raises(ValueError):
        replace(DEFAULTS, **changes)


@pytest.fixture
def no_runtime_typecheck():
    """Switch off the jaxtyping/beartype hook of pytest, so `MDAConfig` sees what a run sees."""
    jaxtyping.config.update("jaxtyping_disable", True)
    yield
    jaxtyping.config.update("jaxtyping_disable", False)


OPTIONS = [f.name for f in fields(MDAConfig) if isinstance(f.type, TypeAliasType)]


def test_options_are_the_literal_fields():
    assert set(OPTIONS) == {"residual", "bic", "check", "design", "design_space", "llm_effort"}


@pytest.mark.parametrize("name", OPTIONS)
@pytest.mark.usefixtures("no_runtime_typecheck")
def test_unsupported_options_raise_at_load_time(name):
    # Annotations are not checked at runtime, so `__post_init__` checks each option against its
    # `Literal` alias: a bad manifest fails when it is loaded, not midway through a run
    with pytest.raises(ValueError, match=name):
        replace(DEFAULTS, **{name: "bogus"})
    with pytest.raises(ValueError, match=name):
        MDAConfig.from_dict({name: "bogus"})


def test_boundary_values_are_allowed():
    # No experiments, no random starts (a warm start remains), no expansion rounds, expansion
    # at any error, and a cap that does not change
    replace(DEFAULTS, n_active=0, n_starts=0, r_m=0, n_e_max=0, tau_e=0.0, n_min=24, n_max=24)
