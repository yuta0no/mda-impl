import math
import warnings

import numpy as np
import pytest

from mda.bench import World, WorldSpec
from mda.hypothesis import Hypothesis, HypothesisError, Param, compile_hypothesis
from mda.metrics import rmsle

MM = Hypothesis(
    name="michaelis_menten",
    expr="kcat * Enz * C_A / (Km + C_A)",
    params=(Param("kcat", 1e-2, 1e3), Param("Km", 1e-3, 1e3)),
)


def _X(n: int = 50) -> np.ndarray:
    return World(WorldSpec("c0_michaelis_menten")).heldout(seed=0, n=n)


def test_reproduces_world_truth():
    world = World(WorldSpec("c0_michaelis_menten"))
    p = world.oracle._law_params
    X = world.heldout(seed=42, n=1000)
    y = compile_hypothesis(MM)(np.array([p["kcat"], p["Km"]]), X)
    assert rmsle(y, world.truth(X)) < 1e-12


def test_matches_numpy():
    h = Hypothesis(
        name="all_ops",
        expr="-a * exp(-T / 300) + sqrt(C_A) ** 2 - log(Enz) / +b",
        params=(Param("a", 0.0, 1.0), Param("b", 1.0, 2.0)),
    )
    X = _X()
    C_A, Enz, T = X[:, 0], X[:, 4], X[:, 5]
    expected = -0.5 * np.exp(-T / 300) + np.sqrt(C_A) ** 2 - np.log(Enz) / 1.5
    assert np.allclose(compile_hypothesis(h)(np.array([0.5, 1.5]), X), expected)


@pytest.mark.parametrize("expr", ["2", "k", "k * 2"])
def test_constant_expression_has_shape_n(expr):
    params = (Param("k", 0.0, 10.0),) if "k" in expr else ()
    y = compile_hypothesis(Hypothesis("const", expr, params))(np.full(len(params), 3.0), _X(7))
    assert y.shape == (7,)
    y = compile_hypothesis(Hypothesis("const", expr, params))(np.full(len(params), 3.0), _X(20))
    assert y.shape == (20,)
    assert y.dtype == np.float64


def test_numerical_errors_return_nonfinite_without_warnings():
    h = Hypothesis("bad", "log(C_I - C_I) + 1 / (C_A - C_A) + exp(1000 * T)", ())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = compile_hypothesis(h)(np.array([]), _X())
    assert not np.isfinite(y).any()


def test_wrong_theta_shape_raises():
    with pytest.raises(ValueError, match="theta"):
        compile_hypothesis(MM)(np.array([1.0]), _X())


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "C_A.real",
        "(lambda: 1)()",
        "abs(C_A)",
        "exp(C_A, 2)",
        "exp(x=C_A)",
        "exp",
        "C_A if C_A > 1 else 1",
        "C_A % 2",
        "C_A // 2",
        "[C_A][0]",
        "True",
        "1j",
        "'C_A'",
        "unknown * C_A",
        "",
        "C_A +",
    ],
)
def test_rejects_unsupported_expr(expr):
    with pytest.raises(HypothesisError):
        compile_hypothesis(Hypothesis("bad", expr, ()))


@pytest.mark.parametrize(
    "params",
    [
        (Param("k", 0.0, 1.0), Param("k", 0.0, 1.0)),  # duplicate
        (Param("C_A", 0.0, 1.0),),  # clashes with an input
        (Param("exp", 0.0, 1.0),),  # clashes with a function
        (Param("1k", 0.0, 1.0),),  # not an identifier
        (Param("lambda", 0.0, 1.0),),  # keyword
        (Param("k", 2.0, 1.0),),  # low > high
        (Param("k", 0.0, math.inf),),  # non-finite bound
        (Param("k", 0.0, 1.0), Param("unused", 0.0, 1.0)),  # declared but unused
    ],
)
def test_rejects_invalid_params(params):
    expr = " + ".join(p.name for p in params[:1]) + " * C_A"
    with pytest.raises(HypothesisError):
        compile_hypothesis(Hypothesis("bad", expr, params))
