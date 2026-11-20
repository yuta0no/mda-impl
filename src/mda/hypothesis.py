import ast
import json
import keyword
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mda.arrays import Conditions, Rates, Theta
from mda.bench import INPUT_NAMES


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float


type Params = tuple[Param, ...]


@dataclass(frozen=True)
class Hypothesis:
    """Hypothesis in the LLM's output format, which is specified in `src/mda/prompts/chem_propose.txt`"""

    name: str
    expr: str
    params: Params


class HypothesisError(ValueError):
    """Raised when a hypothesis cannot be compiled."""


class ProposalParseError(ValueError):
    """Raised when an LLM reply contains no `{"hypotheses": [...]}` object at all."""


FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
}
_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.Pow: np.power,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: np.negative,
    ast.UAdd: np.positive,
}

# A compiled sub-expression: (theta, X) -> array of shape (n,) or a scalar
type _Node = Callable[[np.ndarray, np.ndarray], np.ndarray | float]


def compile_hypothesis(h: Hypothesis) -> Callable[[Theta, Conditions], Rates]:
    """Compile `h.expr` into a vectorised function `f(theta, X) -> rates`.

    Allowed syntax: numbers, `+`, `-`, `*`, `/`, `**`, unary `-`/`+`, and calls to `exp`, `log`, `sqrt`.
    Allowed names: the inputs `C_A, C_I, C_B, C_P, Enz, T, pH` (the columns of `X`) and the names
    declared in `h.params`. `theta[i]` is the value of `h.params[i]`.

    Raises `HypothesisError` for anything else, for invalid parameter declarations, and for
    declared parameters that `expr` does not use. The expression is never passed to `eval`.
    Numerical problems (division by zero, overflow, log of a negative number) do not raise:
    they return inf or nan. Whitespace, including line breaks, is ignored.
    """
    params = _check_params(h.params)
    try:
        # Collapse whitespace so line breaks outside brackets parse. Safe because the grammar has
        # no string literals, and joining with a space never merges two tokens.
        tree = ast.parse(" ".join(h.expr.split()), mode="eval")
        used: set[str] = set()
        root = _build(tree.body, {n: j for j, n in enumerate(INPUT_NAMES)}, params, used)
    except HypothesisError:
        raise
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise HypothesisError(f"cannot parse expr {h.expr!r}: {e}") from e
    if unused := [p.name for p in h.params if p.name not in used]:
        raise HypothesisError(f"declared parameters not used in expr: {unused}")
    n_params = len(h.params)

    def f(theta: Theta, X: Conditions) -> Rates:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (n_params,):
            raise ValueError(f"expected theta of shape ({n_params},), got {theta.shape}")
        with np.errstate(all="ignore"):
            y = np.asarray(root(theta, X), dtype=float)
        return np.broadcast_to(y, (X.shape[0],)).copy()

    return f


def _check_params(params: Params) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, p in enumerate(params):
        if not p.name.isidentifier() or keyword.iskeyword(p.name):
            raise HypothesisError(f"invalid parameter name {p.name!r}")
        if p.name in index:
            raise HypothesisError(f"duplicate parameter name {p.name!r}")
        if p.name in INPUT_NAMES or p.name in FUNCTIONS:
            raise HypothesisError(f"parameter name {p.name!r} clashes with an input or function")
        if not (math.isfinite(p.low) and math.isfinite(p.high) and p.low <= p.high):
            raise HypothesisError(f"invalid bounds for {p.name!r}: [{p.low}, {p.high}]")
        index[p.name] = i
    return index


def _build(node: ast.expr, inputs: dict[str, int], params: dict[str, int], used: set[str]) -> _Node:
    match node:
        case ast.Constant(value=bool()):
            pass  # bool is a subclass of int; reject it below
        case ast.Constant(value=int() | float() as value):
            c = float(value)
            return lambda theta, X: c
        case ast.Name(id=name) if name in inputs:
            j = inputs[name]
            return lambda theta, X: X[:, j]
        case ast.Name(id=name) if name in params:
            i = params[name]
            used.add(name)
            return lambda theta, X: theta[i]
        case ast.Name(id=name):
            raise HypothesisError(f"unknown name {name!r}")
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY_OPS:
            binop = _BINARY_OPS[type(op)]
            lhs, rhs = _build(left, inputs, params, used), _build(right, inputs, params, used)
            return lambda theta, X: binop(lhs(theta, X), rhs(theta, X))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            unop = _UNARY_OPS[type(op)]
            arg = _build(operand, inputs, params, used)
            return lambda theta, X: unop(arg(theta, X))
        case ast.Call(func=ast.Name(id=name), args=[arg_node], keywords=[]) if name in FUNCTIONS:
            func = FUNCTIONS[name]
            arg = _build(arg_node, inputs, params, used)
            return lambda theta, X: func(arg(theta, X))
    raise HypothesisError(f"unsupported syntax: {ast.unparse(node)!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_hypotheses(text: str) -> tuple[list[Hypothesis], list[str]]:
    """Parse an LLM reply in the format of `src/mda/prompts/chem_propose.txt`.

    Accepts bare JSON, JSON inside a ``` code fence, or JSON surrounded by prose.
    Each hypothesis is validated individually and must pass `compile_hypothesis`; invalid ones are
    dropped. Returns `(hypotheses, rejected)`, where `rejected` holds one reason per dropped entry.

    Raises `ProposalParseError` only when the reply as a whole is unusable: no JSON object can be
    decoded, or it has no `hypotheses` list. A reply whose hypotheses are all invalid returns
    `([], reasons)`; the caller decides whether to retry.
    """
    data = _decode(text)
    if not isinstance(data, dict) or not isinstance(items := data.get("hypotheses"), list):
        raise ProposalParseError('reply has no "hypotheses" list')
    hypotheses: list[Hypothesis] = []
    rejected: list[str] = []
    for k, item in enumerate(items):
        try:
            h = _to_hypothesis(item)
            compile_hypothesis(h)
        except HypothesisError as e:
            rejected.append(f"hypothesis {k}: {e}")
            continue
        hypotheses.append(h)
    return hypotheses, rejected


def _decode(text: str) -> Any:
    candidates = [text, *_FENCE.findall(text)]
    if (start := text.find("{")) != -1 and (end := text.rfind("}")) > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ProposalParseError(f"no JSON object found in reply: {text[:200]!r}")


def _to_hypothesis(item: Any) -> Hypothesis:
    if not isinstance(item, dict):
        raise HypothesisError(f"expected an object, got {type(item).__name__}")
    name, expr, params = item.get("name"), item.get("expr"), item.get("params")
    if not isinstance(name, str):
        raise HypothesisError('"name" must be a string')
    if not isinstance(expr, str):
        raise HypothesisError(f'{name!r}: "expr" must be a string')
    if not isinstance(params, list):
        raise HypothesisError(f'{name!r}: "params" must be a list')
    return Hypothesis(name=name, expr=expr, params=tuple(_to_param(name, p) for p in params))


def _to_param(hname: str, p: Any) -> Param:
    if not isinstance(p, dict) or not isinstance(pname := p.get("name"), str):
        raise HypothesisError(f'{hname!r}: each param must be an object with a string "name"')
    return Param(
        pname, _to_float(hname, pname, p.get("low")), _to_float(hname, pname, p.get("high"))
    )


def _to_float(hname: str, pname: str, v: Any) -> float:
    # JSON integers arrive as int, which beartype rejects for `float` fields; bool is an int subclass
    if isinstance(v, bool) or not isinstance(v, int | float | str):
        raise HypothesisError(f"{hname!r}: bound of {pname!r} must be a number, got {v!r}")
    try:
        return float(v)
    except ValueError as e:
        raise HypothesisError(f"{hname!r}: bound of {pname!r} must be a number, got {v!r}") from e
