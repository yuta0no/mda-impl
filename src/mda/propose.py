"""Proposers: where the agent's new hypotheses come from (spec §4, Alg. 11 and 12).

`MDAAgent` asks its proposer for hypotheses once at initialisation and once per expansion.
`NullProposer` proposes nothing, which gives the fixed-library arm: the seed library alone, with
no LLM call (spec §8). `LLMProposer` asks an LLM, with the prompt of App. G.1
(`prompts/chem_propose.txt`), which gives the hybrid arm.
"""

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from mda.arrays import Conditions, PerModel, Rates
from mda.bench import INPUT_NAMES
from mda.config import DEFAULTS, MDAConfig
from mda.fit import residual_input_spearman
from mda.hypothesis import (
    HYPOTHESES_SCHEMA,
    Hypothesis,
    ProposalParseError,
    normalise_expr,
    parse_hypotheses,
)
from mda.llm import ClaudeLLM, LLMClient
from mda.pool import Candidate

type ProposalKind = Literal["init", "expand"]


@dataclass(frozen=True)
class CallRecord:
    """One LLM request made for a proposal, as the run log records it."""

    call_id: str  # the request's `call_id` in `LLMClient.complete`, its key in the reply cache
    request_id: str | None = None  # the provider's ID of the request


@dataclass(frozen=True)
class Proposal:
    """What one `Proposer.propose` call returned."""

    hypotheses: tuple[Hypothesis, ...] = ()
    calls: tuple[CallRecord, ...] = ()  # every LLM request made, retries included


@runtime_checkable  # so that runtime type checkers can check a `Proposer` argument
class Proposer(Protocol):
    def propose(
        self,
        kind: ProposalKind,
        pool: Sequence[Candidate],
        X: Conditions,
        y: Rates,
        *,
        weights: PerModel | None,
        budget_left: int,
    ) -> Proposal:
        """Propose hypotheses to add to the pool, given the data (X, y) collected so far.

        `kind` is "init" for the call at initialisation (Alg. 11) and "expand" for an expansion
        (Alg. 12). `pool` is the agent's candidates, in pool order. At initialisation they are
        the seed library, not yet fitted, and `weights` is None. At an expansion they are fitted
        to (X, y) and `weights` are their model weights. `budget_left` is the number of
        experiments still to run.

        The arguments are for reading: the agent passes a snapshot of its pool, its data as
        arrays marked read-only, and a copy of its weights. A proposer changes the pool only
        through what it returns.

        The proposer decides how many hypotheses to return, and every one must compile:
        `Pool.add` raises `HypothesisError` otherwise. It need not remove duplicates of the pool,
        which `Pool.add` drops. An empty proposal is valid: the agent goes on with the pool it has.
        """
        ...


class NullProposer:
    """Proposes nothing: the pool stays the seed library (the fixed-library arm, spec §8)."""

    def propose(
        self,
        kind: ProposalKind,
        pool: Sequence[Candidate],
        X: Conditions,
        y: Rates,
        *,
        weights: PerModel | None,
        budget_left: int,
    ) -> Proposal:
        return Proposal()


# The prompt (spec §9, D6)

TEMPLATE_PATH = Path(__file__).parent / "prompts" / "chem_propose.txt"

# Significant digits of every number in the prompt: the observations and the residuals (D6).
# The observations carry 1% noise, so a fifth digit would be noise, and rounding to four moves a
# value by at most 0.05%
_DIGITS = 4


@dataclass(frozen=True)
class _Template:
    """The fixed lines of `chem_propose.txt`, which the prompts are assembled from."""

    system: str
    context: str  # the world description
    experiments: str  # "Experiments (inputs -> r0):"
    tried: str  # "Forms ALREADY TRIED ...", which opens the refining section
    refine: str  # "Refine by proposing a DIFFERENT ...", which closes it
    status: str  # the budget/phase line, with the placeholders <B> and <explore|refine>
    request: str  # "Propose N distinct ...", with the placeholder N
    spec: str  # the expression language and the reply format


def _load_template(text: str) -> _Template:
    """Split the template into its fixed lines, checking its layout.

    The file is App. G.1 verbatim: a `SYSTEM:` block, a `USER (...)` line describing the order of
    the parts, then the parts separated by blank lines. A line in square brackets says when the
    part after it is present, and an indented line in angle brackets stands for content. Neither
    is sent. Raises `ValueError` when the file does not have this layout, so that an edit to it
    cannot silently drop a part of the prompt.
    """
    blocks = [b.split("\n") for b in text.strip("\n").split("\n\n")]
    shape = [len(b) for b in blocks]
    if shape != [2, 1, 1, 2, 4, 2, 2]:
        raise ValueError(f"unexpected layout of {TEMPLATE_PATH.name}: block sizes {shape}")
    system, user, (context,), experiments, refining, status, request = blocks
    layout_ok = (
        system[0] == "SYSTEM:"
        and user[0].startswith("USER (")
        and _is_placeholder(experiments[1])
        and _is_marker(refining[0])
        and _is_placeholder(refining[2])
        and _is_marker(status[0])
        and "<B>" in status[1]
        and "<explore|refine>" in status[1]
        and len(re.findall(r"\bN\b", request[0])) == 1
    )
    if not layout_ok:
        raise ValueError(f"unexpected layout of {TEMPLATE_PATH.name}")
    return _Template(
        system=system[1],
        context=context,
        experiments=experiments[0],
        tried=refining[1],
        refine=refining[3],
        status=status[1],
        request=request[0],
        spec=request[1],
    )


def _is_marker(line: str) -> bool:
    return line.startswith("[") and line.endswith("]")


def _is_placeholder(line: str) -> bool:
    return line.startswith("  <") and line.endswith(">")


_TEMPLATE = _load_template(TEMPLATE_PATH.read_text())

SYSTEM_PROMPT = _TEMPLATE.system
"""The system prompt of `chem_propose.txt`, the same for every call."""


def render_prompt(
    kind: ProposalKind,
    pool: Sequence[Candidate],
    X: Conditions,
    y: Rates,
    *,
    weights: PerModel | None,
    budget_left: int,
    n: int,
    tau_fit: float = DEFAULTS.tau_fit,
    phase: bool = DEFAULTS.prompt_phase,
) -> str:
    """The user prompt of `chem_propose.txt` for one proposer call, asking for `n` laws (D6).

    The parts, separated by blank lines, are the world description; the experiments, one line
    per observation in the order of the data; at an expansion only, the refining section and,
    with `phase`, the budget/phase line; then the request for `n` laws and the expression
    language. Every fixed word is the template's.

    The refining section lists every form of the pool, best first (see `map_index`), each with
    its in-sample median relative residual and the input its residual correlates with most (see
    `correlated_input`); the best form's line adds the template's advice to consider a factor in
    that input. The phase is "refine" when the best form's median
    relative residual is at most `tau_fit`, and "explore" otherwise, including when there is no
    best form. Numbers have 4 significant digits.

    `pool`, `weights` and `budget_left` are as `Proposer.propose` receives them; at
    initialisation only `X`, `y` and `n` are used. Raises `ValueError` for an expansion without
    one weight per candidate, or with a candidate that has no fit.
    """
    t = _TEMPLATE
    parts = [t.context, "\n".join([t.experiments, *_experiment_lines(X, y)])]
    if kind == "expand":
        if weights is None or len(weights) != len(pool):
            raise ValueError("an expansion prompt needs one weight per candidate of the pool")
        best = map_index(weights)
        parts.append("\n".join([t.tried, *_form_lines(pool, weights, best, X, y), t.refine]))
        if phase:
            stage = "refine" if best is not None and pool[best].fit_err <= tau_fit else "explore"
            parts.append(
                t.status.replace("<B>", str(budget_left)).replace("<explore|refine>", stage)
            )
    elif kind != "init":
        raise ValueError(f"unknown proposal kind {kind!r}")
    request = re.sub(r"\bN\b", str(n), t.request, count=1)
    parts.append("\n".join([request, t.spec]))
    return "\n\n".join(parts)


def map_index(weights: PerModel) -> int | None:
    """The index of the MAP candidate, as `MDAAgent` chooses it: the largest weight, the first
    candidate on ties, and None when every weight is 0 (no candidate has a finite fit)."""
    if not np.any(weights > 0):
        return None
    return int(np.argmax(weights))


def correlated_input(X: Conditions, y: Rates, y_hat: Rates) -> str | None:
    """The input the signed residual y - y_hat correlates with most (D6, App. G).

    The input with the largest absolute Spearman correlation (`residual_input_spearman`), leaving
    out those whose correlation is 0 or NaN: a constant input, a constant residual or a non-finite
    one. Ties go to the input earlier in `INPUT_NAMES`. None when no input correlates at all.
    """
    rho = residual_input_spearman(X, y, y_hat)
    strength = np.where(np.isfinite(rho), np.abs(rho), 0.0)
    j = int(np.argmax(strength))  # the first of the largest
    return INPUT_NAMES[j] if strength[j] > 0 else None


def _num(v: float) -> str:
    return f"{v:.{_DIGITS}g}"


def _experiment_lines(X: Conditions, y: Rates) -> list[str]:
    """One line per observation: "  C_A=.., C_I=.., ... -> r0=..", as the template shows it."""
    lines = []
    for x, r in zip(X, y, strict=True):
        inputs = ", ".join(f"{name}={_num(v)}" for name, v in zip(INPUT_NAMES, x, strict=True))
        lines.append(f"  {inputs} -> r0={_num(r)}")
    return lines


def _form_lines(
    pool: Sequence[Candidate], weights: PerModel, best: int | None, X: Conditions, y: Rates
) -> list[str]:
    """The pool's forms, by decreasing weight (ties in pool order).

    A form is its dedupe key (D13): one line, whatever the proposer's formatting, and exactly
    what counts as re-proposing it. A form with a finite fit shows its residual and, when it
    correlates with any input, the input it correlates with most (App. G: "which input each
    residual correlates most with"). A form whose fit failed has no residual to show and says
    so. The best form is marked, and its line carries the template's advice to consider a factor
    in its input, reworded for one input ("that variable" for "those variable(s)").
    """
    lines = []
    for i in np.argsort(-weights, kind="stable"):
        c = pool[i]
        if c.fit is None:
            raise ValueError(f"candidate {c.hypothesis.name!r} has no fit")
        if math.isfinite(c.fit_err):
            line = f"  {c.key} -- median relative residual {_num(c.fit_err)}"
            if (name := correlated_input(X, y, c.fit.y_hat)) is not None:
                line += f"; its residual correlates most with {name}"
                if i == best:
                    line += " -- consider a factor in that variable"
        else:
            line = f"  {c.key} -- fit failed (no finite prediction on the data)"
        lines.append(line + (" (current best form)" if i == best else ""))
    return lines


# The proposer (spec §4, D14, D15, D17)


class LLMProposer:
    """Proposes hypotheses by asking an LLM, for one (arm, world, seed) cell.

    Each `propose` call renders the prompt (`render_prompt`), asking for `config.n_init` laws at
    initialisation and `config.n_new` at an expansion, and sends it with `SYSTEM_PROMPT`, and with
    `HYPOTHESES_SCHEMA` when `config.structured_output` is on (D15).

    A reply is unusable when it is not `ok` (cut at `max_tokens`, refused, or stopped for another
    reason) or `parse_hypotheses` finds no hypotheses list in it. Then the call is made again, up
    to `config.max_attempts` attempts, after which the proposal is empty: the agent goes on with
    the pool it has, at initialisation the seed library (D14). A usable reply is never retried,
    even when it yields nothing new. Its valid hypotheses are deduplicated against the pool and
    within the reply by expression (D13), and the first N kept, in reply order.

    Call IDs are `<call_prefix>/call<kk>/try<a>`: `kk` is the index of the `propose` call in the
    cell, from 00 for the initialisation, and `a` the attempt, from 1. Every attempt therefore
    has its own reply-cache entry (D17), and a rerun with the same cache replays the same
    sequence of attempts. `Proposal.calls` holds one `CallRecord` per attempt, in order. A call
    asking for 0 laws makes no request, but still takes its index.

    One instance serves one cell: the first call must be the initialisation, and there is only
    one. `LLMError` from the client, a request that failed, is raised.
    """

    def __init__(self, llm: LLMClient, *, call_prefix: str, config: MDAConfig = DEFAULTS) -> None:
        """`call_prefix` names the cell, as `<arm>/<world>/s<seed>`."""
        self._llm = llm
        self._call_prefix = call_prefix
        self._config = config
        self._n_calls = 0

    @property
    def llm(self) -> LLMClient:
        return self._llm

    @property
    def call_prefix(self) -> str:
        return self._call_prefix

    @property
    def config(self) -> MDAConfig:
        return self._config

    @property
    def n_calls(self) -> int:
        """The number of `propose` calls so far, which is the index of the next one."""
        return self._n_calls

    def propose(
        self,
        kind: ProposalKind,
        pool: Sequence[Candidate],
        X: Conditions,
        y: Rates,
        *,
        weights: PerModel | None,
        budget_left: int,
    ) -> Proposal:
        cfg = self._config
        if kind not in ("init", "expand"):
            raise ValueError(f"unknown proposal kind {kind!r}")
        if (kind == "init") != (self._n_calls == 0):
            raise RuntimeError(
                "an LLMProposer serves one cell: its first call, and only that one, is 'init'"
            )
        k = self._n_calls
        self._n_calls += 1
        n = cfg.n_init if kind == "init" else cfg.n_new
        if n == 0:
            return Proposal()

        user = render_prompt(
            kind,
            pool,
            X,
            y,
            weights=weights,
            budget_left=budget_left,
            n=n,
            tau_fit=cfg.tau_fit,
            phase=cfg.prompt_phase,
        )
        schema = HYPOTHESES_SCHEMA if cfg.structured_output else None
        calls: list[CallRecord] = []
        for attempt in range(1, cfg.max_attempts + 1):
            call_id = f"{self._call_prefix}/call{k:02d}/try{attempt}"
            reply = self._llm.complete(SYSTEM_PROMPT, user, schema=schema, call_id=call_id)
            calls.append(CallRecord(call_id, reply.request_id))
            if not reply.ok:
                continue
            try:
                hypotheses, _ = parse_hypotheses(reply.text)
            except ProposalParseError:
                continue
            return Proposal(_new_hypotheses(hypotheses, pool, n), tuple(calls))
        return Proposal((), tuple(calls))


def _new_hypotheses(
    hypotheses: Sequence[Hypothesis], pool: Sequence[Candidate], n: int
) -> tuple[Hypothesis, ...]:
    """The first `n` of `hypotheses` whose expression is in neither the pool nor an earlier one
    of `hypotheses` (D13). Duplicates are dropped before the cap, so they do not take places."""
    seen = {c.key for c in pool}
    kept: list[Hypothesis] = []
    for h in hypotheses:
        key = normalise_expr(h.expr)
        if key in seen:
            continue
        seen.add(key)
        kept.append(h)
    return tuple(kept[:n])


def claude_proposer(
    config: MDAConfig,
    *,
    call_prefix: str,
    cache_dir: str | os.PathLike[str] | None = None,
    transcript: str | os.PathLike[str] | None = None,
) -> LLMProposer:
    """An `LLMProposer` over a `ClaudeLLM` built from the config's `llm_*` fields (spec §9).

    For the hybrid arm. A client cannot be pickled, so with multiprocessing each worker calls
    this for its own cells. `cache_dir` and `transcript` are passed to the client (D17).
    """
    llm = ClaudeLLM(
        config.llm_model,
        effort=config.llm_effort,
        max_tokens=config.llm_max_tokens,
        max_retries=config.llm_max_retries,
        cache_dir=cache_dir,
        transcript=transcript,
    )
    return LLMProposer(llm, call_prefix=call_prefix, config=config)
