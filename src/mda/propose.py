"""Proposers: where the agent's new hypotheses come from (spec §4, Alg. 11 and 12).

`MDAAgent` asks its proposer for hypotheses once at initialisation and once per expansion.
`NullProposer` proposes nothing, which gives the fixed-library arm: the seed library alone, with
no LLM call (spec §8).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from mda.arrays import Conditions, PerModel, Rates
from mda.hypothesis import Hypothesis
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
