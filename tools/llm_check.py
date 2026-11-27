"""Check the Claude client by hand: count proposer-prompt tokens, or make one real call.

    uv run python tools/llm_check.py count   # free: the official count_tokens endpoint
    uv run python tools/llm_check.py call    # one real call (under $0.01), cached under cache/

Both read ANTHROPIC_API_KEY from `.env`. The prompts are rendered by `mda.propose.render_prompt`
from realistic data, the observations of one world. `count` measures the smallest prompt (the
init call, 3 observations, no refining section) and the largest (an expansion at the last step:
25 observations and a full pool of 24 fitted forms, with the budget/phase line).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from mda.bench import World, WorldSpec, from_unit
from mda.config import DEFAULTS
from mda.hypothesis import HYPOTHESES_SCHEMA, Hypothesis, Param, parse_hypotheses
from mda.library import SEED_LIBRARY
from mda.llm import ClaudeLLM, LLMClient
from mda.pool import Origin, Pool
from mda.posterior import bic_profiled, weights
from mda.propose import SYSTEM_PROMPT, render_prompt
from mda.run import passive_design

# The repository root: `cache/` and `runs/` are git-ignored there, wherever the tool is run from
ROOT = Path(__file__).resolve().parents[1]
WORLD = "c10_mm_competitive_arrhenius"


def data(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`n` observations of `WORLD`: D0, then random conditions standing in for designed ones."""
    world = World(WorldSpec(WORLD))
    X = np.vstack([passive_design(), from_unit(np.random.default_rng(0).uniform(size=(n - 3, 7)))])
    return X, np.array([world.observe(x) for x in X])


def pool_forms(n: int) -> list[Hypothesis]:
    """`n` distinct forms: the seeds, then the seeds with an extra inhibition or Arrhenius factor."""
    extras = [
        ("inhibited", "/ (1 + C_I/Ki)", Param("Ki", 1e-3, 1e3)),
        ("arrhenius", "* exp(-Ea/(8.314*T))", Param("Ea", 1e3, 2e5)),
    ]
    forms = list(SEED_LIBRARY)
    for tag, factor, param in extras:
        for h in SEED_LIBRARY:
            # A seed that already has the parameter shares it with the factor
            params = h.params + tuple(
                p for p in [param] if p.name not in {q.name for q in h.params}
            )
            forms.append(Hypothesis(f"{h.name}_{tag}", f"({h.expr}) {factor}", params))
    return forms[:n]


def init_prompt() -> str:
    X, y = data(3)
    # The init prompt does not show the pool
    return render_prompt(
        "init", (), X, y, weights=None, budget_left=DEFAULTS.n_active, n=DEFAULTS.n_init
    )


def expand_prompt(n_obs: int = 25, n_forms: int = 24) -> str:
    """The prompt of an expansion after the last experiment, when no budget is left."""
    X, y = data(n_obs)
    pool = Pool()
    pool.add(pool_forms(n_forms), Origin.SEED, added_at=n_obs)
    pool.refit(X, y, rng=np.random.default_rng(0), n_starts=1)
    w = weights(bic_profiled(pool.rss, pool.n_free, n_obs))
    return render_prompt("expand", tuple(pool), X, y, weights=w, budget_left=0, n=DEFAULTS.n_new)


def count(llm: LLMClient) -> None:
    print(f"model: {llm.model}")
    print(f"{'prompt':<28} {'chars':>6} {'tokens':>7} {'+ schema':>9}")
    for name, user in [
        ("init (3 obs)", init_prompt()),
        ("expand (25 obs, 24 forms)", expand_prompt()),
    ]:
        plain = llm.count_tokens(SYSTEM_PROMPT, user)
        with_schema = llm.count_tokens(SYSTEM_PROMPT, user, schema=HYPOTHESES_SCHEMA)
        print(f"{name:<28} {len(SYSTEM_PROMPT) + len(user):>6} {plain:>7} {with_schema:>9}")


def call(llm: LLMClient) -> None:
    for attempt in ("first", "second"):
        reply = llm.complete(
            SYSTEM_PROMPT, init_prompt(), schema=HYPOTHESES_SCHEMA, call_id="smoke/init"
        )
        print(
            f"{attempt} call: cached={reply.cached} stop={reply.stop_reason} "
            f"request_id={reply.request_id} usage={reply.usage} cost=${reply.cost:.5f}"
        )
    if not reply.ok:
        sys.exit(f"unusable reply: stop_reason={reply.stop_reason} detail={reply.stop_detail}")
    hypotheses, rejected = parse_hypotheses(reply.text)
    print(f"{len(hypotheses)} valid hypotheses, {len(rejected)} rejected")
    for h in hypotheses:
        print(f"  {h.name}: {h.expr}")
    for r in rejected:
        print(f"  rejected: {r}")
    print(f"sent {llm.n_sent} request(s), total cost ${llm.cost:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["count", "call"])
    args = parser.parse_args()
    load_dotenv()
    llm = ClaudeLLM(cache_dir=ROOT / "cache", transcript=ROOT / "runs/smoke/transcript.jsonl")
    {"count": count, "call": call}[args.command](llm)


if __name__ == "__main__":
    sys.exit(main())
